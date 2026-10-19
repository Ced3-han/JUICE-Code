"""Write the OpenDatasets/dalle-3-dataset parquets as text-to-image shards.

Input: the downloaded dataset folder (the parquets anywhere below it, sorted by path).
Five parquets make one shard: shard-000.tar holds parquets 0-4, and so on. A row gives
`image` (bytes, a path, base64 text or a byte list) and `caption` (else
`synthetic_caption`). Rows without image bytes are skipped and listed in errors.txt.

Sample (key = {image_hash}-{index:012d}; the index starts at shard * 1,000,000):
  <key>.jpg  the image bytes as stored in the parquet
  <key>.txt  the caption (stripped) + "\\n"

    python -m scripts.data.stage_dalle3 --src data/src/dalle-3-dataset --out data/wds/dalle3_t2i
"""

from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import tarfile
from math import ceil
from pathlib import Path


def coerce_image_bytes(raw, base_dir: Path) -> bytes | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        if raw.get("bytes") is not None:
            return coerce_image_bytes(raw["bytes"], base_dir)
        if raw.get("path"):
            path = base_dir / raw["path"]
            return path.read_bytes() if path.exists() else None
        return None
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return bytes(raw)
    if isinstance(raw, str):
        if raw.startswith("data:image"):
            try:
                return base64.b64decode(raw.split(",", 1)[1], validate=False)
            except Exception:
                return None
        try:
            return base64.b64decode(raw, validate=True)
        except Exception:
            pass
        try:
            literal = ast.literal_eval(raw)
            if isinstance(literal, (list, tuple)):
                return bytes(literal)
        except Exception:
            pass
    if isinstance(raw, (list, tuple)):
        try:
            return bytes(raw)
        except Exception:
            return None
    return None


def add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def main(argv=None) -> None:
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="the downloaded dataset folder")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    src, out = Path(a.src), Path(a.out)
    parquets = sorted(src.rglob("*.parquet"))
    if not parquets:
        raise SystemExit(f"no .parquet files under {src}")
    out.mkdir(parents=True, exist_ok=True)
    errors, per_shard = [], []
    for shard_idx in range(ceil(len(parquets) / 5)):
        shard_path = out / f"shard-{shard_idx:03d}.tar"
        if shard_path.exists():
            raise SystemExit(f"{shard_path} exists")
        written, global_idx = 0, shard_idx * 1_000_000
        with tarfile.open(shard_path.with_name(f".tmp_{shard_path.name}"), "w") as tar:
            for parquet_path in parquets[shard_idx * 5: (shard_idx + 1) * 5]:
                row_offset = 0
                for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=1024):
                    rows = batch.to_pylist()
                    for local_offset, row in enumerate(rows):
                        image_bytes = coerce_image_bytes(row.get("image"), src)
                        if not image_bytes:
                            errors.append(f"{parquet_path.name}:{row_offset + local_offset}:image")
                            continue
                        caption = str(row.get("caption") or row.get("synthetic_caption") or "")
                        key = f"{row.get('image_hash') or 'sample'}-{global_idx:012d}"
                        add(tar, f"{key}.jpg", image_bytes)
                        add(tar, f"{key}.txt", (caption.strip() + "\n").encode("utf-8"))
                        global_idx += 1
                        written += 1
                    row_offset += len(rows)
        shard_path.with_name(f".tmp_{shard_path.name}").rename(shard_path)
        per_shard.append(written)
        print(f"wrote {shard_path} ({written} samples)", flush=True)
    if errors:
        (out / "errors.txt").write_text("\n".join(errors) + "\n", encoding="utf-8")
    (out / "pack_summary.json").write_text(json.dumps(
        {"parquets": len(parquets), "per_shard": per_shard, "total_packed": sum(per_shard), "errors": len(errors),
         "tars": len(per_shard)}, indent=1))
    print(f"{sum(per_shard)} rows in {len(per_shard)} shards, {len(errors)} errors (paper: 19,024 rows, 14 shards)")


if __name__ == "__main__":
    main()
