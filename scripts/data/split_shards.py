"""Split each shard into K shards of consecutive samples (<stem>_00.tar ... <stem>_{K-1}.tar).

Sample i of n goes to part i // ceil(n / K). The member bytes and headers (name, size,
mtime, mode) are copied; nothing is decoded. The paper splits each VQAv2 shard (10,000
rows) into 13 parts, so that the caption rows and the question rows mix more evenly in
each batch.

    python -m scripts.data.split_shards --parts 13 --out data/wds/vqav2_train_v2_s13 data/wds/vqav2_train_v2/vqav2_train_000*.tar
"""

from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path


def split_tar(src: Path, out_dir: Path, k: int) -> list[int]:
    """Split one tar into k parts. Returns the samples per part."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(src, "r|") as t:
        members = [(m, t.extractfile(m).read() if m.isfile() else None) for m in t]
    keys: list[str] = []
    for m, _ in members:
        key = m.name.rsplit(".", 1)[0]
        if not keys or keys[-1] != key:
            keys.append(key)
    per = (len(keys) + k - 1) // k
    part = {key: i // per for i, key in enumerate(keys)}
    outs = [tarfile.open(out_dir / f"{src.stem}_{j:02d}.tar", "w") for j in range(k)]
    counts = [0] * k
    for m, data in members:
        j = part[m.name.rsplit(".", 1)[0]]
        info = tarfile.TarInfo(name=m.name)
        info.size = len(data) if data is not None else 0
        info.mtime, info.mode = m.mtime, m.mode
        outs[j].addfile(info, io.BytesIO(data) if data is not None else None)
    for o in outs:
        o.close()
    for key in keys:
        counts[part[key]] += 1
    return counts


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("tars", nargs="+")
    a = ap.parse_args(argv)
    for src in a.tars:
        print(f"{Path(src).name}: {split_tar(Path(src), Path(a.out), a.parts)}", flush=True)


if __name__ == "__main__":
    main()
