"""Pack the ShareGPT-4o-Image text-to-image rows into training shards.

Input: the folder of FreedomIntelligence/ShareGPT-4o-Image with text_to_image.json and
text_to_image_part_*.tar. The PNG members are numbered in the order of the sorted part
tars; --shard K/N packs the members with number % N == K. The image is converted to RGB,
reduced to a longest side of 1024 px (LANCZOS) when it is larger, and saved as JPEG
quality 95. The caption is `input_prompt` of text_to_image.json (no trailing newline).

Sample (key = "sharegpt4o_" + PNG stem), 2,500 rows per tar:
  <key>.jpg, <key>.txt

    python -m scripts.data.pack_sharegpt4o --src data/src/ShareGPT-4o-Image --shard 0/8 --out data/wds/sharegpt4o_t2i
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

from scripts.data.common import parse_shard
from scripts.data.shards import ShardWriter


def jpeg_bytes(fileobj, max_side: int) -> bytes:
    from PIL import Image

    img = Image.open(fileobj).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    return buf.getvalue()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder with text_to_image.json + text_to_image_part_*.tar")
    ap.add_argument("--shard", default="0/1", help="K/N")
    ap.add_argument("--out", required=True, help="all shards write into this folder")
    ap.add_argument("--samples-per-tar", type=int, default=2500)
    ap.add_argument("--max-side", type=int, default=1024)
    a = ap.parse_args(argv)
    k, n = parse_shard(a.shard)
    src, out = Path(a.src), Path(a.out)
    if any(out.glob(f"sharegpt4o_s{k:02d}_*.tar")):
        raise SystemExit(f"{out} already holds shard {k}")
    prompt = {m["output_image"]: m["input_prompt"] for m in json.load(open(src / "text_to_image.json"))}
    writer = ShardWriter(out, f"sharegpt4o_s{k:02d}", a.samples_per_tar)
    i, written, missing_prompt = -1, 0, 0
    for part in sorted(src.glob("text_to_image_part_*.tar")):
        with tarfile.open(part) as tf:
            for m in tf:
                if not m.isfile() or not m.name.endswith(".png"):
                    continue
                i += 1
                if i % n != k:
                    continue
                p = prompt.get(m.name)
                if p is None:
                    missing_prompt += 1
                    continue
                writer.write("sharegpt4o_" + Path(m.name).stem,
                             {"jpg": jpeg_bytes(tf.extractfile(m), a.max_side), "txt": p.encode("utf-8")})
                written += 1
    summary = {"shard": a.shard, "written": written, "missing_prompt": missing_prompt, "n_meta": len(prompt),
               "max_side": a.max_side, "jpeg_q": 95, "tars": len(writer.close())}
    (out / f"summary_s{k:02d}.json").write_text(json.dumps(summary, indent=1))
    print(f"shard {a.shard}: written={written} missing_prompt={missing_prompt} (paper: 45,717 rows over 8 shards)")


if __name__ == "__main__":
    main()
