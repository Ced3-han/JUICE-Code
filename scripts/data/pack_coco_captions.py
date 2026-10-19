"""Pack the COCO train2014 human captions into training shards (5 rows per image).

Only train2014 is used: it has no image of val2014 or val2017 (the packer checks this with
the file names of the two val folders; train2017 is not usable, it contains val2014 images).
Images are written in image-id order, their captions in annotation order.

Sample (key = {image_id:012d}_{caption index}), 2,000 rows per tar:
  <key>.jpg  the train2014 image bytes
  <key>.txt  the caption (stripped, newlines replaced by spaces) + "\\n"
There is no json sidecar: these rows are plain caption rows.

    python -m scripts.data.pack_coco_captions --captions data/src/coco/annotations/captions_train2014.json \\
        --images-dir data/src/coco/train2014 --val2014-dir data/src/coco/val2014 \\
        --val2017-dir data/src/coco/val2017 --out data/wds/coco_train2014_caption
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from scripts.data.common import empty_out_dir, write_summary
from scripts.data.shards import ShardWriter


def image_ids(folder: str) -> set[int]:
    return {int(os.path.splitext(f)[0].split("_")[-1]) for f in os.listdir(folder) if f.endswith(".jpg")}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captions", required=True, help="annotations/captions_train2014.json")
    ap.add_argument("--images-dir", required=True, help="train2014 image folder")
    ap.add_argument("--val2014-dir", required=True)
    ap.add_argument("--val2017-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples-per-tar", type=int, default=2000)
    ap.add_argument("--limit-images", type=int, default=0, help="smoke test: first N images only")
    args = ap.parse_args(argv)

    ann = json.loads(Path(args.captions).read_text())
    caps: dict[int, list[str]] = defaultdict(list)
    for a in ann["annotations"]:
        caps[int(a["image_id"])].append(a["caption"].strip().replace("\n", " "))
    id2file = {int(im["id"]): im["file_name"] for im in ann["images"]}
    train_ids = sorted(id2file)
    for name, folder in (("val2014", args.val2014_dir), ("val2017", args.val2017_dir)):
        val = image_ids(folder)
        if len(val) < 1000:
            raise SystemExit(f"{folder}: only {len(val)} images; give the {name} image folder")
        if val.intersection(train_ids):
            raise SystemExit(f"{len(val.intersection(train_ids))} caption images are in {name}")
    if args.limit_images:
        train_ids = train_ids[: args.limit_images]

    out_dir = empty_out_dir(args.out)
    writer = ShardWriter(out_dir, "coco_train2014_cap", args.samples_per_tar)
    n_rows = n_imgs = 0
    missing: list[int] = []
    for iid in train_ids:
        p = Path(args.images_dir) / id2file[iid]
        if not p.exists():
            missing.append(iid)
            continue
        img = p.read_bytes()
        n_imgs += 1
        for ci, c in enumerate(caps[iid]):
            if c:
                writer.write(f"{iid:012d}_{ci}", {"jpg": img, "txt": (c + "\n").encode("utf-8")})
                n_rows += 1
    counts = {"images_in_annotations": len(id2file), "images_packed": n_imgs, "missing_images": len(missing),
              "packed": n_rows}
    write_summary(out_dir, {"counts": counts, "missing_sample": missing[:10], "total_packed": n_rows,
                            "tars": len(writer.close())})


if __name__ == "__main__":
    main()
