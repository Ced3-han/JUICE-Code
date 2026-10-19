"""Build the probe shards of one question-answering benchmark from its public files.

    python scripts/eval/build_vqa_probes.py --bench okvqa --src data/OK-VQA/data \
        --images data/coco/val2014 --out probes

Writes <out>/<bench>/probe_shard{0..3}.json (VQAv2 n1000: probe_shard0.json only),
<out>/<bench>/drops.json and, for benchmarks with reference answers,
<out>/<bench>/annotations.json. See docs/EVAL_VQA.md for the --src folder of each benchmark.

--images: the COCO folder for okvqa, pope, vqav2_n1000 (val2014) and vqav2_testdev
(test2015). For the other benchmarks the images are exported from the parquet files
into this folder (default <out>/<bench>/images). Probe items keep the image path as
written here, so give an absolute path if the probes move.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from juice.eval.vqa_probes import BENCHES, MCQ_BENCHES, build, load_tokenizer
from juice.pipeline import SHORT_ANSWER
from juice.representations import TEXT_CANVAS, TEXT_ENCODER_ID, TEXT_ENCODER_REVISION


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, choices=BENCHES)
    ap.add_argument("--src", required=True, help="folder with the downloaded benchmark files")
    ap.add_argument("--images", default=None, help="COCO image folder, or the export folder of the images")
    ap.add_argument("--coco-val2017", default=None, help="vqav2_n1000 only: the COCO val2017 image folder (5,000 files)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out) / args.bench
    images = args.images or str(out / "images")
    if args.bench in ("okvqa", "pope", "vqav2_n1000", "vqav2_testdev") and not args.images:
        ap.error(f"--images (the COCO folder) is required for {args.bench}")
    if args.bench == "vqav2_n1000" and not args.coco_val2017:
        ap.error("--coco-val2017 is required for vqav2_n1000")

    result = build(args.bench, args.src, images, load_tokenizer(), coco_val2017=args.coco_val2017)
    shards, drops = result["shards"], result["drops"]
    n_kept = sum(len(s) for s in shards)
    funnel = {"total": n_kept + len(drops), "kept": n_kept, "dropped": len(drops),
              "dropped_by_reason": dict(Counter(d["reason"] for d in drops))}
    meta = {
        "bench": args.bench,
        "tokenizer": f"{TEXT_ENCODER_ID}@{TEXT_ENCODER_REVISION}",
        "prompt_length": TEXT_CANVAS,
        "question_suffix": "" if args.bench in MCQ_BENCHES else SHORT_ANSWER,
        "funnel": funnel,
        "n_shards": len(shards),
    }
    out.mkdir(parents=True, exist_ok=True)
    for k, items in enumerate(shards):
        (out / f"probe_shard{k}.json").write_text(json.dumps({"meta": {**meta, "shard": k, "n": len(items)}, "items": items}))
    (out / "drops.json").write_text(json.dumps({"funnel": funnel, "drops": drops}))
    if result["annotations"] is not None:
        (out / "annotations.json").write_text(json.dumps({"annotations": result["annotations"]}))
    print(json.dumps({"bench": args.bench, "out": str(out), **funnel, "shards": [len(s) for s in shards]}))


if __name__ == "__main__":
    main()
