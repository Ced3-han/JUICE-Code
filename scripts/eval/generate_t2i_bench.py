"""Generate the GenEval or DPG-Bench images of the paper.

    python scripts/eval/generate_t2i_bench.py --benchmark geneval --model <folder or hub id> \
        --prompts geneval/prompts/evaluation_metadata.jsonl --out images/geneval
    python scripts/eval/generate_t2i_bench.py --benchmark dpg --model <folder or hub id> \
        --prompts ELLA/dpg_bench/dpg_bench.csv --out images/dpg

Protocol of the paper's table: 4 images per prompt in one batch with the
generator seed seed + 1000 * index (seed 0); the steps, the guidance scale, the generic
caption of the guidance reference and the precision of config.json ("defaults" /
"text_to_image": 25 steps, cfg 5.0, "a photo.", fp32); native 224 x 224 images, no upsampling. The seeds do not
depend on --num-shards. A prompt with complete output files is skipped, unless you
give --overwrite. The file gen_protocol.json in --out records the settings; the script
stops if you resume into a folder with different settings.

--upsample N saves LANCZOS-resized N x N images. The paper did not use it.
--indices selects prompt indices (for checks); the seeds stay the same.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from juice.eval.t2i_bench import (
    SAMPLES_PER_PROMPT,
    load_dpg_prompts,
    load_geneval_prompts,
    output_complete,
    prompt_seed,
    sha256_file,
    shard_items,
    write_dpg,
    write_geneval,
)


def check_or_write_protocol(out: Path, protocol: dict, overwrite: bool) -> None:
    path = out / "gen_protocol.json"
    if path.exists() and not overwrite:
        old = json.loads(path.read_text())
        if old != protocol:
            diff = {k: [old.get(k), protocol.get(k)] for k in sorted(set(old) | set(protocol)) if old.get(k) != protocol.get(k)}
            raise SystemExit(f"{path} records other settings (old, new): {diff}. Use a new --out or --overwrite.")
        return
    out.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".gen_protocol.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(protocol, indent=2, sort_keys=True))
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, choices=["geneval", "dpg"])
    ap.add_argument("--model", required=True, help="model folder or Hugging Face id")
    ap.add_argument("--prompts", required=True,
                    help="GenEval: evaluation_metadata.jsonl; DPG-Bench: dpg_bench.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None, help="default: config.json (25)")
    ap.add_argument("--cfg-scale", type=float, default=None, help="default: config.json (5.0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--upsample", type=int, default=0, help="0 = native 224 (the paper)")
    ap.add_argument("--indices", type=int, nargs="+", default=None, help="only these prompt indices (checks)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    load = load_geneval_prompts if args.benchmark == "geneval" else load_dpg_prompts
    items = load(args.prompts)
    if args.indices is not None:
        items = [items[i] for i in sorted(set(args.indices))]
    local = shard_items(items, args.num_shards, args.shard)
    out = Path(args.out)

    from juice.pipeline import JuicePipeline  # after the argument checks: loading takes time

    pipe = JuicePipeline.from_pretrained(args.model, device=args.device)
    d = pipe.defaults["text_to_image"]
    steps = args.steps or d["steps"]
    cfg_scale = d["cfg_scale"] if args.cfg_scale is None else args.cfg_scale
    check_or_write_protocol(out, {
        "benchmark": args.benchmark, "model_source": pipe.config.get("source"),
        "prompts_sha256": sha256_file(args.prompts), "steps": steps, "cfg_scale": cfg_scale,
        "null_caption": d["null_caption"], "precision": d["precision"], "seed": args.seed,
        "samples_per_prompt": SAMPLES_PER_PROMPT, "upsample": args.upsample,
    }, args.overwrite)
    if not args.overwrite:
        local = [it for it in local if not output_complete(args.benchmark, out, it)]
    print(f"{args.benchmark}: {len(local)} prompts to make (shard {args.shard}/{args.num_shards}, "
          f"{len(items)} selected), steps {steps}, cfg {cfg_scale}, seed {args.seed}", flush=True)

    for n, item in enumerate(local, start=1):
        t0 = time.time()
        images = pipe.text_to_image(item.prompt, num_images=SAMPLES_PER_PROMPT,
                                    seed=prompt_seed(args.seed, item.index), steps=steps, cfg_scale=cfg_scale)
        if args.upsample:
            from PIL import Image

            images = [im.resize((args.upsample, args.upsample), Image.LANCZOS) for im in images]
        if args.benchmark == "geneval":
            write_geneval(out, item, images)
        else:
            write_dpg(out, item, images)
        print(f"[{n}/{len(local)}] {item.item_id} {time.time() - t0:.1f}s", flush=True)
    print(f"done: {len(local)} prompts -> {out}", flush=True)


if __name__ == "__main__":
    main()
