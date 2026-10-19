"""Answer the questions of one probe shard with JUICE.

    python scripts/eval/run_vqa.py --model <folder or hub id> \
        --probe probes/gqa/probe_shard0.json --out answers/gqa/shard0.json

Protocol of the paper's table: one process per shard; items in shard order,
batches of 128; batch generator torch.Generator().manual_seed(seed + start), where
start is the index of the first item of the batch in the shard; seed 0; 16 steps,
cfg 3, the precision of config.json; answers read by the parallel text head with the
first-EOS rule. A different batch size or shard split gives other noise, so other
(equally valid) answers.

To use more GPUs than shards, split a shard by batches (--skip-batches, --max-batches):
the seed of a batch depends only on its start index, so the answers do not change.

The question prefix is the probe's token ids [0, answer_tok_start). --save-states
writes the sampled text states of each batch (float16, batch_<start>.pt).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from juice.pipeline import JuicePipeline
from juice.representations import load_image


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True, help="probe_shard<k>.json")
    ap.add_argument("--out", required=True, help="answers json")
    ap.add_argument("--image-root", default="", help="folder that relative image paths start from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--skip-batches", type=int, default=0, help="start at this batch of the shard (split a shard over GPUs)")
    ap.add_argument("--max-batches", type=int, default=0, help="stop after this many batches (0 = all)")
    ap.add_argument("--save-states", default=None, help="folder for the sampled text states")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    probe = json.loads(Path(args.probe).read_text())
    items = probe["items"]
    pipe = JuicePipeline.from_pretrained(args.model, device=args.device)
    if args.save_states:
        Path(args.save_states).mkdir(parents=True, exist_ok=True)
    rows = []
    starts = list(range(0, len(items), args.batch))[args.skip_batches:]
    if args.max_batches:
        starts = starts[: args.max_batches]
    for start in starts:
        t0 = time.time()
        batch = items[start:start + args.batch]
        prefixes = [it["token_ids_64"][: it["answer_tok_start"]] for it in batch]
        pixels = torch.stack([load_image(Path(args.image_root) / it["image_file"]) for it in batch])
        states = pipe.answer_states(pixels, prefixes, torch.Generator().manual_seed(args.seed + start))
        answers = pipe.read_answers(states, [len(p) for p in prefixes])
        rows += [{"question_id": it["question_id"], "answer": a} for it, a in zip(batch, answers)]
        if args.save_states:
            torch.save({"start": start, "question_ids": [it["question_id"] for it in batch],
                        "txt": states.to(torch.float16).cpu()}, Path(args.save_states) / f"batch_{start:06d}.pt")
        print(f"batch {start}: {len(batch)} items {time.time() - t0:.1f}s", flush=True)

    d = pipe.defaults["answer"]
    meta = {"model": args.model, "source": pipe.config.get("source"), "probe": str(args.probe),
            "bench": probe.get("meta", {}).get("bench"), "shard": probe.get("meta", {}).get("shard"),
            "seed": args.seed, "batch": args.batch, "steps": d["steps"], "cfg_scale": d["cfg_scale"],
            "precision": d["precision"], "n_items": len(items), "n_answered": len(rows),
            "skip_batches": args.skip_batches, "max_batches": args.max_batches}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"meta": meta, "rows": rows}))
    print(f"wrote {len(rows)} answers -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
