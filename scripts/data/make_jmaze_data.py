"""Generate the JMaze data: puzzle splits and training shards.

Splits (the settings of the paper are the defaults):
  train     200,000 mazes, sizes 5, 6, 7 in turn
  test_id   500 mazes, sizes 5, 6, 7, no (maze, start, goal) of train   <- evaluation split
  test_ood  500 slots, sizes 3, 4, no (maze, start, goal) of train (some slots stay empty)

Each puzzle slot draws its own random stream from sha256(seed | split | index | attempt),
so the output does not depend on the number of workers.

Outputs under --out:
  <split>.jsonl                       one puzzle per line (juice.tasks.maze.make_candidate)
  shards/<split>/jmaze_<split>_*.tar  with --shards: WebDataset samples, 10,000 per tar:
      <key>.png        solved image (blue route) = image target
      <key>.input.png  input image (no route)    = source of the known patches
      <key>.txt        "Question: <q> Answer: <a>"
      <key>.json       answer_tok_start, answer_tok_end, n_content_tokens (the answer span
                       in content tokens), img_known_patches (256 ints, 1 = known patch),
                       and the puzzle (id, n, blocks, start, goal, moves, ...)

    python -m scripts.data.make_jmaze_data --out data/jmaze --workers 8 --shards train
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

from juice.tasks import maze

_TOKENIZER = None


def _tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        from juice.representations import TEXT_CANVAS, TEXT_ENCODER_ID, TEXT_ENCODER_REVISION

        _TOKENIZER = AutoTokenizer.from_pretrained(TEXT_ENCODER_ID, revision=TEXT_ENCODER_REVISION, model_max_length=TEXT_CANVAS)
    return _TOKENIZER


def _train_slot(job: tuple):
    seed, index, n = job
    return maze.make_puzzle(seed, "train", index, n, _tokenizer())


def _sample(rec: dict) -> tuple[str, dict[str, bytes]]:
    from scripts.data.shards import png_bytes

    blocks, start, goal = tuple(rec["blocks"]), tuple(rec["start"]), tuple(rec["goal"])
    meta = {k: rec[k] for k in ("id", "n", "blocks", "start", "goal", "moves", "maze_hash", "topology",
                                "n_shortest_routes", "question", "answer", "answer_tok_start",
                                "answer_tok_end", "n_content_tokens")}
    meta["img_known_patches"] = maze.known_patch_flags(blocks, start, goal)
    return rec["key"], {
        "png": png_bytes(maze.render_maze(blocks, start, goal, [tuple(c) for c in rec["path_cells"]])),
        "input.png": png_bytes(maze.render_maze(blocks, start, goal)),
        "txt": (rec["caption"].strip() + "\n").encode("utf-8"),
        "json": json.dumps(meta, ensure_ascii=False).encode("utf-8"),
    }


def make_test_split(seed: int, split: str, sizes, count: int, forbidden: set) -> list[dict]:
    """Serial generation: a candidate whose key is in train or already in this split is rejected."""
    records, seen = [], set(forbidden)
    for index in range(count):
        rec = maze.make_puzzle(seed, split, index, sizes[index % len(sizes)], _tokenizer(), seen)
        if rec is not None:
            seen.add(rec["maze_hash"])
            records.append(rec)
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=200_000)
    ap.add_argument("--test-id", type=int, default=500)
    ap.add_argument("--test-ood", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shards", default="", help="comma-separated splits to pack into tar shards (e.g. train)")
    ap.add_argument("--samples-per-shard", type=int, default=10_000)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    _tokenizer()  # load once; forked workers inherit it
    t0 = time.time()
    jobs = [(args.seed, i, maze.TRAIN_SIZES[i % len(maze.TRAIN_SIZES)]) for i in range(args.n_train)]
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        train = [r for r in pool.imap(_train_slot, jobs, chunksize=256) if r is not None]
    print(f"train: {len(train)} puzzles ({time.time() - t0:.0f}s)", flush=True)
    train_keys = {r["maze_hash"] for r in train}
    splits = {
        "train": train,
        "test_id": make_test_split(args.seed, "test_id", maze.TRAIN_SIZES, args.test_id, train_keys),
        "test_ood": make_test_split(args.seed, "test_ood", maze.OOD_SIZES, args.test_ood, train_keys),
    }
    for split, records in splits.items():
        with (args.out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{split}: {len(records)} puzzles -> {args.out / f'{split}.jsonl'}", flush=True)

    from scripts.data.shards import ShardWriter

    for split in [s for s in args.shards.split(",") if s]:
        writer = ShardWriter(args.out / "shards" / split, f"jmaze_{split}", args.samples_per_shard)
        with ctx.Pool(args.workers) as pool:
            for key, members in pool.imap(_sample, splits[split], chunksize=64):
                writer.write(key, members)
        paths = writer.close()
        (args.out / "shards" / f"{split}.txt").write_text("".join(f"{p.resolve()}\n" for p in paths))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
