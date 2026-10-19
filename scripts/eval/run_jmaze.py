"""Evaluate a JMaze release folder on a puzzle split.

For each puzzle the model gets the input maze image and the question, and makes
the solved image and the move string together (16 Euler steps, no guidance).
Metric of the paper: joint_consistent = the text is a valid shortest route, the
image shows a valid shortest route, and both are the same route. The paper reports
the mean over noise seeds 1 and 2 on test_id (500 puzzles).

    python -m scripts.eval.run_jmaze --model my-jmaze-siglip2 --puzzles data/jmaze/test_id.jsonl \
        --seeds 1 2 --out outputs/jmaze_siglip2

Writes <out>/seed<k>.json (summary + one row per puzzle) and <out>/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from juice.tasks import maze
from juice.tasks.planning import Planner, noise_generator, summarize

KEYS = ("joint_consistent", "text_valid", "img_valid", "joint_valid", "agree", "joint",
        "text_exact", "img_exact", "text_parse_fail", "img_block_acc", "img_wall_acc")


def evaluate(planner: Planner, puzzles: list[dict], seed: int, steps: int | None = None) -> dict:
    rows, batch_size, t0 = [], int(planner.settings["batch_size"]), time.time()
    for start in range(0, len(puzzles), batch_size):
        batch = puzzles[start:start + batch_size]
        images = [maze.render_maze(p["blocks"], p["start"], p["goal"]) for p in batch]
        known = [maze.known_patch_flags(p["blocks"], p["start"], p["goal"]) for p in batch]
        out = planner.solve(images, [p["question"] for p in batch], known, noise_generator(seed, start), steps)
        for p, answer, image in zip(batch, out.answers, out.images):
            labels = maze.parse_blocks_from_image(image, int(p["n"]))
            scores = maze.score_puzzle(p["blocks"], p["start"], p["goal"], labels, maze.parse_answer(answer))
            rows.append({"id": int(p["id"]), "gold_answer": p["answer"], "pred_answer": answer, **scores})
        print(f"seed {seed}: {len(rows)}/{len(puzzles)} joint_consistent so far "
              f"{sum(r['joint_consistent'] for r in rows) / len(rows):.4f} ({time.time() - t0:.0f}s)", flush=True)
    by_size = {}
    for r in rows:
        by_size.setdefault(r["n"], []).append(r)
    return {"seed": seed, "steps": steps or int(planner.settings["steps"]), "summary": summarize(rows, KEYS),
            "per_size": {str(n): summarize(rs, KEYS) for n, rs in sorted(by_size.items())}, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="release folder (model + task_head)")
    ap.add_argument("--puzzles", required=True, help="split jsonl from scripts/data/make_jmaze_data.py")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--steps", type=int, default=None, help="Euler steps (default: the folder's setting, 16)")
    ap.add_argument("--n", type=int, default=0, help="first n puzzles only (0 = all)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    puzzles = [json.loads(line) for line in Path(args.puzzles).read_text().splitlines() if line.strip()]
    puzzles = puzzles[: args.n] if args.n else puzzles
    planner = Planner(args.model, args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    per_seed = {}
    for seed in args.seeds:
        result = evaluate(planner, puzzles, seed, args.steps)
        result.update(model=str(args.model), puzzles=str(args.puzzles))
        (args.out / f"seed{seed}.json").write_text(json.dumps(result, indent=2))
        per_seed[seed] = result["summary"]["joint_consistent"]
        print(f"seed {seed}: joint_consistent = {per_seed[seed]:.4f}", flush=True)
    metrics = {"metric": "joint_consistent", "per_seed": per_seed,
               "mean": round(sum(per_seed.values()) / len(per_seed), 6), "n_puzzles": len(puzzles)}
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
