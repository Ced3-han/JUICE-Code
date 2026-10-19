"""Evaluate a VSP release folder on the official VSP maze maps (levels 3 to 8, 100 maps each).

Get the benchmark: clone https://github.com/UCSB-NLP-Chang/Visual-Spatial-Planning (MIT
License) and pass <repo>/maze/task-main/maps. The maps give the puzzles (table text) and
the cell sprites (images).

For each map the model gets the input map image and the question, and makes the
plan text and the solved image together (16 Euler steps, no guidance). Metric of
record: the official VSP success rule on the text plan, per level; the paper
reports the mean over levels 3 to 6 (noise seed 0).

    python -m scripts.eval.run_vsp --model my-vsp-siglip2 --vsp-maps VSP/maze/task-main/maps \
        --seed 0 --out outputs/vsp_siglip2

Writes <out>/seed<k>.json (summary + one row per map) and <out>/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from juice.tasks import vsp
from juice.tasks.planning import Planner, noise_generator, summarize

KEYS = ("text_success", "text_shortest", "text_exact", "text_parse_fail", "img_valid", "img_exact",
        "img_block_acc", "agree", "joint", "joint_valid", "joint_consistent")


def evaluate(planner: Planner, records: list[dict], tiles: dict, seed: int, steps: int | None = None) -> dict:
    rows, batch_size, t0 = [], int(planner.settings["batch_size"]), time.time()
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        images = [vsp.render(r["grid"], tiles) for r in batch]
        out = planner.solve(images, [r["question"] for r in batch], [r["img_known_patches"] for r in batch],
                            noise_generator(seed, start), steps)
        for r, answer, image in zip(batch, out.answers, out.images):
            scores = vsp.score(r["grid"], r["moves"], vsp.route_cells_from_image(image, r["grid"]), answer)
            rows.append({"id": r["id"], "vsp_level": r["vsp_level"], "vsp_map_id": r["vsp_map_id"],
                         "gold_answer": r["answer"], "pred_answer": answer, **scores})
        print(f"seed {seed}: {len(rows)}/{len(records)} ({time.time() - t0:.0f}s)", flush=True)
    levels = {}
    for row in rows:
        levels.setdefault(row["vsp_level"], []).append(row)
    per_level = {str(level): summarize(rs, KEYS) for level, rs in sorted(levels.items())}
    bench = [per_level[str(level)]["text_success"] for level in vsp.BENCHMARK_LEVELS if str(level) in per_level]
    return {"seed": seed, "steps": steps or int(planner.settings["steps"]), "summary": summarize(rows, KEYS),
            "per_level": per_level,
            "success_avg_levels_3_6": round(sum(bench) / len(bench), 6) if len(bench) == 4 else None,
            "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="release folder (model + task_head)")
    ap.add_argument("--vsp-maps", required=True, help="<VSP repo>/maze/task-main/maps")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=None, help="Euler steps (default: the folder's setting, 16)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    planner = Planner(args.model, args.device)
    records = vsp.official_records(args.vsp_maps, planner.pipe.text.tokenizer)
    tiles = vsp.cell_tiles(vsp.load_sprites(args.vsp_maps))
    args.out.mkdir(parents=True, exist_ok=True)
    metrics = {"metric": "text_success (official VSP rule)", "per_seed": {}}
    for seed in args.seeds:
        result = evaluate(planner, records, tiles, seed, args.steps)
        result.update(model=str(args.model), vsp_maps=str(args.vsp_maps))
        (args.out / f"seed{seed}.json").write_text(json.dumps(result, indent=2))
        metrics["per_seed"][seed] = {"per_level": {k: v["text_success"] for k, v in result["per_level"].items()},
                                     "avg_levels_3_6": result["success_avg_levels_3_6"]}
        print(f"seed {seed}: {json.dumps(metrics['per_seed'][seed])}", flush=True)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
