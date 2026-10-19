"""Solve one planning puzzle and save the solved image.

JMaze (a puzzle from a split made by scripts/data/make_jmaze_data.py):
    python -m scripts.inference.plan --model my-jmaze-siglip2 --puzzles data/jmaze/test_id.jsonl \
        --index 0 --out maze0.png
VSP (an official map):
    python -m scripts.inference.plan --model my-vsp-siglip2 --vsp-maps VSP/maze/task-main/maps \
        --level 5 --map-id 3 --out vsp_5_3.png

Prints the question, the answer text and its scores, and writes the input image and
the generated solved image side by side. One puzzle is one batch, so the noise (and
the result) can differ from the batch of 25 that the evaluation scripts use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from juice.tasks import maze, vsp
from juice.tasks.planning import Planner, noise_generator


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--puzzles", default=None, help="JMaze split jsonl")
    ap.add_argument("--index", type=int, default=0, help="line of --puzzles")
    ap.add_argument("--vsp-maps", default=None, help="<VSP repo>/maze/task-main/maps")
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--map-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    planner = Planner(args.model, args.device)
    if planner.task == "jmaze":
        lines = Path(args.puzzles).read_text().splitlines()
        p = json.loads(lines[args.index])
        image = maze.render_maze(p["blocks"], p["start"], p["goal"])
        known = maze.known_patch_flags(p["blocks"], p["start"], p["goal"])
    else:
        records = vsp.official_records(args.vsp_maps, planner.pipe.text.tokenizer, levels=[args.level])
        p = records[args.map_id]
        image = vsp.render(p["grid"], vsp.cell_tiles(vsp.load_sprites(args.vsp_maps)))
        known = p["img_known_patches"]
    out = planner.solve([image], [p["question"]], [known], noise_generator(args.seed, 0))
    answer, solved = out.answers[0], out.images[0]
    if planner.task == "jmaze":
        labels = maze.parse_blocks_from_image(solved, int(p["n"]))
        scores = maze.score_puzzle(p["blocks"], p["start"], p["goal"], labels, maze.parse_answer(answer))
        keys = ("text_valid", "img_valid", "agree", "joint_consistent")
    else:
        scores = vsp.score(p["grid"], p["moves"], vsp.route_cells_from_image(solved, p["grid"]), answer)
        keys = ("text_success", "img_valid", "agree", "joint_consistent")
    print(f"question: {p['question']}\nanswer:   {answer}\ngold:     {p['answer']}")
    print("scores:  ", {k: scores[k] for k in keys})
    pixels = (np.clip(solved, 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(np.concatenate([np.asarray(image), pixels], axis=1)).save(args.out)
    print(f"wrote {args.out} (left: input, right: generated)")


if __name__ == "__main__":
    main()
