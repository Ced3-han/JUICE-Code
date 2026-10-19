"""Export a planning fine-tune (JMaze or VSP) into a release folder.

A planning release folder is a normal release folder plus the linear task head:

    model.safetensors, config.json   MM-RiT weights and settings (juice.export)
    task_head/                       linear answer head (juice.heads.LinearTextHead)

The paper's planning results use the raw "model" weights of the fine-tune
checkpoint, not an EMA copy. --task-head is a task-head folder (model.safetensors +
config.json); it is copied after a load check.

    python -m scripts.export_planning --checkpoint checkpoint_005000.pt --task-head task_head_jmaze \
        --preset jmaze --out my-jmaze-siglip2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from juice.export import copy_head, export


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--task-head", type=Path, required=True, help="task-head folder (model.safetensors + config.json)")
    ap.add_argument("--preset", required=True, choices=["jmaze", "vsp"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights", default="model", choices=["model", "ema", "ema_fast"])
    ap.add_argument("--sha256", default=None, help="known sha256 of --checkpoint (skips hashing a large file)")
    args = ap.parse_args()
    export(args.checkpoint, args.weights, args.out, args.preset, args.sha256)
    copy_head(args.task_head, args.out / "task_head", "task")


if __name__ == "__main__":
    main()
