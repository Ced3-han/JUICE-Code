"""Train the linear task head of a planning model (JMaze or VSP) on one GPU.

    python -m scripts.train.train_task_head --config configs/train/task_head_jmaze.yaml
    python -m scripts.train.train_task_head --config configs/train/task_head_vsp.yaml

The configs run 1,500 steps at batch 64 on the training shards of the task (docs/PLANNING.md,
section 3). The head is written to <output_dir>/<run_name>/head/ (config.json + model.safetensors);
load it with juice.heads.LinearTextHead.from_folder, or give the folder to
scripts/export_planning.py --task-head.

--set takes key=value pairs (YAML values) that replace config keys, for example
--set output_dir=/scratch/juice seed=1.
"""

from __future__ import annotations

import argparse

import yaml

from juice.train.task_head import TaskHeadConfig, train_task_head


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="replace config keys")
    ap.add_argument("--device", default=None, help="default: cuda if available")
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many steps (for tests)")
    args = ap.parse_args()

    overrides = {}
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set needs KEY=VALUE, got {item!r}")
        overrides[key] = yaml.safe_load(value)
    train_task_head(TaskHeadConfig.from_yaml(args.config, **overrides), device=args.device, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
