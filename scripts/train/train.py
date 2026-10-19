"""Train JUICE with one config file.

    torchrun --nproc_per_node 8 scripts/train/train.py --config configs/train/sft_siglip2.yaml

The global batch is micro_batch_size x grad_accum_steps x GPUs and must equal batch_size.
With a different number of GPUs, change grad_accum_steps:

    torchrun --nproc_per_node 4 scripts/train/train.py --config configs/train/sft_siglip2.yaml \
        --set grad_accum_steps=32

--set takes key=value pairs (YAML values) that replace config keys. --resume continues
a run from one of its checkpoints (weights, EMA, optimizer and step).
"""

from __future__ import annotations

import argparse

import yaml

from juice.train.config import TrainConfig
from juice.train.trainer import train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="replace config keys")
    ap.add_argument("--resume", default=None, help="training checkpoint to continue from")
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many steps (for tests)")
    args = ap.parse_args()

    overrides = {}
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set needs KEY=VALUE, got {item!r}")
        overrides[key] = yaml.safe_load(value)
    train(TrainConfig.from_yaml(args.config, **overrides), resume=args.resume, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
