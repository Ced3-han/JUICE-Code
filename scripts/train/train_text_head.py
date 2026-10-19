"""Train the parallel text head (the text readout for question answering) on one GPU.

    python scripts/train/train_text_head.py --config configs/train/text_head.yaml

The config runs the two stages of the released head (8,000 steps each, batch 64).
Stage k is written to <output_dir>/<run_name>/stage_<k>/ (config.json + model.safetensors);
the last stage is the head. Load it with juice.heads.ParallelTextHead.from_folder.

--init-from starts stage 1 from a text-head folder instead of the pretrained Flan-T5-Small.
--set takes key=value pairs (YAML values) that replace config keys, for example
--set stages=1 output_dir=/scratch/juice.
"""

from __future__ import annotations

import argparse

import yaml

from juice.train.text_head import TextHeadConfig, train_text_head


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="replace config keys")
    ap.add_argument("--init-from", default=None, help="text-head folder to start stage 1 from")
    ap.add_argument("--device", default=None, help="default: cuda if available")
    ap.add_argument("--max-steps", type=int, default=None, help="stop each stage after this many steps (for tests)")
    args = ap.parse_args()

    overrides = {}
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set needs KEY=VALUE, got {item!r}")
        overrides[key] = yaml.safe_load(value)
    if args.init_from is not None:
        overrides["init_from"] = args.init_from
    train_text_head(TextHeadConfig.from_yaml(args.config, **overrides), device=args.device, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
