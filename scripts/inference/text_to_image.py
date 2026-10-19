"""Generate images from a text prompt.

    python -m scripts.inference.text_to_image --model JUICE-2B-SigLIP2 \
        --prompt "a red bicycle next to a blue door" --num-images 4 --out bicycle

Writes <out>_<k>.png (224 x 224). The defaults are the settings of the paper
(config.json "defaults": 25 steps, guidance 5, generic caption "a photo." as the guidance reference).
Images are made in batches of 4; batch k uses the seed (seed + 4k), as in the paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from juice.pipeline import JuicePipeline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model folder or Hugging Face repo id")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--num-images", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None, help="default: config.json (25)")
    ap.add_argument("--cfg-scale", type=float, default=None, help="default: config.json (5.0)")
    ap.add_argument("--out", required=True, help="output path prefix")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    pipe = JuicePipeline.from_pretrained(args.model, device=args.device)
    images = pipe.text_to_image(args.prompt, num_images=args.num_images, seed=args.seed,
                                steps=args.steps, cfg_scale=args.cfg_scale)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for k, image in enumerate(images):
        path = f"{args.out}_{k}.png"
        image.save(path)
        print(path)


if __name__ == "__main__":
    main()
