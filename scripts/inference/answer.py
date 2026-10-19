"""Answer questions about images.

    python -m scripts.inference.answer --model JUICE-2B-SigLIP2 \
        --image photo.jpg --question "What color is the door?"

    python -m scripts.inference.answer --model JUICE-2B-SigLIP2 --short \
        --image photo.jpg --question "How many people are there?"

One question per image; give --image and --question once each, or several times in
the same order. --short adds the short-answer instruction of the paper's benchmarks
("Answer the question using a single word or phrase."). For a multiple-choice
question, put the options and "Answer with the option's letter from the given
choices directly." in --question, as docs/EVAL_VQA.md shows.

The defaults are the settings of the paper (config.json "defaults": 16 steps,
guidance 3, an answer window of 12 slots, the parallel text head).
"""

from __future__ import annotations

import argparse

from juice.pipeline import SHORT_ANSWER, JuicePipeline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model folder or Hugging Face repo id")
    ap.add_argument("--image", action="append", required=True)
    ap.add_argument("--question", action="append", required=True)
    ap.add_argument("--short", action="store_true", help="add the short-answer instruction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if len(args.image) != len(args.question):
        ap.error("give one --question for each --image")

    questions = [f"{q}\n{SHORT_ANSWER}" if args.short else q for q in args.question]
    pipe = JuicePipeline.from_pretrained(args.model, device=args.device)
    for image, question, answer in zip(args.image, args.question, pipe.answer(args.image, questions, seed=args.seed)):
        print(f"{image}\tQ: {question}\tA: {answer}")


if __name__ == "__main__":
    main()
