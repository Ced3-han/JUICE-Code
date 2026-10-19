"""Answers of VQAv2 test-dev -> the EvalAI submission file (no ground truth is used).

    python scripts/eval/vqav2_testdev_submission.py --answers answers/vqav2_testdev/shard*.json \
        --questions data/vqav2 --out answers/vqav2_testdev/submission.json

The test-dev phase expects one entry for every test2015 question (447,793): test-dev
questions get the model answer; all other questions, and test-dev questions without an
answer, get "". --questions is the folder with
v2_OpenEnded_mscoco_test2015_questions.json and v2_OpenEnded_mscoco_test-dev2015_questions.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", nargs="+", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-partial", action="store_true", help="write the file although test-dev answers are missing")
    args = ap.parse_args()

    pred: dict[int, str] = {}
    for p in args.answers:
        for r in json.loads(Path(p).read_text())["rows"]:
            q = int(r["question_id"])
            if q in pred:
                raise SystemExit(f"question {q} has two answers")
            pred[q] = str(r["answer"]).strip()
    folder = Path(args.questions)
    testdev = [int(q["question_id"]) for q in json.loads((folder / "v2_OpenEnded_mscoco_test-dev2015_questions.json").read_text())["questions"]]
    test_all = [int(q["question_id"]) for q in json.loads((folder / "v2_OpenEnded_mscoco_test2015_questions.json").read_text())["questions"]]
    extra = set(pred) - set(testdev)
    if extra:
        raise SystemExit(f"{len(extra)} answers are not test-dev questions")
    missing = [q for q in testdev if q not in pred]
    if missing and not args.allow_partial:
        raise SystemExit(f"{len(missing)} of {len(testdev)} test-dev questions have no answer")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps([{"question_id": q, "answer": pred.get(q, "")} for q in test_all]))
    print(f"wrote {len(test_all)} entries ({len(pred)} answered, {len(missing)} test-dev missing) -> {args.out}")


if __name__ == "__main__":
    main()
