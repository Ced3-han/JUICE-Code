"""Score the answers of one benchmark and print its row of the question-answering table.

    python scripts/eval/score_vqa.py --bench gqa --probe-dir probes/gqa \
        --answers answers/gqa/shard0.json answers/gqa/shard1.json answers/gqa/shard2.json answers/gqa/shard3.json \
        --out answers/gqa/metrics.json

Table metric: VQAv2 n1000 / OK-VQA / VizWiz / POPE / GQA = VQA soft accuracy over the
probe items; MME = perception and cognition scores; MMBench / AI2D / MMMU /
ScienceQA-IMG / RealWorldQA = letter accuracy. The metrics file also holds the
accuracy with the dropped questions counted as wrong, per-type accuracies and, for
POPE, accuracy and F1 per split. Every probe item must have an answer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from juice.eval.vqa_probes import BENCHES, MCQ_BENCHES
from juice.eval.vqa_scoring import full_set_accuracy, mcq_accuracy, mme_scores, pope_metrics, vqa_accuracy

NAMES = {"vqav2_n1000": "VQAv2 n1000", "okvqa": "OKVQA", "vizwiz": "VizWiz", "pope": "POPE", "mme": "MME-P / MME-C",
         "gqa": "GQA", "mmbench": "MMBench", "ai2d": "AI2D", "mmmu": "MMMU", "scienceqa_img": "ScienceQA-IMG",
         "realworldqa": "RealWorldQA"}


def load_answers(paths: list[str]) -> list[dict]:
    """Rows of all answer files, shard 0 first."""
    files = [json.loads(Path(p).read_text()) for p in paths]
    files.sort(key=lambda f: f["meta"].get("shard") or 0)
    return [r for f in files for r in f["rows"]]


def score(bench: str, rows: list[dict], probe_dir: Path) -> dict:
    items = [it for p in sorted(probe_dir.glob("probe_shard*.json")) for it in json.loads(p.read_text())["items"]]
    drops = json.loads((probe_dir / "drops.json").read_text())
    if bench in MCQ_BENCHES:
        m = mcq_accuracy(rows, items)
        return {**m, "row": f"| {NAMES[bench]} | {m['accuracy']:.4f} |"}
    annotations = json.loads((probe_dir / "annotations.json").read_text())["annotations"]
    if bench == "mme":
        m = mme_scores(rows, items)
        return {**m, "row": f"| {NAMES[bench]} | {m['perception']:.1f} / {m['cognition']:.1f} |"}
    m = vqa_accuracy(rows, annotations)
    m["n_dropped"] = drops["funnel"]["dropped"]
    m["full_set_accuracy"] = full_set_accuracy(m["accuracy"], m["n"], m["n_dropped"])
    if bench == "pope":
        m["pope"] = pope_metrics(rows, annotations)
    return {**m, "row": f"| {NAMES[bench]} | {m['accuracy']:.4f} |"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, choices=[b for b in BENCHES if b in NAMES])
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument("--answers", nargs="+", required=True, help="answer files of all shards")
    ap.add_argument("--out", default=None, help="metrics json")
    args = ap.parse_args()

    probe_dir = Path(args.probe_dir)
    rows = load_answers(args.answers)
    want = {it["question_id"] for p in probe_dir.glob("probe_shard*.json") for it in json.loads(p.read_text())["items"]}
    got = [r["question_id"] for r in rows]
    if len(got) != len(set(got)) or set(got) != want:
        raise SystemExit(f"answers do not cover the probe: {len(set(got) & want)} of {len(want)} items answered, "
                         f"{len(got) - len(set(got))} duplicates, {len(set(got) - want)} unknown ids")
    metrics = {"bench": args.bench, **score(args.bench, rows, probe_dir)}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(metrics, indent=1))
    print(metrics["row"])


if __name__ == "__main__":
    main()
