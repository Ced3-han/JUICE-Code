"""Pack the VQAv2 train questions (lmms-lab/VQAv2 train parquets) into training shards.

Each question gets the LLaVA short-answer cue: "{question} Answer the question using a
single word or phrase.". The answer is `multiple_choice_answer`. A row is kept when its
answer span fits (1..8 tokens, content + EOS <= 128). Only COCO train2014 images are
accepted: a row with another image path stops the packer. Rows are written in parquet
order (sorted file names), 10,000 per tar: vqav2_train_00000.tar ... vqav2_train_00044.tar.
The paper trains on tars 00000-00042 (430,000 rows), each split into 13 parts
(split_shards.py).

Sample (key = 12-digit question_id):
  <key>.jpg   COCO train2014 image bytes from the parquet
  <key>.txt   "Question: {q} Answer: {a}\\n"
  <key>.json  question_id, image_id, answer, answer_type, answer_tok_start, answer_tok_end,
              n_content_tokens

    python -m scripts.data.pack_vqav2 --parquet-dir data/src/VQAv2/data --out data/wds/vqav2_train_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.data.common import (SHORT_ANSWER_CUE, answer_span, empty_out_dir, fits, load_t5_tokenizer,
                                 write_summary)
from scripts.data.shards import ShardWriter

TRAIN_PATH_PREFIX = "COCO_train2014_"


def iter_rows(parquets: list[Path], columns: list[str] | None = None):
    import pyarrow.parquet as pq

    for path in parquets:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=columns):
            yield from batch.to_pylist()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet-dir", required=True, help="folder with train-*.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples-per-tar", type=int, default=10_000)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many packed rows (smoke test)")
    ap.add_argument("--count-only", action="store_true",
                    help="run the answer filters only (reads no images and does not check the image paths)")
    args = ap.parse_args(argv)

    parquets = sorted(Path(args.parquet_dir).glob("train-*.parquet"))
    if not parquets:
        raise SystemExit(f"no train-*.parquet under {args.parquet_dir}")
    tokenizer = load_t5_tokenizer()
    writer = None if args.count_only else ShardWriter(empty_out_dir(args.out), "vqav2_train", args.samples_per_tar)
    counts = {"total": 0, "boundary_ok": 0, "packed": 0}
    seen_qids: set[int] = set()
    columns = ["question", "multiple_choice_answer"] if args.count_only else None
    for row in iter_rows(parquets, columns):
        if args.limit and counts["packed"] >= args.limit:
            break
        counts["total"] += 1
        if not args.count_only and not str(row["image"].get("path") or "").startswith(TRAIN_PATH_PREFIX):
            raise SystemExit(f"image path {row['image'].get('path')!r} is not COCO train2014; use the train parquets only")
        question = f"{row['question']} {SHORT_ANSWER_CUE}"
        answer = str(row["multiple_choice_answer"])
        span = answer_span(tokenizer, question, answer)
        if span is None:
            continue
        counts["boundary_ok"] += 1
        if not fits(span, 1, 8):
            continue
        counts["packed"] += 1
        if args.count_only:
            continue
        qid = int(row["question_id"])
        if qid in seen_qids:
            raise SystemExit(f"duplicate question_id {qid}")
        seen_qids.add(qid)
        sidecar = {"question_id": qid, "image_id": int(row["image_id"]), "answer": answer,
                   "answer_type": str(row.get("answer_type", "")),
                   "answer_tok_start": int(span["answer_tok_start"]), "answer_tok_end": int(span["answer_tok_end"]),
                   "n_content_tokens": int(span["n_content_tokens"])}
        writer.write(f"{qid:012d}", {
            "jpg": bytes(row["image"]["bytes"]),
            "txt": (span["caption"].strip() + "\n").encode("utf-8"),
            "json": json.dumps(sidecar, ensure_ascii=False).encode("utf-8"),
        })
    summary = {"question_suffix": SHORT_ANSWER_CUE, "counts": counts, "total_packed": counts["packed"]}
    if args.count_only:
        print(json.dumps(summary), flush=True)
        return
    summary["tars"] = len(writer.close())
    write_summary(Path(args.out), summary)


if __name__ == "__main__":
    main()
