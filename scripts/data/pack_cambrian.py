"""Pack Cambrian-7M question-answer rows into training shards.

Input: <root>/jsons/cambrian7m_mmu.jsonl (make it with make_cambrian_mmu_jsonl.py) and the
images under <root>/images/. The ScienceQA rows use the same packer on the jsonl of
convert_scienceqa_mcq.py.

Pass 1 (cheap filters, file order):
  * keep the rows whose image path starts with one of --sources ("gqa/...", "coco/...");
    with --source-file-prefix, also the jsonl field `source` must start with that prefix;
  * drop the 1,024 lines of the held-out probe (sha1 of the raw line, see
    cambrian7m_mmu_valprobe1024.linehash);
  * coco rows: --coco-val-mode exclude (default) drops the rows whose 12-digit image id is
    a COCO val2014 or val2017 image; only keeps exactly those rows; keep does not filter;
  * split each conversation into (human, gpt) pairs. The question loses the <image> tag.
    v1 (default): the question also loses the LLaVA cue "Answer the question using a single
    word or phrase.". v2 (--keep-prompt-suffix): the question keeps the cue, and when the
    first turn of a conversation has the cue, every pair of that conversation gets it;
  * --skip-cued-rows drops pairs whose question has the cue; --only-mcq keeps only pairs
    with "option's letter" in the question; without --truncate-answers, a pair whose
    answer has more words than --max-answer-tokens is dropped (a word is >= 1 token).
Shuffle all candidates (seed 4242); --shard K/N keeps candidates[K::N].
Pass 2: keep a pair when its answer span fits (1..8 tokens by default, content + EOS <= 128).
With --truncate-answers sentence, a long answer is cut at the last sentence end that fits.

Sample (key = {source}_{jsonl line:07d}_{pair:02d}):
  <key>.<ext>  the image bytes (extension of the file, lower case, jpeg -> jpg)
  <key>.txt    "Question: {q} Answer: {a}\\n"
  <key>.json   source, cambrian_line, pair_idx, image, answer, answer_tok_start,
               answer_tok_end, n_content_tokens [, qa_window] [, truncation record]

    python -m scripts.data.pack_cambrian --root data/src/Cambrian-10M \\
        --coco-val2014 data/src/coco/val2014 --coco-val2017 data/src/coco/val2017 \\
        --sources vg --shard 0/8 --out data/wds/cambrian-s8/vg/s00 --tar-prefix cambrian_vg_s00
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from scripts.data.common import (CANVAS, SHORT_ANSWER_CUE, SHUFFLE_SEED, answer_span, empty_out_dir, fits,
                                 load_t5_tokenizer, parse_shard, truncate_at_sentence, write_summary)
from scripts.data.shards import ShardWriter

DEFAULT_LINEHASH = Path(__file__).with_name("cambrian7m_mmu_valprobe1024.linehash")
IMG_TAG = re.compile(r"<image>\s*", flags=re.IGNORECASE)
CUE = re.compile(r"\s*Answer the question using a single word or phrase\.\s*$", flags=re.IGNORECASE)
COCO_ID = re.compile(r"(\d{12})\.[a-zA-Z]+$")


def coco_val_ids(dirs: list[str]) -> set[str]:
    """12-digit image ids of the files in the COCO val2014 and val2017 folders."""
    ids: set[str] = set()
    for d in dirs:
        matched = 0
        for p in Path(d).iterdir():
            m = COCO_ID.search(p.name)
            if m:
                ids.add(m.group(1))
                matched += 1
        if matched < 1000:
            raise SystemExit(f"{d}: only {matched} COCO image files; give the val2014 and val2017 image folders")
    return ids


def collect_candidates(jsonl: Path, args, excluded_hashes: set[str], val_ids: set[str], counts: dict) -> list[tuple]:
    """Pass 1: (source, line, pair, image path, question, answer) in file order."""
    candidates = []
    with open(jsonl, "rb") as f:
        for line_idx, raw in enumerate(f):
            counts["lines"] += 1
            try:
                d = json.loads(raw)
                img_rel = d.get("image") or ""
            except Exception:
                continue
            src = img_rel.split("/", 1)[0] if "/" in img_rel else ""
            if src not in args.sources:
                continue
            if args.source_file_prefix and not str(d.get("source") or "").startswith(args.source_file_prefix):
                continue
            counts["lines_source_kept"] += 1
            if hashlib.sha1(raw).hexdigest() in excluded_hashes:
                counts["lines_valprobe_excluded"] += 1
                continue
            if src == "coco" and args.coco_val_mode != "keep":
                m = COCO_ID.search(img_rel)
                on_val = bool(m and m.group(1) in val_ids)
                if on_val != (args.coco_val_mode == "only"):
                    counts["rows_coco_val_filtered"] += 1
                    continue
            conv = d.get("conversations") or []
            # LLaVA-1.5 gives the cue once, in the first human turn; a split pair would lose it.
            conv_cue = bool(conv) and conv[0].get("from") == "human" and \
                CUE.search(IMG_TAG.sub("", str(conv[0].get("value", "")))) is not None
            for pair_idx in range(0, len(conv) - 1, 2):
                if conv[pair_idx].get("from") != "human" or conv[pair_idx + 1].get("from") != "gpt":
                    continue
                counts["pairs"] += 1
                q = IMG_TAG.sub("", conv[pair_idx]["value"]).strip()
                if args.keep_prompt_suffix:
                    if conv_cue and CUE.search(q) is None:
                        q = f"{q} {SHORT_ANSWER_CUE}"
                else:
                    q = CUE.sub("", q).strip()
                a = conv[pair_idx + 1]["value"].strip()
                if args.skip_cued_rows and CUE.search(q) is not None:
                    continue
                if not q or not a or (args.truncate_answers == "none" and len(a.split()) > args.max_answer_tokens):
                    continue
                if args.only_mcq and "option's letter" not in q:
                    continue
                candidates.append((src, line_idx, pair_idx // 2, img_rel, q, a))
    return candidates


def fit_pair(tokenizer, q: str, a: str, args, counts: dict):
    """Pass 2 for one pair: (answer, span, truncation record) or None."""
    span = answer_span(tokenizer, q, a)
    if span is None:
        return None
    trunc = None
    if args.truncate_answers == "sentence" and (span["answer_span_len"] > args.max_answer_tokens
                                                or span["n_content_tokens"] + 1 > CANVAS):
        cut = truncate_at_sentence(tokenizer, q, a, args.max_answer_tokens)
        if cut is None:
            counts["truncate_failed"] += 1
            return None
        trunc = {"truncated": True, "n_sentences_kept": cut[2], "n_sentences_total": cut[3],
                 "answer_original_tokens": int(span["answer_span_len"])}
        a, span = cut[0], cut[1]
        counts["truncated"] += 1
    if not fits(span, args.min_answer_tokens, args.max_answer_tokens):
        return None
    return a, span, trunc


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Cambrian-10M folder (jsons/, images/)")
    ap.add_argument("--jsonl", default="jsons/cambrian7m_mmu.jsonl", help="relative to --root")
    ap.add_argument("--exclude-linehash", default=str(DEFAULT_LINEHASH), help="'' = exclude nothing")
    ap.add_argument("--sources", required=True, help="comma list of image folders, e.g. gqa,coco")
    ap.add_argument("--source-file-prefix", default="")
    ap.add_argument("--coco-val-mode", choices=["exclude", "only", "keep"], default="exclude")
    ap.add_argument("--coco-val2014", help="COCO val2014 image folder (not needed with --coco-val-mode keep)")
    ap.add_argument("--coco-val2017", help="COCO val2017 image folder")
    ap.add_argument("--keep-prompt-suffix", action="store_true", help="prompt v2 (SFT packs)")
    ap.add_argument("--skip-cued-rows", action="store_true")
    ap.add_argument("--only-mcq", action="store_true")
    ap.add_argument("--truncate-answers", choices=["none", "sentence"], default="none")
    ap.add_argument("--min-answer-tokens", type=int, default=1)
    ap.add_argument("--max-answer-tokens", type=int, default=8)
    ap.add_argument("--qa-window", type=int, default=0, help="write qa_window into each sidecar (0 = no field)")
    ap.add_argument("--shard", default="", help="K/N")
    ap.add_argument("--samples-per-tar", type=int, default=2500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tar-prefix", required=True)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many packed rows (smoke test)")
    ap.add_argument("--count-only", action="store_true", help="run the filters, read and write no images")
    args = ap.parse_args(argv)
    args.sources = {s.strip() for s in args.sources.split(",") if s.strip()}
    if args.qa_window and args.qa_window < args.max_answer_tokens + 1:
        raise SystemExit(f"--qa-window must be >= --max-answer-tokens + 1 = {args.max_answer_tokens + 1}")
    shard_k, shard_n = parse_shard(args.shard)

    root = Path(args.root)
    excluded = set(Path(args.exclude_linehash).read_text().split()) if args.exclude_linehash else set()
    val_ids: set[str] = set()
    if args.coco_val_mode != "keep" and "coco" in args.sources:
        if not (args.coco_val2014 and args.coco_val2017):
            raise SystemExit("coco rows need --coco-val2014 and --coco-val2017 (or --coco-val-mode keep)")
        val_ids = coco_val_ids([args.coco_val2014, args.coco_val2017])
    out_dir = None if args.count_only else empty_out_dir(args.out)

    counts = {k: 0 for k in ("lines", "lines_source_kept", "lines_valprobe_excluded", "rows_coco_val_filtered",
                             "pairs", "candidates", "candidates_in_shard", "truncated", "truncate_failed",
                             "image_missing", "packed")}
    candidates = collect_candidates(root / args.jsonl, args, excluded, val_ids, counts)
    counts["candidates"] = len(candidates)
    random.Random(SHUFFLE_SEED).shuffle(candidates)
    candidates = candidates[shard_k::shard_n]
    counts["candidates_in_shard"] = len(candidates)
    print(f"pass 1: {len(candidates)} candidates in shard {shard_k}/{shard_n}", flush=True)

    tokenizer = load_t5_tokenizer()
    writer = None if args.count_only else ShardWriter(out_dir, args.tar_prefix, args.samples_per_tar)
    for src, line_idx, pair_idx, img_rel, q, a in candidates:
        if args.limit and counts["packed"] >= args.limit:
            break
        fitted = fit_pair(tokenizer, q, a, args, counts)
        if fitted is None:
            continue
        a, span, trunc = fitted
        if args.count_only:
            counts["packed"] += 1
            continue
        img_path = root / "images" / img_rel
        try:
            img_bytes = img_path.read_bytes()
        except OSError:
            counts["image_missing"] += 1
            continue
        ext = img_path.suffix.lower().lstrip(".") or "jpg"
        ext = "jpg" if ext == "jpeg" else ext
        sidecar = {"source": src, "cambrian_line": line_idx, "pair_idx": pair_idx, "image": img_rel, "answer": a,
                   "answer_tok_start": int(span["answer_tok_start"]), "answer_tok_end": int(span["answer_tok_end"]),
                   "n_content_tokens": int(span["n_content_tokens"]),
                   **({"qa_window": int(args.qa_window)} if args.qa_window > 0 else {}), **(trunc or {})}
        writer.write(f"{src}_{line_idx:07d}_{pair_idx:02d}", {
            ext: img_bytes,
            "txt": (span["caption"].strip() + "\n").encode("utf-8"),
            "json": json.dumps(sidecar, ensure_ascii=False).encode("utf-8"),
        })
        counts["packed"] += 1

    summary = {"sources": sorted(args.sources), "shard": {"k": shard_k, "n": shard_n},
               "filters": {k: getattr(args, k) for k in (
                   "source_file_prefix", "coco_val_mode", "keep_prompt_suffix", "skip_cued_rows", "only_mcq",
                   "truncate_answers", "min_answer_tokens", "max_answer_tokens", "qa_window", "exclude_linehash")},
               "shuffle_seed": SHUFFLE_SEED, "counts": counts, "total_packed": counts["packed"]}
    if args.count_only:
        print(json.dumps(summary), flush=True)
        return
    summary["tars"] = len(writer.close())
    write_summary(out_dir, summary)


if __name__ == "__main__":
    main()
