"""Turn caption shards into prompted-caption question-answer rows (SFT).

Each caption row (<key>.<image> + <key>.txt) becomes "Question: {prompt} Answer: {caption}".
The prompt is a template of the style, chosen by the sha1 of the source key
(common.pick_prompt): concise = LLaVA-Pretrain brief-description instructions,
detailed = LLaVA detail_23k instructions. White space in the caption is collapsed.
A row is kept when its span fits (--min/--max-answer-tokens, content + EOS <= 128).
With --truncate sentence, a long caption is cut at the last sentence end that fits.

--shard K/N packs the source tars [K::N] of the manifest, in manifest order.
--limit-rows stops after N packed rows (the paper takes the first 156,250 or 187,500
rows of each blip3o shard), so the manifest order matters: keep it sorted by name.

Sample (key = {source tar stem}__{source key}, "/" -> "_"), 2,500 rows per tar:
  <key>.<ext>  the source image bytes
  <key>.txt    "Question: {prompt} Answer: {caption}\\n"
  <key>.json   source, style, src_tar, src_key, prompt, answer, answer_tok_start,
               answer_tok_end, n_content_tokens, qa_window [, truncation record]

    python -m scripts.data.pack_prompted_captions --src-manifest data/manifests/coco_train2014_caption.txt \\
        --style concise --qa-window 48 --max-answer-tokens 40 --shard 0/4 \\
        --source-name capc_coco2014 --out data/wds/sft_cap/capc_coco2014/s00 --tar-prefix sft_capc_coco2014_s00
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from scripts.data.common import (CANVAS, TEMPLATES, answer_span, empty_out_dir, fits, load_t5_tokenizer,
                                 parse_shard, pick_prompt, truncate_at_sentence, write_summary)
from scripts.data.shards import ShardWriter

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def iter_tar_rows(path: str, read_images: bool = True):
    """(key, image member name, image bytes, caption) for each key with an image and a .txt."""
    with tarfile.open(path) as tar:
        pending: dict[str, dict] = {}
        for m in tar:
            if not m.isfile():
                continue
            stem, dot, ext = m.name.rpartition(".")
            if not dot:
                continue
            ext = "." + ext.lower()
            if ext == ".txt":
                pending.setdefault(stem, {})["txt"] = tar.extractfile(m).read().decode("utf-8", "ignore")
            elif ext in IMG_EXTS:
                pending.setdefault(stem, {})["img"] = (m.name, tar.extractfile(m).read() if read_images else b"")
            else:
                continue
            rec = pending[stem]
            if "txt" in rec and "img" in rec:
                del pending[stem]
                yield stem, rec["img"][0], rec["img"][1], rec["txt"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-manifest", required=True, help="one caption tar per line")
    ap.add_argument("--style", choices=sorted(TEMPLATES), required=True)
    ap.add_argument("--qa-window", type=int, required=True)
    ap.add_argument("--max-answer-tokens", type=int, required=True)
    ap.add_argument("--min-answer-tokens", type=int, default=1)
    ap.add_argument("--truncate", choices=["none", "sentence"], default="none")
    ap.add_argument("--shard", default="", help="K/N over the tars of the manifest")
    ap.add_argument("--limit-rows", type=int, default=0, help="stop after this many packed rows (0 = all)")
    ap.add_argument("--source-name", required=True, help="the `source` field of the sidecar")
    ap.add_argument("--samples-per-tar", type=int, default=2500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tar-prefix", required=True)
    ap.add_argument("--count-only", action="store_true", help="run the filters, read no images, write nothing")
    args = ap.parse_args(argv)
    if args.qa_window < args.max_answer_tokens + 1:
        raise SystemExit(f"--qa-window must be >= --max-answer-tokens + 1 = {args.max_answer_tokens + 1}")

    tars = [l.strip() for l in Path(args.src_manifest).read_text().splitlines() if l.strip() and not l.startswith("#")]
    shard_k, shard_n = parse_shard(args.shard)
    tars = tars[shard_k::shard_n]
    tokenizer = load_t5_tokenizer()
    out_dir = None if args.count_only else empty_out_dir(args.out)
    writer = None if args.count_only else ShardWriter(out_dir, args.tar_prefix, args.samples_per_tar)
    counts = {"tars_in": len(tars), "rows_in": 0, "boundary_drop": 0, "truncated": 0, "truncate_failed": 0,
              "length_drop": 0, "packed": 0}
    for tpath in tars:
        if args.limit_rows and counts["packed"] >= args.limit_rows:
            break
        for src_key, img_name, img_bytes, caption in iter_tar_rows(tpath, read_images=not args.count_only):
            counts["rows_in"] += 1
            caption = " ".join(caption.split())
            if not caption:
                continue
            prompt = pick_prompt(args.style, src_key)
            span = answer_span(tokenizer, prompt, caption)
            if span is None:
                counts["boundary_drop"] += 1
                continue
            trunc = None
            if args.truncate == "sentence" and (span["answer_span_len"] > args.max_answer_tokens
                                                or span["n_content_tokens"] + 1 > CANVAS):
                cut = truncate_at_sentence(tokenizer, prompt, caption, args.max_answer_tokens)
                if cut is None:
                    counts["truncate_failed"] += 1
                    continue
                caption, span = cut[0], cut[1]
                trunc = {"truncated": True, "n_sentences_kept": cut[2], "n_sentences_total": cut[3]}
                counts["truncated"] += 1
            if not fits(span, args.min_answer_tokens, args.max_answer_tokens):
                counts["length_drop"] += 1
                continue
            if args.count_only:
                counts["packed"] += 1
                if args.limit_rows and counts["packed"] >= args.limit_rows:
                    break
                continue
            ext = img_name.rsplit(".", 1)[-1].lower()
            ext = "jpg" if ext == "jpeg" else ext
            sidecar = {"source": args.source_name, "style": args.style, "src_tar": str(tpath), "src_key": src_key,
                       "prompt": prompt, "answer": caption, "answer_tok_start": int(span["answer_tok_start"]),
                       "answer_tok_end": int(span["answer_tok_end"]), "n_content_tokens": int(span["n_content_tokens"]),
                       "qa_window": int(args.qa_window), **(trunc or {})}
            writer.write(f"{Path(tpath).stem}__{src_key}".replace("/", "_"), {
                ext: img_bytes,
                "txt": (span["caption"].strip() + "\n").encode("utf-8"),
                "json": json.dumps(sidecar, ensure_ascii=False).encode("utf-8"),
            })
            counts["packed"] += 1
            if args.limit_rows and counts["packed"] >= args.limit_rows:
                break
    summary = {"style": args.style, "src_manifest": args.src_manifest, "shard": {"k": shard_k, "n": shard_n},
               "counts": counts, "total_packed": counts["packed"]}
    if args.count_only:
        print(json.dumps(summary), flush=True)
        return
    summary["tars"] = len(writer.close())
    write_summary(out_dir, summary)


if __name__ == "__main__":
    main()
