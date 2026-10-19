"""Shared helpers of the training-data packers (scripts/data/pack_*.py).

Question-answering rows use the caption "Question: {q} Answer: {a}". The json sidecar of
a row gives the answer span in Flan-T5 content tokens (no EOS): answer_tok_start,
answer_tok_end and n_content_tokens. A row fits when
    min_answer_tokens <= answer tokens <= max_answer_tokens  and  content + EOS <= 128.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from juice.tasks.planning import answer_span as _answer_span

CANVAS = 128                 # text slots of the model (juice.representations.TEXT_CANVAS)
SHUFFLE_SEED = 4242          # seed of every row shuffle of the paper packs
SHORT_ANSWER_CUE = "Answer the question using a single word or phrase."  # LLaVA-1.5 instruction
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def load_t5_tokenizer():
    """Flan-T5-Small tokenizer at the revision of the paper."""
    from juice.representations import load_tokenizer

    return load_tokenizer()


def answer_span(tokenizer, question: str, answer: str) -> dict | None:
    """Answer span of "Question: {question} Answer: {answer}", or None when the tokens of
    the question prefix are not a strict prefix of the caption tokens."""
    span = _answer_span(tokenizer, question, answer)
    if span is not None:
        span["answer_span_len"] = span["answer_tok_end"] - span["answer_tok_start"]
    return span


def fits(span: dict, min_answer_tokens: int, max_answer_tokens: int) -> bool:
    return (min_answer_tokens <= span["answer_span_len"] <= max_answer_tokens
            and span["n_content_tokens"] + 1 <= CANVAS)


def truncate_at_sentence(tokenizer, question: str, answer: str, max_answer_tokens: int):
    """Longest sentence prefix of `answer` whose span fits (answer <= max_answer_tokens and
    content + EOS <= 128). Sentences end at '.', '!' or '?' followed by white space.
    Returns (answer, span, sentences kept, sentences total), or None when not even the
    first sentence fits. The full answer is never a candidate (the caller tried it)."""
    sents = [s for s in SENT_SPLIT.split(answer.strip()) if s.strip()]
    for k in range(len(sents) - 1, 0, -1):
        cand = " ".join(s.strip() for s in sents[:k])
        span = answer_span(tokenizer, question, cand)
        if span is None:
            continue
        if span["answer_span_len"] <= max_answer_tokens and span["n_content_tokens"] + 1 <= CANVAS:
            return cand, span, k, len(sents)
    return None


# Prompt sets of the prompted-caption rows.
# Concise: the LLaVA-Pretrain brief-description instructions. Detailed: LLaVA detail_23k.
CONCISE_TEMPLATES = [
    "Describe the image concisely.",
    "Provide a brief description of the given image.",
    "Offer a succinct explanation of the picture presented.",
    "Summarize the visual content of the image.",
    "Give a short and clear explanation of the subsequent image.",
    "Share a concise interpretation of the image provided.",
    "Present a compact description of the photo's key features.",
    "Relay a brief, clear account of the picture shown.",
    "Render a clear and concise summary of the photo.",
    "Write a terse but informative summary of the picture.",
    "Create a compact narrative representing the image presented.",
]
DETAILED_TEMPLATES = [
    "Describe the following image in detail.",
    "Provide a detailed description of the given image.",
    "Give an elaborate explanation of the image you see.",
    "Share a comprehensive rundown of the presented image.",
    "Offer a thorough analysis of the image.",
    "Explain the various aspects of the image before you.",
    "Clarify the contents of the displayed image with great detail.",
    "Characterize the image using a well-detailed description.",
    "Break down the elements of the image in a detailed manner.",
    "Walk through the important details of the image.",
    "Portray the image with a rich, descriptive narrative.",
    "Narrate the contents of the image with precision.",
    "Analyze the image in a comprehensive and detailed manner.",
    "Illustrate the image through a descriptive explanation.",
    "Examine the image closely and share its details.",
    "Write an exhaustive depiction of the given image.",
]
TEMPLATES = {"concise": CONCISE_TEMPLATES, "detailed": DETAILED_TEMPLATES}


def pick_prompt(style: str, key: str) -> str:
    """The prompt of a row: a template chosen by the sha1 of the source key (no RNG state)."""
    templates = TEMPLATES[style]
    return templates[int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % len(templates)]


def parse_shard(text: str) -> tuple[int, int]:
    """"K/N" -> (K, N); "" -> (0, 1)."""
    if not text:
        return 0, 1
    k, n = (int(x) for x in text.split("/"))
    if not 0 <= k < n:
        raise SystemExit(f"--shard {text}: need 0 <= K < N")
    return k, n


def empty_out_dir(path: str | Path) -> Path:
    """Make the output folder. Refuse a folder that already holds shards (no resume)."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("*.tar")) or (out / "pack_summary.json").exists():
        raise SystemExit(f"{out} already holds a pack; remove it or use a new --out")
    return out


def write_summary(out_dir: Path, summary: dict) -> None:
    (Path(out_dir) / "pack_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"DONE {json.dumps(summary.get('counts', {}))} -> {out_dir}", flush=True)
