"""Question-answering probes: public benchmark files -> probe items.

A probe item is one question, one image and the text slots of

    "Question: {question} Answer: {answer}"

as Flan-T5 token ids (no special tokens), filled with EOS to the 128-slot canvas.
Only the prefix, token ids [0, answer_tok_start), enters the model (see
JuicePipeline.answer_states); the reference answer tokens only fix the item layout.
The field name "token_ids_64" is historical (the first probes had 64 slots); it
holds 128 ids.

Benchmarks of the question-answering table of the paper:
  short answer     vqav2_n1000, okvqa, vizwiz, pope, mme, gqa   (and vqav2_testdev, no answers)
  multiple choice  mmbench, ai2d, mmmu, scienceqa_img, realworldqa

Short-answer questions end with SHORT_ANSWER (MME keeps its own "Please answer yes or
no." before it). Multiple-choice prompts list the lettered options and end with
MCQ_INSTRUCTION; the answer is one letter.

Funnel. A short-answer item is dropped when the question/answer token boundary is not
clean, when the canonical answer (mode of the reference answers) has more than 8
tokens, or when the caption plus EOS does not fit 128 slots. A multiple-choice item is
dropped when its prompt does not fit, and for the reasons in its row reader. Every
drop is recorded with its reason.

Order. Items are sorted by question id (multiple choice: by the id as a string) and
split round robin into shards (item k goes to shard k % 4). The order is part of the
protocol: the sampler seeds every batch by the position of its first item in the shard.
VQAv2 n1000 is one shard of 1000 items.
"""

from __future__ import annotations

import ast
import glob
import json
import random
import re
from collections import Counter
from pathlib import Path

from juice.pipeline import QA_TEMPLATE, SHORT_ANSWER
from juice.representations import EOS_ID, TEXT_CANVAS, TEXT_ENCODER_ID, TEXT_ENCODER_REVISION

MCQ_INSTRUCTION = "Answer with the option's letter from the given choices directly."
LETTERS = "ABCDEFGHI"
MAX_ANSWER_TOKENS = 8
QA_WINDOW = 12  # answer slots that the sampler generates
NUM_SHARDS = 4

SHORT_ANSWER_BENCHES = ("vqav2_n1000", "okvqa", "vizwiz", "pope", "mme", "gqa")
MCQ_BENCHES = ("mmbench", "ai2d", "mmmu", "scienceqa_img", "realworldqa")
BENCHES = SHORT_ANSWER_BENCHES + MCQ_BENCHES + ("vqav2_testdev",)


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TEXT_ENCODER_ID, revision=TEXT_ENCODER_REVISION, model_max_length=TEXT_CANVAS)


# ----------------------------------------------------------------------------- text layout


def answer_span(tokenizer, question: str, answer: str) -> dict | None:
    """Token ids of "Question: q Answer: a" and the answer span; None if the boundary is not clean.

    The prefix ids must be a strict token prefix of the full ids (SentencePiece can
    merge across the boundary); the answer occupies [answer_tok_start, answer_tok_end).
    """
    caption = f"{QA_TEMPLATE.format(question=question)} {answer}"
    prefix = tokenizer(QA_TEMPLATE.format(question=question), add_special_tokens=False)["input_ids"]
    full = tokenizer(caption, add_special_tokens=False)["input_ids"]
    if full[: len(prefix)] != prefix or len(full) <= len(prefix):
        return None
    return {"caption": caption, "ids": full, "answer_tok_start": len(prefix), "answer_tok_end": len(full)}


def make_item(question_id: int, image_id: int, image_file: str, question: str, answer: str, span: dict) -> dict:
    ids = span["ids"]
    return {
        "question_id": question_id,
        "image_id": image_id,
        "image_file": image_file,
        "question": question,
        "answer": answer,
        "caption": span["caption"],
        "token_ids_64": ids + [EOS_ID] * (TEXT_CANVAS - len(ids)),
        "answer_tok_start": span["answer_tok_start"],
        "answer_tok_end": span["answer_tok_end"],
        "n_content_tokens": len(ids),
    }


def shard_items(items: list[dict], num_shards: int = NUM_SHARDS) -> list[list[dict]]:
    return [items[s::num_shards] for s in range(num_shards)]


# ----------------------------------------------------------------------------- readers


def _files(src: str | Path, pattern: str) -> list[str]:
    files = sorted(glob.glob(str(Path(src) / pattern)))
    if not files:
        raise FileNotFoundError(f"no files match {Path(src) / pattern}")
    return files


def _parquet_rows(files: list[str], batch_size: int = 256):
    import pyarrow.parquet as pq

    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
            yield from batch.to_pylist()


def _image_bytes(cell) -> tuple[bytes | None, str]:
    """Hugging Face image cell -> (bytes, source file name)."""
    if isinstance(cell, dict):
        b = cell.get("bytes")
        return (b, _s(cell.get("path"))) if b else (None, "")
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell), ""
    return None, ""


def _s(v) -> str:
    """Table cell -> stripped string; None and 'nan' -> ''."""
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("", "nan", "none", "null") else s


def _extension(b: bytes) -> str:
    head = bytes(b[:12])
    if head[:4] == b"\x89PNG":
        return ".png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return ".webp"
    if head[:3] == b"GIF":
        return ".gif"
    return ".jpg"


# Short-answer readers yield: question_id, question, answers (reference list),
# answer_type, question_type, image_name, image_bytes (None = a COCO file on disk).


def okvqa_rows(src):
    """lmms-lab/OK-VQA, split val2014 (5,046 questions on COCO val2014 images)."""
    for r in _parquet_rows(_files(src, "val2014-*.parquet")):
        yield {"question_id": int(r["question_id"]), "question": str(r["question"]), "answers": [str(a) for a in r["answers"]],
               "answer_type": str(r.get("answer_type") or "other"), "question_type": str(r.get("question_type") or ""),
               "image_name": r["image"]["path"], "image_bytes": None}


def vizwiz_rows(src):
    """lmms-lab/VizWiz-VQA, split val (4,319 questions)."""
    for r in _parquet_rows(_files(src, "val-*.parquet")):
        yield {"question_id": int(str(r["question_id"]).rsplit("_", 1)[-1]), "question": str(r["question"]),
               "answers": [str(a) for a in r["answers"]], "answer_type": str(r.get("category") or "other"),
               "question_type": str(r.get("question_type") or ""),
               "image_name": r["image"]["path"], "image_bytes": r["image"]["bytes"]}


def pope_rows(src):
    """POPE COCO files coco_pope_{random,popular,adversarial}.json (3 x 3,000 yes/no questions).

    Question ids are offset by 100000 per split; the one label is repeated 10 times so
    that the VQA soft accuracy is the exact-match accuracy.
    """
    for si, split in enumerate(("random", "popular", "adversarial")):
        with open(Path(src) / f"coco_pope_{split}.json") as f:
            for line in f:
                r = json.loads(line)
                yield {"question_id": si * 100000 + int(r["question_id"]), "question": str(r["text"]),
                       "answers": [str(r["label"])] * 10, "answer_type": split, "question_type": "",
                       "image_name": r["image"], "image_bytes": None}


def mme_rows(src):
    """lmms-lab/MME, split test (2,374 yes/no questions, 2 per image, 14 categories).

    The source question ids are not unique, so the id is the row number (from 1). The
    image file is "<category>__<source path with / -> _>"; the MME scorer reads the
    category and the image pairing from this name.
    """
    for seq, r in enumerate(_parquet_rows(_files(src, "test-*.parquet")), start=1):
        cat = str(r["category"])
        yield {"question_id": seq, "question": str(r["question"]), "answers": [str(r["answer"]).lower()] * 10,
               "answer_type": cat, "question_type": "",
               "image_name": f"{cat}__{str(r['image']['path']).replace('/', '_')}", "image_bytes": r["image"]["bytes"]}


def gqa_rows(src):
    """lmms-lab/GQA testdev_balanced (12,578 questions on 398 images); answer_type = structural type."""
    images = {}
    for r in _parquet_rows(_files(src, "testdev_balanced_images/*.parquet"), batch_size=64):
        img = r["image"]
        images[str(r["id"])] = img["bytes"] if isinstance(img, dict) else img
    for r in _parquet_rows(_files(src, "testdev_balanced_instructions/*.parquet")):
        iid = str(r["imageId"])
        if iid not in images:
            continue
        types = r.get("types") or {}
        yield {"question_id": int(str(r["id"]).rsplit("_", 1)[-1]), "question": str(r["question"]),
               "answers": [str(r["answer"])] * 10, "answer_type": str(types.get("structural") or "other"),
               "question_type": str(types.get("semantic") or ""), "image_name": f"{iid}.jpg", "image_bytes": images[iid]}


def vqav2_testdev_rows(src):
    """Official VQAv2 test-dev2015 questions (107,394; no public answers: placeholder answer "a")."""
    qs = json.load(open(Path(src) / "v2_OpenEnded_mscoco_test-dev2015_questions.json"))["questions"]
    for r in sorted(qs, key=lambda r: int(r["question_id"])):
        yield {"question_id": int(r["question_id"]), "question": str(r["question"]), "answers": ["a"],
               "answer_type": "unknown", "question_type": "", "image_name": f"COCO_test2015_{int(r['image_id']):012d}.jpg",
               "image_bytes": None}


SHORT_ANSWER_READERS = {"okvqa": okvqa_rows, "vizwiz": vizwiz_rows, "pope": pope_rows, "mme": mme_rows, "gqa": gqa_rows,
                        "vqav2_testdev": vqav2_testdev_rows}


# ----------------------------------------------------------------------------- short answer


def build_short_answer(rows, tokenizer, image_dir: str | Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Rows -> (items sorted by question id, drops, annotations of every row).

    image_dir holds the COCO files (rows without bytes) or receives the exported images.
    The annotations keep all reference answers for the VQA soft accuracy.
    """
    image_dir = Path(image_dir)
    items, drops, annotations = [], [], []
    for r in rows:
        qid = r["question_id"]
        image_file = str(image_dir / r["image_name"])
        if r["image_bytes"] is None:
            if not Path(image_file).is_file():
                drops.append({"question_id": qid, "reason": "image_missing"})
                continue
        elif not Path(image_file).is_file():
            image_dir.mkdir(parents=True, exist_ok=True)
            Path(image_file).write_bytes(r["image_bytes"])
        canonical = Counter(r["answers"]).most_common(1)[0][0]  # ties: first seen
        annotations.append({"question_id": qid, "answers": [{"answer": a} for a in r["answers"]],
                            "multiple_choice_answer": canonical, "answer_type": r["answer_type"],
                            "question_type": r["question_type"]})
        question = f"{r['question']} {SHORT_ANSWER}"
        span = answer_span(tokenizer, question, canonical)
        if span is None:
            drops.append({"question_id": qid, "reason": "boundary"})
            continue
        if not 1 <= span["answer_tok_end"] - span["answer_tok_start"] <= MAX_ANSWER_TOKENS:
            drops.append({"question_id": qid, "reason": "answer_len"})
            continue
        if len(span["ids"]) + 1 > TEXT_CANVAS:
            drops.append({"question_id": qid, "reason": "caption_len"})
            continue
        items.append(make_item(qid, qid, image_file, question, canonical, span))
    items.sort(key=lambda it: it["question_id"])
    return items, drops, annotations


def build_vqav2_n1000(src, val2014_dir, val2017_dir, tokenizer, n: int = 1000, seed: int = 0):
    """VQAv2 val2014 n1000: a fixed random subset of the questions on COCO val2017 images.

    Selection (on the plain question): answer = multiple_choice_answer; clean boundary,
    1-8 answer tokens, caption fits; image id in val2017 (the 5,000 files in
    val2017_dir); sorted by question id; random.Random(seed).sample(n); sorted again.
    Then SHORT_ANSWER is appended and the span is computed again (an item that stops
    fitting, or whose answer reaches past the 12-slot window, is dropped).
    Images are the val2014 files. Returns (items, drops, annotations of the items).
    """
    questions = json.load(open(Path(src) / "v2_OpenEnded_mscoco_val2014_questions.json"))["questions"]
    ann = {int(a["question_id"]): a for a in json.load(open(Path(src) / "v2_mscoco_val2014_annotations.json"))["annotations"]}
    val2017 = {int(p.stem) for p in Path(val2017_dir).glob("*.jpg") if p.stem.isdigit()}
    population = []
    for q in questions:
        qid, image_id = int(q["question_id"]), int(q["image_id"])
        if qid not in ann or image_id not in val2017:
            continue
        image_file = str(Path(val2014_dir) / f"COCO_val2014_{image_id:012d}.jpg")
        if not Path(image_file).is_file():
            continue
        answer = str(ann[qid]["multiple_choice_answer"])
        span = answer_span(tokenizer, str(q["question"]), answer)
        if span is None or not 1 <= span["answer_tok_end"] - span["answer_tok_start"] <= MAX_ANSWER_TOKENS:
            continue
        if len(span["ids"]) + 1 > TEXT_CANVAS:
            continue
        population.append({"question_id": qid, "image_id": image_id, "image_file": image_file,
                           "question": str(q["question"]), "answer": answer})
    population.sort(key=lambda r: r["question_id"])
    picked = sorted(random.Random(seed).sample(population, n), key=lambda r: r["question_id"])
    items, drops = [], []
    for r in picked:
        question = f"{r['question']} {SHORT_ANSWER}"
        span = answer_span(tokenizer, question, r["answer"])
        if span is None:
            drops.append({"question_id": r["question_id"], "reason": "boundary"})
        elif len(span["ids"]) + 1 > TEXT_CANVAS:
            drops.append({"question_id": r["question_id"], "reason": "caption_len"})
        elif len(span["ids"]) >= span["answer_tok_start"] + QA_WINDOW:
            drops.append({"question_id": r["question_id"], "reason": "window"})
        else:
            items.append(make_item(r["question_id"], r["image_id"], r["image_file"], question, r["answer"], span))
    annotations = [{"question_id": it["question_id"], "answers": [{"answer": a["answer"]} for a in ann[it["question_id"]]["answers"]],
                    "multiple_choice_answer": ann[it["question_id"]]["multiple_choice_answer"],
                    "answer_type": ann[it["question_id"]]["answer_type"],
                    "question_type": ann[it["question_id"]]["question_type"]} for it in items]
    return items, drops, annotations


# ----------------------------------------------------------------------------- multiple choice
# MCQ readers yield: qid, question, options, gold_index, img_bytes, img_name, drop (None or a
# reason), and optionally str_qid / group / category.


def _options_cell(cell) -> list[str]:
    """Options as a list or as the string of a Python list."""
    if cell is None:
        return []
    if isinstance(cell, (list, tuple)):
        return [_s(x) for x in cell]
    s = str(cell).strip()
    if s.startswith("["):
        try:
            return [_s(x) for x in ast.literal_eval(s)]
        except (ValueError, SyntaxError):
            return []
    return []


def _first_present(row: dict, names):
    for n in names:
        if n in row:
            return row[n]
    return None


def mmbench_rows(src):
    """lmms-lab/MMBench en/dev (4,329 questions). The hint goes before the question, as in lmms-eval."""
    for r in _parquet_rows(_files(src, "dev-*.parquet"), batch_size=64):
        idx = int(r["index"])
        opts, letters = [], []
        for letter in "ABCD":
            v = _s(r.get(letter))
            if v:
                opts.append(v)
                letters.append(letter)
        gold = _s(r.get("answer")).upper()
        q, hint = _s(r.get("question")), _s(r.get("hint"))
        b, _ = _image_bytes(r.get("image"))
        yield {"qid": idx, "group": idx % 1000000, "category": _s(r.get("category")) or "other",
               "question": f"{hint}\n{q}" if hint else q, "options": opts,
               "gold_index": letters.index(gold) if gold in letters else -1, "img_bytes": b, "img_name": f"mmbench_{idx}",
               "drop": None if len(opts) >= 2 else "few_options"}


def ai2d_rows(src):
    """lmms-lab/ai2d test (3,088 questions). The id is the row number (from 1)."""
    for seq, r in enumerate(_parquet_rows(_files(src, "test-*.parquet"), batch_size=64), start=1):
        opts = _options_cell(_first_present(r, ("options", "choices")))
        ans = _s(_first_present(r, ("answer", "label")))
        gi = int(ans) if ans.isdigit() else (LETTERS.index(ans.upper()) if ans.upper() in LETTERS else -1)
        b, name = _image_bytes(r.get("image"))
        qid = _s(r.get("question_id")) or _s(r.get("id"))
        yield {"qid": int(re.sub(r"\D", "", qid) or seq) if qid else seq, "group": None, "category": None,
               "question": _s(r.get("question")), "options": opts, "gold_index": gi, "img_bytes": b,
               "img_name": f"ai2d_{Path(name).stem}" if name else f"ai2d_{seq}",
               "drop": None if len(opts) >= 2 else "few_options"}


def mmmu_rows(src):
    """MMMU/MMMU validation, 30 subject folders (900 questions).

    Kept: single-image multiple-choice items with text options. The id is
    (subject slot in sorted order + 1) * 10000 + the item number of the source id.
    "<image k>" placeholders are removed from the question.
    """
    import pyarrow.parquet as pq

    for si, path in enumerate(_files(src, "*/validation-*.parquet")):
        subject = Path(path).parent.name
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for r in batch.to_pylist():
                rid = _s(r.get("id"))
                seq = (si + 1) * 10000 + int(re.sub(r"\D", "", rid.split("_")[-1]) or 0)
                imgs = [_image_bytes(r.get(f"image_{i}")) for i in range(1, 8)]
                imgs = [b for b, _ in imgs if b]
                opts = _options_cell(r.get("options"))
                gold = _s(r.get("answer")).upper()
                drop = None
                if _s(r.get("question_type")) != "multiple-choice":
                    drop = "not_multiple_choice"
                elif len(imgs) != 1:
                    drop = "no_image" if not imgs else "multi_image"
                elif any("<img" in o.lower() or "<image" in o.lower() for o in opts):
                    drop = "option_has_image"
                elif len(opts) < 2:
                    drop = "few_options"
                q = re.sub(r"<image\s*\d*>", " ", _s(r.get("question")))
                yield {"qid": seq, "str_qid": rid, "group": None, "category": subject,
                       "question": re.sub(r"[ \t]+", " ", q).strip(), "options": opts,
                       "gold_index": LETTERS.index(gold) if gold in LETTERS else -1,
                       "img_bytes": imgs[0] if imgs else None, "img_name": f"mmmu_{rid}", "drop": drop}


def scienceqa_rows(src):
    """lmms-lab/ScienceQA, ScienceQA-IMG test (2,017 questions). The hint goes before the question."""
    for seq, r in enumerate(_parquet_rows(_files(src, "test-*.parquet"), batch_size=64), start=1):
        opts = _options_cell(_first_present(r, ("choices", "options")))
        ans = _s(_first_present(r, ("answer", "label")))
        b, _ = _image_bytes(r.get("image"))
        q, hint = _s(r.get("question")), _s(r.get("hint"))
        yield {"qid": seq, "group": None, "category": _s(r.get("subject")) or "other",
               "question": f"{hint}\n{q}" if hint else q, "options": opts, "gold_index": int(ans) if ans.isdigit() else -1,
               "img_bytes": b, "img_name": f"sqa_{seq}", "drop": None if b else "no_image"}


_RWQA_OPTION = re.compile(r"^\s*([A-D])[.)]\s*(.+?)\s*$")


def realworldqa_rows(src):
    """lmms-lab/RealWorldQA test (765 questions). Kept: questions with >= 2 lettered option lines.

    The options are lines "A. ..." inside the question text; the source instruction
    lines ("Please answer ...") are removed.
    """
    for seq, r in enumerate(_parquet_rows(_files(src, "test-*.parquet"), batch_size=64), start=1):
        raw_q = _s(_first_present(r, ("question", "query")))
        ans = _s(_first_present(r, ("answer", "label")))
        b, name = _image_bytes(r.get("image"))
        name = name or _s(r.get("image_path"))
        body, opts = [], []
        for line in raw_q.splitlines():
            m = _RWQA_OPTION.match(line)
            if m and LETTERS[len(opts)] == m.group(1):
                opts.append(m.group(2))
            elif not opts:
                body.append(line)
        body = [ln for ln in body if "please answer" not in ln.lower()]
        gi = -1
        if ans.upper() in LETTERS[: len(opts)] and len(ans) == 1:
            gi = LETTERS.index(ans.upper())
        elif ans.strip().lower() in [o.strip().lower() for o in opts]:
            gi = [o.strip().lower() for o in opts].index(ans.strip().lower())
        drop = "open_ended" if len(opts) < 2 else "bad_answer" if gi < 0 else "no_image" if not b else None
        yield {"qid": seq, "group": None, "category": None, "question": "\n".join(x for x in body if x.strip()).strip(),
               "options": opts, "gold_index": gi, "img_bytes": b,
               "img_name": f"rwqa_{Path(name).stem}" if name else f"rwqa_{seq}", "drop": drop}


MCQ_READERS = {"mmbench": mmbench_rows, "ai2d": ai2d_rows, "mmmu": mmmu_rows, "scienceqa_img": scienceqa_rows,
               "realworldqa": realworldqa_rows}


def mcq_question(question: str, options: list[str]) -> str:
    """Question, lettered options, instruction: the lmms-eval generative MCQ prompt."""
    return "\n".join([question.strip()] + [f"{LETTERS[i]}. {o}" for i, o in enumerate(options)] + [MCQ_INSTRUCTION])


def build_mcq(rows, tokenizer, image_dir: str | Path) -> tuple[list[dict], list[dict]]:
    """Rows -> (items sorted by str(question id), drops). Images of kept items go to image_dir."""
    image_dir = Path(image_dir)
    items, drops = [], []
    for n_raw, r in enumerate(rows, start=1):
        qid = r.get("qid") or n_raw
        if r["drop"]:
            drops.append({"question_id": qid, "reason": r["drop"]})
            continue
        if not 0 <= r["gold_index"] < len(r["options"]):
            drops.append({"question_id": qid, "reason": "bad_answer"})
            continue
        if not r["img_bytes"]:
            drops.append({"question_id": qid, "reason": "no_image"})
            continue
        if len(r["options"]) > len(LETTERS):
            drops.append({"question_id": qid, "reason": "too_many_options"})
            continue
        letter = LETTERS[r["gold_index"]]
        question = mcq_question(r["question"], r["options"])
        span = answer_span(tokenizer, question, letter)
        if span is None:
            drops.append({"question_id": qid, "reason": "boundary"})
            continue
        if len(span["ids"]) + 1 > TEXT_CANVAS:
            drops.append({"question_id": qid, "reason": "prompt_too_long"})
            continue
        image_file = image_dir / f"{r['img_name']}{_extension(r['img_bytes'])}"
        if not image_file.is_file():
            image_dir.mkdir(parents=True, exist_ok=True)
            image_file.write_bytes(r["img_bytes"])
        item = make_item(qid, qid, str(image_file), question, letter, span)
        item.update({"options": r["options"], "n_options": len(r["options"]), "answer_letter": letter,
                     "gold_index": r["gold_index"]})
        if r.get("str_qid"):
            item["source_id"] = r["str_qid"]
        if r.get("group") is not None:
            item["circular_group"] = r["group"]
        if r.get("category"):
            item["category"] = r["category"]
        items.append(item)
    items.sort(key=lambda it: str(it["question_id"]))
    return items, drops


# ----------------------------------------------------------------------------- one benchmark


def build(bench: str, src, image_dir, tokenizer, coco_val2017=None) -> dict:
    """Build one benchmark. Returns {"shards", "drops", "annotations"}.

    image_dir: COCO folder (okvqa, pope, vqav2_n1000: val2014; vqav2_testdev: test2015)
    or the folder that receives the exported images (all other benchmarks).
    Every source row becomes one item or one drop.
    """
    annotations = None
    if bench == "vqav2_n1000":
        items, drops, annotations = build_vqav2_n1000(src, image_dir, coco_val2017, tokenizer)
        return {"shards": [items], "drops": drops, "annotations": annotations}
    if bench in MCQ_READERS:
        items, drops = build_mcq(MCQ_READERS[bench](src), tokenizer, image_dir)
    else:
        items, drops, annotations = build_short_answer(SHORT_ANSWER_READERS[bench](src), tokenizer, image_dir)
        if bench == "vqav2_testdev":
            annotations = None  # no ground truth
    return {"shards": shard_items(items), "drops": drops, "annotations": annotations}
