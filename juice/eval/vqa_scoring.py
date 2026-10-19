"""Scorers of the question-answering table.

  vqa_accuracy   VQAv2 / OK-VQA / VizWiz / POPE / GQA: the official VQA soft accuracy
                 (POPE and GQA store their one label 10 times, so it is exact match).
  mme_scores     MME perception and cognition (Fu et al. 2023): per category
                 100 * (accuracy + accuracy+), summed over the categories.
  mcq_accuracy   MMBench / AI2D / MMMU / ScienceQA-IMG / RealWorldQA: the option letter
                 is parsed from the answer as in lmms-eval; an unparsed answer is wrong.
  pope_metrics   POPE accuracy, precision, recall and F1 per split (yes = positive).

The table reports accuracy over the evaluated items (the probe items). Items that the
probe funnel dropped can also be counted as wrong: full_set_accuracy.

Sums run in the order of the rows (shard 0 first), as in the paper's scoring, so the
floating-point results are equal to the paper's values.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict

# ----------------------------------------------------------------------------- VQA soft accuracy
# Port of the official evaluator (GT-Vision-Lab/VQA, vqaEval.py, BSD) with its quirks:
# the period regex has a malformed lookbehind, normalization is skipped when the 10
# reference answers are all equal, and punctuation is tested on the original string.

CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've",
    "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
    "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
    "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
    "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's",
    "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
    "somethingd've": "something'd've", "something'dve": "something'd've",
    "somethingll": "something'll", "thats": "that's", "thered": "there'd",
    "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're",
    "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're",
    "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've",
    "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's",
    "whered": "where'd", "wheres": "where's", "whereve": "where've", "whod": "who'd",
    "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's",
    "whove": "who've", "whyll": "why'll", "whyre": "why're", "whys": "why's",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", "yall": "y'all",
    "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd",
    "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll",
    "youre": "you're", "youve": "you've",
}
MANUAL_MAP = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
              "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
ARTICLES = ["a", "an", "the"]
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]


def _process_punctuation(in_text: str) -> str:
    out_text = in_text
    for p in PUNCT:
        if (p + " " in in_text or " " + p in in_text) or re.search(COMMA_STRIP, in_text) is not None:
            out_text = out_text.replace(p, "")
        else:
            out_text = out_text.replace(p, " ")
    return PERIOD_STRIP.sub("", out_text, re.UNICODE)


def _process_digit_article(in_text: str) -> str:
    words = [MANUAL_MAP.get(w, w) for w in in_text.lower().split()]
    words = [w for w in words if w not in ARTICLES]
    return " ".join(CONTRACTIONS.get(w, w) for w in words)


def _clean(s: str) -> str:
    return s.replace("\n", " ").replace("\t", " ").strip()


def vqa_item_accuracy(pred: str, references: list[str]) -> float:
    """Official accuracy of one answer: mean over annotators i of min(1, #{j != i: ref_j == pred} / 3)."""
    refs = [_clean(a) for a in references]
    res = _clean(pred)
    if len(set(refs)) > 1:
        refs = [_process_digit_article(_process_punctuation(a)) for a in refs]
        res = _process_digit_article(_process_punctuation(res))
    accs = [min(1.0, sum(o == res for o in refs[:i] + refs[i + 1:]) / 3.0) for i in range(len(refs))]
    return sum(accs) / len(accs)


def vqa_accuracy(rows: list[dict], annotations: list[dict]) -> dict:
    """rows: [{question_id, answer}] in shard order; annotations: the probe annotations file."""
    by_qid = {a["question_id"]: a for a in annotations}
    accs, per_type = [], defaultdict(list)
    for r in rows:
        a = by_qid[r["question_id"]]
        acc = vqa_item_accuracy(r["answer"], [d["answer"] for d in a["answers"]])
        accs.append(acc)
        per_type[a["answer_type"]].append(acc)
    return {"n": len(accs), "accuracy": sum(accs) / len(accs),
            "per_answer_type": {t: {"n": len(v), "accuracy": sum(v) / len(v)} for t, v in sorted(per_type.items())}}


def full_set_accuracy(accuracy: float, n_evaluated: int, n_dropped: int) -> float:
    """Accuracy with every dropped question counted as wrong."""
    return accuracy * n_evaluated / (n_evaluated + n_dropped)


# ----------------------------------------------------------------------------- POPE


def pope_metrics(rows: list[dict], annotations: list[dict]) -> dict:
    """Per split and pooled: accuracy, precision, recall, F1, yes rate. A non-yes/no answer is wrong and "not yes"."""
    by_qid = {a["question_id"]: a for a in annotations}
    counts = {k: dict(tp=0, fp=0, fn=0, tn=0, other=0, n=0, yes=0, correct=0) for k in ("random", "popular", "adversarial", "all")}
    for r in rows:
        a = by_qid[r["question_id"]]
        gt, pred = a["multiple_choice_answer"].strip().lower().rstrip("."), r["answer"].strip().lower().rstrip(".")
        for d in (counts[a["answer_type"]], counts["all"]):
            d["n"] += 1
            d["other"] += pred not in ("yes", "no")
            d["yes"] += pred == "yes"
            d["correct"] += pred == gt
            key = ("tp" if pred == "yes" else "fn") if gt == "yes" else ("fp" if pred == "yes" else "tn")
            d[key] += 1
    out = {}
    for k, d in counts.items():
        if not d["n"]:
            continue
        precision = d["tp"] / (d["tp"] + d["fp"]) if d["tp"] + d["fp"] else None  # None: undefined
        recall = d["tp"] / (d["tp"] + d["fn"]) if d["tp"] + d["fn"] else None
        f1 = None if precision is None or recall is None else (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        out[k] = {"n": d["n"], "accuracy": d["correct"] / d["n"], "precision": precision, "recall": recall, "f1": f1,
                  "yes_rate": d["yes"] / d["n"], "non_yes_no": d["other"]}
    return out


# ----------------------------------------------------------------------------- MME

MME_PERCEPTION = ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"]
MME_COGNITION = ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]


def mme_scores(rows: list[dict], items: list[dict]) -> dict:
    """MME perception (max 2000) and cognition (max 800).

    items: all probe items (shard order). The category and the image of an item come
    from its image file name "<category>__<name>"; the two questions of an image form
    a pair for accuracy+. A question without an answer is wrong.
    """
    pred = {r["question_id"]: r["answer"] for r in rows}
    per_image, per_category = defaultdict(list), defaultdict(list)
    for it in items:
        image = os.path.basename(it["image_file"])
        category = image.split("__", 1)[0]
        qid = it["question_id"]
        ok = qid in pred and vqa_item_accuracy(pred[qid], [str(it["answer"]).lower()] * 10) >= 1.0
        per_image[(category, image)].append(ok)
        per_category[category].append(ok)
    categories = {}
    for c in MME_PERCEPTION + MME_COGNITION:
        qs = per_category[c]
        images = [v for (cc, _), v in per_image.items() if cc == c]
        acc, acc_plus = sum(qs) / len(qs), sum(all(v) for v in images) / len(images)
        categories[c] = {"n_questions": len(qs), "n_images": len(images), "accuracy": acc, "accuracy_plus": acc_plus,
                         "score": 100 * (acc + acc_plus)}
    return {"perception": sum(categories[c]["score"] for c in MME_PERCEPTION),
            "cognition": sum(categories[c]["score"] for c in MME_COGNITION),
            "exact_accuracy": sum(map(sum, per_category.values())) / len(items), "categories": categories}


# ----------------------------------------------------------------------------- multiple choice

_LETTER = re.compile(r"(?<![a-z0-9])([a-i])(?![a-z0-9])", re.IGNORECASE)
LETTERS = "ABCDEFGHI"


def parse_mcq_answer(text: str, n_options: int, options: list[str]) -> str | None:
    """Option letter of a generated answer, or None.

    1. The first standalone letter A-I that is an option of this item.
    2. Else, an exact (case-insensitive, stripped) match of the whole answer to one option text.
    """
    valid = set(LETTERS[:n_options])
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    for m in _LETTER.finditer(text):
        if m.group(1).upper() in valid:
            return m.group(1).upper()
    low = text.strip(" .!?,:;'\"").lower()
    for i, opt in enumerate(options[:n_options]):
        if low and low == re.sub(r"\s+", " ", str(opt)).strip().strip(" .!?,:;'\"").lower():
            return LETTERS[i]
    return None


def mcq_accuracy(rows: list[dict], items: list[dict]) -> dict:
    """Letter accuracy over the answered probe items (unparsed = wrong)."""
    by_qid = {str(it["question_id"]): it for it in items}
    n_correct, n_unparsed, per_category = 0, 0, defaultdict(list)
    for r in rows:
        it = by_qid[str(r["question_id"])]
        letter = parse_mcq_answer(r["answer"], int(it["n_options"]), list(it["options"]))
        ok = int(letter == str(it["answer_letter"]).upper())
        n_correct += ok
        n_unparsed += letter is None
        if it.get("category"):
            per_category[str(it["category"])].append(ok)
    n = len(rows)
    return {"n": n, "accuracy": n_correct / n, "n_correct": n_correct, "n_unparsed": n_unparsed,
            "chance": sum(1.0 / int(by_qid[str(r["question_id"])]["n_options"]) for r in rows) / n,
            "per_category": {c: {"n": len(v), "accuracy": sum(v) / len(v)} for c, v in sorted(per_category.items())}}
