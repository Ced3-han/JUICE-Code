"""CPU tests of the question-answering probe builders (juice/eval/vqa_probes.py).

Most tests use a character-level stand-in tokenizer. One test checks the real
Flan-T5 tokenizer against an item of the paper's n1000 probe; it is skipped when the tokenizer
is not in the local Hugging Face cache.
"""

import tempfile
import unittest
from pathlib import Path

from juice.eval import vqa_probes as vp
from juice.pipeline import SHORT_ANSWER
from juice.representations import EOS_ID, TEXT_CANVAS


class CharTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


class ReversingTokenizer:
    """Token ids that do not keep the prefix: every boundary is unclean."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in reversed(text)]}


def short_row(qid, answers, image_name="a.jpg", question="What is it?"):
    return {"question_id": qid, "question": question, "answers": answers, "answer_type": "other",
            "question_type": "", "image_name": image_name, "image_bytes": b"\xff\xd8 not an image"}


class LayoutTest(unittest.TestCase):
    def test_span_and_item(self):
        tok = CharTokenizer()
        span = vp.answer_span(tok, "Why?", "ok")
        self.assertEqual(span["caption"], "Question: Why? Answer: ok")
        self.assertEqual(span["answer_tok_start"], len("Question: Why? Answer:"))
        self.assertEqual(span["answer_tok_end"], len("Question: Why? Answer: ok"))
        item = vp.make_item(7, 7, "x.jpg", "Why?", "ok", span)
        self.assertEqual(len(item["token_ids_64"]), TEXT_CANVAS)
        self.assertEqual(item["token_ids_64"][item["n_content_tokens"]:], [EOS_ID] * (TEXT_CANVAS - item["n_content_tokens"]))
        self.assertIsNone(vp.answer_span(ReversingTokenizer(), "Why?", "ok"))

    def test_shards_are_round_robin(self):
        self.assertEqual(vp.shard_items(list(range(10))), [[0, 4, 8], [1, 5, 9], [2, 6], [3, 7]])

    def test_mcq_question(self):
        self.assertEqual(vp.mcq_question(" Q? ", ["x", "y"]), "Q?\nA. x\nB. y\n" + vp.MCQ_INSTRUCTION)


class ShortAnswerTest(unittest.TestCase):
    def test_funnel_order_and_annotations(self):
        tok = CharTokenizer()  # one token per character: answers longer than 8 characters are dropped
        rows = [short_row(3, ["no", "yes", "yes", "no"]), short_row(1, ["red"] * 10), short_row(2, ["a long answer"] * 10)]
        with tempfile.TemporaryDirectory() as d:
            items, drops, ann = vp.build_short_answer(rows, tok, d)
            self.assertTrue((Path(d) / "a.jpg").is_file())  # exported image
        self.assertEqual([it["question_id"] for it in items], [1, 3])
        self.assertEqual(drops, [{"question_id": 2, "reason": "answer_len"}])
        self.assertEqual(len(ann), 3)
        self.assertEqual(items[1]["answer"], "no")  # mode, ties -> first seen
        self.assertEqual(items[0]["question"], f"What is it? {SHORT_ANSWER}")

    def test_missing_coco_image_is_dropped(self):
        row = {**short_row(1, ["x"] * 10), "image_bytes": None}
        with tempfile.TemporaryDirectory() as d:
            items, drops, ann = vp.build_short_answer([row], CharTokenizer(), d)
        self.assertEqual((items, drops, ann), ([], [{"question_id": 1, "reason": "image_missing"}], []))


class MCQTest(unittest.TestCase):
    def row(self, qid, options, gold, drop=None):
        return {"qid": qid, "question": "Pick.", "options": options, "gold_index": gold, "img_bytes": b"\x89PNG....",
                "img_name": f"img{qid}", "drop": drop, "category": "c"}

    def test_funnel_and_string_order(self):
        rows = [self.row(9, ["x", "y"], 1), self.row(10, ["x", "y"], 0), self.row(11, ["x"], 0, drop="few_options"),
                self.row(12, ["x", "y"], 5), self.row(13, ["x" * 200, "y"], 0)]
        with tempfile.TemporaryDirectory() as d:
            items, drops = vp.build_mcq(rows, CharTokenizer(), d)
            self.assertTrue(Path(items[0]["image_file"]).name == "img10.png")
        self.assertEqual([it["question_id"] for it in items], [10, 9])  # sorted as strings
        self.assertEqual([it["answer"] for it in items], ["A", "B"])
        self.assertEqual([d["reason"] for d in drops], ["few_options", "bad_answer", "prompt_too_long"])

    def test_realworldqa_reader(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow not installed")
        q = ["Is the door open?\nA. yes\nB. no\nPlease answer directly with a single word or number.", "How many cars?"]
        table = pa.table({"image": [{"bytes": b"\x89PNG..", "path": "p1.png"}, {"bytes": b"\x89PNG..", "path": "p2.png"}],
                          "image_path": ["p1.png", "p2.png"], "question": q, "answer": ["B", "3"]})
        with tempfile.TemporaryDirectory() as d:
            pq.write_table(table, Path(d) / "test-00000-of-00001.parquet")
            rows = list(vp.realworldqa_rows(d))
        self.assertEqual((rows[0]["question"], rows[0]["options"], rows[0]["gold_index"]), ("Is the door open?", ["yes", "no"], 1))
        self.assertEqual((rows[0]["drop"], rows[1]["drop"], rows[0]["img_name"]), (None, "open_ended", "rwqa_p1"))


class RealTokenizerTest(unittest.TestCase):
    def test_paper_vqav2_item(self):
        try:
            tok = vp.load_tokenizer()
        except Exception as e:  # noqa: BLE001 - offline machine without the cached tokenizer
            self.skipTest(f"tokenizer unavailable: {e}")
        span = vp.answer_span(tok, f"What color is the floor? {SHORT_ANSWER}", "brown")
        # VQAv2 val2014 question 139002, item 0 of the paper's n1000 probe.
        self.assertEqual(span["ids"], [11860, 10, 363, 945, 19, 8, 1501, 58, 11801, 8, 822, 338, 3, 9, 712, 1448, 42,
                                       9261, 5, 11801, 10, 4216])
        self.assertEqual((span["answer_tok_start"], span["answer_tok_end"]), (21, 22))


if __name__ == "__main__":
    unittest.main()
