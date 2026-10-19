"""CPU tests of the question-answering scorers (juice/eval/vqa_scoring.py)."""

import unittest

from juice.eval.vqa_scoring import (
    MME_COGNITION,
    MME_PERCEPTION,
    full_set_accuracy,
    mcq_accuracy,
    mme_scores,
    parse_mcq_answer,
    pope_metrics,
    vqa_accuracy,
    vqa_item_accuracy,
)


class VQAAccuracyTest(unittest.TestCase):
    def test_soft_accuracy_is_min_of_k_over_3(self):
        for k, want in [(0, 0.0), (1, 0.3), (2, 0.6), (3, 0.9), (4, 1.0), (10, 1.0)]:
            refs = ["cat"] * k + ["dog"] * (10 - k)
            self.assertAlmostEqual(vqa_item_accuracy("cat", refs), want, places=12)

    def test_normalization(self):
        refs = ["2"] * 5 + ["two dogs"] * 5
        self.assertEqual(vqa_item_accuracy("two", refs), 1.0)       # number words -> digits
        refs = ["dog"] * 5 + ["cat"] * 5
        self.assertEqual(vqa_item_accuracy("a dog.", refs), 1.0)    # articles and periods
        self.assertEqual(vqa_item_accuracy("dog!", refs), 1.0)      # punctuation

    def test_no_normalization_when_references_agree(self):
        # Official quirk: with 10 equal references the answer is compared as is.
        self.assertEqual(vqa_item_accuracy("yes", ["yes"] * 10), 1.0)
        self.assertEqual(vqa_item_accuracy("yes.", ["yes"] * 10), 0.0)

    def test_dataset_accuracy_and_full_set(self):
        ann = [{"question_id": 1, "answers": [{"answer": "yes"}] * 10, "answer_type": "yes/no"},
               {"question_id": 2, "answers": [{"answer": "red"}] * 10, "answer_type": "other"}]
        m = vqa_accuracy([{"question_id": 1, "answer": "yes"}, {"question_id": 2, "answer": "blue"}], ann)
        self.assertEqual(m["accuracy"], 0.5)
        self.assertEqual(m["per_answer_type"]["yes/no"]["accuracy"], 1.0)
        self.assertEqual(full_set_accuracy(0.5, 2, 2), 0.25)


class PopeTest(unittest.TestCase):
    def test_metrics(self):
        ann = [{"question_id": i, "multiple_choice_answer": gt, "answer_type": "random"}
               for i, gt in enumerate(["yes", "yes", "no", "no"])]
        rows = [{"question_id": i, "answer": p} for i, p in enumerate(["yes", "no", "yes", "maybe"])]
        m = pope_metrics(rows, ann)["random"]
        self.assertEqual(m["accuracy"], 0.25)
        self.assertEqual((m["precision"], m["recall"], m["f1"]), (0.5, 0.5, 0.5))
        self.assertEqual(m["non_yes_no"], 1)


class MMETest(unittest.TestCase):
    def test_perception_and_cognition(self):
        items, rows, qid = [], [], 0
        for c in MME_PERCEPTION + MME_COGNITION:
            for q, ans in enumerate(["yes", "no"]):
                qid += 1
                items.append({"question_id": qid, "image_file": f"/x/{c}__img.png", "answer": ans})
                # existence: both right (score 200); other categories: second question wrong (score 50)
                rows.append({"question_id": qid, "answer": ans if (c == "existence" or q == 0) else "yes"})
        m = mme_scores(rows, items)
        self.assertEqual(m["categories"]["existence"]["score"], 200.0)
        self.assertEqual(m["categories"]["count"]["score"], 50.0)
        self.assertEqual(m["perception"], 200.0 + 9 * 50.0)
        self.assertEqual(m["cognition"], 4 * 50.0)

    def test_missing_answer_is_wrong(self):
        items = [{"question_id": i + 1, "image_file": f"{c}__a.png", "answer": "yes"}
                 for i, c in enumerate(MME_PERCEPTION + MME_COGNITION)]
        self.assertEqual(mme_scores([], items)["perception"], 0.0)


class MCQTest(unittest.TestCase):
    def test_parse(self):
        opts = ["red", "blue", "green"]
        self.assertEqual(parse_mcq_answer("b", 3, opts), "B")
        self.assertEqual(parse_mcq_answer("(c)", 3, opts), "C")
        self.assertEqual(parse_mcq_answer("blue", 3, opts), "B")        # option text
        self.assertIsNone(parse_mcq_answer("d", 3, opts))               # not an option of this item
        self.assertIsNone(parse_mcq_answer("", 3, opts))
        self.assertEqual(parse_mcq_answer("the answer is c", 3, opts), "C")

    def test_accuracy(self):
        items = [{"question_id": 1, "n_options": 2, "options": ["x", "y"], "answer_letter": "A", "category": "k"},
                 {"question_id": 2, "n_options": 4, "options": ["p", "q", "r", "s"], "answer_letter": "D"}]
        m = mcq_accuracy([{"question_id": 1, "answer": "a"}, {"question_id": 2, "answer": "zzz"}], items)
        self.assertEqual((m["accuracy"], m["n_unparsed"]), (0.5, 1))
        self.assertEqual(m["chance"], (0.5 + 0.25) / 2)


if __name__ == "__main__":
    unittest.main()
