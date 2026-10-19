"""CPU unit tests of the training-data packers (no network, no real tokenizer).

    python -m unittest tests.test_data_packing -v
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from scripts.data import common, make_manifests, pack_cambrian, pack_prompted_captions
from scripts.data.clean_journeydb import clean_tar
from scripts.data.make_cambrian_mmu_jsonl import convert_line
from scripts.data.split_shards import split_tar

REPO = Path(__file__).resolve().parents[1]


class WordTokenizer:
    """Stand-in for the T5 tokenizer: one id per space-separated word."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [sum(map(ord, w)) % 1000 + 2 for w in text.split()]}


def write_tar(path: Path, samples: list[tuple[str, dict[str, bytes]]]) -> None:
    with tarfile.open(path, "w") as t:
        for key, members in samples:
            for suffix, data in members.items():
                info = tarfile.TarInfo(f"{key}.{suffix}")
                info.size = len(data)
                t.addfile(info, io.BytesIO(data))


def read_tar(path: Path) -> list[tuple[str, bytes]]:
    with tarfile.open(path) as t:
        return [(m.name, t.extractfile(m).read()) for m in t if m.isfile()]


class SpanTest(unittest.TestCase):
    def test_span_of_the_answer(self):
        span = common.answer_span(WordTokenizer(), "what is it?", "a red car")
        self.assertEqual(span["caption"], "Question: what is it? Answer: a red car")
        self.assertEqual((span["answer_tok_start"], span["answer_tok_end"], span["n_content_tokens"]), (5, 8, 8))
        self.assertEqual(span["answer_span_len"], 3)

    def test_boundary_merge_is_dropped(self):
        class Merging(WordTokenizer):
            def __call__(self, text, add_special_tokens=False):  # "Answer:" changes when an answer follows
                ids = super().__call__(text)["input_ids"]
                return {"input_ids": ids if text.endswith("Answer:") else ids[:2] + [7] + ids[3:]}

        self.assertIsNone(common.answer_span(Merging(), "q", "a b"))

    def test_fits(self):
        span = {"answer_span_len": 8, "n_content_tokens": 127}
        self.assertTrue(common.fits(span, 1, 8))
        self.assertFalse(common.fits({**span, "answer_span_len": 9}, 1, 8))
        self.assertFalse(common.fits({**span, "n_content_tokens": 128}, 1, 8))
        self.assertFalse(common.fits({**span, "answer_span_len": 8}, 9, 100))

    def test_sentence_truncation(self):
        tok = WordTokenizer()
        answer = "One two three. Four five six! Seven eight nine ten? Eleven."
        cut, span, kept, total = common.truncate_at_sentence(tok, "q", answer, max_answer_tokens=7)
        self.assertEqual((cut, kept, total), ("One two three. Four five six!", 2, 4))
        self.assertEqual(span["answer_span_len"], 6)
        self.assertIsNone(common.truncate_at_sentence(tok, "q", "One two three. Four.", max_answer_tokens=2))
        # The full answer is never a candidate: one sentence cannot be cut.
        self.assertIsNone(common.truncate_at_sentence(tok, "q", "One two.", max_answer_tokens=100))


class PromptTest(unittest.TestCase):
    def test_templates(self):
        self.assertEqual(len(common.CONCISE_TEMPLATES), 11)
        self.assertEqual(len(common.DETAILED_TEMPLATES), 16)
        self.assertEqual(common.CONCISE_TEMPLATES[0], "Describe the image concisely.")
        self.assertEqual(common.DETAILED_TEMPLATES[0], "Describe the following image in detail.")

    def test_prompt_is_a_hash_of_the_key(self):
        key = "000000000009_0"
        h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)
        self.assertEqual(common.pick_prompt("concise", key), common.CONCISE_TEMPLATES[h % 11])
        self.assertEqual(common.pick_prompt("detailed", key), common.DETAILED_TEMPLATES[h % 16])
        self.assertEqual(common.pick_prompt("concise", key), common.pick_prompt("concise", key))

    def test_parse_shard(self):
        self.assertEqual(common.parse_shard(""), (0, 1))
        self.assertEqual(common.parse_shard("3/8"), (3, 8))
        with self.assertRaises(SystemExit):
            common.parse_shard("8/8")


def cambrian_args(**kw):
    base = dict(sources={"gqa", "coco"}, source_file_prefix="", coco_val_mode="exclude", keep_prompt_suffix=False,
                skip_cued_rows=False, only_mcq=False, truncate_answers="none", max_answer_tokens=8)
    base.update(kw)
    return argparse.Namespace(**base)


def conv(*turns):
    return [{"from": "human" if i % 2 == 0 else "gpt", "value": v} for i, v in enumerate(turns)]


class CambrianPass1Test(unittest.TestCase):
    CUE = common.SHORT_ANSWER_CUE

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        rows = [
            {"image": "gqa/1.jpg", "conversations": conv(f"<image>\nIs it red?\n{self.CUE}", "Yes", "What color?", "red")},
            {"image": "coco/train2017/000000000139.jpg", "conversations": conv("Q?\nA. x\nB. y\nAnswer with the option's letter.", "A")},
            {"image": "coco/train2017/000000000285.jpg", "conversations": conv("Q2?", "B")},
            {"image": "vg/5.jpg", "conversations": conv("Q3?", "C")},
            {"image": "gqa/2.jpg", "conversations": conv("Long?", "one two three four five six seven eight nine")},
        ]
        self.jsonl = Path(self.tmp.name) / "rows.jsonl"
        self.jsonl.write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.lines = self.jsonl.read_bytes().splitlines(keepends=True)

    def tearDown(self):
        self.tmp.cleanup()

    def run_pass1(self, args, excluded=(), val_ids=("000000000285",)):
        counts = {k: 0 for k in ("lines", "lines_source_kept", "lines_valprobe_excluded", "rows_coco_val_filtered", "pairs")}
        return pack_cambrian.collect_candidates(self.jsonl, args, set(excluded), set(val_ids), counts)

    def test_v1_strips_the_cue_and_word_filter(self):
        got = self.run_pass1(cambrian_args())
        self.assertEqual([(c[1], c[2], c[4], c[5]) for c in got],
                         [(0, 0, "Is it red?", "Yes"), (0, 1, "What color?", "red"),
                          (1, 0, "Q?\nA. x\nB. y\nAnswer with the option's letter.", "A")])

    def test_v2_keeps_and_propagates_the_cue(self):
        got = self.run_pass1(cambrian_args(keep_prompt_suffix=True))
        self.assertEqual(got[0][4], f"Is it red?\n{self.CUE}")
        self.assertEqual(got[1][4], f"What color? {self.CUE}")
        skipped = self.run_pass1(cambrian_args(keep_prompt_suffix=True, skip_cued_rows=True))
        self.assertEqual([c[1] for c in skipped], [1])

    def test_coco_val_modes_and_mcq(self):
        only = self.run_pass1(cambrian_args(sources={"coco"}, coco_val_mode="only"))
        self.assertEqual([c[1] for c in only], [2])
        keep = self.run_pass1(cambrian_args(sources={"coco"}, coco_val_mode="keep"))
        self.assertEqual([c[1] for c in keep], [1, 2])
        mcq = self.run_pass1(cambrian_args(only_mcq=True))
        self.assertEqual([c[1] for c in mcq], [1])

    def test_linehash_and_truncation_mode(self):
        h = hashlib.sha1(self.lines[0]).hexdigest()
        self.assertEqual([c[1] for c in self.run_pass1(cambrian_args(), excluded=[h])], [1])
        long = self.run_pass1(cambrian_args(sources={"gqa"}, truncate_answers="sentence"))
        self.assertIn(4, [c[1] for c in long])

    def test_mmu_jsonl_line(self):
        line = json.dumps({"image": "a/b.jpg", "conversations": [{"from": "human", "value": "<image>\nHi"}]})
        self.assertEqual(json.loads(convert_line(line))["conversations"][0]["value"], "Hi")
        self.assertIsNone(convert_line(json.dumps({"conversations": []})))


class PackerEndToEndTest(unittest.TestCase):
    def test_cambrian_sample_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            (root / "jsons").mkdir(parents=True)
            (root / "images" / "gqa").mkdir(parents=True)
            (root / "images" / "gqa" / "1.JPEG").write_bytes(b"img")
            row = {"image": "gqa/1.JPEG", "conversations": conv("<image>\nIs it red?", "yes")}
            (root / "jsons" / "rows.jsonl").write_text(json.dumps(row) + "\n")
            with mock.patch.object(pack_cambrian, "load_t5_tokenizer", WordTokenizer):
                pack_cambrian.main(["--root", str(root), "--jsonl", "jsons/rows.jsonl", "--exclude-linehash", "",
                                    "--sources", "gqa", "--qa-window", "12", "--out", f"{tmp}/out", "--tar-prefix", "p"])
            members = read_tar(Path(tmp) / "out" / "p_00000.tar")
            self.assertEqual([n for n, _ in members], ["gqa_0000000_00.jpg", "gqa_0000000_00.txt", "gqa_0000000_00.json"])
            self.assertEqual(members[1][1], b"Question: Is it red? Answer: yes\n")
            self.assertEqual(json.loads(members[2][1]), {
                "source": "gqa", "cambrian_line": 0, "pair_idx": 0, "image": "gqa/1.JPEG", "answer": "yes",
                "answer_tok_start": 5, "answer_tok_end": 6, "n_content_tokens": 6, "qa_window": 12})

    def test_prompted_caption_sample_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "caps_00000.tar"
            write_tar(src, [("k/1", {"jpg": b"a", "txt": b"A  dog.\n"}), ("k/2", {"txt": b"no image"}),
                            ("k/3", {"png": b"b", "txt": b"A cat."})])
            (Path(tmp) / "m.txt").write_text(f"{src}\n")
            with mock.patch.object(pack_prompted_captions, "load_t5_tokenizer", WordTokenizer):
                pack_prompted_captions.main(["--src-manifest", f"{tmp}/m.txt", "--style", "concise", "--qa-window", "48",
                                             "--max-answer-tokens", "40", "--source-name", "capc", "--limit-rows", "1",
                                             "--out", f"{tmp}/out", "--tar-prefix", "c"])
            members = read_tar(Path(tmp) / "out" / "c_00000.tar")
            self.assertEqual([n for n, _ in members], ["caps_00000__k_1.jpg", "caps_00000__k_1.txt", "caps_00000__k_1.json"])
            prompt = common.pick_prompt("concise", "k/1")
            self.assertEqual(members[1][1].decode(), f"Question: {prompt} Answer: A dog.\n")
            side = json.loads(members[2][1])
            self.assertEqual((side["src_key"], side["answer"], side["qa_window"]), ("k/1", "A dog.", 48))


class ShardToolsTest(unittest.TestCase):
    def test_split_keeps_order_and_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "s.tar"
            samples = [(f"{i:03d}", {"jpg": bytes([i]), "txt": b"t", "json": b"{}"}) for i in range(5)]
            write_tar(src, samples)
            self.assertEqual(split_tar(src, Path(tmp) / "out", 2), [3, 2])
            parts = [read_tar(Path(tmp) / "out" / f"s_{j:02d}.tar") for j in range(2)]
            self.assertEqual(parts[0] + parts[1], read_tar(src))

    def test_clean_drops_image_and_caption(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "j.tar"
            write_tar(src, [("01", {"jpg": b"a", "txt": b"x"}), ("02", {"jpg": b"bad", "txt": b"y"})])
            kept, dropped = clean_tar(src, Path(tmp) / "j_clean.tar", ["02.jpg"])
            self.assertEqual((kept, dropped), (2, 2))
            self.assertEqual([n for n, _ in read_tar(Path(tmp) / "j_clean.tar")], ["01.jpg", "01.txt"])

    def test_replicas_are_hard_links_in_copy_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = []
            for i in range(2):
                p = Path(tmp) / f"b{i}.tar"
                p.write_bytes(b"x")
                base.append(str(p))
            lines = make_manifests.make_replica("r_x3", base, 3, Path(tmp) / "links")
            self.assertEqual([str(Path(l).relative_to(tmp)) for l in lines],
                             [f"links/r_x3/c{c:02d}/b{i}.tar" for c in range(3) for i in range(2)])
            self.assertTrue(all(os.stat(l).st_ino == os.stat(base[i % 2]).st_ino for i, l in enumerate(lines)))
            self.assertEqual(len(set(lines)), 6)
            with self.assertRaises(SystemExit):
                make_manifests.write_manifest(Path(tmp), "dup", [base[0], base[0]])


class ManifestTableTest(unittest.TestCase):
    def test_table_covers_the_configs_with_the_paper_counts(self):
        rows = {name: r for name, (_, r) in make_manifests.PACKS.items()}
        for name, (base, copies) in make_manifests.REPLICAS.items():
            rows[name] = rows[base] * copies
        known = set(rows) | set(make_manifests.SOURCE_POOLS) | set(make_manifests.PLANNING)
        for path in glob.glob(str(REPO / "configs/train/*.yaml")):
            cfg = yaml.safe_load(Path(path).read_text())
            for manifest, count in zip(cfg["train_manifests"], cfg["train_manifest_counts"]):
                name = Path(manifest).stem
                self.assertEqual(manifest, f"data/manifests/{name}.txt")
                self.assertIn(name, known, path)
                if name in rows:
                    self.assertEqual(rows[name], count, f"{path}: {name}")


if __name__ == "__main__":
    unittest.main()
