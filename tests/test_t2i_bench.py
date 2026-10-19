"""CPU tests of the text-to-image benchmark helpers (juice/eval/t2i_bench.py)."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from juice.eval.t2i_bench import (
    PromptItem,
    dpg_grid_path,
    geneval_sample_path,
    load_dpg_prompts,
    load_geneval_prompts,
    make_grid,
    output_complete,
    prompt_seed,
    shard_items,
    write_dpg,
    write_geneval,
)


class SeedTest(unittest.TestCase):
    def test_seed_of_prompt(self):
        # The paper's generator: seed + 1000 * index + made, with made = 0 for one batch of 4.
        self.assertEqual(prompt_seed(0, 0), 0)
        self.assertEqual(prompt_seed(0, 552), 552000)
        self.assertEqual(prompt_seed(7, 3), 3007)

    def test_seeds_do_not_depend_on_shard_count(self):
        items = [PromptItem(i, f"{i:05d}", f"p{i}") for i in range(553)]
        reference = {it.index: prompt_seed(0, it.index) for it in items}
        for n in (1, 2, 3, 4, 7, 8):
            seen = {}
            for s in range(n):
                for it in shard_items(items, n, s):
                    seen[it.index] = prompt_seed(0, it.index)
            self.assertEqual(seen, reference)  # every prompt once, same seed

    def test_shard_is_round_robin(self):
        self.assertEqual(shard_items(list(range(10)), 3, 1), [1, 4, 7])
        with self.assertRaises(ValueError):
            shard_items([1, 2], 2, 2)


class PromptTest(unittest.TestCase):
    def test_geneval_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evaluation_metadata.jsonl"
            rows = [{"tag": "single_object", "include": [{"class": "bench", "count": 1}], "prompt": "a photo of a bench"},
                    {"tag": "colors", "include": [{"class": "cow", "count": 1, "color": "red"}], "prompt": "a photo of a red cow"}]
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            items = load_geneval_prompts(path)
        self.assertEqual([(it.index, it.item_id, it.prompt) for it in items],
                         [(0, "00000", "a photo of a bench"), (1, "00001", "a photo of a red cow")])
        self.assertEqual(items[1].metadata, rows[1])

    def test_dpg_prompts_one_item_per_id_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dpg_bench.csv"
            with path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["item_id", "text", "proposition_id", "question_natural_language"])
                w.writerow(["partiprompts97", 'A "quoted", long\nprompt', 1, "Is there a space?"])
                w.writerow(["partiprompts97", 'A "quoted", long\nprompt', 2, "Is there a man?"])
                w.writerow(["countbench3", "three cats", 1, "Are there cats?"])
            items = load_dpg_prompts(path)
        self.assertEqual([(it.index, it.item_id, it.prompt) for it in items],
                         [(0, "partiprompts97", 'A "quoted", long\nprompt'), (1, "countbench3", "three cats")])


def solid(color, size=224):
    return Image.new("RGB", (size, size), color)


class LayoutTest(unittest.TestCase):
    def test_grid_order(self):
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        grid = make_grid([solid(c) for c in colors])
        self.assertEqual(grid.size, (448, 448))
        for c, xy in zip(colors, [(0, 0), (224, 0), (0, 224), (224, 224)]):
            self.assertEqual(grid.getpixel((xy[0] + 5, xy[1] + 5)), c)
        with self.assertRaises(ValueError):
            make_grid([solid((0, 0, 0))])

    def test_geneval_layout(self):
        item = PromptItem(7, "00007", "a photo of a cow", {"tag": "single_object", "prompt": "a photo of a cow"})
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(output_complete("geneval", tmp, item))
            write_geneval(tmp, item, [solid((k, k, k)) for k in range(4)])
            self.assertEqual(geneval_sample_path(tmp, item, 3), Path(tmp) / "00007" / "samples" / "00003.png")
            self.assertEqual(sorted(p.name for p in (Path(tmp) / "00007" / "samples").iterdir()),
                             ["00000.png", "00001.png", "00002.png", "00003.png"])
            meta = (Path(tmp) / "00007" / "metadata.jsonl").read_text()
            self.assertEqual(json.loads(meta), item.metadata)
            self.assertTrue(meta.endswith("\n"))
            self.assertTrue(output_complete("geneval", tmp, item))

    def test_dpg_layout(self):
        item = PromptItem(0, "partiprompts97", "x")
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(output_complete("dpg", tmp, item))
            write_dpg(tmp, item, [solid((k, k, k)) for k in range(4)])
            self.assertEqual(dpg_grid_path(tmp, item), Path(tmp) / "partiprompts97.png")
            self.assertEqual(Image.open(dpg_grid_path(tmp, item)).size, (448, 448))
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["partiprompts97.png"])  # no temp files left
            self.assertTrue(output_complete("dpg", tmp, item))


if __name__ == "__main__":
    unittest.main()
