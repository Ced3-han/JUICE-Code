"""CPU tests of the task-head trainer (juice/train/task_head.py); tiny synthetic data, no network.

The end-to-end test needs the Flan-T5-Small weights in the local Hugging Face cache; it is
skipped when they are not there.
"""

from __future__ import annotations

import io
import json
import math
import tarfile
import tempfile
import unittest
from pathlib import Path

import torch

from juice.heads import LinearTextHead
from juice.representations import TEXT_ENCODER_ID, TEXT_ENCODER_REVISION, TextBatch
from juice.train.task_head import TaskHeadConfig, save_task_head, task_head_loss, train_task_head
from juice.train.text_head import MIX_P_MEAN, MIX_P_STD, corrupt

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "train"


def _t5_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return all(isinstance(try_to_load_from_cache(TEXT_ENCODER_ID, name, revision=TEXT_ENCODER_REVISION), str)
                   for name in ("config.json", "model.safetensors", "spiece.model", "tokenizer_config.json"))
    except Exception:
        return False


def _write_shard(path: Path, texts: list[str]) -> None:
    from PIL import Image

    with tarfile.open(path, "w") as tar:
        for i, text in enumerate(texts):
            buf = io.BytesIO()
            Image.new("RGB", (32, 32), (i * 20 % 255, 0, 0)).save(buf, format="PNG")
            for ext, data in (("png", buf.getvalue()), ("txt", text.encode("utf-8"))):
                info = tarfile.TarInfo(f"{i:08d}.{ext}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))


class TaskHeadConfigTest(unittest.TestCase):
    def test_configs_of_the_planning_heads(self):
        for task, manifest, count in (("jmaze", "jmaze_multi_train", 200000), ("vsp", "vsp_train", 55939)):
            cfg = TaskHeadConfig.from_yaml(CONFIGS / f"task_head_{task}.yaml")
            self.assertEqual(cfg.run_name, f"task_head_{task}")
            self.assertEqual((cfg.train_manifests, cfg.train_manifest_counts),
                             ([f"data/manifests/{manifest}.txt"], [count]))
            self.assertEqual((cfg.steps, cfg.batch_size, cfg.learning_rate, cfg.weight_decay), (1500, 64, 1e-3, 0.01))
            self.assertEqual((cfg.corrupt_frac, cfg.noise_std, cfg.seed), (0.5, 1.0, 67280421310721))
            data = cfg.data_config()
            self.assertEqual((data.seed, data.wds_shard_seed, data.micro_batch_size, data.num_workers, data.shuffle_buffer),
                             (4242, 4242, 64, 2, 1000))

    def test_overrides_and_unknown_keys(self):
        cfg = TaskHeadConfig.from_yaml(CONFIGS / "task_head_jmaze.yaml", seed=1, output_dir="/tmp/x")
        self.assertEqual((cfg.seed, cfg.output_dir), (1, "/tmp/x"))
        self.assertEqual(cfg.to_dict()["seed"], 1)
        with self.assertRaises(ValueError):
            TaskHeadConfig.from_yaml(CONFIGS / "task_head_jmaze.yaml", tail_weight=0.2)  # a text-head key
        with self.assertRaises(ValueError):
            TaskHeadConfig(train_manifests=["a"], train_manifest_counts=[1, 2])


class TaskHeadStepTest(unittest.TestCase):
    def test_corruption_statistics(self):
        torch.manual_seed(0)
        z = torch.randn(1_000_000) * MIX_P_STD + MIX_P_MEAN
        lam_mean = float(torch.sigmoid(z).mean())
        one_minus_lam_rms = float((1 - torch.sigmoid(z)).pow(2).mean().sqrt())
        # noise_std 0: the output is lam, one value per slot (shared by all channels).
        torch.manual_seed(1)
        lam = corrupt(torch.ones(256, 128, 64), noise_std=0.0)
        self.assertEqual(float(lam.std(dim=-1).max()), 0.0)
        self.assertTrue(bool(((lam > 0) & (lam < 1)).all()))
        self.assertAlmostEqual(float(lam.mean()), lam_mean, delta=0.005)
        # Zero states: the output is (1 - lam) * N(0, noise_std^2).
        torch.manual_seed(2)
        out = corrupt(torch.zeros(256, 128, 64), noise_std=1.0)
        self.assertAlmostEqual(float(out.mean()), 0.0, delta=0.005)
        self.assertAlmostEqual(float(out.pow(2).mean().sqrt()), one_minus_lam_rms, delta=0.005)
        torch.manual_seed(2)
        self.assertTrue(torch.equal(out, corrupt(torch.zeros(256, 128, 64), noise_std=1.0)), "torch RNG only")

    def test_loss_uses_only_the_valid_slots(self):
        tb = TextBatch.from_ids(torch.tensor([[5, 6, 7, 1, 0, 0]]), torch.tensor([[1, 1, 1, 1, 0, 0]]))
        logits = torch.zeros(1, 6, 50)
        self.assertAlmostEqual(float(task_head_loss(logits, tb)), math.log(50), places=5)
        logits[0, 4:, 3] = 100.0  # PAD slots carry no loss
        self.assertAlmostEqual(float(task_head_loss(logits, tb)), math.log(50), places=5)

    def test_loss_decreases_on_a_toy_batch(self):
        torch.manual_seed(0)
        head = LinearTextHead(16, 40)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=0.01)
        ids = torch.randint(2, 40, (8, 12))
        mask = (torch.arange(12)[None, :] < torch.randint(3, 12, (8, 1))).long()
        tb = TextBatch.from_ids(ids, mask)
        states = torch.nn.functional.embedding(tb.ids, torch.randn(40, 16))
        losses = []
        for _ in range(30):
            loss = task_head_loss(head(corrupt(states.clone(), 0.2)), tb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        self.assertLess(losses[-1], 0.5 * losses[0])

    def test_saved_folder_loads_into_the_inference_head(self):
        torch.manual_seed(0)
        head = LinearTextHead(16, 40)
        with tempfile.TemporaryDirectory() as tmp:
            save_task_head(head, Path(tmp), {"steps": 0})
            loaded = LinearTextHead.from_folder(tmp)
            config = json.loads((Path(tmp) / "config.json").read_text())
        self.assertEqual((config["dim"], config["vocab_size"], config["training"]), (16, 40, {"steps": 0}))
        for key, value in head.state_dict().items():
            self.assertTrue(torch.equal(loaded.state_dict()[key], value), key)


@unittest.skipUnless(_t5_cached(), "Flan-T5-Small is not in the local Hugging Face cache")
class TaskHeadTrainTest(unittest.TestCase):
    def test_three_steps_on_cpu(self):
        texts = [f"Question: Maze 5x5. Start ({i % 5},0). Goal (1,2). Answer: U R R" for i in range(8)]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _write_shard(tmp / "toy_000.tar", texts)
            (tmp / "toy.txt").write_text("toy_000.tar\n")
            cfg = TaskHeadConfig(train_manifests=[str(tmp / "toy.txt")], train_manifest_counts=[8],
                                 output_dir=str(tmp / "out"), run_name="toy", num_workers=0, batch_size=4,
                                 shuffle_buffer=4, image_size=32, steps=3, log_every=1, seed=0)
            folder = train_task_head(cfg, device="cpu")
            log = [json.loads(line) for line in (tmp / "out" / "toy" / "log.jsonl").read_text().splitlines()]
            head = LinearTextHead.from_folder(folder)
            config = json.loads((folder / "config.json").read_text())
        self.assertEqual([r["step"] for r in log], [1, 2, 3])
        self.assertTrue(all(math.isfinite(r["loss"]) for r in log))
        self.assertEqual((head.linear.in_features, head.linear.out_features), (512, 32128))
        self.assertEqual((config["training"]["steps"], config["training"]["seed"]), (3, 0))


if __name__ == "__main__":
    unittest.main()
