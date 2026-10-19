"""CPU tests of the training code (no data, no pretrained weights)."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from juice.model import MMRiT, MMRiTConfig
from juice.representations import TextBatch
from juice.train.config import TrainConfig
from juice.train.loss import I2T, JOINT, T2I, joint_flow_loss, stratified_regimes, token_time_patterns
from juice.train.trainer import lr_at, update_ema

# configs/train/text_head.yaml is the config of the text-head trainer (tests/test_text_head.py).
CONFIGS = sorted(p for p in (Path(__file__).resolve().parents[1] / "configs" / "train").glob("*.yaml")
                 if p.name != "text_head.yaml" and not p.name.startswith("task_head_"))


def _tiny_model() -> MMRiT:
    torch.manual_seed(0)
    model = MMRiT(MMRiTConfig(image_latent_dim=24, image_grid=4, text_dim=32, text_bottleneck_dim=8, hidden_size=64,
                              depth=2, text_preamble_depth=1, num_heads=2, head_dim=32, time_tokens=4))
    model.init_weights()
    return model


class TrainTest(unittest.TestCase):
    def test_configs_load(self):
        self.assertEqual(len(CONFIGS), 7)
        for path in CONFIGS:
            cfg = TrainConfig.from_yaml(path)
            self.assertEqual(cfg.batch_size % (cfg.micro_batch_size * cfg.grad_accum_steps), 0, path.name)

    def test_unknown_key_fails(self):
        with self.assertRaises(ValueError):
            TrainConfig.from_yaml(CONFIGS[0], no_such_key=1)

    def test_stratified_counts_are_exact(self):
        for probs, batch in (([0.556, 0.222, 0.222], 8), ([0.6, 0.2, 0.2], 8), ([0.0, 1.0, 0.0], 8), ([1, 1, 1], 7)):
            regime = stratified_regimes(probs, batch, "cpu")
            counts = [int((regime == r).sum()) for r in (JOINT, T2I, I2T)]
            self.assertEqual(sum(counts), batch)
            if probs[T2I] == probs[I2T] and probs[JOINT] > 0:
                self.assertEqual(counts[T2I], counts[I2T], f"{probs}: edges must get equal counts")

    def test_token_time_patterns(self):
        t = torch.tensor([0.3, 1.0, 0.5, 0.2])
        eligible = torch.tensor([True, True, False, True])
        tok = token_time_patterns(t, 10, 1.0, 0.0, lambda k: torch.full((k,), 0.9), eligible)
        self.assertTrue(bool((tok[1] == 1.0).all()), "a clean row stays clean")
        self.assertTrue(bool((tok[2] == 0.5).all()), "a row that is not eligible keeps its time")
        for i in (0, 3):  # span pattern: the row time on one span, clean elsewhere
            self.assertTrue(bool(((tok[i] == t[i]) | (tok[i] == 1.0)).all()))
            self.assertGreater(int((tok[i] == t[i]).sum()), 0)

    def test_warmup_then_constant(self):
        cfg = TrainConfig(run_name="t", train_manifests=[], train_manifest_counts=[], learning_rate=1e-4, warmup_steps=250)
        self.assertAlmostEqual(lr_at(cfg, 0), 1e-6 + (1e-4 - 1e-6) / 250)
        self.assertAlmostEqual(lr_at(cfg, 249), 1e-4)
        self.assertEqual(lr_at(cfg, 5000), 1e-4)

    def test_ema(self):
        a, b = torch.nn.Linear(3, 3), torch.nn.Linear(3, 3)
        with torch.no_grad():
            a.weight.fill_(1.0)
            b.weight.fill_(0.0)
        update_ema(a, b, 0.9)
        self.assertTrue(torch.allclose(a.weight, torch.full_like(a.weight, 0.9)))

    def test_loss_and_gradients(self):
        model = _tiny_model()
        cfg = TrainConfig(run_name="t", train_manifests=[], train_manifest_counts=[], qa_answer_window=6,
                          task_mixture=[0.6, 0.2, 0.2], qa_rows_force_i2t=False, qa_span_regimes=["i2t", "product"],
                          img_known_inpaint=True)
        b, n = 6, 20
        ids = torch.randint(2, 100, (b, n))
        mask = (torch.arange(n)[None, :] < torch.tensor([12, 12, 12, 5, 20, 9])[:, None]).long()
        spans = torch.tensor([8, 8, 8, -1, -1, -1])
        known = torch.rand(b, 16) < 0.5
        torch.manual_seed(0)
        loss, metrics = joint_flow_loss(
            model, torch.randn(b, 16, 24), torch.randn(b, n, 32), TextBatch.from_ids(ids, mask), cfg, spans,
            torch.randn(b, n, 32), img_known=known, img_x0_in=torch.randn(b, 16, 24),
        )
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(int(metrics["rows_qa"]) <= 3, True)
        grads = [p.grad for p in model.parameters()]
        self.assertTrue(all(g is not None for g in grads), "every parameter gets a gradient")

    def test_answer_past_window_fails(self):
        model = _tiny_model()
        cfg = TrainConfig(run_name="t", train_manifests=[], train_manifest_counts=[], qa_answer_window=2,
                          task_mixture=[0.0, 0.0, 1.0])
        ids = torch.randint(2, 100, (2, 20))
        mask = (torch.arange(20)[None, :] < 12).long().expand(2, -1)
        with self.assertRaises(ValueError):
            joint_flow_loss(model, torch.randn(2, 16, 24), torch.randn(2, 20, 32), TextBatch.from_ids(ids, mask),
                            cfg, torch.tensor([4, 4]), torch.randn(2, 20, 32))


if __name__ == "__main__":
    unittest.main()
