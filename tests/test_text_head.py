"""CPU tests of the text-head trainer (juice/train/text_head.py); no data, no pretrained weights."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from juice.heads import ParallelTextHead
from juice.representations import EOS_ID, PAD_ID, TEXT_LATENT_STD, TextBatch
from juice.train.text_head import (
    EMBEDDING_KEYS,
    TextHeadConfig,
    TrainableTextHead,
    batch_metrics,
    corrupt,
    first_eos_index,
    head_loss,
    labels_and_weights,
    save_head,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "train"


def _tiny_head(vocab: int = 40) -> TrainableTextHead:
    from transformers import T5Config, T5EncoderModel

    torch.manual_seed(0)
    config = T5Config(vocab_size=vocab, d_model=16, d_kv=8, d_ff=32, num_layers=2, num_heads=2,
                      feed_forward_proj="gated-gelu", dropout_rate=0.0)
    return TrainableTextHead(T5EncoderModel(config), torch.nn.Linear(16, vocab))


class TextHeadTest(unittest.TestCase):
    def test_config_of_the_released_head(self):
        cfg = TextHeadConfig.from_yaml(CONFIGS / "text_head.yaml")
        self.assertEqual((cfg.stages, cfg.steps, cfg.batch_size, cfg.learning_rate), (2, 8000, 64, 1e-4))
        self.assertEqual((cfg.corrupt_frac, cfg.tail_weight, cfg.seed, cfg.num_workers), (0.5, 0.2, 0, 1))
        pre = yaml.safe_load((CONFIGS / "pretrain_siglip2.yaml").read_text())
        self.assertEqual(cfg.train_manifests, pre["train_manifests"])
        self.assertEqual(cfg.train_manifest_counts, pre["train_manifest_counts"])
        data = cfg.data_config()
        self.assertEqual((data.seed, data.wds_shard_seed, data.micro_batch_size, data.num_workers), (4242, 4242, 64, 1))

    def test_unknown_key_fails(self):
        with self.assertRaises(ValueError):
            TextHeadConfig.from_yaml(CONFIGS / "text_head.yaml", no_such_key=1)

    def test_labels_and_weights(self):
        tb = TextBatch.from_ids(torch.tensor([[5, 6, 7, 1, 0, 0]]), torch.tensor([[1, 1, 1, 1, 0, 0]]))
        labels, weights = labels_and_weights(tb, 0.2)
        self.assertEqual(labels.tolist(), [[5, 6, 7, EOS_ID, EOS_ID, EOS_ID]])
        self.assertTrue(torch.allclose(weights, torch.tensor([[1, 1, 1, 1, 0.2, 0.2]])))
        self.assertNotIn(PAD_ID, labels.tolist()[0])

    def test_loss_is_normalized_by_the_weights(self):
        labels = torch.randint(0, 50, (2, 6))
        weights = torch.rand(2, 6) + 0.1
        loss = head_loss(torch.zeros(2, 6, 50), labels, weights)
        self.assertAlmostEqual(float(loss), math.log(50), places=5)

    def test_corrupt_mixes_toward_noise(self):
        torch.manual_seed(0)
        a = corrupt(torch.ones(4, 128, 8))
        torch.manual_seed(0)
        b = corrupt(torch.ones(4, 128, 8))
        self.assertTrue(torch.equal(a, b), "the corruption uses the torch RNG only")
        self.assertFalse(torch.equal(a, torch.ones_like(a)))
        torch.manual_seed(0)
        zero = corrupt(torch.zeros(64, 128, 8))
        self.assertLess(float(zero.std()), 0.2)  # (1 - lam) * N(0, 0.2^2), lam in (0, 1)

    def test_metrics(self):
        tb = TextBatch.from_ids(torch.tensor([[5, 6, 1, 0], [7, 1, 0, 0]]), torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]))
        self.assertEqual(first_eos_index(tb.ids).tolist(), [2, 1])
        logits = torch.nn.functional.one_hot(torch.tensor([[5, 6, 1, 1], [7, 3, 1, 1]]), 10).float()
        m = batch_metrics(logits, tb)
        self.assertAlmostEqual(m["tok_acc"], 4 / 5)
        self.assertAlmostEqual(m["exact_match"], 0.5)
        self.assertAlmostEqual(m["eos_pos_mae"], 0.5)

    def test_saved_folder_loads_into_the_inference_head(self):
        head = _tiny_head().eval()
        with tempfile.TemporaryDirectory() as tmp:
            save_head(head, Path(tmp), {"steps": 0})
            loaded = ParallelTextHead.from_folder(tmp)
        saved = {k: v for k, v in head.state_dict().items() if k not in EMBEDDING_KEYS}
        for key, value in saved.items():
            self.assertTrue(torch.equal(loaded.state_dict()[key], value), key)
        # The trainer reads T5 states; the inference head reads normalized states (T5 state / 0.2).
        normalized = torch.randn(2, 128, 16)
        with torch.no_grad():
            self.assertTrue(torch.equal(loaded(normalized), head(normalized.float() * TEXT_LATENT_STD)))


if __name__ == "__main__":
    unittest.main()
