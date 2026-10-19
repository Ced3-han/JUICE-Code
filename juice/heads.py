"""Text readout: generated text states -> tokens.

JUICE generates text as Flan-T5 encoder states, not as tokens. A readout head maps
the 128 generated states to token ids in one forward pass (no autoregression):

    ParallelTextHead   Flan-T5-Small encoder stack (8 bidirectional layers) + LM head.
                       Used for all question-answering results of the paper.
    LinearTextHead     one linear layer per slot. Used for the planning tasks
                       (JMaze, VSP); see juice/tasks/planning.py.

The answer is read with the "first EOS" rule: argmax tokens from the first answer
slot up to the first predicted EOS inside the answer window. The rule uses no
ground-truth length.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from juice.representations import EOS_ID, TEXT_LATENT_STD

# The head reads states, not token ids, so the T5 token-embedding table is not stored.
_EMBEDDING_KEYS = {"encoder.shared.weight", "encoder.encoder.embed_tokens.weight"}


class ParallelTextHead(nn.Module):
    def __init__(self, t5_config: dict, vocab_size: int):
        super().__init__()
        from transformers import T5Config, T5EncoderModel

        self.encoder = T5EncoderModel(T5Config.from_dict(t5_config))
        self.lm_head = nn.Linear(int(t5_config["d_model"]), vocab_size)

    @classmethod
    def from_folder(cls, folder: str | Path) -> "ParallelTextHead":
        from safetensors.torch import load_file

        folder = Path(folder)
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        head = cls(config["t5_config"], int(config["vocab_size"]))
        missing, unexpected = head.load_state_dict(load_file(str(folder / "model.safetensors")), strict=False)
        if set(missing) - _EMBEDDING_KEYS or unexpected:
            raise RuntimeError(f"text head weights do not match: missing={missing} unexpected={unexpected}")
        return head.eval().requires_grad_(False)

    def forward(self, txt_states: torch.Tensor) -> torch.Tensor:
        """Normalized text states [B, L, 512] -> logits [B, L, vocab]. Every slot sees every slot."""
        x = (txt_states.float() * TEXT_LATENT_STD).to(dtype=self.lm_head.weight.dtype)
        mask = torch.ones(x.shape[:2], dtype=torch.long, device=x.device)
        return self.lm_head(self.encoder(inputs_embeds=x, attention_mask=mask).last_hidden_state)


def first_eos_answer(token_ids: torch.Tensor, start: int, window: int, eos_id: int = EOS_ID) -> list[int]:
    """Tokens from `start` up to (not including) the first EOS inside [start, start + window)."""
    out: list[int] = []
    for p in range(start, min(start + window, token_ids.shape[0])):
        t = int(token_ids[p])
        if t == eos_id:
            break
        out.append(t)
    return out


def clean_answer(text: str) -> str:
    """Decoded answer -> scored answer: strip, lowercase, drop trailing " .!?,"."""
    return text.strip().lower().rstrip(" .!?,")


class LinearTextHead(nn.Module):
    """One linear layer: normalized text state [512] -> token logits.

    The planning tasks (JMaze, VSP) read their answers with this head. It was fit
    on the T5 states of the task captions, so it only knows the task vocabulary.
    It reads each slot alone (no attention between slots).
    """

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(dim, vocab_size)

    @classmethod
    def from_folder(cls, folder: str | Path) -> "LinearTextHead":
        from safetensors.torch import load_file

        folder = Path(folder)
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        head = cls(int(config["dim"]), int(config["vocab_size"]))
        head.load_state_dict(load_file(str(folder / "model.safetensors")), strict=True)
        return head.eval().requires_grad_(False)

    def forward(self, txt_states: torch.Tensor) -> torch.Tensor:
        """Normalized text states [B, L, 512] -> logits [B, L, vocab]."""
        return self.linear(txt_states.float())
