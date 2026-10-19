"""Joint planning inference, shared by JMaze and VSP.

A planning puzzle gives an input image and a question. The model makes the
solved image and the answer text together, in one run of the joint sampler:

  image  the patches that the input image already shows correctly (background,
         walls or holes, start, goal) are known: they stay clean at t = 1 with the
         encoder states of the input image. The other patches are generated.
  text   [question prefix (clean) | answer window, 64 slots (generated) |
         tail (clean, hidden from attention)]. The clean slots hold the T5
         states of "Question: <q> Answer:" + EOS, as in question answering.

Readouts:
  text   a linear head (juice.heads.LinearTextHead) takes the argmax token of each
         slot from the first answer slot on, up to the first EOS.
  image  the RAE decoder makes pixels; each task reads its route from the pixels.

Training captions use the template "Question: {question} Answer: {answer}"; the
answer span is the token range of the answer in that caption.

The batch that starts at puzzle index `start` draws its noise from the seed
seed + 1000003 * start (noise_generator), as in the paper's runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from juice.representations import PAD_ID, TEXT_CANVAS, TextBatch

NOISE_SEED_STRIDE = 1_000_003


def noise_generator(seed: int, start: int) -> torch.Generator:
    """Noise generator of the batch that starts at puzzle index `start`."""
    return torch.Generator().manual_seed(seed + NOISE_SEED_STRIDE * start)


def caption(question: str, answer: str) -> str:
    return f"Question: {question} Answer: {answer}"


def answer_span(tokenizer, question: str, answer: str) -> dict | None:
    """Token range of the answer inside the training caption (content tokens, no EOS).

    Returns None when the tokens of "Question: <q> Answer:" are not a strict prefix
    of the tokens of the full caption (a SentencePiece merge across the boundary).
    """
    def ids(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    full = ids(caption(question, answer))
    prefix = ids(f"Question: {question} Answer:")
    if full[: len(prefix)] != prefix or len(full) <= len(prefix):
        return None
    return {"caption": caption(question, answer), "answer_tok_start": len(prefix),
            "answer_tok_end": len(full), "n_content_tokens": len(full)}


def images_to_unit(images) -> torch.Tensor:
    """PIL images or uint8 [H, W, 3] arrays -> float32 [B, 3, H, W] in [-1, 1].

    Computed as x / 255 * 2 - 1, the rule of the paper's planning runs. (It can
    differ from juice.representations.to_unit_range in the last bit.)
    """
    arr = np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in images])
    return torch.from_numpy(arr).permute(0, 3, 1, 2) * 2 - 1


def images_to_float(decoded: torch.Tensor) -> np.ndarray:
    """Decoder output [B, 3, H, W] in [-1, 1] -> float32 [B, H, W, 3] in [0, 1] for the route readouts."""
    return ((decoded.detach().float().cpu().clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).numpy()


class PlanOutput(NamedTuple):
    answers: list[str]      # decoded answer text, stripped
    images: np.ndarray      # [B, 224, 224, 3] float32 in [0, 1]


class Planner:
    """A planning release folder (model + task_head) ready for joint sampling."""

    def __init__(self, folder: str | Path, device: str | torch.device = "cuda"):
        from juice.heads import LinearTextHead
        from juice.pipeline import JuicePipeline

        self.pipe = JuicePipeline(folder, device)
        self.settings = self.pipe.defaults["plan"]
        self.head = LinearTextHead.from_folder(Path(folder) / "task_head").to(self.pipe.device)

    @property
    def task(self) -> str:
        return self.settings["task"]

    @torch.no_grad()
    def solve(self, images, questions: list[str], known_patches, generator: torch.Generator,
              steps: int | None = None) -> PlanOutput:
        """Solve one batch.

        images: input renders (PIL or uint8 arrays, 224 px); questions: one per image;
        known_patches: [B, 256] 0/1 (1 = the patch keeps the input image state).
        """
        from juice.heads import first_eos_answer
        from juice.pipeline import precision
        from juice.sampling import sample_joint

        pipe, s = self.pipe, self.settings
        device, window, b = pipe.device, int(s["answer_window"]), len(questions)
        prefixes = [pipe.question_ids(q) for q in questions]
        spans = torch.tensor([len(p) for p in prefixes])
        if int(spans.max()) + window > TEXT_CANVAS:
            raise ValueError("question prefix too long for the text canvas")
        ids = torch.full((b, TEXT_CANVAS), PAD_ID, dtype=torch.long)
        for i, p in enumerate(prefixes):
            ids[i, : len(p)] = torch.tensor(p, dtype=torch.long)
        pos = torch.arange(TEXT_CANVAS)[None, :]
        tail = pos >= (spans[:, None] + window)
        tb = TextBatch.from_ids(ids, pos < (spans[:, None] + 1))
        with precision(s["precision"]):
            img = pipe.image_encoder(images_to_unit(images).to(device)).float()
            txt = pipe.text.encode(tb.to(device), dtype=torch.float32).float()
            sample = sample_joint(
                pipe.model, img, txt,
                img_known=torch.as_tensor(np.asarray(known_patches), dtype=torch.bool),
                txt_known=(pos < spans[:, None]) | tail,
                steps=steps or int(s["steps"]), cfg_scale=float(s["cfg_scale"]),
                t_shift=pipe.flow["t_shift"], generator=generator, txt_key_mask=~tail,
                noise_scale=pipe.flow["noise_scale"], text_noise_scale=pipe.flow["text_noise_scale"],
                text_t_shift=pipe.flow["text_t_shift"],
            )
            decoded = pipe.decoder(sample.img)
            tokens = self.head(sample.txt).argmax(dim=-1).cpu()
        answers = []
        for row, span in zip(tokens, spans.tolist()):
            answer_ids = first_eos_answer(row, span, TEXT_CANVAS)   # no window limit: read to the canvas end
            answers.append(pipe.text.decode(answer_ids).strip() if answer_ids else "")
        return PlanOutput(answers=answers, images=images_to_float(decoded))


def summarize(rows: list[dict], keys) -> dict:
    """Mean of each score key over the rows (booleans count as 0/1), rounded to 6 digits."""
    return {"count": len(rows), **{k: round(sum(float(r[k]) for r in rows) / len(rows), 6) for k in keys}}
