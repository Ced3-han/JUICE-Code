"""High-level inference: text-to-image generation and question answering.

    from juice.pipeline import JuicePipeline
    pipe = JuicePipeline.from_pretrained("path/or/hf-repo", device="cuda")
    images = pipe.text_to_image("a red bicycle next to a blue door", num_images=4, seed=0)
    answers = pipe.answer(["photo.jpg"], ["What color is the door?"])

The defaults are the settings of the paper (config.json, "defaults").
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import torch

from juice.heads import ParallelTextHead, clean_answer, first_eos_answer
from juice.model import MMRiT, MMRiTConfig
from juice.representations import (
    PAD_ID,
    TEXT_CANVAS,
    T5TextEncoder,
    TextBatch,
    build_image_decoder,
    build_image_encoder,
    load_image,
    to_pil,
    to_unit_range,
)
from juice.sampling import sample_joint, sample_text_to_image

QA_TEMPLATE = "Question: {question} Answer:"
# Instruction that the paper appends to open questions of the short-answer benchmarks.
SHORT_ANSWER = "Answer the question using a single word or phrase."
HEAD_CHUNK = 64  # text-head batch of the paper's runs


@contextmanager
def precision(mode: str):
    """"fp32": TF32 matmuls off. "tf32": TF32 matmuls and convolutions on."""
    if mode not in ("fp32", "tf32"):
        raise ValueError(f"precision must be 'fp32' or 'tf32', got {mode!r}")
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = mode == "tf32"
    if mode == "tf32":
        torch.backends.cudnn.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


class JuicePipeline:
    def __init__(self, folder: str | Path, device: str | torch.device = "cuda"):
        from safetensors.torch import load_file

        self.folder = Path(folder)
        self.device = torch.device(device)
        self.config = json.loads((self.folder / "config.json").read_text(encoding="utf-8"))
        self.flow = self.config["flow"]
        self.defaults = self.config["defaults"]
        self.space = self.config["image_space"]
        self.model = MMRiT(MMRiTConfig(**self.config["architecture"]))
        self.model.load_state_dict(load_file(str(self.folder / "model.safetensors")), strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        self.text = T5TextEncoder().to(self.device)
        self._image_encoder = self._decoder = self._text_head = None

    @classmethod
    def from_pretrained(cls, name_or_path: str, device: str | torch.device = "cuda") -> "JuicePipeline":
        path = Path(name_or_path)
        if not path.is_dir():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(name_or_path))
        return cls(path, device)

    # Components that a task may not need are loaded on first use.
    @property
    def image_encoder(self):
        if self._image_encoder is None:
            self._image_encoder = build_image_encoder(self.space).to(self.device)
        return self._image_encoder

    @property
    def decoder(self):
        if self._decoder is None:
            self._decoder = build_image_decoder(self.space).to(self.device)
        return self._decoder

    @property
    def text_head(self) -> ParallelTextHead:
        if self._text_head is None:
            self._text_head = ParallelTextHead.from_folder(self.folder / "text_head").to(self.device)
        return self._text_head

    # ------------------------------------------------------------------ text to image

    def _encode_condition_bf16(self, tb: TextBatch) -> torch.Tensor:
        # Text-to-image runs the T5 encoder in bfloat16, as in the paper's runs.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            return self.text.encode(tb.to(self.device), dtype=torch.bfloat16).float()

    @torch.no_grad()
    def text_to_image_states(self, prompt: str, num_images: int = 4, seed: int = 0,
                             steps: int | None = None, cfg_scale: float | None = None) -> torch.Tensor:
        """Image states [num_images, 256, D]. Images are made in batches of 4; batch k uses seed + 4k."""
        d = self.defaults["text_to_image"]
        with precision(d["precision"]):
            null_tb = self.text.tokenize([d["null_caption"]])
            null_txt = self._encode_condition_bf16(null_tb)
            out = []
            for made in range(0, num_images, 4):
                tb = self.text.tokenize([prompt] * min(4, num_images - made))
                states = sample_text_to_image(
                    self.model, self._encode_condition_bf16(tb), tb.valid.to(self.device),
                    steps=steps or d["steps"], cfg_scale=d["cfg_scale"] if cfg_scale is None else cfg_scale,
                    t_shift=self.flow["t_shift"], generator=torch.Generator().manual_seed(seed + made),
                    null_txt=null_txt, null_mask=null_tb.valid.to(self.device),
                    noise_scale=self.flow["noise_scale"], text_noise_scale=self.flow["text_noise_scale"],
                )
                out.append(states)
            return torch.cat(out)

    @torch.no_grad()
    def text_to_image(self, prompt: str, num_images: int = 4, seed: int = 0, **kwargs) -> list:
        states = self.text_to_image_states(prompt, num_images, seed, **kwargs)
        with precision(self.defaults["text_to_image"]["precision"]):
            return [im for k in range(0, len(states), 4) for im in to_pil(self.decoder(states[k:k + 4]))]

    # ------------------------------------------------------------------ question answering

    def question_ids(self, question: str) -> list[int]:
        """Token ids of "Question: <question> Answer:" (no EOS): the clean text prefix."""
        return self.text.tokenizer(QA_TEMPLATE.format(question=question), add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def answer_states(self, pixels: torch.Tensor, prefixes: list[list[int]], generator: torch.Generator,
                      steps: int | None = None, cfg_scale: float | None = None) -> torch.Tensor:
        """Generate the answer slots for one batch.

        pixels: uint8 [B, 3, 224, 224]; prefixes: token ids of each question prefix.
        Text layout: [prefix (clean) | answer window (generated) | tail (clean, hidden)].
        The clean slots hold the T5 states of "prefix + EOS". The model never sees the
        answer length: the EOS and the answer are generated inside the window.
        Returns the final text states [B, 128, 512].
        """
        d = self.defaults["answer"]
        b, window = len(prefixes), int(d["answer_window"])
        spans = torch.tensor([len(p) for p in prefixes])
        if int(spans.max()) + 1 > TEXT_CANVAS:
            raise ValueError(f"question prefix too long for the {TEXT_CANVAS}-slot canvas")
        ids = torch.full((b, TEXT_CANVAS), PAD_ID, dtype=torch.long)
        for i, p in enumerate(prefixes):
            ids[i, : len(p)] = torch.tensor(p, dtype=torch.long)
        pos = torch.arange(TEXT_CANVAS)[None, :]
        tb = TextBatch.from_ids(ids, pos < (spans[:, None] + 1))
        tail = pos >= (spans[:, None] + window)
        with precision(d["precision"]):
            img = self.image_encoder(to_unit_range(pixels.to(self.device))).float()
            txt = self.text.encode(tb.to(self.device), dtype=torch.float32).float()
            sample = sample_joint(
                self.model, img, txt,
                img_known=torch.ones(b, img.shape[1], dtype=torch.bool),
                txt_known=(pos < spans[:, None]) | tail,
                steps=steps or d["steps"], cfg_scale=d["cfg_scale"] if cfg_scale is None else cfg_scale,
                t_shift=self.flow["t_shift"], generator=generator, txt_key_mask=~tail,
                noise_scale=self.flow["noise_scale"], text_noise_scale=self.flow["text_noise_scale"],
                text_t_shift=self.flow["text_t_shift"],
            )
        return sample.txt

    @torch.no_grad()
    def read_answers(self, txt_states: torch.Tensor, spans: list[int]) -> list[str]:
        """Text states -> answer strings with the parallel head and the first-EOS rule.

        The paper's runs stored the states as float16 and ran the head in
        bfloat16 autocast with batches of 64; both steps are kept here.
        """
        window = int(self.defaults["answer"]["answer_window"])
        states = txt_states.to(torch.float16)
        answers = []
        for k in range(0, states.shape[0], HEAD_CHUNK):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                pred = self.text_head(states[k:k + HEAD_CHUNK].to(self.device)).float().argmax(dim=-1).cpu()
            for row, span in zip(pred, spans[k:k + HEAD_CHUNK]):
                answers.append(clean_answer(self.text.decode(first_eos_answer(row, span, window))))
        return answers

    @torch.no_grad()
    def answer(self, images: list, questions: list[str], seed: int = 0, batch_size: int = 128, **kwargs) -> list[str]:
        """Answer one question per image. Batch k uses the generator seed (seed + k * batch_size)."""
        if len(images) != len(questions):
            raise ValueError("pass one question per image")
        answers: list[str] = []
        for start in range(0, len(images), batch_size):
            pixels = torch.stack([load_image(im) for im in images[start:start + batch_size]])
            prefixes = [self.question_ids(q) for q in questions[start:start + batch_size]]
            states = self.answer_states(pixels, prefixes, torch.Generator().manual_seed(seed + start), **kwargs)
            answers += self.read_answers(states, [len(p) for p in prefixes])
        return answers
