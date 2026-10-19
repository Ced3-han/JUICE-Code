"""The frozen representation spaces that JUICE models.

Image state: 256 tokens from a frozen vision encoder at 224 x 224 px. Each token is
normalized by a parameter-free LayerNorm (eps 1e-6), as in Scale-RAE.
  - SigLIP2-SO400M/14: last encoder layer, before the post-LayerNorm, D = 1152.
  - WebSSL-DINO300M:   last hidden state, CLS token dropped, D = 1024.
Text state: 128 slots of Flan-T5-Small encoder states (D = 512), divided by 0.2.

Text slot contract (all text in JUICE uses it):
  content tokens + exactly one EOS + PAD tail, 128 slots in total.
  The encoder attends only to content + EOS ("valid" slots). PAD slots still get
  encoder states, because the denoiser reads all 128 slots when it generates text.

Nothing in this file is trained. The pixel decoders only make images visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

# Hugging Face revisions used for every result in the paper.
TEXT_ENCODER_ID = "google/flan-t5-small"
TEXT_ENCODER_REVISION = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
TEXT_CANVAS = 128          # text slots
TEXT_LATENT_STD = 0.2      # scalar text codec: state = T5 state / 0.2
EOS_ID, PAD_ID = 1, 0      # Flan-T5 special tokens


@dataclass(frozen=True)
class ImageSpace:
    name: str
    encoder_id: str
    encoder_revision: str
    decoder_repo: str
    decoder_revision: str
    latent_dim: int
    pixel_mean: tuple[float, float, float]   # pixel statistics of the encoder's processor
    pixel_std: tuple[float, float, float]


IMAGE_SPACES = {
    "siglip2": ImageSpace(
        name="siglip2",
        encoder_id="google/siglip2-so400m-patch14-224",
        encoder_revision="78e403963a4f6a3640d07803284752326fdf4edf",
        decoder_repo="nyu-visionx/siglip2_decoder",
        decoder_revision="4a728df72b21ff7ec1d81b4235047c1731b0edae",
        latent_dim=1152,
        pixel_mean=(0.5, 0.5, 0.5),
        pixel_std=(0.5, 0.5, 0.5),
    ),
    "webssl_dino": ImageSpace(
        name="webssl_dino",
        encoder_id="facebook/webssl-dino300m-full2b-224",
        encoder_revision="8529cdb3fb75014932af3b896455fc21c386168e",
        decoder_repo="nyu-visionx/webssl300m_decoder",
        decoder_revision="0e6a28d0744b31ac35624ecabadde20b540d8fa0",
        latent_dim=1024,
        pixel_mean=(0.485, 0.456, 0.406),
        pixel_std=(0.229, 0.224, 0.225),
    ),
}


# ----------------------------------------------------------------------------- images


def load_image(image, size: int = 224) -> torch.Tensor:
    """Path or PIL image -> uint8 [3, size, size]: bicubic resize of the short side, center crop."""
    from PIL import Image
    from torchvision import transforms

    if not isinstance(image, Image.Image):
        image = Image.open(image)
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.PILToTensor(),
    ])
    return transform(image.convert("RGB"))


def to_unit_range(pixels: torch.Tensor) -> torch.Tensor:
    """uint8 [B, 3, H, W] -> float32 in [-1, 1]."""
    return pixels.to(torch.float32).mul(1.0 / 127.5).add(-1.0)


def to_pil(images: torch.Tensor) -> list:
    """float [B, 3, H, W] in [-1, 1] -> PIL images (the rounding rule of the benchmark images)."""
    from PIL import Image

    images = (images.detach().float().clamp(-1, 1) * 127.5 + 128.0).clamp(0, 255).to(torch.uint8)
    return [Image.fromarray(a) for a in images.permute(0, 2, 3, 1).cpu().numpy()]


class SigLIP2Encoder(nn.Module):
    def __init__(self, space: ImageSpace):
        super().__init__()
        from transformers import SiglipModel

        self.vision = SiglipModel.from_pretrained(space.encoder_id, revision=space.encoder_revision).vision_model

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images in [-1, 1] (= the SigLIP processor space) -> [B, 256, 1152]."""
        features = self.vision(images, output_hidden_states=True).hidden_states[-1]
        return F.layer_norm(features, (features.shape[-1],), eps=1e-6)


class WebSSLDinoEncoder(nn.Module):
    def __init__(self, space: ImageSpace):
        super().__init__()
        from transformers import Dinov2Model

        self.vision = Dinov2Model.from_pretrained(space.encoder_id, revision=space.encoder_revision)
        self.register_buffer("pixel_mean", torch.tensor(space.pixel_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(space.pixel_std).view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images in [-1, 1] -> ImageNet statistics -> [B, 256, 1024] (CLS token dropped)."""
        x = (images * 0.5 + 0.5 - self.pixel_mean.to(images.dtype)) / self.pixel_std.to(images.dtype)
        features = self.vision(x).last_hidden_state[:, 1:]
        return F.layer_norm(features, (features.shape[-1],), eps=1e-6)


def build_image_encoder(space_name: str) -> nn.Module:
    space = IMAGE_SPACES[space_name]
    encoder = SigLIP2Encoder(space) if space_name == "siglip2" else WebSSLDinoEncoder(space)
    return encoder.eval().requires_grad_(False)


def build_image_decoder(space_name: str) -> nn.Module:
    from huggingface_hub import hf_hub_download

    from juice.rae_decoder import RAEDecoder

    space = IMAGE_SPACES[space_name]
    config = hf_hub_download(space.decoder_repo, "config.json", revision=space.decoder_revision)
    weights = hf_hub_download(space.decoder_repo, "model.pt", revision=space.decoder_revision)
    return RAEDecoder.from_files(config, weights, space.latent_dim, space.pixel_mean, space.pixel_std)


# ----------------------------------------------------------------------------- text


class TextBatch(NamedTuple):
    """Token ids in the slot contract and its masks (all [B, L])."""

    ids: torch.Tensor       # long: content + one EOS + PAD
    valid: torch.Tensor     # bool: content + EOS
    content: torch.Tensor   # bool: valid without the EOS
    eos: torch.Tensor       # bool: the EOS slot
    pad: torch.Tensor       # bool: not valid
    lengths: torch.Tensor   # long [B]: number of valid slots (>= 1)

    @staticmethod
    def from_ids(input_ids: torch.Tensor, attention_mask: torch.Tensor, eos_id: int = EOS_ID, pad_id: int = PAD_ID) -> "TextBatch":
        """Tokenizer output -> contract. Forces the EOS onto the last valid slot and PAD after it."""
        ids = input_ids.long().clone()
        lengths = attention_mask.to(dtype=torch.bool).long().sum(dim=1).clamp_(min=1, max=ids.shape[1])
        pos = torch.arange(ids.shape[1], device=ids.device)[None, :]
        valid = pos < lengths[:, None]
        eos = pos == (lengths[:, None] - 1)
        ids = torch.where(eos, torch.full_like(ids, eos_id), ids)
        ids = torch.where(valid, ids, torch.full_like(ids, pad_id))
        return TextBatch(ids=ids, valid=valid, content=valid & ~eos, eos=eos, pad=~valid, lengths=lengths)

    def to(self, device) -> "TextBatch":
        return TextBatch(*(t.to(device) for t in self))


def load_tokenizer(model_id: str = TEXT_ENCODER_ID, revision: str = TEXT_ENCODER_REVISION, canvas: int = TEXT_CANVAS):
    """The Flan-T5 tokenizer alone (data loaders need no encoder)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, revision=revision, model_max_length=canvas)


class T5TextEncoder(nn.Module):
    """Flan-T5-Small encoder + tokenizer under the slot contract."""

    def __init__(self, model_id: str = TEXT_ENCODER_ID, revision: str = TEXT_ENCODER_REVISION, canvas: int = TEXT_CANVAS):
        super().__init__()
        from transformers import T5EncoderModel

        self.canvas = canvas
        self.model = T5EncoderModel.from_pretrained(model_id, revision=revision).eval().requires_grad_(False)
        self.tokenizer = load_tokenizer(model_id, revision, canvas)

    def tokenize(self, texts: list[str]) -> TextBatch:
        tok = self.tokenizer(texts, max_length=self.canvas, padding="max_length", truncation=True, return_tensors="pt")
        return TextBatch.from_ids(tok["input_ids"], tok["attention_mask"])

    @torch.no_grad()
    def encode(self, tb: TextBatch, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """TextBatch -> normalized text states [B, L, 512] (T5 state / 0.2), in `dtype`."""
        raw = self.model(input_ids=tb.ids, attention_mask=tb.valid.long()).last_hidden_state.to(dtype=dtype)
        return raw / TEXT_LATENT_STD

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(torch.tensor(ids, dtype=torch.long), skip_special_tokens=True)
