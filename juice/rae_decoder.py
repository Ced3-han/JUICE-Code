# Adapted from Scale-RAE (https://github.com/ZitengWangNYU/Scale-RAE),
# MIT License, Copyright (c) 2026 Shengbang Tong, Boyang Zheng, Ziteng Wang.
# Scale-RAE adapts the ViT-MAE decoder of HuggingFace Transformers,
# Apache License 2.0, Copyright 2022 Facebook AI and The HuggingFace Inc. team.
# Changes in this file: decode-only path, written without transformers internals.
# Parameter names are unchanged, so the published decoder weights
# (nyu-visionx/siglip2_decoder, nyu-visionx/webssl300m_decoder) load with strict=True.
"""Pixel decoder for frozen representation states (RAE decoder).

JUICE generates images as encoder states (256 tokens). This ViT decoder maps the
states back to a 224 x 224 RGB image. It is used only to look at samples; it
takes no part in training.
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F
from torch import nn


class _SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[:-1] + (self.num_heads, self.head_dim)).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._heads(self.query(x)), self._heads(self.key(x)), self._heads(self.value(x))
        out = F.scaled_dot_product_attention(q, k, v, None, 0.0, is_causal=False, scale=None)
        out = out.permute(0, 2, 1, 3).contiguous()
        return out.view(out.shape[:-2] + (self.num_heads * self.head_dim,))


class _Dense(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.dense = nn.Linear(dim_in, dim_out)


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.attention = _SelfAttention(dim, num_heads)
        self.output = _Dense(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output.dense(self.attention(x))


class _Layer(nn.Module):
    """Pre-norm ViT block (LayerNorm -> attention -> residual -> LayerNorm -> GELU MLP -> residual)."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, eps: float):
        super().__init__()
        self.attention = _Attention(dim, num_heads)
        self.intermediate = _Dense(dim, mlp_dim)
        self.output = _Dense(mlp_dim, dim)
        self.layernorm_before = nn.LayerNorm(dim, eps=eps)
        self.layernorm_after = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(self.layernorm_before(x)) + x
        y = F.gelu(self.intermediate.dense(self.layernorm_after(x)))
        return self.output.dense(y) + x


class RAEDecoder(nn.Module):
    """[B, 256, D] encoder states -> [B, 3, 224, 224] images in [-1, 1]."""

    def __init__(self, config: dict, latent_dim: int, pixel_mean, pixel_std, num_patches: int = 256):
        super().__init__()
        width = int(config["decoder_hidden_size"])
        eps = float(config["layer_norm_eps"])
        self.patch_size = int(config["patch_size"])
        self.image_size = int(config["image_size"])
        self.num_channels = int(config["num_channels"])
        self.decoder_embed = nn.Linear(latent_dim, width)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, width), requires_grad=False)
        self.decoder_layers = nn.ModuleList(
            _Layer(width, int(config["decoder_num_attention_heads"]), int(config["decoder_intermediate_size"]), eps)
            for _ in range(int(config["decoder_num_hidden_layers"]))
        )
        self.decoder_norm = nn.LayerNorm(width, eps=eps)
        self.decoder_pred = nn.Linear(width, self.patch_size**2 * self.num_channels)
        self.trainable_cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(1, 3, 1, 1), persistent=False)

    @classmethod
    def from_files(cls, config_path: str, weights_path: str, latent_dim: int, pixel_mean, pixel_std) -> "RAEDecoder":
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        decoder = cls(config, latent_dim, pixel_mean, pixel_std)
        decoder.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=True)
        return decoder.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.float()
        # The decoder was trained with a CLS slot in front; its input is dropped after
        # the first projection and replaced by the learned CLS token.
        x = self.decoder_embed(torch.cat([torch.zeros_like(latents[:, :1]), latents], dim=1))
        x = torch.cat([self.trainable_cls_token.expand(x.shape[0], -1, -1), x[:, 1:]], dim=1)
        x = x + self.decoder_pos_embed
        for layer in self.decoder_layers:
            x = layer(x)
        patches = self.decoder_pred(self.decoder_norm(x))[:, 1:]
        b, grid, p, c = patches.shape[0], self.image_size // self.patch_size, self.patch_size, self.num_channels
        pixels = patches.reshape(b, grid, grid, p, p, c).permute(0, 5, 1, 3, 2, 4).reshape(b, c, grid * p, grid * p)
        pixels = pixels * self.pixel_std + self.pixel_mean
        return pixels.clamp(0, 1) * 2.0 - 1.0
