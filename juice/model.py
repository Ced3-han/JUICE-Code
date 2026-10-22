"""MM-RiT: the two-stream flow-matching Transformer of JUICE (paper Sec. 3.3, App. A.3).

The network reads a noisy image state and a noisy text state, each with its own
time, and predicts the clean state of both streams:

    (z_i_hat, z_t_hat) = f(z_i^{t_i}, z_t^{t_t}; t_i, t_t)            (paper Eq. 2)

Time convention: t = 0 is pure noise and t = 1 is clean data. A clean condition
enters at t = 1. A time can be one value per sample ([B]) or one value per token
([B, N]); per-token times use the additive token-wise time embedding.

Layout of one forward pass:
  image tokens   [B, 256, D_img] -> linear -> + fixed 2D sin-cos table
  text tokens    [B, L, D_txt]   -> linear bottleneck (D_txt -> 128 -> width)
  text preamble  2 text-only blocks
  time tokens    4 image-time + 4 text-time tokens appended to the text side
  joint blocks   per-modality Q/K/V, norms, projections and SwiGLU FFNs, one
                 bidirectional attention over [text, time tokens, image]
                 (paper Eq. 7); 2D RoPE on image tokens, 1D RoPE on the text side
  outputs        per-modality RMSNorm + linear head back to D_img / D_txt

Parameter names match the research checkpoints, so `load_state_dict` reads the
exported weights without a key map.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

# 1D RoPE position of the first time token. Text canvases longer than this keep
# the time tokens here instead of moving them to the end of the canvas. The
# released models use a 128-slot canvas, so all text-side tokens use one RoPE.
TIME_TOKEN_ROPE_ANCHOR = 128


@dataclass
class MMRiTConfig:
    """Architecture of one MM-RiT backbone (paper Table 12)."""

    image_latent_dim: int = 1152      # SigLIP2-SO400M: 1152, WebSSL-DINO300M: 1024
    image_grid: int = 16              # 224 px / 14 px patches
    text_dim: int = 512               # Flan-T5-Small encoder width
    text_bottleneck_dim: int = 128
    hidden_size: int = 1792
    depth: int = 26                   # joint (two-stream) blocks
    text_preamble_depth: int = 2      # text-only blocks before the joint blocks
    num_heads: int = 28
    head_dim: int = 64
    mlp_ratio: float = 2.6667
    time_tokens: int = 4              # in-context time tokens per modality
    token_time: bool = True           # additive token-wise time embeddings
    attention_impl: str = "sdpa"      # "sdpa" | "einsum" (same math)

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = math.ceil(hidden_dim / 8) * 8
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def timestep_embedding(t: torch.Tensor, dim: int = 256, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, 256).to(dtype=next(self.parameters()).dtype))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_1d_rope(x: torch.Tensor, start: int = 0, theta: int = 10000) -> torch.Tensor:
    """x: [B, H, N, d]; positions start, ..., start + N - 1."""
    _, _, n, d = x.shape
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    pos = torch.arange(start, start + n, device=x.device, dtype=torch.float32)
    angles = torch.einsum("n,f->nf", pos, inv)
    angles = torch.cat([angles, angles], dim=-1)
    return x * angles.cos()[None, None].to(x.dtype) + rotate_half(x) * angles.sin()[None, None].to(x.dtype)


def apply_2d_rope(x: torch.Tensor, grid: int, theta: int = 10000) -> torch.Tensor:
    """x: [B, H, grid*grid, d]; row-major patches. Half of each head encodes the row,
    half encodes the column."""
    _, _, n, d = x.shape
    if n != grid * grid:
        raise ValueError(f"expected {grid * grid} image tokens, got {n}")
    rope_dim = d // 2
    inv = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, device=x.device, dtype=torch.float32) / rope_dim))
    freqs = torch.einsum("n,f->nf", torch.arange(grid, device=x.device, dtype=torch.float32), inv)
    f_h, f_w = torch.broadcast_tensors(freqs[:, None, :], freqs[None, :, :])
    angles = torch.cat([f_h, f_w], dim=-1)
    angles = torch.cat([angles, angles], dim=-1).reshape(n, d)
    return x * angles.cos()[None, None].to(x.dtype) + rotate_half(x) * angles.sin()[None, None].to(x.dtype)


def sincos_2d(embed_dim: int, grid: int) -> torch.Tensor:
    """Fixed 2D sin-cos table [grid*grid, embed_dim], added to the image tokens."""
    y, x = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    omega = torch.arange(embed_dim // 4, dtype=torch.float32) / (embed_dim // 4)
    omega = 1.0 / (10000**omega)
    out_y = torch.einsum("n,d->nd", y.flatten().float(), omega)
    out_x = torch.einsum("n,d->nd", x.flatten().float(), omega)
    return torch.cat([out_x.sin(), out_x.cos(), out_y.sin(), out_y.cos()], dim=1)


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, head_dim: int, impl: str, bias: torch.Tensor | None
) -> torch.Tensor:
    """Bidirectional attention. bias: additive [B, 1, 1, K] (0 = keep, -inf = drop) or None."""
    if impl == "sdpa":
        return F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False, scale=head_dim**-0.5)
    if impl == "einsum":
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * (head_dim**-0.5)
        if bias is not None:
            scores = scores + bias
        return torch.einsum("bhqk,bhkd->bhqd", torch.softmax(scores, dim=-1), v)
    raise ValueError(f"unknown attention_impl={impl!r}")


def key_padding_bias(keep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """bool keep-mask [B, K] -> additive bias [B, 1, 1, K]."""
    bias = torch.zeros(keep.shape, dtype=dtype, device=keep.device).masked_fill(~keep, float("-inf"))
    return bias[:, None, None, :]


class TextBlock(nn.Module):
    """Text-only pre-norm block (the two preamble layers)."""

    def __init__(self, cfg: MMRiTConfig):
        super().__init__()
        self.num_heads, self.head_dim, self.attention_impl = cfg.num_heads, cfg.head_dim, cfg.attention_impl
        inner = cfg.num_heads * cfg.head_dim
        self.norm1 = RMSNorm(cfg.hidden_size)
        self.norm2 = RMSNorm(cfg.hidden_size)
        self.qkv = nn.Linear(cfg.hidden_size, inner * 3)
        self.proj = nn.Linear(inner, cfg.hidden_size)
        self.mlp = SwiGLU(cfg.hidden_size, int(cfg.hidden_size * cfg.mlp_ratio))
        self.q_norm = RMSNorm(cfg.head_dim)
        self.k_norm = RMSNorm(cfg.head_dim)

    def forward(self, txt: torch.Tensor, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, _ = txt.shape
        qkv = self.qkv(self.norm1(txt)).view(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = self.q_norm(qkv[:, :, 0]), self.k_norm(qkv[:, :, 1]), qkv[:, :, 2]
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        q, k = apply_1d_rope(q), apply_1d_rope(k)
        bias = None if key_mask is None else key_padding_bias(key_mask, q.dtype)
        out = attention(q, k, v, self.head_dim, self.attention_impl, bias)
        txt = txt + self.proj(out.transpose(1, 2).reshape(b, n, -1))
        return txt + self.mlp(self.norm2(txt))


class JointBlock(nn.Module):
    """Two-stream block: modality-specific weights, one joint attention (paper Eq. 7)."""

    def __init__(self, cfg: MMRiTConfig, n_time_tokens: int):
        super().__init__()
        self.num_heads, self.head_dim, self.attention_impl = cfg.num_heads, cfg.head_dim, cfg.attention_impl
        self.image_grid = cfg.image_grid
        self.n_time_tokens = n_time_tokens
        inner = cfg.num_heads * cfg.head_dim
        h = cfg.hidden_size
        self.img_norm1, self.img_norm2 = RMSNorm(h), RMSNorm(h)
        self.txt_norm1, self.txt_norm2 = RMSNorm(h), RMSNorm(h)
        self.img_qkv = nn.Linear(h, inner * 3)
        self.txt_qkv = nn.Linear(h, inner * 3)
        self.q_norm = RMSNorm(cfg.head_dim)
        self.k_norm = RMSNorm(cfg.head_dim)
        self.img_proj = nn.Linear(inner, h)
        self.txt_proj = nn.Linear(inner, h)
        self.img_mlp = SwiGLU(h, int(h * cfg.mlp_ratio))
        self.txt_mlp = SwiGLU(h, int(h * cfg.mlp_ratio))

    def _rope_text(self, x: torch.Tensor) -> torch.Tensor:
        """1D RoPE over [text slots, time tokens]; see TIME_TOKEN_ROPE_ANCHOR."""
        n_slots = x.shape[2] - self.n_time_tokens
        if n_slots <= TIME_TOKEN_ROPE_ANCHOR:
            return apply_1d_rope(x)
        return torch.cat(
            [apply_1d_rope(x[:, :, :n_slots]), apply_1d_rope(x[:, :, n_slots:], start=TIME_TOKEN_ROPE_ANCHOR)], dim=2
        )

    def forward(
        self, img: torch.Tensor, txt: torch.Tensor, txt_key_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, li, _ = img.shape
        lt = txt.shape[1]
        qi, ki, vi = self.img_qkv(self.img_norm1(img)).view(b, li, 3, self.num_heads, self.head_dim).unbind(2)
        qt, kt, vt = self.txt_qkv(self.txt_norm1(txt)).view(b, lt, 3, self.num_heads, self.head_dim).unbind(2)
        q = torch.cat([qt, qi], dim=1).transpose(1, 2)
        k = torch.cat([kt, ki], dim=1).transpose(1, 2)
        v = torch.cat([vt, vi], dim=1).transpose(1, 2)
        q_txt, q_img = self.q_norm(q[:, :, :lt]), self.q_norm(q[:, :, lt:])
        k_txt, k_img = self.k_norm(k[:, :, :lt]), self.k_norm(k[:, :, lt:])
        q = torch.cat([self._rope_text(q_txt), apply_2d_rope(q_img, self.image_grid)], dim=2)
        k = torch.cat([self._rope_text(k_txt), apply_2d_rope(k_img, self.image_grid)], dim=2)
        bias = None
        if txt_key_mask is not None:
            keep = torch.cat([txt_key_mask.to(torch.bool), txt_key_mask.new_ones((b, li), dtype=torch.bool)], dim=1)
            bias = key_padding_bias(keep, q.dtype)
        out = attention(q, k, v, self.head_dim, self.attention_impl, bias).transpose(1, 2)
        txt = txt + self.txt_proj(out[:, :lt].reshape(b, lt, -1))
        img = img + self.img_proj(out[:, lt:].reshape(b, li, -1))
        img = img + self.img_mlp(self.img_norm2(img))
        txt = txt + self.txt_mlp(self.txt_norm2(txt))
        return img, txt


class MMRiT(nn.Module):
    """Multimodal Representation Diffusion Transformer (clean-state prediction)."""

    def __init__(self, cfg: MMRiTConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_size
        self.latent_in = nn.Linear(cfg.image_latent_dim, h)
        self.register_buffer("pos_embed", sincos_2d(h, cfg.image_grid).unsqueeze(0), persistent=False)
        self.txt_bottleneck_in = nn.Linear(cfg.text_dim, cfg.text_bottleneck_dim, bias=False)
        self.txt_bottleneck_out = nn.Linear(cfg.text_bottleneck_dim, h, bias=True)
        self.t_embed = TimestepEmbedder(h)        # image time
        self.txt_t_embed = TimestepEmbedder(h)    # text time
        self.img_t_tokens = nn.Parameter(torch.zeros(1, cfg.time_tokens, h))
        self.txt_t_tokens = nn.Parameter(torch.zeros(1, cfg.time_tokens, h))
        if cfg.token_time:
            self.img_tok_t_proj = nn.Linear(h, h)
            self.txt_tok_t_proj = nn.Linear(h, h)
        self.txt_blocks = nn.ModuleList([TextBlock(cfg) for _ in range(cfg.text_preamble_depth)])
        self.blocks = nn.ModuleList([JointBlock(cfg, 2 * cfg.time_tokens) for _ in range(cfg.depth)])
        self.final_norm = RMSNorm(h)
        self.final = nn.Linear(h, cfg.image_latent_dim)
        self.txt_final_norm = RMSNorm(h)
        self.txt_final = nn.Linear(h, cfg.text_dim)

    @torch.no_grad()
    def init_weights(self) -> None:
        """Initialization for training from scratch (pretraining). Call it once, right
        after construction; loading weights does not need it.

        Same distributions as the paper's runs: Xavier-uniform for the SwiGLU layers
        and the input projections, N(0, 0.02) for the timestep MLPs, N(0, 1/hidden) for
        the time tokens, and zeros for the two output layers and the per-token time
        projections (so the first prediction is zero). Attention layers keep the
        PyTorch default.
        """
        for m in self.modules():
            if isinstance(m, SwiGLU):
                for linear in (m.w1, m.w3, m.w2):
                    nn.init.xavier_uniform_(linear.weight)
            elif isinstance(m, TimestepEmbedder):
                for linear in (m.mlp[0], m.mlp[2]):
                    nn.init.normal_(linear.weight, std=0.02)
                    nn.init.zeros_(linear.bias)
        for linear in (self.latent_in, self.txt_bottleneck_in, self.txt_bottleneck_out):
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        for tokens in (self.img_t_tokens, self.txt_t_tokens):
            tokens.copy_(torch.randn_like(tokens) * self.cfg.hidden_size**-0.5)
        zero_init = [self.final, self.txt_final]
        if self.cfg.token_time:
            zero_init += [self.img_tok_t_proj, self.txt_tok_t_proj]
        for linear in zero_init:
            nn.init.zeros_(linear.weight)
            nn.init.zeros_(linear.bias)

    @property
    def num_image_tokens(self) -> int:
        return self.cfg.image_grid**2

    def forward(
        self,
        img: torch.Tensor,
        t_img: torch.Tensor,
        txt: torch.Tensor,
        t_txt: torch.Tensor,
        txt_key_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the clean image and text states.

        img:          [B, 256, D_img] noisy (or clean) normalized image state
        t_img:        [B] or [B, 256] image time(s)
        txt:          [B, L, D_txt] noisy (or clean) normalized text state
        t_txt:        [B] or [B, L] text time(s)
        txt_key_mask: optional bool [B, L]; False slots are hidden from all attention.
                      Pass the content+EOS mask when the text is a clean condition and
                      None when the text is a generation target (the model must not
                      see the true length).
        Returns (image x0 [B, 256, D_img], text x0 [B, L, D_txt]), both float32.
        """
        n_txt = txt.shape[1]
        img_tokens = self.latent_in(img) + self.pos_embed.to(device=img.device, dtype=img.dtype)
        txt_tokens = self.txt_bottleneck_out(self.txt_bottleneck_in(txt))
        if t_img.ndim == 2 or t_txt.ndim == 2 or self.cfg.token_time:
            if not self.cfg.token_time:
                raise ValueError("per-token times need token_time=True")
            ti = t_img if t_img.ndim == 2 else t_img[:, None].expand(-1, img_tokens.shape[1])
            tt = t_txt if t_txt.ndim == 2 else t_txt[:, None].expand(-1, n_txt)
            img_tokens = img_tokens + self.img_tok_t_proj(
                self.t_embed(ti.reshape(-1)).view(ti.shape[0], ti.shape[1], -1)
            ).to(dtype=img_tokens.dtype)
            txt_tokens = txt_tokens + self.txt_tok_t_proj(
                self.txt_t_embed(tt.reshape(-1)).view(tt.shape[0], tt.shape[1], -1)
            ).to(dtype=txt_tokens.dtype)
            # The in-context time tokens carry the mean time of their stream.
            t_img = t_img.mean(dim=1) if t_img.ndim == 2 else t_img
            t_txt = t_txt.mean(dim=1) if t_txt.ndim == 2 else t_txt
        for block in self.txt_blocks:
            txt_tokens = block(txt_tokens, key_mask=txt_key_mask)
        dtype = txt_tokens.dtype
        img_time = self.img_t_tokens.to(dtype=dtype) + self.t_embed(t_img).to(dtype=dtype)[:, None]
        txt_time = self.txt_t_tokens.to(dtype=dtype) + self.txt_t_embed(t_txt).to(dtype=dtype)[:, None]
        b = txt_tokens.shape[0]
        txt_tokens = torch.cat([txt_tokens, img_time.expand(b, -1, -1), txt_time.expand(b, -1, -1)], dim=1)
        side_mask = None
        if txt_key_mask is not None:
            n_time = txt_tokens.shape[1] - n_txt
            side_mask = torch.cat(
                [txt_key_mask.to(torch.bool), txt_key_mask.new_ones((b, n_time), dtype=torch.bool)], dim=1
            )
        for block in self.blocks:
            img_tokens, txt_tokens = block(img_tokens, txt_tokens, txt_key_mask=side_mask)
        img_x0 = self.final(self.final_norm(img_tokens)).float()
        txt_x0 = self.txt_final(self.txt_final_norm(txt_tokens[:, :n_txt])).float()
        return img_x0, txt_x0
