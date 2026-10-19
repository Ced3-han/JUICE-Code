"""Euler samplers on the joint flow (paper Sec. 3.5, App. A.6.2).

Time convention: t = 0 is noise, t = 1 is data; z^t = t * z + (1 - t) * noise.
The network predicts the clean state x0_hat; the sampler uses the velocity
v = (x0_hat - z^t) / max(1 - t, 0.05). The image grid is shifted with
s(t) = t / (shift + (1 - shift) * t); the text grid is not shifted.

Two samplers cover all tasks:
  sample_text_to_image  text is a clean condition; only the image moves.
  sample_joint          per-token times: "known" tokens stay clean at t = 1, the
                        other tokens of both modalities move from noise to data.
                        Used for question answering, captioning and planning.
Noise is drawn on the CPU from `generator` in a fixed order, so a seed gives the
same sample on any GPU count.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

T_EPS = 0.05


def shift_time(t: torch.Tensor, shift: float) -> torch.Tensor:
    return t / (shift + (1.0 - shift) * t)


def time_grid(steps: int, shift: float, device) -> torch.Tensor:
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    return shift_time(ts, shift) if shift != 1.0 else ts


def velocity(x0_pred: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """x0 prediction -> velocity. t broadcasts against x ([B, 1, 1] or [B, N, 1])."""
    return (x0_pred - x) / (1.0 - t).clamp_min(T_EPS)


def _noise(shape, scale: float, generator: torch.Generator, device) -> torch.Tensor:
    return (torch.randn(tuple(shape), generator=generator) * scale).to(device=device, dtype=torch.float32)


@torch.no_grad()
def sample_text_to_image(
    model,
    txt: torch.Tensor,
    txt_mask: torch.Tensor,
    *,
    steps: int,
    cfg_scale: float,
    t_shift: float,
    generator: torch.Generator,
    null_txt: torch.Tensor | None = None,
    null_mask: torch.Tensor | None = None,
    noise_scale: float = 1.0,
    text_noise_scale: float = 2.0,
) -> torch.Tensor:
    """Generate image states [B, 256, D] for clean text states txt [B, L, 512].

    The text is a clean condition at t_txt = 1 and sees only its valid slots (txt_mask).
    Classifier-free guidance needs an unconditional branch:
      null_txt given  -> the clean states of a null caption ("a photo.") at t_txt = 1;
                         the paper's text-to-image results use this branch.
      null_txt = None -> time swap: the text slots hold pure noise at t_txt = 0.
    """
    if (null_txt is None) != (null_mask is None):
        raise ValueError("pass null_txt and null_mask together")
    model.eval()
    b, device = txt.shape[0], txt.device
    n_img, d_img = model.num_image_tokens, model.cfg.image_latent_dim
    x = _noise((b, n_img, d_img), noise_scale, generator, device)
    txt_noise = None
    if null_txt is None or null_mask is None:
        txt_noise = _noise(txt.shape, text_noise_scale, generator, device)
    else:
        null_txt = null_txt.to(device=device, dtype=txt.dtype).expand(b, -1, -1)
        null_mask = null_mask.to(device=device).expand(b, -1)
    ts = time_grid(steps, t_shift, device)
    ones, zeros = torch.ones(b, device=device), torch.zeros(b, device=device)
    for i in range(steps):
        t0, t1 = ts[i], ts[i + 1]
        t = torch.full((b,), float(t0), device=device)
        v = velocity(model(x, t, txt, ones, txt_key_mask=txt_mask)[0], x, t[:, None, None])
        if cfg_scale != 1.0:
            if txt_noise is not None:
                x0_unc = model(x, t, txt_noise, zeros)[0]
            else:
                x0_unc = model(x, t, null_txt, ones, txt_key_mask=null_mask)[0]
            v_unc = velocity(x0_unc, x, t[:, None, None])
            v = v_unc + (v - v_unc) * cfg_scale
        x = x + v * (t1 - t0)
    return x


class JointSample(NamedTuple):
    img: torch.Tensor   # [B, 256, D] final image states (known tokens = input)
    txt: torch.Tensor   # [B, L, 512] final text states (known slots = input)


@torch.no_grad()
def sample_joint(
    model,
    img: torch.Tensor,
    txt: torch.Tensor,
    img_known: torch.Tensor,
    txt_known: torch.Tensor,
    *,
    steps: int,
    cfg_scale: float,
    t_shift: float,
    generator: torch.Generator,
    txt_key_mask: torch.Tensor | None = None,
    noise_scale: float = 1.0,
    text_noise_scale: float = 2.0,
    text_t_shift: float = 1.0,
) -> JointSample:
    """Fill the unknown tokens of both modalities.

    img [B, 256, D], txt [B, L, 512]: clean states; only the known tokens are read.
    img_known [B, 256], txt_known [B, L]: bool masks of the clean (condition) tokens.
    txt_key_mask: optional bool [B, L]; False slots are hidden from attention.

    Guidance (cfg_scale != 1) uses the time swap in both directions: the image
    velocity is guided against "text = noise at t = 0", the text velocity against
    "image = noise at t = 0".
    """
    model.eval()
    b, device = img.shape[0], img.device
    img, txt = img.float(), txt.float()
    img_known, txt_known = img_known.to(device=device, dtype=torch.bool), txt_known.to(device=device, dtype=torch.bool)
    if txt_key_mask is not None:
        txt_key_mask = txt_key_mask.to(device=device, dtype=torch.bool)
    # Fixed draw order: start noise (image, text), then the time-swap nulls (image, text).
    img_start = _noise(img.shape, noise_scale, generator, device)
    txt_start = _noise(txt.shape, text_noise_scale, generator, device)
    img_null = _noise(img.shape, noise_scale, generator, device)
    txt_null = _noise(txt.shape, text_noise_scale, generator, device)
    img_ts, txt_ts = time_grid(steps, t_shift, device), time_grid(steps, text_t_shift, device)

    img_move = (~img_known).float()[:, :, None]
    txt_move = (~txt_known).float()[:, :, None]
    x = torch.where(img_known[:, :, None], img, img_start)
    z = torch.where(txt_known[:, :, None], txt, txt_start)
    zeros = torch.zeros(b, device=device)
    img_ones = torch.ones(b, img.shape[1], device=device)
    txt_ones = torch.ones(b, txt.shape[1], device=device)
    for i in range(steps):
        t_img = torch.where(img_known, img_ones, torch.full_like(img_ones, float(img_ts[i])))
        t_txt = torch.where(txt_known, txt_ones, torch.full_like(txt_ones, float(txt_ts[i])))
        img_x0, txt_x0 = model(x, t_img, z, t_txt, txt_key_mask=txt_key_mask)
        v_img = velocity(img_x0, x, t_img[:, :, None])
        v_txt = velocity(txt_x0, z, t_txt[:, :, None])
        if cfg_scale != 1.0:
            v_img_unc = velocity(model(x, t_img, txt_null, zeros, txt_key_mask=txt_key_mask)[0], x, t_img[:, :, None])
            v_txt_unc = velocity(model(img_null, zeros, z, t_txt, txt_key_mask=txt_key_mask)[1], z, t_txt[:, :, None])
            v_img = v_img_unc + (v_img - v_img_unc) * cfg_scale
            v_txt = v_txt_unc + (v_txt - v_txt_unc) * cfg_scale
        x = x + v_img * (img_ts[i + 1] - img_ts[i]) * img_move
        z = z + v_txt * (txt_ts[i + 1] - txt_ts[i]) * txt_move
    x = torch.where(img_known[:, :, None], img, x)
    z = torch.where(txt_known[:, :, None], txt, z)
    return JointSample(img=x, txt=z)
