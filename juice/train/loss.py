"""The JUICE training objective (paper Sec. 3.4, App. A.6.1).

One flow-matching loss trains generation, understanding and joint generation. Time
convention: t = 0 is noise, t = 1 is data; z^t = t * z + (1 - t) * noise.

Routing. Every row of a micro-batch gets one of three regimes. The counts follow
`task_mixture` exactly (largest-remainder allocation); the positions are shuffled.
    joint   both streams are noised with independent times; both carry a loss
    t2i     the text is a clean condition (t_txt = 1); only the image carries a loss
    i2t     the image is a clean condition (t_img = 1); only the text carries a loss
With `qa_rows_force_i2t` (pretraining and SFT), a question-answering row (its sample
sets answer_tok_start >= 1) is always i2t. The planning fine-tunes keep the mixture
for QA rows too, so a plan can also be generated jointly with its image.

Times. The image time is sigmoid(N(mu, sigma)) shifted by `t_shift`; the text time is
sigmoid(N(text_t_mu, text_t_sigma)). On target streams a row can then draw a token
pattern: with probability `token_t_span_prob` only a random span (10-90 % of the
tokens) keeps the row time and the rest is clean; with `token_t_indep_prob` every
token draws its own time.

Question-answering rows. The prefix "Question: ... Answer:" is clean (t = 1) and holds
the T5 states of the question alone, so the clean slots carry no answer information.
Only the answer window [s, s + K) is generated (K = the row's `qa_window` or
`qa_answer_window`). Later slots are clean, carry no loss and are hidden from attention.
PAD slots inside the window get a small extra loss (`qa_window_tail_weight`).

Image-known rows (visual planning). Patches that the task gives (walls, start, goal)
are clean at t = 1 and hold the states of the input render; the model generates the
other patches.

Loss. The network predicts clean states; both streams are trained in velocity space,
v = (x0 - z^t) / max(1 - t, 0.05). Each stream is averaged over its own active tokens
(text: content + EOS only). Total = lambda_img * L_img + lambda_txt * L_txt.

All random draws use the global torch RNG in a fixed order, so a seed reproduces a step.
"""

from __future__ import annotations

import torch

from juice.representations import TextBatch
from juice.sampling import shift_time

T_EPS = 0.05
JOINT, T2I, I2T = 0, 1, 2
REGIME_CODES = {"product": JOINT, "t2i": T2I, "i2t": I2T}


def sample_time(n: int, mu: float, sigma: float, shift: float, device) -> torch.Tensor:
    """Logit-normal time on (0, 1), then the resolution shift (t = 1 is data)."""
    t = torch.sigmoid(torch.randn(n, device=device) * sigma + mu).clamp(1e-5, 1.0 - 1.0e-5)
    return shift_time(t, shift) if shift != 1.0 else t


def stratified_regimes(probs: list[float], batch: int, device) -> torch.Tensor:
    """Exact regime counts for `batch` rows (largest remainder), in shuffled positions.

    When the t2i and i2t weights are equal, both edges always get the same count: the
    leftover rows go one to each edge while two or more remain, and a final odd row
    goes to the joint regime.
    """
    p = torch.tensor(probs, dtype=torch.float64)
    p = p / p.sum()
    exact = p * batch
    base = exact.floor().long()
    rem = batch - int(base.sum())
    if probs[T2I] == probs[I2T] and (probs[JOINT] > 0 or rem % 2 == 0):
        if rem >= 2:
            base[T2I] += 1
            base[I2T] += 1
            rem -= 2
        if rem:
            base[JOINT] += 1
    elif rem > 0:
        order = torch.argsort(exact - base.to(exact.dtype), descending=True)
        base[order[:rem]] += 1
    regime = torch.repeat_interleave(torch.arange(3), base).to(device)
    return regime[torch.randperm(batch, device=device)]


def sample_regimes_and_times(cfg, batch: int, device, spans: torch.Tensor | None):
    """(t_img [B], t_txt [B], regime [B]). The clean stream of an edge row is set to t = 1."""
    t_img = sample_time(batch, cfg.t_lognorm_mu, cfg.t_lognorm_sigma, cfg.t_shift, device)
    t_txt = sample_time(batch, cfg.text_t_mu, cfg.text_t_sigma, cfg.text_t_shift, device)
    regime = stratified_regimes(list(cfg.task_mixture), batch, device)
    if cfg.qa_rows_force_i2t and spans is not None:
        # QA rows are i2t by construction (pretraining and SFT). The draw above still ran,
        # so the other rows keep the same regimes and times.
        regime = torch.where(spans >= 1, torch.full_like(regime, I2T), regime)
    ones = torch.ones_like(t_img)
    t_img = torch.where(regime == I2T, ones, t_img)
    t_txt = torch.where(regime == T2I, ones, t_txt)
    return t_img, t_txt, regime


def token_time_patterns(t_row: torch.Tensor, n: int, span_prob: float, indep_prob: float,
                        resample, eligible: torch.Tensor) -> torch.Tensor:
    """Row times [B] -> token times [B, n] (span / independent / uniform pattern per row)."""
    b, device = t_row.shape[0], t_row.device
    tok = t_row[:, None].expand(b, n).clone()
    noised = (t_row < 1.0 - 1e-6) & eligible
    u = torch.rand(b, device=device)
    span_rows = noised & (u < span_prob)
    indep_rows = noised & (u >= span_prob) & (u < span_prob + indep_prob)
    if bool(indep_rows.any()):
        tok = torch.where(indep_rows[:, None], resample(b * n).view(b, n).to(tok.dtype), tok)
    if bool(span_rows.any()):
        frac = torch.empty(b, device=device).uniform_(0.1, 0.9)
        length = (frac * n).long().clamp(1, n)
        start = (torch.rand(b, device=device) * (n - length + 1).float()).long()
        idx = torch.arange(n, device=device)[None, :]
        in_span = (idx >= start[:, None]) & (idx < (start + length)[:, None])
        row_tok = torch.where(in_span, t_row[:, None].expand(b, n), torch.ones_like(tok))
        tok = torch.where(span_rows[:, None], row_tok, tok)
    return tok


def _rows_in(regime: torch.Tensor, names: list[str]) -> torch.Tensor:
    sel = torch.zeros_like(regime, dtype=torch.bool)
    for name in names:
        sel = sel | (regime == REGIME_CODES[name])
    return sel


def joint_flow_loss(
    model,
    img_latents: torch.Tensor,
    txt_x0: torch.Tensor,
    text: TextBatch,
    cfg,
    answer_tok_start: torch.Tensor,
    qa_cond_states: torch.Tensor,
    qa_window_row: torch.Tensor | None = None,
    img_known: torch.Tensor | None = None,
    img_x0_in: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Loss of one micro-batch.

    img_latents [B, 256, D] and txt_x0 [B, L, 512]: clean states of the sample.
    text: the sample's TextBatch. answer_tok_start [B]: answer start slot (-1 = no QA).
    qa_cond_states [B, L, 512]: states of "question prefix + EOS" (used on QA rows).
    qa_window_row [B] (optional): per-row answer window, <= 0 = use cfg.qa_answer_window.
    img_known [B, 256] bool and img_x0_in [B, 256, D] (optional): image-known patches and
    the states of the input render.
    """
    b, device = img_latents.shape[0], img_latents.device
    spans = answer_tok_start.to(device=device, dtype=torch.long)
    t_img, t_txt, regime = sample_regimes_and_times(cfg, b, device, spans)
    img_target = (regime != I2T).float()
    txt_target = (regime != T2I).float()
    qa_row = _rows_in(regime, list(cfg.qa_span_regimes)) & (spans >= 1)

    # Token times: the image pattern is drawn first, then the text pattern (RNG order).
    def resample_img(k: int) -> torch.Tensor:
        return sample_time(k, cfg.t_lognorm_mu, cfg.t_lognorm_sigma, cfg.t_shift, device)

    def resample_txt(k: int) -> torch.Tensor:
        return sample_time(k, cfg.text_t_mu, cfg.text_t_sigma, cfg.text_t_shift, device)

    n_img, n_txt = img_latents.shape[1], txt_x0.shape[1]
    t_img_tok = token_time_patterns(t_img, n_img, cfg.token_t_span_prob, cfg.token_t_indep_prob,
                                    resample_img, eligible=regime != I2T)
    # QA rows keep one time on the answer region (no pattern).
    t_txt_tok = token_time_patterns(t_txt, n_txt, cfg.token_t_span_prob, cfg.token_t_indep_prob,
                                    resample_txt, eligible=(regime != T2I) & ~qa_row)
    idx = torch.arange(n_txt, device=device)[None, :]
    qa_post = None
    if bool(qa_row.any()):
        prefix = idx < spans[:, None]
        t_txt_tok = torch.where(qa_row[:, None] & prefix, torch.ones_like(t_txt_tok), t_txt_tok)
        # Clean slots of QA rows (prefix and PAD tail) hold the question-only states.
        swap = qa_row[:, None] & (prefix | text.pad)
        txt_x0 = torch.where(swap.unsqueeze(-1), qa_cond_states.float(), txt_x0)
        window = torch.full_like(spans, int(cfg.qa_answer_window))
        if qa_window_row is not None:
            row_window = qa_window_row.to(device=device, dtype=torch.long)
            window = torch.where(row_window > 0, row_window, window)
        qa_post = qa_row[:, None] & (idx >= (spans + window)[:, None])
        if bool((qa_post & text.valid).any()):
            raise ValueError("an answer token falls past its answer window; raise the window or fix the data")
        t_txt_tok = torch.where(qa_post, torch.ones_like(t_txt_tok), t_txt_tok)
    img_known_tok = None
    if cfg.img_known_inpaint and img_known is not None:
        if img_x0_in is None:
            raise ValueError("img_known needs img_x0_in (the states of the input render)")
        known_row = _rows_in(regime, list(cfg.img_known_regimes)) & img_known.any(dim=1)
        img_known_tok = known_row[:, None] & img_known
        t_img_tok = torch.where(img_known_tok, torch.ones_like(t_img_tok), t_img_tok)

    tb_i, tb_t = t_img_tok.unsqueeze(-1), t_txt_tok.unsqueeze(-1)
    img_xt = img_latents * tb_i + (torch.randn_like(img_latents) * cfg.noise_scale) * (1.0 - tb_i)
    if img_known_tok is not None:
        img_xt = torch.where(img_known_tok.unsqueeze(-1), img_x0_in.to(img_xt.dtype), img_xt)
    txt_zt = txt_x0 * tb_t + (torch.randn_like(txt_x0) * cfg.text_noise_scale) * (1.0 - tb_t)

    # A clean text condition (t2i rows) shows only its valid slots; a generated text shows
    # all slots, so the model never sees the true length.
    txt_key_mask = torch.where((regime == T2I).view(-1, 1), text.valid, torch.ones_like(text.valid))
    if qa_post is not None:
        txt_key_mask = txt_key_mask & ~qa_post
    img_pred, txt_pred = model(img_xt, t_img_tok, txt_zt, t_txt_tok, txt_key_mask=txt_key_mask)

    # Image: x0 prediction, velocity-space MSE over noised tokens of image-target rows.
    img_v = (img_pred - img_xt) / (1.0 - tb_i).clamp_min(T_EPS)
    img_v_tgt = (img_latents - img_xt) / (1.0 - tb_i).clamp_min(T_EPS)
    img_tok_active = (t_img_tok < 1.0 - 1e-6).float() * img_target.view(-1, 1)
    loss_img = ((img_v - img_v_tgt).pow(2).mean(dim=-1) * img_tok_active).sum() / img_tok_active.sum().clamp_min(1.0)

    # Text: the same on noised content + EOS slots of text-target rows.
    v_pred = (txt_pred - txt_zt) / (1.0 - tb_t).clamp_min(T_EPS)
    v_tgt = (txt_x0 - txt_zt) / (1.0 - tb_t).clamp_min(T_EPS)
    l2 = (v_pred - v_tgt).pow(2).mean(dim=-1)
    txt_tok_active = (t_txt_tok < 1.0 - 1e-6).float()
    txt_mask = txt_target.view(-1, 1) * txt_tok_active * text.valid.float()
    loss_txt_flow = (l2 * txt_mask).sum() / txt_mask.sum().clamp_min(1.0)
    loss_txt = loss_txt_flow
    loss_tail = torch.zeros((), device=device)
    if cfg.qa_window_tail_weight > 0.0 and qa_post is not None:
        # PAD slots inside the generated window learn the question-only PAD state.
        in_window = qa_row[:, None] & (idx >= spans[:, None]) & ~qa_post
        tail_mask = txt_target.view(-1, 1) * txt_tok_active * (in_window & text.pad).float()
        loss_tail = (l2 * tail_mask).sum() / tail_mask.sum().clamp_min(1.0)
        loss_txt = loss_txt + cfg.qa_window_tail_weight * loss_tail

    loss = cfg.lambda_img * loss_img + cfg.lambda_txt * loss_txt
    # The same keys on every micro-batch (the trainer averages them over GPUs).
    metrics = {
        "loss_txt_qa_tail": loss_tail.detach(),
        "loss": loss.detach(),
        "loss_img": loss_img.detach(),
        "loss_txt": loss_txt.detach(),
        "loss_txt_flow": loss_txt_flow.detach(),
        "img_active_tokens": img_tok_active.sum().detach(),
        "txt_flow_tokens": txt_mask.sum().detach(),
        "rows_joint": (regime == JOINT).sum().detach(),
        "rows_t2i": (regime == T2I).sum().detach(),
        "rows_i2t": (regime == I2T).sum().detach(),
        "rows_qa": qa_row.sum().detach(),
        "img_known_rows": (img_known_tok.any(dim=1).sum() if img_known_tok is not None
                           else torch.zeros((), device=device)).detach(),
    }
    return loss, metrics
