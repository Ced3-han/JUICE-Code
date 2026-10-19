"""The training loop (pretraining, SFT and task fine-tuning use the same loop).

One optimizer step = grad_accum_steps micro-batches on every GPU:
    1. encode the micro-batch with the frozen encoders (bfloat16 autocast, no gradient)
    2. loss (juice/train/loss.py) under bfloat16 autocast, divided by grad_accum_steps
    3. backward (DDP synchronizes the gradients only on the last micro-batch)
then: learning rate (linear warmup from 1e-6, then constant), gradient clipping at
max_grad_norm, fused AdamW (beta1 0.9), and two EMA copies of the weights.

TF32 is on for matmuls and convolutions. The model is wrapped in DDP and then compiled.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from juice.model import MMRiT, MMRiTConfig
from juice.representations import IMAGE_SPACES, T5TextEncoder, TextBatch, build_image_encoder, to_unit_range
from juice.train.config import TrainConfig
from juice.train.data import make_loader
from juice.train.loss import joint_flow_loss

WARMUP_START_LR = 1e-6


def lr_at(cfg: TrainConfig, step: int) -> float:
    if step < cfg.warmup_steps:
        return WARMUP_START_LR + (step + 1) / max(cfg.warmup_steps, 1) * (cfg.learning_rate - WARMUP_START_LR)
    return cfg.learning_rate


@torch.no_grad()
def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    ema_params = [p.data for p in ema.parameters()]
    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, [p.detach() for p in model.parameters()], alpha=1.0 - decay)


def build_model(cfg: TrainConfig) -> MMRiT:
    return MMRiT(MMRiTConfig(image_latent_dim=IMAGE_SPACES[cfg.image_space].latent_dim))


def load_weights(path: str | Path) -> dict[str, torch.Tensor]:
    """Model weights from a model folder, a .safetensors file or a training checkpoint ("model")."""
    path = Path(path)
    if path.is_dir():
        path = path / "model.safetensors"
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)["model"]


@torch.no_grad()
def encode_micro_batch(batch: dict, device, image_encoder, text_encoder, need_known: bool) -> dict:
    """Frozen-encoder states of one micro-batch (bfloat16 autocast, as in the paper's runs)."""
    tb = TextBatch.from_ids(batch["input_ids"].to(device), batch["attention_mask"].to(device))
    spans = batch["answer_tok_start"].to(device)
    out = {"text": tb, "spans": spans, "qa_window": batch.get("qa_window")}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out["img"] = image_encoder(to_unit_range(batch["pixel_values"].to(device)))
        out["txt"] = text_encoder.encode(tb, dtype=torch.bfloat16)
        out["qa_cond"] = None
        if bool((spans >= 1).any()):
            # Condition states of QA rows: T5 states of "question prefix + EOS" alone,
            # so the clean prefix slots carry no information about the answer.
            pos = torch.arange(tb.ids.shape[1], device=device)[None, :]
            start = spans.clamp(min=0)[:, None]
            prefix_ids = torch.where(pos < start, tb.ids, torch.zeros_like(tb.ids))
            out["qa_cond"] = text_encoder.encode(TextBatch.from_ids(prefix_ids, pos < start + 1), dtype=torch.bfloat16)
        out["img_known"] = out["img_in"] = None
        if need_known:
            if "img_known_patches" not in batch:
                raise RuntimeError("img_known_inpaint is on, but the data has no img_known_patches sidecar")
            out["img_known"] = batch["img_known_patches"].to(device)
            out["img_in"] = image_encoder(to_unit_range(batch["img_input"].to(device)))
    if out["qa_window"] is not None:
        out["qa_window"] = out["qa_window"].to(device)
    return out


def micro_step_loss(model, enc: dict, cfg: TrainConfig) -> tuple[torch.Tensor, dict]:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return joint_flow_loss(
            model, enc["img"], enc["txt"], enc["text"], cfg, enc["spans"], enc["qa_cond"],
            qa_window_row=enc["qa_window"], img_known=enc["img_known"], img_x0_in=enc["img_in"],
        )


def checkpoint_payload(cfg: TrainConfig, model, ema, ema_fast, opt, step: int) -> dict:
    return {
        "model": model.state_dict(), "ema": ema.state_dict(), "ema_fast": ema_fast.state_dict(),
        "optimizer": opt.state_dict(), "step": step, "config": cfg.to_dict(),
        "architecture": model.cfg.to_dict(),
    }


def _save(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def train(cfg: TrainConfig, resume: str | None = None, max_steps: int | None = None) -> None:
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if cfg.batch_size != cfg.micro_batch_size * cfg.grad_accum_steps * world:
        raise ValueError(f"batch_size {cfg.batch_size} != micro {cfg.micro_batch_size} x accum "
                         f"{cfg.grad_accum_steps} x GPUs {world}; change grad_accum_steps")
    torch.manual_seed(cfg.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out_dir = Path(cfg.output_dir) / cfg.run_name
    ckpt_dir = out_dir / "checkpoints"
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    text_encoder = T5TextEncoder().to(device)
    image_encoder = build_image_encoder(cfg.image_space).to(device)
    model = build_model(cfg).to(device)
    start_step, opt_state = 0, None
    if resume:
        payload = torch.load(resume, map_location="cpu", weights_only=True, mmap=True)
        model.load_state_dict(payload["model"], strict=True)
        start_step, opt_state = int(payload["step"]), payload["optimizer"]
    elif cfg.init_from:
        model.load_state_dict(load_weights(cfg.init_from), strict=True)
    else:
        model.init_weights()
    ema = copy.deepcopy(model).requires_grad_(False)
    ema_fast = copy.deepcopy(model).requires_grad_(False)
    if resume:
        ema.load_state_dict(payload["ema"], strict=True)
        ema_fast.load_state_dict(payload["ema_fast"], strict=True)
        del payload
    raw_model = model

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        if cfg.ddp_bf16_compress:
            from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import bf16_compress_hook

            model.register_comm_hook(state=None, hook=bf16_compress_hook)
    if cfg.compile_model:
        model = torch.compile(model)
    opt = torch.optim.AdamW(
        (p for p in raw_model.parameters() if p.requires_grad), lr=cfg.learning_rate,
        betas=(0.9, cfg.adam_beta2), weight_decay=cfg.weight_decay, fused=True,
    )
    if opt_state is not None:
        opt.load_state_dict(opt_state)

    # A resumed run reads a new data order (the resume step is part of the stream seed).
    loader = make_loader(cfg, rank, world, text_encoder.tokenizer, resume_key=start_step)
    data = iter(loader)
    end_step = cfg.num_steps if max_steps is None else min(cfg.num_steps, start_step + max_steps)
    log_file = (out_dir / "log.jsonl").open("a") if rank == 0 else None
    t_start = time.time()
    for step in range(start_step, end_step):
        sums: dict[str, torch.Tensor] = {}
        for accum in range(cfg.grad_accum_steps):
            enc = encode_micro_batch(next(data), device, image_encoder, text_encoder, cfg.img_known_inpaint)
            last = accum == cfg.grad_accum_steps - 1
            sync = contextlib.nullcontext() if (last or not distributed) else model.no_sync()
            with sync:
                loss, metrics = micro_step_loss(model, enc, cfg)
                (loss / cfg.grad_accum_steps).backward()
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + value.float()
        lr = lr_at(cfg, step)
        for group in opt.param_groups:
            group["lr"] = lr
        params = [p for p in raw_model.parameters() if p.grad is not None]
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm, error_if_nonfinite=True)
        opt.step()
        opt.zero_grad(set_to_none=True)
        update_ema(ema, raw_model, cfg.ema_decay)
        update_ema(ema_fast, raw_model, cfg.ema_fast_decay)

        done = step + 1
        if done % cfg.log_every == 0 or done == end_step:
            keys = sorted(sums)
            values = torch.stack([sums[k] for k in keys]) / cfg.grad_accum_steps
            if distributed:
                dist.all_reduce(values)
                values /= world
            if rank == 0:
                record = {"step": done, "lr": lr, "grad_norm": float(grad_norm),
                          "sec_per_step": (time.time() - t_start) / (done - start_step)}
                record.update({k: round(float(v), 6) for k, v in zip(keys, values)})
                print(json.dumps(record), flush=True)
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
        if rank == 0 and (done % cfg.ckpt_every == 0 or done in cfg.ckpt_milestones or done == end_step):
            payload = checkpoint_payload(cfg, raw_model, ema, ema_fast, opt, done)
            _save(payload, ckpt_dir / f"checkpoint_{done:06d}.pt")
            rolling = sorted(p for p in ckpt_dir.glob("checkpoint_*.pt")
                             if int(p.stem.split("_")[1]) not in cfg.ckpt_milestones)
            for old in rolling[:-cfg.save_keep]:
                old.unlink()
    if log_file is not None:
        log_file.close()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
