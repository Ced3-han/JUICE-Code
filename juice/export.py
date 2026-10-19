"""Convert a training checkpoint into a release folder.

A training checkpoint (.pt) holds {"model", "ema", "ema_fast", "config", ...}.
A release folder holds one weight set and the settings to use it:

    model.safetensors   MM-RiT weights, float32
    config.json         architecture, representation space, sampler settings and
                        the inference defaults of the paper

Usage:
    python -m juice.export --checkpoint checkpoint_015000.pt --weights ema_fast \
        --out my-sft-siglip2 --text-head outputs/train/text_head/stage_2

--text-head copies a text-head folder (written by scripts/train/train_text_head.py)
into <out>/text_head after a load check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

from juice.model import MMRiT, MMRiTConfig
from juice.representations import TEXT_LATENT_STD

# Inference defaults of the paper, per release preset. "precision" applies to the
# model and the encoders: "fp32" = TF32 off, "tf32" = TF32 matmuls (as in the paper's runs).
# "default" serves the pretrained and the SFT models.
PRESETS = {
    "default": {
        "text_to_image": {"steps": 25, "cfg_scale": 5.0, "null_caption": "a photo.", "precision": "fp32"},
        "answer": {"steps": 16, "cfg_scale": 3.0, "answer_window": 12},
    },
    # Planning fine-tunes (JMaze, VSP): coupled joint sampler, no guidance, TF32 off,
    # batches of 25 puzzles (the batch of the paper's runs).
    "jmaze": {"plan": {"task": "jmaze", "steps": 16, "cfg_scale": 1.0, "answer_window": 64,
                       "precision": "fp32", "batch_size": 25}},
    "vsp": {"plan": {"task": "vsp", "steps": 16, "cfg_scale": 1.0, "answer_window": 64,
                     "precision": "fp32", "batch_size": 25}},
}
QA_PRECISION = {"siglip2": "tf32", "webssl_dino": "fp32"}  # precision of the paper's QA results

# Tensors of the paper's research checkpoints that the joint inference path never reads.
UNUSED_KEYS = ("txt_embed.weight", "mask_token", "pooled_embed.weight", "txt_decode_proj.weight",
               "txt_decode_proj.bias", "txt_unembed.weight", "txt_unembed.bias")


def _require(cfg: dict, key: str, expected) -> None:
    if cfg.get(key) != expected:
        raise ValueError(f"checkpoint config {key}={cfg.get(key)!r}; this release supports {expected!r}")


def release_config(train_cfg: dict, preset: str) -> dict:
    """Map a training config onto the release config (fails on unsupported variants)."""
    for key, value in {
        "image_latent": True, "text_stream": True, "time_cond": "in_context", "token_time": True,
        "image_prediction": "x", "text_codec": "t5", "text_codec_norm": "scalar", "text_pad_contract": "strict",
        "pad_tail_mode": "encoder", "compat_checkpoint_arch": False, "attn_isolate_txt": False,
        "t5_name": "google/flan-t5-small", "txt_len": 128, "text_encoder_layer": None,
    }.items():
        _require(train_cfg, key, value)
    if train_cfg["text_hidden_size"] != train_cfg["hidden_size"]:
        raise ValueError("text_hidden_size must equal hidden_size")
    space = {"rae": "siglip2", "webssl_dino": "webssl_dino"}[train_cfg["image_codec"]]
    arch = MMRiTConfig(
        image_latent_dim=int(train_cfg["image_latent_dim"]),
        image_grid=int(train_cfg["image_size"]) // int(train_cfg["patch_size"]),
        text_dim=int(train_cfg["t5_hidden_size"]),
        text_bottleneck_dim=int(train_cfg["text_bottleneck_dim"]),
        hidden_size=int(train_cfg["hidden_size"]),
        depth=int(train_cfg["depth_double"]),
        text_preamble_depth=int(train_cfg["text_preamble_depth"]),
        num_heads=int(train_cfg["num_heads"]),
        head_dim=int(train_cfg["head_dim"]),
        mlp_ratio=float(train_cfg["mlp_ratio"]),
        time_tokens=int(train_cfg["time_tokens"]),
        token_time=True,
        attention_impl=str(train_cfg["attention_impl"]),
    )
    defaults = json.loads(json.dumps(PRESETS[preset]))
    if "answer" in defaults:
        defaults["answer"]["precision"] = QA_PRECISION[space]
    return {
        "architecture": arch.to_dict(),
        "image_space": space,
        "text_latent_std": float(train_cfg["text_latent_std"]),
        "flow": {
            "t_shift": float(train_cfg["t_shift"]),
            "text_t_shift": float(train_cfg.get("text_t_shift", 1.0)),
            "noise_scale": float(train_cfg["noise_scale"]),
            "text_noise_scale": float(train_cfg["text_noise_scale"]),
        },
        "defaults": defaults,
    }


def trainer_release_config(train_cfg: dict, architecture: dict, preset: str) -> dict:
    """Release config of a checkpoint written by scripts/train/train.py."""
    space = train_cfg["image_space"]
    defaults = json.loads(json.dumps(PRESETS[preset]))
    if "answer" in defaults:
        defaults["answer"]["precision"] = QA_PRECISION[space]
    return {
        "architecture": MMRiTConfig(**architecture).to_dict(),
        "image_space": space,
        "text_latent_std": TEXT_LATENT_STD,
        "flow": {key: float(train_cfg[key]) for key in ("t_shift", "text_t_shift", "noise_scale", "text_noise_scale")},
        "defaults": defaults,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def export(checkpoint: Path, weights: str, out: Path, preset: str, sha256: str | None = None) -> None:
    from safetensors.torch import save_file

    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    if "architecture" in payload:  # checkpoint of scripts/train/train.py
        config = trainer_release_config(payload["config"], payload["architecture"], preset)
    else:  # checkpoint of the paper's research code
        config = release_config(payload["config"], preset)
    state = {k: v for k, v in payload[weights].items() if k not in UNUSED_KEYS}
    model = MMRiT(MMRiTConfig(**config["architecture"]))
    model.load_state_dict(state, strict=True)  # proves the key set before anything is written
    tensors = {k: v.detach().to(torch.float32).contiguous().clone() for k, v in state.items()}
    config["source"] = {"checkpoint": checkpoint.name, "weights": weights, "step": int(payload.get("step", -1)),
                        "sha256": sha256 or _sha256(checkpoint)}
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(tensors)} tensors, {sum(t.numel() for t in tensors.values()) / 1e9:.3f} B params)")


def copy_head(folder: Path, out: Path, kind: str) -> None:
    """Copy a head folder (model.safetensors + config.json) to out after a load check.

    kind "text": ParallelTextHead (scripts/train/train_text_head.py); kind "task": LinearTextHead.
    """
    from juice.heads import LinearTextHead, ParallelTextHead

    {"text": ParallelTextHead, "task": LinearTextHead}[kind].from_folder(folder)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("model.safetensors", "config.json"):
        shutil.copy2(folder / name, out / name)
    print(f"copied {folder} -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--weights", default="ema_fast", choices=["model", "ema", "ema_fast"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--preset", default="default", choices=sorted(PRESETS))
    ap.add_argument("--sha256", default=None, help="known sha256 of --checkpoint (skips hashing a large file)")
    ap.add_argument("--text-head", type=Path, default=None, help="text-head folder to copy into <out>/text_head")
    args = ap.parse_args()
    export(args.checkpoint, args.weights, args.out, args.preset, args.sha256)
    if args.text_head is not None:
        copy_head(args.text_head, args.out / "text_head", "text")


if __name__ == "__main__":
    main()
