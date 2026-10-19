"""Training configuration: one flat dataclass, loaded from a YAML file.

The YAML files in configs/train/ hold the settings of the paper's runs. Unknown keys
are an error, so a typo cannot silently fall back to a default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from juice.representations import IMAGE_SPACES, TEXT_CANVAS


@dataclass
class TrainConfig:
    run_name: str
    train_manifests: list[str]           # manifest files (one .tar shard path per line)
    train_manifest_counts: list[int]     # samples per manifest (balances the data workers)

    # Model and initialization
    image_space: str = "siglip2"         # "siglip2" or "webssl_dino"
    init_from: str | None = None         # model folder (model.safetensors) or a training checkpoint
    output_dir: str = "outputs/train"

    # Data
    image_size: int = 224
    text_canvas: int = TEXT_CANVAS
    shuffle_buffer: int = 1000
    num_workers: int = 8                 # data workers per GPU
    prefetch: int = 4
    wds_shard_seed: int = 4242

    # Objective (juice/train/loss.py)
    task_mixture: list[float] = field(default_factory=lambda: [0.556, 0.222, 0.222])  # joint, t2i, i2t
    t_lognorm_mu: float = 0.0
    t_lognorm_sigma: float = 1.0
    t_shift: float = 8.485               # image time shift: 8.485 SigLIP2, 8.0 WebSSL-DINO
    text_t_mu: float = -1.5
    text_t_sigma: float = 0.8
    text_t_shift: float = 1.0
    noise_scale: float = 1.0             # image noise std
    text_noise_scale: float = 2.0        # text noise std
    token_t_span_prob: float = 0.15
    token_t_indep_prob: float = 0.10
    lambda_img: float = 1.0
    lambda_txt: float = 0.25
    qa_answer_window: int = 12
    qa_window_tail_weight: float = 0.1
    qa_span_regimes: list[str] = field(default_factory=lambda: ["i2t"])
    qa_rows_force_i2t: bool = True
    img_known_inpaint: bool = False
    img_known_regimes: list[str] = field(default_factory=lambda: ["product", "t2i"])

    # Optimization
    seed: int = 4242
    batch_size: int = 4096               # global batch = micro_batch_size * grad_accum_steps * GPUs
    micro_batch_size: int = 8
    grad_accum_steps: int = 64
    num_steps: int = 60_000
    learning_rate: float = 4e-4
    warmup_steps: int = 5000             # linear warmup from 1e-6, then constant
    adam_beta2: float = 0.95
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    ema_decay: float = 0.99995
    ema_fast_decay: float = 0.9996       # the released weights are this EMA
    ddp_bf16_compress: bool = False      # bf16 gradient all-reduce
    compile_model: bool = True

    # Logs and checkpoints
    log_every: int = 50
    ckpt_every: int = 2500               # rolling checkpoints (the last save_keep are kept)
    ckpt_milestones: list[int] = field(default_factory=list)  # kept forever
    save_keep: int = 3

    def __post_init__(self) -> None:
        if self.image_space not in IMAGE_SPACES:
            raise ValueError(f"image_space must be one of {sorted(IMAGE_SPACES)}, got {self.image_space!r}")
        if len(self.train_manifests) != len(self.train_manifest_counts):
            raise ValueError("train_manifests and train_manifest_counts must have the same length")
        if len(self.task_mixture) != 3 or min(self.task_mixture) < 0 or sum(self.task_mixture) <= 0:
            raise ValueError(f"task_mixture must be 3 non-negative weights, got {self.task_mixture}")
        for name in list(self.qa_span_regimes) + list(self.img_known_regimes):
            if name not in ("product", "t2i", "i2t"):
                raise ValueError(f"unknown regime {name!r} (product, t2i, i2t)")
        if self.qa_answer_window <= 0:
            raise ValueError("qa_answer_window must be positive")

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "TrainConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        raw.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"{path}: unknown config keys {unknown}")
        return cls(**raw)

    def to_dict(self) -> dict:
        return asdict(self)
