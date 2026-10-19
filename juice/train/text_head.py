"""Training of the parallel text head (juice/heads.py: ParallelTextHead).

The head maps 128 text states to 128 token ids in one forward pass. It is a copy of
the Flan-T5-Small encoder stack (8 layers) that reads the states as input embeddings,
plus an LM head. Every slot sees every slot (all-ones attention), so the head never
sees the length of the caption.

Training data: the texts (captions and "Question: ... Answer: ..." rows) of the
pretraining pool, read by the stream of juice/train/data.py. The images are not used.

One step:
    1. Put a batch of texts into the slot contract (content + EOS + PAD).
    2. Encode it with the frozen Flan-T5-Small encoder (bfloat16 autocast on GPU).
       The head reads these T5 states as they are (not divided by 0.2).
    3. Corrupt the first corrupt_frac of the rows, slot by slot:
           state <- lam * state + (1 - lam) * noise,
           lam = sigmoid(N(0.8, 0.8^2)),  noise ~ N(0, 0.2^2).
    4. Loss: cross entropy at all 128 slots, normalized by the weight sum.
           content and EOS slots -> their token, weight 1
           PAD slots             -> EOS,         weight tail_weight
       The PAD term teaches the head to repeat EOS after the text. So a reader can
       stop at the first predicted EOS.
    5. Clip the gradient norm at 1.0. AdamW step (weight decay 0).

Start: the encoder stack of the pretrained Flan-T5-Small, and an LM head equal to
the token embedding x 512**-0.5 (bias 0), or a text-head folder (init_from).
The released head: stage 1 from the pretrained start, then stage 2 from stage 1.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import nn

from juice.representations import EOS_ID, TEXT_ENCODER_ID, TEXT_ENCODER_REVISION, T5TextEncoder, TextBatch
from juice.train.config import TrainConfig
from juice.train.data import make_loader

# Corruption of the head input (T5 state space): noise std and the logit-normal mix.
NOISE_STD = 0.2
MIX_P_MEAN = 0.8
MIX_P_STD = 0.8
# The head reads states, not token ids: these tensors are not saved (as in juice/export.py).
EMBEDDING_KEYS = ("encoder.shared.weight", "encoder.encoder.embed_tokens.weight")


@dataclass
class TextHeadConfig:
    train_manifests: list[str]           # manifest files (one .tar shard path per line)
    train_manifest_counts: list[int]     # samples per manifest
    run_name: str = "text_head"
    output_dir: str = "outputs/train"
    init_from: str | None = None         # text-head folder for stage 1 (None: pretrained Flan-T5-Small)

    # Data (the stream order depends on all of these; keep num_workers at 1)
    data_seed: int = 4242
    wds_shard_seed: int = 4242
    shuffle_buffer: int = 1000
    num_workers: int = 1
    prefetch: int = 4
    image_size: int = 224

    # Optimization (per stage)
    stages: int = 2                      # stage k > 1 starts from stage k - 1
    steps: int = 8000
    batch_size: int = 64
    learning_rate: float = 1e-4
    corrupt_frac: float = 0.5            # fraction of the rows that get the corruption
    tail_weight: float = 0.2             # loss weight of the PAD slots (label EOS)
    max_grad_norm: float = 1.0
    seed: int = 0                        # torch seed (corruption noise, dropout)
    log_every: int = 50

    def __post_init__(self) -> None:
        if len(self.train_manifests) != len(self.train_manifest_counts):
            raise ValueError("train_manifests and train_manifest_counts must have the same length")
        if self.stages < 1 or self.steps < 1:
            raise ValueError("stages and steps must be positive")
        if not 0.0 <= self.corrupt_frac <= 1.0:
            raise ValueError(f"corrupt_frac must be in [0, 1], got {self.corrupt_frac}")

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "TextHeadConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        raw.update({k: v for k, v in overrides.items() if v is not None})
        unknown = sorted(set(raw) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"{path}: unknown config keys {unknown}")
        return cls(**raw)

    def to_dict(self) -> dict:
        return asdict(self)

    def data_config(self) -> TrainConfig:
        """Settings of the text stream, in the form that juice/train/data.py reads."""
        return TrainConfig(
            run_name=self.run_name, train_manifests=list(self.train_manifests),
            train_manifest_counts=list(self.train_manifest_counts), image_size=self.image_size,
            shuffle_buffer=self.shuffle_buffer, num_workers=self.num_workers, prefetch=self.prefetch,
            wds_shard_seed=self.wds_shard_seed, seed=self.data_seed, micro_batch_size=self.batch_size,
        )


class TrainableTextHead(nn.Module):
    """The modules and parameter names of ParallelTextHead; reads T5 states (not divided by 0.2)."""

    def __init__(self, encoder: nn.Module, lm_head: nn.Linear):
        super().__init__()
        self.encoder = encoder
        self.lm_head = lm_head

    def forward(self, t5_states: torch.Tensor) -> torch.Tensor:
        x = t5_states.to(dtype=self.lm_head.weight.dtype)
        mask = torch.ones(x.shape[:2], dtype=torch.long, device=x.device)
        return self.lm_head(self.encoder(inputs_embeds=x, attention_mask=mask).last_hidden_state)


def build_head(init_from: str | Path | None = None) -> TrainableTextHead:
    """Pretrained Flan-T5-Small encoder stack + LM head (token embedding x d_model**-0.5),
    then the weights of a text-head folder if init_from is given."""
    from transformers import T5EncoderModel

    encoder = T5EncoderModel.from_pretrained(TEXT_ENCODER_ID, revision=TEXT_ENCODER_REVISION)
    dim, vocab = int(encoder.config.d_model), int(encoder.config.vocab_size)
    lm_head = nn.Linear(dim, vocab)
    with torch.no_grad():
        # T5 scales the stack output by d_model**-0.5 before its tied LM head.
        lm_head.weight.copy_(encoder.shared.weight * (float(dim) ** -0.5))
        lm_head.bias.zero_()
    head = TrainableTextHead(encoder, lm_head)
    if init_from is not None:
        from safetensors.torch import load_file

        folder = Path(init_from)
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if int(config["vocab_size"]) != vocab or int(config["t5_config"]["d_model"]) != dim:
            raise ValueError(f"{folder}: vocab/d_model {config['vocab_size']}/{config['t5_config']['d_model']} "
                             f"!= {vocab}/{dim}")
        missing, unexpected = head.load_state_dict(load_file(str(folder / "model.safetensors")), strict=False)
        if set(missing) - set(EMBEDDING_KEYS) or unexpected:
            raise RuntimeError(f"{folder}: text head weights do not match: missing={missing} unexpected={unexpected}")
    return head


def corrupt(states: torch.Tensor, noise_std: float = NOISE_STD) -> torch.Tensor:
    """Per-slot logit-normal mix of the states toward N(0, noise_std^2)."""
    lam = torch.sigmoid(torch.randn(states.shape[0], states.shape[1], 1, device=states.device) * MIX_P_STD + MIX_P_MEAN)
    noise = torch.randn_like(states) * noise_std
    return lam * states + (1 - lam) * noise


def labels_and_weights(tb: TextBatch, tail_weight: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Valid slots: their token, weight 1. PAD slots: EOS, weight tail_weight."""
    labels = torch.where(tb.valid, tb.ids, torch.full_like(tb.ids, EOS_ID))
    weights = torch.where(tb.valid, torch.ones_like(labels, dtype=torch.float32),
                          torch.full_like(labels, float(tail_weight), dtype=torch.float32))
    return labels, weights


def head_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                         reduction="none").reshape(labels.shape)
    return (ce * weights).sum() / weights.sum().clamp_min(1e-6)


def first_eos_index(ids: torch.Tensor) -> torch.Tensor:
    """Slot of the first EOS of each row ([B, L] -> [B]); L if a row has no EOS."""
    is_eos = ids == EOS_ID
    length = torch.full((ids.shape[0],), ids.shape[1], device=ids.device, dtype=torch.long)
    return torch.where(is_eos.any(dim=1), is_eos.long().argmax(dim=1), length)


@torch.no_grad()
def batch_metrics(logits: torch.Tensor, tb: TextBatch) -> dict[str, float]:
    """Token accuracy on the valid slots, exact match of the valid span, first-EOS slot error."""
    pred = logits.argmax(dim=-1)
    return {
        "tok_acc": float(((pred == tb.ids) & tb.valid).sum()) / max(1.0, float(tb.valid.sum())),
        "exact_match": float(((pred == tb.ids) | ~tb.valid).all(dim=1).float().mean()),
        "eos_pos_mae": float((first_eos_index(pred) - first_eos_index(tb.ids)).abs().float().mean()),
    }


def save_head(head: TrainableTextHead, folder: Path, training: dict) -> None:
    """Write folder/model.safetensors + folder/config.json (the format of ParallelTextHead.from_folder)."""
    from safetensors.torch import save_file

    from juice.heads import ParallelTextHead

    tensors = {k: v.detach().to(torch.float32).cpu().contiguous().clone()
               for k, v in head.state_dict().items() if k not in EMBEDDING_KEYS}
    config = {"t5_config": head.encoder.config.to_dict(), "vocab_size": int(head.lm_head.out_features),
              "training": training}
    folder.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(folder / "model.safetensors"), metadata={"format": "pt"})
    (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    ParallelTextHead.from_folder(folder)  # proves that the folder loads


def train_stage(cfg: TextHeadConfig, out: Path, init_from: str | Path | None = None, device=None,
                max_steps: int | None = None) -> Path:
    """One stage: cfg.steps steps from init_from (None: pretrained start). Writes the head to `out`."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.seed)
    text_encoder = T5TextEncoder().to(device)            # frozen: makes the states
    head = build_head(init_from).to(device)              # trained
    params = [p for p in head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=0.0)
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
    data = iter(make_loader(cfg.data_config(), 0, 1, text_encoder.tokenizer))
    n_corrupt = int(cfg.corrupt_frac * cfg.batch_size)
    steps = cfg.steps if max_steps is None else min(cfg.steps, max_steps)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[text-head] {sum(p.numel() for p in params) / 1e6:.1f}M parameters, init_from={init_from}, "
          f"{steps} steps, device={device}", flush=True)
    head.train()
    t_start = time.time()
    with (out / "log.jsonl").open("w", encoding="utf-8") as log:
        for step in range(steps):
            batch = next(data)
            tb = TextBatch.from_ids(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            with torch.no_grad(), amp:
                states = text_encoder.model(input_ids=tb.ids, attention_mask=tb.valid.long()).last_hidden_state.float()
            if n_corrupt > 0:
                states[:n_corrupt] = corrupt(states[:n_corrupt])
            labels, weights = labels_and_weights(tb, cfg.tail_weight)
            with amp:
                logits = head(states)
            loss = head_loss(logits, labels, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % max(1, cfg.log_every) == 0 or step == steps - 1:
                record = {"step": step, "loss": round(float(loss), 4)}
                record.update({k: round(v, 4) for k, v in batch_metrics(logits, tb).items()})
                record["elapsed"] = round(time.time() - t_start, 1)
                print(json.dumps(record), flush=True)
                log.write(json.dumps(record) + "\n")
                log.flush()
    del data
    training = {"init_from": None if init_from is None else str(init_from), "steps": steps,
                **{k: getattr(cfg, k) for k in ("batch_size", "learning_rate", "corrupt_frac", "tail_weight", "seed")},
                "noise_std": NOISE_STD, "mix_p_mean": MIX_P_MEAN, "mix_p_std": MIX_P_STD}
    save_head(head, out, training)
    print(f"[text-head] wrote {out}", flush=True)
    return out


def train_text_head(cfg: TextHeadConfig, device=None, max_steps: int | None = None) -> Path:
    """All stages; stage k is written to <output_dir>/<run_name>/stage_<k>. Returns the last folder."""
    root = Path(cfg.output_dir) / cfg.run_name
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n", encoding="utf-8")
    init = cfg.init_from
    for stage in range(1, cfg.stages + 1):
        init = train_stage(cfg, root / f"stage_{stage}", init, device, max_steps)
    return init
