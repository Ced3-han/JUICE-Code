"""Training of the linear task head of the planning models (juice/heads.py: LinearTextHead).

The planning models (JMaze, VSP) read their answers with one linear layer, applied to each slot:
normalized text state [512] -> Flan-T5 token logits [32128]. The head is trained alone,
without the JUICE model, on the texts of the task's training shards
("Question: ... Answer: ..."). The images of the shards are not used.

One step:
    1. Put a batch of texts into the slot contract (content + EOS + PAD, 128 slots).
    2. Encode it with the frozen Flan-T5-Small encoder (bfloat16 autocast on GPU) and
       divide by 0.2: the state space in which JUICE generates text.
    3. Corrupt the first corrupt_frac of the rows, slot by slot:
           state <- lam * state + (1 - lam) * noise,
           lam = sigmoid(N(0.8, 0.8^2)),  noise ~ N(0, noise_std^2).
    4. Loss: cross entropy over the content and EOS slots (mean over these slots).
       PAD slots carry no loss.
    5. AdamW step (no gradient clipping). The head is float32 (no autocast).

Start: nn.Linear default initialization, drawn on the CPU after torch.manual_seed(seed),
so it does not depend on the device. The corruption noise comes from the torch RNG of
the device (the CUDA generator on a GPU, also seeded by torch.manual_seed).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from juice.heads import LinearTextHead
from juice.representations import T5TextEncoder, TextBatch
from juice.train.data import make_loader
from juice.train.text_head import MIX_P_MEAN, MIX_P_STD, TextHeadConfig, batch_metrics, corrupt


@dataclass
class TaskHeadConfig:
    train_manifests: list[str]           # manifest files (one .tar shard path per line)
    train_manifest_counts: list[int]     # samples per manifest
    run_name: str = "task_head"
    output_dir: str = "outputs/train"

    # Data (the stream order depends on all of these and on num_workers)
    data_seed: int = 4242
    wds_shard_seed: int = 4242
    shuffle_buffer: int = 1000
    num_workers: int = 2
    prefetch: int = 4
    image_size: int = 224

    # Optimization
    steps: int = 1500
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    corrupt_frac: float = 0.5            # fraction of the rows that get the corruption
    noise_std: float = 1.0               # corruption noise std in the normalized state space
    seed: int = 67280421310721           # torch seed (initialization, corruption noise)
    log_every: int = 50

    def __post_init__(self) -> None:
        if len(self.train_manifests) != len(self.train_manifest_counts):
            raise ValueError("train_manifests and train_manifest_counts must have the same length")
        if self.steps < 1 or self.batch_size < 1:
            raise ValueError("steps and batch_size must be positive")
        if not 0.0 <= self.corrupt_frac <= 1.0:
            raise ValueError(f"corrupt_frac must be in [0, 1], got {self.corrupt_frac}")

    # YAML loading, the settings dict and the data-stream settings are those of the text head.
    from_yaml = classmethod(TextHeadConfig.from_yaml.__func__)
    to_dict = TextHeadConfig.to_dict
    data_config = TextHeadConfig.data_config


def task_head_loss(logits: torch.Tensor, tb: TextBatch) -> torch.Tensor:
    """Mean cross entropy over the valid (content + EOS) slots."""
    return F.cross_entropy(logits[tb.valid], tb.ids[tb.valid])


def save_task_head(head: LinearTextHead, folder: Path, training: dict) -> None:
    """Write folder/model.safetensors + folder/config.json (the format of LinearTextHead.from_folder)."""
    from safetensors.torch import save_file

    tensors = {k: v.detach().to(torch.float32).cpu().contiguous().clone() for k, v in head.state_dict().items()}
    config = {"dim": int(head.linear.in_features), "vocab_size": int(head.linear.out_features), "training": training}
    folder.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(folder / "model.safetensors"), metadata={"format": "pt"})
    (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    LinearTextHead.from_folder(folder)  # proves that the folder loads


def train_task_head(cfg: TaskHeadConfig, device=None, max_steps: int | None = None) -> Path:
    """Train the head; write <output_dir>/<run_name>/{config.json, log.jsonl, head/}. Returns the head folder."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    root = Path(cfg.output_dir) / cfg.run_name
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n", encoding="utf-8")

    torch.manual_seed(cfg.seed)
    text_encoder = T5TextEncoder().to(device)            # frozen: makes the states
    t5 = text_encoder.model.config
    head = LinearTextHead(int(t5.d_model), int(t5.vocab_size)).to(device)  # initialized on the CPU
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
    data = iter(make_loader(cfg.data_config(), 0, 1, text_encoder.tokenizer))
    n_corrupt = int(cfg.corrupt_frac * cfg.batch_size)
    steps = cfg.steps if max_steps is None else min(cfg.steps, max_steps)
    print(f"[task-head] linear {head.linear.in_features} -> {head.linear.out_features}, {steps} steps, "
          f"device={device}", flush=True)
    t_start = time.time()
    with (root / "log.jsonl").open("w", encoding="utf-8") as log:
        for step in range(1, steps + 1):
            batch = next(data)
            tb = TextBatch.from_ids(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            with torch.no_grad(), amp:
                states = text_encoder.encode(tb, torch.bfloat16)
            states = states.float()
            if n_corrupt > 0:
                states[:n_corrupt] = corrupt(states[:n_corrupt], cfg.noise_std)
            logits = head(states)
            loss = task_head_loss(logits, tb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step == 1 or step % max(1, cfg.log_every) == 0 or step == steps:
                record = {"step": step, "loss": float(loss)}
                record.update({k: round(v, 4) for k, v in batch_metrics(logits, tb).items()})
                record["elapsed"] = round(time.time() - t_start, 1)
                print(json.dumps(record), flush=True)
                log.write(json.dumps(record) + "\n")
                log.flush()
    del data
    training = {"steps": steps,
                **{k: getattr(cfg, k) for k in ("batch_size", "learning_rate", "weight_decay", "corrupt_frac",
                                                "noise_std", "seed")},
                "mix_p_mean": MIX_P_MEAN, "mix_p_std": MIX_P_STD}
    folder = root / "head"
    save_task_head(head, folder, training)
    print(f"[task-head] wrote {folder}", flush=True)
    return folder
