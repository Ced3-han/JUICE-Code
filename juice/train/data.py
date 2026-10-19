"""Training data: a seeded, repeating stream over WebDataset shards.

Shards. Each manifest file lists .tar shards (one path per line; relative paths start
at the manifest's folder). A sample holds an image (jpg/jpeg/png/webp/bmp), a caption
(txt) and an optional json sidecar:
    answer_tok_start   first answer slot of a "Question: ... Answer: ..." row (QA rows)
    qa_window          answer window of the row (SFT packs)
    img_known_patches  256 flags of the patches that the task gives (planning packs),
                       with the input render as the extra member "input.png"

Order. The configs give the sample count of every manifest. Each shard gets the mean
count of its manifest, and the shards are split once over all (rank, worker) slots by
a deterministic longest-first balance, after one seeded shuffle. A slot then reads its
own shards forever: one seeded shuffle of its shard list per cycle, then a seeded
1000-sample shuffle buffer. So the stream is a pure function of (seed, world size,
number of workers) and the shards.
"""

from __future__ import annotations

import io
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset

IMAGE_KEYS = ("jpg", "jpeg", "png", "webp", "bmp")


def read_manifest(path: str | Path) -> list[str]:
    path = Path(path).expanduser()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        entries = payload.get("tars") if isinstance(payload, dict) else payload
    else:
        entries = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    out = []
    for entry in entries:
        p = Path(str(entry)).expanduser()
        out.append(str(p if p.is_absolute() else (path.parent / p).resolve()))
    return out


def resolve_shards(manifests: list[str], counts: list[int]) -> tuple[list[str], list[float]]:
    """All shards of all manifests, and the expected sample count of each shard."""
    if len(manifests) != len(counts):
        raise ValueError(f"{len(manifests)} manifests but {len(counts)} sample counts")
    shards, weights = [], []
    for manifest, count in zip(manifests, counts):
        entries = read_manifest(manifest)
        if not entries or int(count) <= 0:
            raise ValueError(f"manifest {manifest}: {len(entries)} shards, count {count}")
        shards += entries
        weights += [float(count) / len(entries)] * len(entries)
    seen = set()
    for shard in shards:
        key = str(Path(shard).resolve())
        if key in seen:
            raise ValueError(f"shard listed twice: {shard} (use hard links for replicas)")
        seen.add(key)
        if not Path(shard).is_file():
            raise FileNotFoundError(shard)
    return shards, weights


def balanced_slot_assignment(urls: list[str], weights: list[float], num_slots: int) -> list[list[str]]:
    """Longest-first balance of the shards over the slots. Each slot keeps the input order."""
    order = sorted(range(len(urls)), key=lambda i: (-float(weights[i]), i))
    slot_indices: list[list[int]] = [[] for _ in range(num_slots)]
    loads = [0.0] * num_slots
    for i in order:
        target = min(range(num_slots), key=lambda s: (loads[s], s))
        slot_indices[target].append(i)
        loads[target] += float(weights[i])
    return [[urls[i] for i in sorted(slot)] for slot in slot_indices]


def _json_field(payload, key: str):
    if isinstance(payload, bytes):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return None
    return payload.get(key) if isinstance(payload, dict) else None


def _int_or(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _image(sample: dict) -> Image.Image:
    for key in IMAGE_KEYS:
        if sample.get(key) is not None:
            return sample[key].convert("RGB")
    # A few source images are MPO JPEGs or kept a URL query in the member name
    # ("jpg?1"); the decoder leaves them as bytes.
    raw = sample.get("mpo")
    if isinstance(raw, (bytes, bytearray)):
        return Image.open(io.BytesIO(raw)).convert("RGB")
    for key, raw in sample.items():
        if "?" in key and key.split("?", 1)[0] in IMAGE_KEYS and isinstance(raw, (bytes, bytearray)):
            return Image.open(io.BytesIO(raw)).convert("RGB")
    raise ValueError(f"no image in sample {sample.get('__key__')}")


def _caption(sample: dict) -> str:
    caption = sample.get("txt")
    if caption is None and "json" in sample:
        caption = conversation_prompt(sample["json"])
    if isinstance(caption, bytes):
        caption = caption.decode("utf-8", errors="replace")
    return "" if caption is None else str(caption)


# Scale-RAE packs give the prompt as conversations[0].value wrapped in one of these
# instruction templates (prefix, suffix); we strip the wrapper (longest prefix first).
_TEMPLATES = sorted([
    ("Design an image based on this description:", ""), ("Visualize this for me:", ""),
    ("Create an image based on:", ""), ("I'd like to see an image of", ""),
    ("Could you generate an image of", ""), ("Could you create an image showing", ""),
    ("Can you make an image of", ""), ("Generate a visual representation of", ""),
    ("Make a visual representation of", ""), ("Create a picture that shows", ""),
    ("Make an image that depicts", ""), ("I need an image depicting", ""),
    ("Please create an image of", ""), ("I'd like an image showing", ""),
    ("Create a picture showing", ""), ("Produce an image depicting", ""),
    ("Show me a picture of", ""), ("I'd like a depiction of", ""),
    ("Create a visualization of", ""), ("I need a picture showing", ""),
    ("Make a picture of", ""), ("Render an image of", ""), ("Generate a picture of", ""),
    ("I want an image of", ""), ("Generate an image of", ""), ("Generate imagery of", ""),
    ("Create a visual for", ""), ("Please visualize", ""), ("Draw", ""),
    ("I'd like to see", "as an image"), ("Show", "as an image"), ("Can you visualize", "for me?"),
], key=lambda pair: len(pair[0]), reverse=True)


def conversation_prompt(payload) -> str:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        payload = json.loads(payload)
    text = str(payload["conversations"][0].get("value", "")).strip()
    for prefix, suffix in _TEMPLATES:
        if prefix.endswith(":"):
            if not text.startswith(prefix):
                continue
            candidate = text[len(prefix):]
        elif text.startswith(prefix + " "):
            candidate = text[len(prefix) + 1:]
        else:
            continue
        if suffix:
            if not candidate.rstrip().endswith(suffix):
                continue
            candidate = candidate.rstrip()[: -len(suffix)]
        if candidate.strip():
            return candidate.strip()
    return text


def _identity(src):
    # The shards are already split over (rank, worker) slots above.
    yield from src


class ShardStream(IterableDataset):
    def __init__(self, cfg, rank: int, world_size: int, tokenizer, resume_key: int = 0):
        super().__init__()
        from torchvision import transforms

        self.cfg, self.rank, self.world_size, self.tokenizer = cfg, rank, world_size, tokenizer
        self.shards, weights = resolve_shards(list(cfg.train_manifests), list(cfg.train_manifest_counts))
        self.weight = dict(zip(self.shards, weights))
        # A resumed run draws a new permutation instead of replaying the samples it has seen.
        self.stream_seed = int(cfg.seed) + int(resume_key) * 7_368_787
        self.transform = transforms.Compose([
            transforms.Resize(cfg.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(cfg.image_size),
            transforms.PILToTensor(),
        ])

    def process(self, sample: dict) -> dict:
        caption = _caption(sample)
        tok = self.tokenizer(caption, max_length=self.cfg.text_canvas, padding="max_length",
                             truncation=True, return_tensors="pt")
        sidecar = sample.get("json")
        out = {
            "pixel_values": self.transform(_image(sample)),
            "input_ids": tok["input_ids"][0].long(),
            "attention_mask": tok["attention_mask"][0].long(),
            "answer_tok_start": _int_or(_json_field(sidecar, "answer_tok_start"), -1),
        }
        window = _int_or(_json_field(sidecar, "qa_window"), None)
        if window is not None and window > 0 and out["answer_tok_start"] >= 1:
            out["qa_window"] = window
        known = _json_field(sidecar, "img_known_patches")
        input_image = sample.get("input.png")
        if isinstance(known, (list, tuple)) and known and isinstance(input_image, Image.Image):
            out["img_input"] = self.transform(input_image.convert("RGB"))
            out["img_known_patches"] = torch.tensor([int(v) for v in known], dtype=torch.bool)
        return out

    def _slot(self) -> tuple[int, int]:
        info = torch.utils.data.get_worker_info()
        worker_id, num_workers = (info.id, info.num_workers) if info is not None else (0, 1)
        return self.rank * num_workers + worker_id, self.world_size * num_workers

    def __iter__(self):
        import webdataset as wds

        slot, num_slots = self._slot()
        shuffled = list(self.shards)
        random.Random(self.stream_seed).shuffle(shuffled)
        own = balanced_slot_assignment(shuffled, [self.weight[u] for u in shuffled], num_slots)[slot]
        if not own:
            raise RuntimeError(f"{len(self.shards)} shards < {num_slots} (rank, worker) slots; use fewer workers")
        cycle = 0
        while True:
            order = list(own)
            random.Random(self.stream_seed + 104_729 * (slot + 1) + cycle * 1_000_003).shuffle(order)
            pipeline = (
                wds.WebDataset(order, handler=wds.reraise_exception, empty_check=False, shardshuffle=False,
                               nodesplitter=_identity, workersplitter=_identity, repeat=False,
                               seed=int(self.cfg.wds_shard_seed) + self.rank + cycle * 1_000_003)
                .shuffle(self.cfg.shuffle_buffer,
                         rng=random.Random(self.stream_seed + self.rank + cycle * 1_000_003))
                .decode("pil", handler=wds.reraise_exception)
                .map(self.process, handler=wds.reraise_exception)
            )
            yielded = False
            for sample in pipeline:
                yielded = True
                yield sample
            if not yielded:
                raise RuntimeError(f"no samples in shards {own[:3]}")
            cycle += 1


def collate(batch: list[dict]) -> dict:
    out = {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "answer_tok_start": torch.tensor([int(x["answer_tok_start"]) for x in batch], dtype=torch.long),
    }
    if any("qa_window" in x for x in batch):
        out["qa_window"] = torch.tensor([int(x.get("qa_window", -1)) for x in batch], dtype=torch.long)
    if any("img_known_patches" in x for x in batch):
        n = next(int(x["img_known_patches"].numel()) for x in batch if "img_known_patches" in x)
        out["img_known_patches"] = torch.stack(
            [x.get("img_known_patches", torch.zeros(n, dtype=torch.bool)) for x in batch])
        out["img_input"] = torch.stack([x.get("img_input", x["pixel_values"]) for x in batch])
    return out


def make_loader(cfg, rank: int, world_size: int, tokenizer, resume_key: int = 0) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(cfg.wds_shard_seed) * 100_003 + rank)
    return DataLoader(
        ShardStream(cfg, rank, world_size, tokenizer, resume_key),
        batch_size=cfg.micro_batch_size, num_workers=cfg.num_workers, collate_fn=collate,
        pin_memory=False, persistent_workers=cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch if cfg.num_workers > 0 else None,
        generator=generator, drop_last=True,
    )
