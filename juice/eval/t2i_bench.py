"""Text-to-image benchmarks (GenEval, DPG-Bench): prompts, seeds and the output layout.

The layout is the layout that the official scorers read:

  GenEval    <out>/<index:05d>/metadata.jsonl       (line <index> of evaluation_metadata.jsonl)
             <out>/<index:05d>/samples/<k:05d>.png  (k = 0..3)
  DPG-Bench  <out>/<item_id>.png                    (2x2 grid of the 4 samples)

Seeds. The 4 samples of prompt <index> come from one batch with the generator seed
seed + 1000 * index. The seed depends only on the prompt index, so the images do not
depend on the number of shards. Shard s of n gets the prompts s, s + n, s + 2n, ...

No model code here: the generation script calls juice.pipeline.JuicePipeline.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

SAMPLES_PER_PROMPT = 4  # the paper: 4 images per prompt (DPG-Bench: one 2x2 grid)
SEED_STRIDE = 1000      # seed of prompt <index> = seed + SEED_STRIDE * index
GENEVAL_PROMPTS = 553
DPG_PROMPTS = 1065


@dataclass(frozen=True)
class PromptItem:
    index: int
    item_id: str
    prompt: str
    metadata: dict | None = None  # GenEval: the row of evaluation_metadata.jsonl


def load_geneval_prompts(path: str | os.PathLike) -> list[PromptItem]:
    """evaluation_metadata.jsonl -> one item per line; item id = 5-digit line number."""
    items = []
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            items.append(PromptItem(i, f"{i:05d}", row["prompt"], row))
    return items


def load_dpg_prompts(path: str | os.PathLike) -> list[PromptItem]:
    """dpg_bench.csv (one row per question) -> one item per item_id, in the order of first use."""
    prompts: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            prompts.setdefault(row["item_id"], row["text"])
    return [PromptItem(i, item_id, text) for i, (item_id, text) in enumerate(prompts.items())]


def prompt_seed(seed: int, index: int) -> int:
    """Generator seed of the 4-image batch of prompt <index>."""
    return seed + SEED_STRIDE * index


def shard_items(items: list, num_shards: int, shard: int) -> list:
    if not 0 <= shard < num_shards:
        raise ValueError("shard must be in [0, num_shards)")
    return items[shard::num_shards]


def geneval_sample_path(out: str | os.PathLike, item: PromptItem, k: int) -> Path:
    return Path(out) / item.item_id / "samples" / f"{k:05d}.png"


def dpg_grid_path(out: str | os.PathLike, item: PromptItem) -> Path:
    return Path(out) / f"{item.item_id}.png"


def make_grid(images: list):
    """4 equal square images -> one 2x2 grid: [0 1 / 2 3]."""
    from PIL import Image

    if len(images) != SAMPLES_PER_PROMPT:
        raise ValueError(f"a DPG-Bench grid needs {SAMPLES_PER_PROMPT} images")
    size = images[0].size[0]
    canvas = Image.new("RGB", (2 * size, 2 * size))
    for image, xy in zip(images, [(0, 0), (size, 0), (0, size), (size, size)]):
        canvas.paste(image, xy)
    return canvas


def save_png(image, path: Path) -> None:
    """Write through a temporary file, so that a stopped job leaves no partial PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    image.save(tmp, format="PNG")
    os.replace(tmp, path)


def write_geneval(out: str | os.PathLike, item: PromptItem, images: list) -> None:
    folder = Path(out) / item.item_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metadata.jsonl").write_text(json.dumps(item.metadata, ensure_ascii=False) + "\n", encoding="utf-8")
    for k, image in enumerate(images):
        save_png(image, geneval_sample_path(out, item, k))


def write_dpg(out: str | os.PathLike, item: PromptItem, images: list) -> None:
    save_png(make_grid(images), dpg_grid_path(out, item))


def output_complete(benchmark: str, out: str | os.PathLike, item: PromptItem) -> bool:
    def nonempty(p: Path) -> bool:
        return p.is_file() and p.stat().st_size > 0

    if benchmark == "geneval":
        return nonempty(Path(out) / item.item_id / "metadata.jsonl") and all(
            nonempty(geneval_sample_path(out, item, k)) for k in range(SAMPLES_PER_PROMPT))
    return nonempty(dpg_grid_path(out, item))


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
