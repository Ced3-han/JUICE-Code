"""Generate the VSP (FrozenLake) training data: puzzle splits and training shards.

Sources:
  * the official VSP maze maps (levels 3 to 8): their layouts are excluded from all
    training data, and their images give the cell sprites;
    https://github.com/UCSB-NLP-Chang/Visual-Spatial-Planning, <repo>/maze/task-main/maps
  * random maps, sizes 3 to 6 (juice.tasks.vsp.random_map), with the BFS route
    (neighbor order L, U, D, R) as the answer;
  * optional: the 6,000 maps of the ThinkMorph Spatial_Navigation set
    (huggingface.co/datasets/ThinkMorph/Spatial_Navigation, train parquet), with
    THEIR answer as the route. Maps equal to a VSP map are excluded.

Splits:
  train     random maps (3:2000, 4:16000, 5:16000, 6:16000) + ThinkMorph, shuffled
  test_id   125 random maps per size, not in train, ThinkMorph or VSP

Outputs under --out:
  <split>.jsonl                     one puzzle per line (juice.tasks.vsp.make_record)
  shards/<split>/vsp_<split>_*.tar  with --shards: WebDataset samples, 2,000 per tar:
      <key>.png        solved image (red route stroke) = image target
      <key>.input.png  input image (no route)          = source of the known patches
      <key>.txt        "Question: <q> Answer: <a>"
      <key>.json       answer_tok_start, answer_tok_end, n_content_tokens, img_known_patches
                       (256 ints, 1 = known patch) and the puzzle (grid, moves, ...)

    python -m scripts.data.make_vsp_data --vsp-maps VSP/maze/task-main/maps \
        --thinkmorph train-00000-of-00001.parquet --out data/vsp --shards train
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from juice.tasks import vsp

SHARD_KEYS = ("id", "source", "n", "grid", "start", "goal", "moves", "layout_hash", "shortest_len",
              "question", "answer", "tm_pid")


def thinkmorph_maps(parquet: str, sprites: dict, excluded: set) -> list[tuple]:
    """(grid, moves, pid) of each ThinkMorph map whose layout is not in `excluded`."""
    import pandas as pd

    df = pd.read_parquet(parquet)
    out = []
    for i in range(len(df)):
        grid = vsp.thinkmorph_grid(df.problem_image_0[i]["bytes"], sprites)
        if vsp.layout_hash(grid) not in excluded:
            out.append((grid, [m.strip() for m in str(df.answer[i]).split(",")], str(df.pid[i])))
    print(f"ThinkMorph: {len(df)} maps, {len(df) - len(out)} excluded (same layout as a VSP map)", flush=True)
    return out


def random_maps(seed: int, quotas: list[tuple[str, dict]], used: set, max_attempts: int) -> dict[str, list]:
    """Solvable random maps with new layouts; one random stream for all splits, in order."""
    rng = random.Random(seed)
    out: dict[str, list] = {}
    attempts = 0
    for split, quota in quotas:
        out[split] = []
        for n, want in quota.items():
            got = 0
            while got < want:
                attempts += 1
                if attempts > max_attempts:
                    raise RuntimeError(f"attempt cap hit at {split} n={n} ({got}/{want})")
                grid = vsp.random_map(n, rng)
                route = vsp.bfs_route(grid) if vsp.layout_hash(grid) not in used else None
                if route is None:
                    continue
                used.add(vsp.layout_hash(grid))
                out[split].append((grid, route[1]))
                got += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vsp-maps", required=True, help="<VSP repo>/maze/task-main/maps")
    ap.add_argument("--thinkmorph", default="", help="ThinkMorph Spatial_Navigation parquet ('' = skip)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-per-size", default="3:2000,4:16000,5:16000,6:16000")
    ap.add_argument("--test-id-per-size", type=int, default=125)
    ap.add_argument("--max-attempts", type=int, default=2_000_000)
    ap.add_argument("--shards", default="", help="comma-separated splits to pack into tar shards (e.g. train)")
    ap.add_argument("--samples-per-shard", type=int, default=2_000)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from juice.representations import TEXT_CANVAS, TEXT_ENCODER_ID, TEXT_ENCODER_REVISION

    tok = AutoTokenizer.from_pretrained(TEXT_ENCODER_ID, revision=TEXT_ENCODER_REVISION, model_max_length=TEXT_CANVAS)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sprites = vsp.load_sprites(args.vsp_maps)
    vsp_layouts = {vsp.layout_hash(r["grid"]) for r in vsp.official_records(args.vsp_maps, tok)}
    tm = thinkmorph_maps(args.thinkmorph, sprites, vsp_layouts) if args.thinkmorph else []

    per_size = {int(k): int(v) for k, v in (kv.split(":") for kv in args.train_per_size.split(","))}
    used = vsp_layouts | {vsp.layout_hash(g) for g, _m, _p in tm}
    gen = random_maps(args.seed, [("test_id", {n: args.test_id_per_size for n in per_size}), ("train", per_size)],
                      used, args.max_attempts)

    items = [("gen", g, m, None) for g, m in gen["train"]] + [("thinkmorph", g, m, p) for g, m, p in tm]
    random.Random(args.seed + 1).shuffle(items)
    splits: dict[str, list[dict]] = {"train": [], "test_id": []}
    for idx, (source, grid, moves, pid) in enumerate(items):
        rec, why = vsp.make_record(tok, "train", idx, source, grid, moves, {"tm_pid": pid} if pid else None)
        if rec is not None:
            splits["train"].append(rec)
    for idx, (grid, moves) in enumerate(gen["test_id"]):
        rec, why = vsp.make_record(tok, "test_id", idx, "gen", grid, moves)
        if rec is not None:
            splits["test_id"].append(rec)
    for split, records in splits.items():
        with (args.out / f"{split}.jsonl").open("w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"{split}: {len(records)} puzzles -> {args.out / f'{split}.jsonl'} ({time.time() - t0:.0f}s)", flush=True)

    from scripts.data.shards import ShardWriter, png_bytes

    tiles = vsp.cell_tiles(sprites)
    for split in [s for s in args.shards.split(",") if s]:
        writer = ShardWriter(args.out / "shards" / split, f"vsp_{split}", args.samples_per_shard)
        for rec in splits[split]:
            meta = {k: rec[k] for k in SHARD_KEYS if k in rec}
            meta.update({k: rec[k] for k in ("answer_tok_start", "answer_tok_end", "n_content_tokens", "img_known_patches")})
            writer.write(rec["key"], {
                "png": png_bytes(vsp.render(rec["grid"], tiles, [tuple(c) for c in rec["route_cells"]])),
                "input.png": png_bytes(vsp.render(rec["grid"], tiles)),
                "txt": (rec["caption"].strip() + "\n").encode("utf-8"),
                "json": json.dumps(meta).encode("utf-8"),
            })
        paths = writer.close()
        (args.out / "shards" / f"{split}.txt").write_text("".join(f"{p.resolve()}\n" for p in paths))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
