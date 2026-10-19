"""Repack the BLIP3o-60k tars into 64 shards of equal size (blip3o_ft60k_r64).

Input: the BLIP3o/BLIP3o-60k folder. We use 10 of its 11 tars. We do not use
mscoco_human.tar: its captions are COCO val2017 reference captions (8,347 of 8,347),
1,322 of them are prompts of the FID-30K evaluation. This removes 8,504 rows.

The keys of a source tar get the tar stem as a prefix ("dalle3/00000"), because each
source tar numbers its keys from 0. All keys are shuffled once (random.Random(0)); key
number i goes to shard i % 64. The shards are written source tar by source tar, keys and
member names in sorted order; the member keeps its size, mtime and mode.

    python -m scripts.data.repack_blip3o60k --src data/src/BLIP3o-60k --out data/wds/blip3o_ft60k_r64
"""

from __future__ import annotations

import argparse
import json
import random
import tarfile
from collections import defaultdict
from pathlib import Path

SOURCE_TARS = ["dalle3.tar", "geneval_train.tar", "human_gestures.tar", "journeyDB.tar", "object_1.tar",
               "object_2.tar", "occupation_1.tar", "occupation_2.tar", "text_1.tar", "text_2.tar"]
N_SHARDS, SEED = 64, 0


def member_index(tars: list[Path]) -> dict[str, dict[str, list[str]]]:
    """{tar name: {key: [member names]}} (regular files only)."""
    index = {}
    for t in tars:
        by_key: dict[str, list[str]] = defaultdict(list)
        with tarfile.open(t) as tf:
            for m in tf.getmembers():
                if m.isfile():
                    by_key[m.name.rsplit(".", 1)[0]].append(m.name)
        index[t.name] = dict(by_key)
    return index


def assign_shards(index: dict[str, dict[str, list[str]]], tar_names: list[str]) -> dict[str, int]:
    keys = [f"{Path(t).stem}/{k}" for t in tar_names for k in sorted(index[t])]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate keys after the tar-stem prefix")
    random.Random(SEED).shuffle(keys)
    return {key: i % N_SHARDS for i, key in enumerate(keys)}


def repack(src: Path, out: Path, only_shards: set[int] | None = None) -> list[int]:
    """Write the shards (all, or only the ones in `only_shards`). Returns rows per shard."""
    tars = [src / name for name in SOURCE_TARS]
    missing = [t.name for t in tars if not t.exists()]
    if missing:
        raise SystemExit(f"missing source tars: {missing}")
    index = member_index(tars)
    assignment = assign_shards(index, SOURCE_TARS)
    shards = sorted(only_shards) if only_shards is not None else list(range(N_SHARDS))
    out.mkdir(parents=True, exist_ok=False)
    outs = {i: tarfile.open(out / f".tmp_shard-{i:05d}.tar", "w") for i in shards}
    counts = [0] * N_SHARDS
    for t in tars:
        stem = Path(t.name).stem
        with tarfile.open(t) as tf:
            for key in sorted(index[t.name]):
                shard = assignment[f"{stem}/{key}"]
                counts[shard] += 1
                if shard not in outs:
                    continue
                for name in sorted(index[t.name][key]):
                    m = tf.getmember(name)
                    info = tarfile.TarInfo(name=f"{stem}/{name}")
                    info.size, info.mtime, info.mode = m.size, m.mtime, m.mode
                    outs[shard].addfile(info, tf.extractfile(m))
        print(f"{t.name} done", flush=True)
    for i, o in outs.items():
        o.close()
        (out / f".tmp_shard-{i:05d}.tar").rename(out / f"shard-{i:05d}.tar")
    return counts


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="BLIP3o-60k folder")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    counts = repack(Path(a.src), Path(a.out))
    (Path(a.out) / "pack_summary.json").write_text(json.dumps(
        {"source_tars": SOURCE_TARS, "seed": SEED, "per_shard": counts, "total_packed": sum(counts),
         "tars": N_SHARDS}, indent=1))
    print(f"{sum(counts)} rows in {N_SHARDS} shards (paper: 50,355 rows)")


if __name__ == "__main__":
    main()
