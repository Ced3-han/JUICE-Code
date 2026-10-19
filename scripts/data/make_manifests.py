"""Write the training manifests data/manifests/<name>.txt that configs/train/*.yaml read.

A manifest lists one shard per line, with an absolute path. The line order is the order of
the paper runs (sorted paths; replicas: copy by copy). The trainer shuffles the shards
with a fixed seed, so the order changes the data order.

Three kinds of manifest:
  * source tars as downloaded: blip3o_long_caption_train, blip3o_short_caption_train
    (all tars of the dataset in name order without the last 16, which are validation
    tars) and journeydb_train (JourneyDB_000-410; the 7 tars of clean_journeydb.py
    replace their source tars);
  * packs under --wds (see PACKS);
  * replicas (_xN): N hard links of each shard of a base manifest, in
    <wds>/links/<name>/c<NN>/. The trainer refuses the same file twice, so a replica must
    be a new path. Hard links use no disk space; they must be on the file system of the
    shards.

The script writes each manifest whose shards exist and prints a table: shards, rows of the
packs (from pack_summary.json), rows of the paper packs. Run it again after more packs are
done. --strict: exit with an error when a manifest is missing or a row count is different.

    python -m scripts.data.make_manifests --src data/src --wds data/wds --out data/manifests \\
        --jmaze data/jmaze --vsp data/vsp --strict
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Downloaded datasets (folder under --src, number of training tars, number of tars in the folder).
SOURCE_POOLS = {
    "blip3o_long_caption_train": ("BLIP3o-Pretrain-Long-Caption", 2875, 2891),
    "blip3o_short_caption_train": ("BLIP3o-Pretrain-Short-Caption", 1815, 1831),
    "journeydb_train": ("BLIP3o-Pretrain-JourneyDB", 411, 419),
}
JOURNEYDB_CLEAN_DIR = "journeydb_clean"  # under --wds

# Packs: manifest name -> (shard pattern under --wds, rows of the paper pack).
PACKS = {
    # Pretraining: Cambrian-S8 (v1 prompts) and COCO
    "cambrian_gqa_x4": ("cambrian-s8/gqa/s*/*.tar", 943_833),
    "cambrian_coco_x4": ("cambrian-s8/coco/s*/*.tar", 634_889),
    "cambrian_lnqa_x4": ("cambrian-s8/lnqa/s*/*.tar", 275_207),
    "cambrian_vizwiz_x4": ("cambrian-s8/vizwiz/s*/*.tar", 20_128),
    "cambrian_oodvqa_x4": ("cambrian-s8/oodvqa/s*/*.tar", 16_630),
    "cambrian_vg": ("cambrian-s8/vg/s*/*.tar", 336_224),
    "cambrian_dvqa": ("cambrian-s8/dvqa/s*/*.tar", 774_888),
    "cambrian_ocr_vqa": ("cambrian-s8/ocr_vqa/s*/*.tar", 317_675),
    "cambrian_clevr": ("cambrian-s8/clevr/s*/*.tar", 351_828),
    "cambrian_tallyqa": ("cambrian-s8/tallyqa/s*/*.tar", 250_279),
    "cambrian_hfdata": ("cambrian-s8/hfdata/s*/*.tar", 313_014),
    "cambrian_Q-Instruct-DB": ("cambrian-s8/Q-Instruct-DB/s*/*.tar", 284_400),
    "cambrian_docvqa": ("cambrian-s8/docvqa/s*/*.tar", 35_240),
    "cambrian_pathvqa": ("cambrian-s8/pathvqa/s*/*.tar", 29_940),
    "cambrian_chartqa": ("cambrian-s8/chartqa/s*/*.tar", 28_272),
    "cambrian_ai2d": ("cambrian-s8/ai2d/s*/*.tar", 15_075),
    "cambrian_scienceqa": ("cambrian-s8/scienceqa/s*/*.tar", 5_867),
    "cambrian_coco_valrows": ("cambrian-s8/coco_valrows/s*/*.tar", 32_010),
    "coco_train2014_caption": ("coco_train2014_caption/*.tar", 414_113),
    # SFT: bases of the replicas and the regenerated packs
    "blip3o_ft60k_r64": ("blip3o_ft60k_r64/shard-*.tar", 50_355),
    "dalle3_t2i": ("dalle3_t2i/shard-*.tar", 19_024),
    "sharegpt4o_t2i": ("sharegpt4o_t2i/*.tar", 45_717),
    "vqav2_train_v2_s13": ("vqav2_train_v2_s13/*.tar", 430_000),
    "cambrian_v2_gqa": ("cambrian-v2/gqa/s*/*.tar", 943_833),
    "cambrian_v2_coco": ("cambrian-v2/coco/s*/*.tar", 634_889),
    "cambrian_v2_ocr_vqa": ("cambrian-v2/ocr_vqa/s*/*.tar", 317_675),
    "cambrian_mcq": ("cambrian_mcq/*.tar", 43_357),
    "scienceqa_train_mcq": ("scienceqa_train_mcq/*.tar", 5_313),
    "sft_capc_coco2014": ("sft_cap/capc_coco2014/s*/*.tar", 413_803),
    "sft_capc_blip3o_short": ("sft_cap/capc_blip3o_short/s*/*.tar", 1_250_000),
    "sft_capd_blip3o_long": ("sft_cap/capd_blip3o_long/s*/*.tar", 1_500_000),
    "sft_capd_blip3o60k": ("sft_cap/capd_blip3o60k/s*/*.tar", 14_233),
    "sft_capd_dalle3": ("sft_cap/capd_dalle3/s*/*.tar", 14_896),
    "sft_long_all": ("sft_long/*/s*/*.tar", 1_718_221),
}
# Replica manifest -> (base manifest, copies).
REPLICAS = {
    "blip3o_ft60k_r64_x75": ("blip3o_ft60k_r64", 75),
    "dalle3_t2i_x52": ("dalle3_t2i", 52),
    "sharegpt4o_t2i_x52": ("sharegpt4o_t2i", 52),
    "vqav2_train_v2_s13_x7": ("vqav2_train_v2_s13", 7),
    "cambrian_v2_gqa_x2": ("cambrian_v2_gqa", 2),
    "cambrian_v2_coco_x2": ("cambrian_v2_coco", 2),
    "cambrian_v2_ocr_vqa_x2": ("cambrian_v2_ocr_vqa", 2),
    "cambrian_vg_x2": ("cambrian_vg", 2),
    "cambrian_vizwiz_x2": ("cambrian_vizwiz_x4", 2),
    "cambrian_mcq_x10": ("cambrian_mcq", 10),
    "scienceqa_train_mcq_x10": ("scienceqa_train_mcq", 10),
    "sft_capc_coco2014_x2": ("sft_capc_coco2014", 2),
    "sft_capd_blip3o60k_x4": ("sft_capd_blip3o60k", 4),
    "sft_capd_dalle3_x4": ("sft_capd_dalle3", 4),
}
# Planning task shards (docs/PLANNING.md): manifest -> (option, pattern under that folder).
PLANNING = {"jmaze_multi_train": ("jmaze", "shards/train/jmaze_train_*.tar"),
            "vsp_train": ("vsp", "shards/train/vsp_train_*.tar")}


def source_pool_tars(src_root: Path, wds: Path, name: str) -> list[str]:
    folder, n_train, n_all = SOURCE_POOLS[name]
    tars = sorted(p for p in (src_root / folder).glob("*.tar") if not p.name.endswith("_clean.tar"))
    if not tars:
        return []
    if len(tars) != n_all:
        raise SystemExit(f"{src_root / folder}: {len(tars)} tars, the paper revision has {n_all}")
    tars = tars[:n_train]
    if name == "journeydb_train":
        from scripts.data.clean_journeydb import BAD_MEMBERS

        clean = {n: wds / JOURNEYDB_CLEAN_DIR / n.replace(".tar", "_clean.tar") for n in BAD_MEMBERS}
        if not all(p.exists() for p in clean.values()):
            print(f"journeydb_train: run clean_journeydb.py first (tars missing in {wds / JOURNEYDB_CLEAN_DIR})")
            return []
        tars = [clean.get(p.name, p) for p in tars]
    return [str(p.resolve()) for p in tars]


def pack_rows(tars: list[str]) -> int | None:
    """Rows of the packs that hold these shards, from their summary files (None if unknown)."""
    total, dirs = 0, sorted({str(Path(t).parent) for t in tars})
    for d in dirs:
        summaries = sorted(Path(d).glob("summary_s*.json")) or [Path(d) / "pack_summary.json"]
        for s in summaries:
            if not s.exists():
                return None
            j = json.loads(s.read_text())
            total += int(j.get("total_packed", j.get("written", 0)))
    return total


def make_replica(name: str, base_lines: list[str], copies: int, links_root: Path) -> list[str]:
    """Hard links <links_root>/<name>/c<NN>/<shard name>; copy-major order."""
    lines = []
    for c in range(copies):
        d = links_root / name / f"c{c:02d}"
        d.mkdir(parents=True, exist_ok=True)
        for s in base_lines:
            t = d / Path(s).name
            if not t.exists():
                os.link(s, t)
            if os.stat(t).st_ino != os.stat(s).st_ino:
                raise SystemExit(f"{t} is not a hard link of {s}")
            lines.append(str(t.absolute()))
    return lines


def write_manifest(out: Path, name: str, lines: list[str]) -> None:
    if len(set(lines)) != len(lines):
        raise SystemExit(f"{name}: a shard is listed twice")
    (out / f"{name}.txt").write_text("\n".join(lines) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/src", help="folder of the downloaded datasets")
    ap.add_argument("--wds", default="data/wds", help="folder of the packs")
    ap.add_argument("--out", default="data/manifests")
    ap.add_argument("--jmaze", default="data/jmaze", help="--out of make_jmaze_data.py")
    ap.add_argument("--vsp", default="data/vsp", help="--out of make_vsp_data.py")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args(argv)
    src, wds, out = Path(a.src).resolve(), Path(a.wds).resolve(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, list[str]] = {}
    expected: dict[str, int | None] = {}
    for name in SOURCE_POOLS:
        manifests[name], expected[name] = source_pool_tars(src, wds, name), None
    for name, (pattern, rows) in PACKS.items():
        manifests[name], expected[name] = [str(p) for p in sorted(wds.glob(pattern))], rows
    for name, (option, pattern) in PLANNING.items():
        manifests[name], expected[name] = [str(p) for p in sorted(Path(getattr(a, option)).resolve().glob(pattern))], None
    for name, (base, copies) in REPLICAS.items():
        rows = expected[base] * copies
        manifests[name] = make_replica(name, manifests[base], copies, wds / "links") if manifests[base] else []
        expected[name] = rows

    problems = 0
    print(f"{'manifest':28s} {'shards':>7s} {'rows':>10s} {'paper rows':>11s}  status")
    for name, lines in manifests.items():
        if not lines:
            print(f"{name:28s} {'-':>7s} {'-':>10s} {expected[name] or '-':>11}  MISSING")
            problems += 1
            continue
        write_manifest(out, name, lines)
        base, copies = REPLICAS.get(name, (name, 1))
        rows = pack_rows(manifests[base]) if name not in SOURCE_POOLS and name not in PLANNING else None
        rows = rows * copies if rows is not None else None
        ok = rows is None or expected[name] is None or rows == expected[name]
        problems += not ok
        print(f"{name:28s} {len(lines):7d} {rows if rows is not None else '-':>10} "
              f"{expected[name] or '-':>11}  {'OK' if ok else 'ROW COUNT DIFFERS'}")
    print(f"{len(manifests) - problems} of {len(manifests)} manifests written and checked -> {out}")
    if a.strict and problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
