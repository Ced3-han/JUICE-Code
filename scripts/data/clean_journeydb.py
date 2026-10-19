"""Remove the 14 images that do not decode from 7 BLIP3o-Pretrain-JourneyDB tars.

A full decode scan of the JourneyDB training tars found 14 bad images in 7 tars
(12 that PIL cannot identify, 2 truncated). For each of the 7 tars, this script writes
<stem>_clean.tar into --out: every member except the bad image and its .txt. The members
keep their tar headers. The source tars do not change. journeydb_train then lists the
clean tar in place of the source tar (make_manifests.py does this).

    python -m scripts.data.clean_journeydb --src data/src/BLIP3o-Pretrain-JourneyDB --out data/wds/journeydb_clean
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

BAD_MEMBERS = {
    "JourneyDB_000.tar": ["00005316.jpg", "00005431.jpg", "00005442.jpg", "00005479.jpg"],
    "JourneyDB_316.tar": ["03169719.jpg"],
    "JourneyDB_320.tar": ["03200586.jpg", "03201707.jpg"],
    "JourneyDB_321.tar": ["03219377.jpg"],
    "JourneyDB_322.tar": ["03224879.jpg", "03225478.jpg", "03227016.jpg"],
    "JourneyDB_323.tar": ["03231536.jpg", "03238232.jpg"],
    "JourneyDB_332.tar": ["03322151.jpg"],
}


def clean_tar(src: Path, dst: Path, bad_members: list[str]) -> tuple[int, int]:
    """Copy src to dst without the bad keys. Returns (kept, dropped) members."""
    bad_keys = {m.rpartition(".")[0] for m in bad_members}
    kept = dropped = 0
    tmp = dst.with_name(f".tmp_{dst.name}")
    with tarfile.open(src, "r") as s, tarfile.open(tmp, "w") as d:
        for member in s.getmembers():
            if member.name.rpartition(".")[0] in bad_keys:
                dropped += 1
                continue
            d.addfile(member, s.extractfile(member) if member.isfile() else None)
            kept += 1
    if dropped != 2 * len(bad_keys):
        raise SystemExit(f"{src.name}: dropped {dropped} members, expected {2 * len(bad_keys)} (image + txt)")
    tmp.rename(dst)
    return kept, dropped


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="BLIP3o-Pretrain-JourneyDB folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", help="clean only these tar names")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, bad in BAD_MEMBERS.items():
        if a.only and name not in a.only:
            continue
        dst = out / name.replace(".tar", "_clean.tar")
        if dst.exists():
            print(f"skip {dst} (exists)")
            continue
        kept, dropped = clean_tar(Path(a.src) / name, dst, bad)
        print(f"{name}: kept {kept} members, dropped {dropped} -> {dst}", flush=True)


if __name__ == "__main__":
    main()
