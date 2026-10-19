"""Make jsons/cambrian7m_mmu.jsonl (the input of pack_cambrian.py) from the Cambrian-10M release.

Rule: keep every row of jsons/Cambrian7M.jsonl that has an "image" field, in file order, and
remove the "<image>" tag (with its adjoining newline) from each conversation value. Write
json.dumps(row, ensure_ascii=False). The paper file has 5,354,769 lines. The held-out probe
exclusion uses the sha1 of each raw line, so the file must be equal byte for byte.

    python -m scripts.data.make_cambrian_mmu_jsonl --root data/src/Cambrian-10M
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_LINES = 5_354_769


def strip_image_tag(value: str) -> str:
    return value.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")


def convert_line(line: str) -> str | None:
    d = json.loads(line)
    if not d.get("image"):
        return None
    for c in d["conversations"]:
        c["value"] = strip_image_tag(c["value"])
    return json.dumps(d, ensure_ascii=False) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Cambrian-10M folder")
    args = ap.parse_args(argv)
    src = Path(args.root) / "jsons" / "Cambrian7M.jsonl"
    dst = Path(args.root) / "jsons" / "cambrian7m_mmu.jsonl"
    if dst.exists():
        raise SystemExit(f"{dst} exists")
    n_in = n_out = 0
    with src.open("r", encoding="utf-8") as f, dst.open("w", encoding="utf-8") as g:
        for line in f:
            n_in += 1
            out = convert_line(line)
            if out is not None:
                g.write(out)
                n_out += 1
    print(f"{src.name}: {n_in} rows -> {dst.name}: {n_out} image rows (paper: {EXPECTED_LINES})")
    if n_out != EXPECTED_LINES:
        print("WARNING: the line count is not the paper count; the packs will differ")


if __name__ == "__main__":
    main()
