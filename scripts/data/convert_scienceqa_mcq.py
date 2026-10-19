"""ScienceQA-IMG train -> a Cambrian-style jsonl and images, for pack_cambrian.py.

Each row becomes one multiple-choice pair in the LLaVA-1.5 form of the Cambrian coco rows:
    human: "<image>\\n{hint}\\n{question}\\nA. ...\\nB. ...\\nAnswer with the option's letter from the given choices directly."
    gpt:   "{letter}"
Rows without an image, without choices, or with more than 9 choices are skipped.
Only the train split is used. The test split is an evaluation benchmark.

Outputs: <root>/jsons/scienceqa_train_mcq.jsonl and <root>/images/scienceqa_mcq/<row>.png.

    python -m scripts.data.convert_scienceqa_mcq \\
        --parquet data/src/ScienceQA/ScienceQA-IMG/train-00000-of-00001.parquet --root data/src/scienceqa_mcq
"""

from __future__ import annotations

import argparse
import json
import os


def human_turn(question: str, hint: str, choices: list[str]) -> str:
    body = (hint + "\n" + question) if hint else question
    opts = "\n".join(f"{chr(65 + k)}. {c}" for k, c in enumerate(choices))
    return f"<image>\n{body}\n{opts}\nAnswer with the option's letter from the given choices directly."


def main(argv=None) -> None:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--root", required=True)
    a = ap.parse_args(argv)
    df = pd.read_parquet(a.parquet)
    os.makedirs(f"{a.root}/jsons", exist_ok=True)
    os.makedirs(f"{a.root}/images/scienceqa_mcq", exist_ok=True)
    n = skipped = 0
    with open(f"{a.root}/jsons/scienceqa_train_mcq.jsonl", "w") as f:
        for i, r in df.iterrows():
            img = r.get("image")
            choices = list(r["choices"])
            ans = int(r["answer"])
            if img is None or not choices or not (0 <= ans < len(choices)) or len(choices) > 9:
                skipped += 1
                continue
            p = f"scienceqa_mcq/{i}.png"
            if not os.path.exists(f"{a.root}/images/{p}"):
                with open(f"{a.root}/images/{p}", "wb") as g:
                    g.write(img["bytes"] if isinstance(img, dict) else img)
            human = human_turn(str(r["question"]).strip(), str(r.get("hint") or "").strip(), choices)
            f.write(json.dumps({"id": f"sqa_train_{i}", "image": p,
                                "conversations": [{"from": "human", "value": human}, {"from": "gpt", "value": chr(65 + ans)}],
                                "source": "scienceqa_img_train"}) + "\n")
            n += 1
    print(f"scienceqa train mcq: {n} rows written, {skipped} skipped -> {a.root} (paper: 6,218 rows)")


if __name__ == "__main__":
    main()
