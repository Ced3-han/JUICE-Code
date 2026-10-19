# JUICE: Unified Multimodal Modeling in Pretrained Representation Spaces

JUICE (Joint Unified multimodal modeling In Continuous Encoder spaces) is one flow-matching
Transformer (MM-RiT) that generates images and text together.
It does not use pixels or token ids. It works in the representation spaces of frozen
pretrained encoders:

- Images: the patch states of SigLIP2 or WebSSL-DINO (256 x 1152 or 256 x 1024).
- Text: the token states of Flan-T5-Small (128 x 512).

One flow-matching objective trains text-to-image generation, question answering
and joint image-and-text generation. A frozen decoder changes image states into
pixels. A small parallel text head (the parallel decoder of the paper) changes text states into words.

- Paper: <ARXIV_LINK> (ICLR 2027)
- Project page: https://github.com/Ced3-han/JUICE-Website
- Code: https://github.com/Ced3-han/JUICE-Code
- Models: https://huggingface.co/collections/CedLJH/juice

## Contents

| Item | Location |
|---|---|
| Model and samplers | `juice/model.py`, `juice/sampling.py`, `juice/pipeline.py` |
| Inference scripts | `scripts/inference/` |
| Benchmark scripts | `scripts/eval/`, guides in `docs/EVAL_T2I.md`, `docs/EVAL_VQA.md`, `docs/PLANNING.md` |
| Training | `scripts/train/train.py`, `configs/train/`, guide in `docs/TRAIN.md` |
| Data preparation | `scripts/data/`, guide in `docs/DATA.md` |
| Released models | `docs/MODEL_ZOO.md` |

## 1. Install

You need Python 3.10 or later, PyTorch 2.5 or later and an NVIDIA GPU.
We tested the code with PyTorch 2.5.1 (CUDA 12.1) on NVIDIA A100 GPUs with 80 GB.

```bash
git clone https://github.com/Ced3-han/JUICE-Code.git
cd JUICE-Code
pip install -e .            # inference
pip install -e ".[train]"   # training (adds webdataset)
pip install -e ".[eval]"    # question-answering benchmarks (adds pyarrow)
```

The first run downloads the frozen encoders and decoders from Hugging Face.
The code pins each download to the revision that the paper used.
See `docs/INSTALL.md` for offline use.

## 2. Inference

Download a model folder (see `docs/MODEL_ZOO.md`), or give its Hugging Face name.

Text-to-image (four 224 x 224 images):

```bash
python -m scripts.inference.text_to_image --model CedLJH/JUICE-2B-SigLIP2 \
    --prompt "a red bicycle next to a blue door" --out bicycle
```

Question answering:

```bash
python -m scripts.inference.answer --model CedLJH/JUICE-2B-SigLIP2 --short \
    --image photo.jpg --question "What color is the door?"
```

Visual planning (JMaze, VSP): we do not release the fine-tuned planning models.
`docs/PLANNING.md` shows how to fine-tune one from an SFT model and run it.

In Python:

```python
from juice.pipeline import JuicePipeline

pipe = JuicePipeline.from_pretrained("CedLJH/JUICE-2B-SigLIP2", device="cuda")
images = pipe.text_to_image("a red bicycle next to a blue door", num_images=4, seed=0)
answers = pipe.answer(["photo.jpg"], ["What color is the door?"])
```

The default settings are the settings of the paper.
They are in the `config.json` file of each model folder.

## 3. Evaluation

| Benchmark | Guide |
|---|---|
| GenEval, DPG-Bench | `docs/EVAL_T2I.md` |
| VQAv2, OK-VQA, VizWiz, POPE, MME, GQA, MMBench, AI2D, MMMU, ScienceQA-IMG, RealWorldQA | `docs/EVAL_VQA.md` |
| JMaze, VSP | `docs/PLANNING.md` |

Each guide gives the expected numbers and the commands that reproduce them.

## 4. Training

Training has three stages: pretraining (60k steps), SFT (15k steps) and
task fine-tuning (5k steps). One trainer does all stages:

```bash
torchrun --nproc_per_node 8 scripts/train/train.py --config configs/train/pretrain_siglip2.yaml
```

Before you train, prepare the data with `docs/DATA.md`.
We do not distribute the training data.
`docs/TRAIN.md` gives the commands and the settings of all stages.

## 5. License

- Code: MIT License (`LICENSE`).
- Model weights: CC BY-NC 4.0.
- Third-party components keep their own licenses. See `THIRD_PARTY_LICENSES.md`.

## 6. Citation

```bibtex
<BIBTEX>
```

## 7. Acknowledgements

This code uses parts of Scale-RAE (image decoders) and MiniT2I.
The frozen encoders are SigLIP2, WebSSL-DINO and Flan-T5.
We thank the authors of GenEval, DPG-Bench, lmms-eval and VSP for their benchmarks.
