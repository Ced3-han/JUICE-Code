# JUICE: Unified Multimodal Modeling in Pretrained Representation Spaces

**[Paper](<ARXIV_LINK>) | [Project Page](https://github.com/Ced3-han/JUICE-Website) | [Models](https://huggingface.co/collections/CedLJH/juice)**

JUICE (Joint Unified multimodal modeling In Continuous Encoder spaces) is one flow-matching
Transformer (MM-RiT, 2.1 B parameters) that generates images and text together.
It does not use pixels or token ids. It works in the representation spaces of frozen
pretrained encoders:

- Images: the patch states of SigLIP2 or WebSSL-DINO (256 x 1152 or 256 x 1024).
- Text: the token states of Flan-T5-Small (128 x 512).

One flow-matching objective trains text-to-image generation, question answering
and joint image-and-text generation. A frozen decoder changes image states into
pixels. A small parallel text head (the parallel decoder of the paper) changes text states into words.

This repository holds the code for inference, evaluation, training (all stages) and data preparation.

## Contents

1. [Environment](#1-environment)
2. [Models](#2-models)
3. [Inference](#3-inference)
4. [Training](#4-training)
5. [Evaluation](#5-evaluation)
6. [Guides and repository structure](#6-guides-and-repository-structure)
7. [License](#7-license)
8. [Citation](#8-citation)
9. [Acknowledgements](#9-acknowledgements)

## 1. Environment

### 1.1 Requirements

| Item | Requirement | We used |
|---|---|---|
| System | Linux, NVIDIA GPU with CUDA | NVIDIA A100 and H100 GPUs with 80 GB |
| Python | 3.10 or later | 3.11 |
| PyTorch | 2.5 or later | 2.5.1 |

### 1.2 Make the environment

```bash
conda create -n juice python=3.11 -y
conda activate juice

# PyTorch. This command installs the PyPI build for CUDA 12.4.
# For a different CUDA version, get the command from https://pytorch.org.
pip install torch==2.5.1 torchvision==0.20.1

# JUICE and all optional parts
git clone https://github.com/Ced3-han/JUICE-Code.git
cd JUICE-Code
pip install -e ".[train,eval]"
```

You can also use a Python virtual environment (`python -m venv`) instead of conda.
If you need only some parts, install only the extras that you need:

| Command | Adds | You need it for |
|---|---|---|
| `pip install -e .` | transformers, huggingface_hub, safetensors, Pillow | inference |
| `pip install -e ".[train]"` | webdataset | training |
| `pip install -e ".[eval]"` | pyarrow | question-answering benchmarks |

`pyproject.toml` keeps transformers at a version earlier than 5.14,
because transformers 5.19 does not operate with PyTorch 2.5.
Do not upgrade huggingface_hub to 2.x: transformers 5.13 needs a version earlier than 2.0.

### 1.3 Check the installation

Run the unit tests (CPU, approximately 20 seconds):

```bash
python -m unittest discover -s tests -t .
```

### 1.4 Downloads

The first run downloads the frozen encoders and decoders from Hugging Face
(6.2 GB for a SigLIP2 model, 3.1 GB for a WebSSL-DINO model) and the JUICE model (8.6 GB).
The code pins each download to the revision that the paper used.
To keep the downloads in a different folder, set `HF_HOME` before the first run:

```bash
export HF_HOME=/path/to/huggingface
```

`docs/INSTALL.md` gives the list of the downloads, the use without a network connection
and the package versions that we tested.

## 2. Models

All models are in the [Hugging Face collection](https://huggingface.co/collections/CedLJH/juice).

| Model | Image encoder | Training | Use |
|---|---|---|---|
| [`CedLJH/JUICE-2B-SigLIP2`](https://huggingface.co/CedLJH/JUICE-2B-SigLIP2) | SigLIP2 | pretraining + SFT | text-to-image, question answering (recommended) |
| [`CedLJH/JUICE-2B-DINO`](https://huggingface.co/CedLJH/JUICE-2B-DINO) | WebSSL-DINO | pretraining + SFT | text-to-image, question answering |
| [`CedLJH/JUICE-2B-SigLIP2-Base`](https://huggingface.co/CedLJH/JUICE-2B-SigLIP2-Base) | SigLIP2 | pretraining | text-to-image, start of a fine-tune |
| [`CedLJH/JUICE-2B-DINO-Base`](https://huggingface.co/CedLJH/JUICE-2B-DINO-Base) | WebSSL-DINO | pretraining | text-to-image, start of a fine-tune |

Each model folder holds the weights that we evaluated (the fast EMA weights),
`config.json` with the inference settings of the paper, and the text head (`text_head/`).
`docs/MODEL_ZOO.md` gives the details and the results.

The scripts download a model when you give its Hugging Face name.
To download a model folder before you use it:

```bash
hf download CedLJH/JUICE-2B-SigLIP2 --local-dir JUICE-2B-SigLIP2
```

## 3. Inference

In all commands, `--model` is a Hugging Face name or a local model folder.
The default settings are the settings of the paper (`config.json` of the model folder).

### 3.1 Text to image

```bash
python -m scripts.inference.text_to_image --model CedLJH/JUICE-2B-SigLIP2 \
    --prompt "a red bicycle next to a blue door" --num-images 4 --out bicycle
```

The command writes `bicycle_0.png` to `bicycle_3.png` (224 x 224).

| Option | Default | Meaning |
|---|---|---|
| `--num-images` | 4 | number of images. The script makes the images in batches of 4. |
| `--seed` | 0 | random seed. Batch k uses the seed (seed + 4k). |
| `--steps` | 25 | number of sampling steps |
| `--cfg-scale` | 5.0 | classifier-free guidance scale |

### 3.2 Question answering

```bash
python -m scripts.inference.answer --model CedLJH/JUICE-2B-SigLIP2 --short \
    --image photo.jpg --question "What color is the door?"
```

The command writes one line for each question: the image, the question and the answer.

- `--short` adds the short-answer instruction of the benchmarks
  ("Answer the question using a single word or phrase.").
- To ask more questions, give `--image` and `--question` again, in the same order (one question for each image).
- For a multiple-choice question, put the options in the question. `docs/EVAL_VQA.md` shows the format.

The default settings are 16 steps, guidance 3 and an answer window of 12 slots.

### 3.3 Python

```python
from juice.pipeline import SHORT_ANSWER, JuicePipeline

pipe = JuicePipeline.from_pretrained("CedLJH/JUICE-2B-SigLIP2", device="cuda")

# Text to image: a list of PIL images (224 x 224)
images = pipe.text_to_image("a red bicycle next to a blue door", num_images=4, seed=0)
images[0].save("bicycle.png")

# Question answering: one question for each image (a file path or a PIL image)
answers = pipe.answer(["photo.jpg"], [f"What color is the door?\n{SHORT_ANSWER}"])
print(answers[0])
```

### 3.4 Visual planning (JMaze, VSP)

We do not release the planning models.
`docs/PLANNING.md` tells you how to make the data, fine-tune an SFT model, train the task head and export
a planning model. Then this command solves one puzzle and writes the solved image:

```bash
python -m scripts.inference.plan --model my-jmaze-siglip2 --puzzles data/jmaze/test_id.jsonl \
    --index 0 --out maze0.png
```

## 4. Training

JUICE has three training stages. One trainer (`scripts/train/train.py`) does all stages.
One YAML file in `configs/train/` holds the settings of one run of the paper.

| Stage | Configs | Starts from | Steps | Global batch | Paper GPUs |
|---|---|---|---|---|---|
| 1. Pretraining | `pretrain_siglip2.yaml`, `pretrain_dino.yaml` | random initialization | 60,000 | 4,096 | 8 |
| 2. SFT | `sft_siglip2.yaml`, `sft_dino.yaml` | stage 1, step 60,000 | 15,000 | 1,024 | 8 |
| 3. Task fine-tune | `jmaze_siglip2.yaml`, `jmaze_dino.yaml`, `vsp_siglip2.yaml` | stage 2, step 15,000 | 5,000 | 1,024 | 8 (VSP: 4) |

Each GPU must have 80 GB of memory.
The text head (question answering) and the task head (planning) have their own trainers. Each uses one GPU.

### 4.1 Prepare the data

We do not distribute the training data. You download the public datasets and run our scripts.

1. Download the datasets: `docs/DATA.md`, section 2 (approximately 6.3 TB).
2. Build the WebDataset packs and the manifests (approximately 2.1 TB more):

   ```bash
   STEP=all JOBS=16 bash scripts/data/prepare_data.sh
   ```

The configs read the manifests in `data/manifests/`.
At the end, the script compares the row counts of the packs with the counts of the paper runs.

### 4.2 Stage 1: pretraining

```bash
torchrun --nproc_per_node 8 scripts/train/train.py --config configs/train/pretrain_siglip2.yaml
```

For WebSSL-DINO, use `configs/train/pretrain_dino.yaml`.
The trainer writes to `outputs/train/<run_name>/`:
`config.json` (the settings), `log.jsonl` (loss and learning rate, every 50 steps)
and `checkpoints/checkpoint_<step>.pt`.

### 4.3 Stage 2: SFT

```bash
torchrun --nproc_per_node 8 scripts/train/train.py --config configs/train/sft_siglip2.yaml
```

The config starts from the raw weights of your stage-1 run at step 60,000
(`outputs/train/pretrain_siglip2/checkpoints/checkpoint_060000.pt`).
To start from a released Base model, give its folder:

```bash
torchrun --nproc_per_node 8 scripts/train/train.py --config configs/train/sft_siglip2.yaml \
    --set init_from=/path/to/JUICE-2B-SigLIP2-Base
```

The released folders hold EMA weights, not raw weights. Thus the result can be slightly different from the paper.

### 4.4 Stage 3: task fine-tune

`docs/PLANNING.md`, section 2, gives the data, the fine-tune, the task head and the export of a planning model.

### 4.5 Text head

The text head reads the answer text from the text states. It trains alone, on the texts of the
pretraining data, and it does not use the JUICE model.
It has two stages of 8,000 steps (approximately 1.3 hours each on one A100 GPU):

```bash
python scripts/train/train_text_head.py --config configs/train/text_head.yaml
```

The released head is `outputs/train/text_head/stage_2`.
One head serves the SigLIP2 and the WebSSL-DINO models.

### 4.6 Make a model folder

The inference scripts read a model folder. To make one from a checkpoint, export one weight set
(the paper evaluated `ema_fast`) and add the text head:

```bash
python -m juice.export --checkpoint outputs/train/sft_siglip2/checkpoints/checkpoint_015000.pt \
    --weights ema_fast --text-head outputs/train/text_head/stage_2 --out my-sft-siglip2

python -m scripts.inference.text_to_image --model my-sft-siglip2 \
    --prompt "a red bicycle next to a blue door" --out test
```

### 4.7 Other options

- **Number of GPUs.** The global batch is `micro_batch_size` (8) x `grad_accum_steps` x number of GPUs.
  It must be equal to `batch_size`. For a different number of GPUs, change `grad_accum_steps`.
  For example, SFT on 4 GPUs:

  ```bash
  torchrun --nproc_per_node 4 scripts/train/train.py --config configs/train/sft_siglip2.yaml \
      --set grad_accum_steps=32
  ```

- **Other settings.** `--set key=value` replaces a config key (the values are YAML),
  for example `--set output_dir=/scratch/juice num_workers=4`.
- **Continue a run.** `--resume` loads the weights, the EMA copies, the optimizer state and the step:

  ```bash
  torchrun --nproc_per_node 8 scripts/train/train.py --config configs/train/pretrain_siglip2.yaml \
      --resume outputs/train/pretrain_siglip2/checkpoints/checkpoint_030000.pt
  ```

`docs/TRAIN.md` gives all the settings, the checkpoint contents and our reproducibility checks.

## 5. Evaluation

| Benchmark | Guide |
|---|---|
| GenEval, DPG-Bench | `docs/EVAL_T2I.md` |
| VQAv2, OK-VQA, VizWiz, POPE, MME, GQA, MMBench, AI2D, MMMU, ScienceQA-IMG, RealWorldQA | `docs/EVAL_VQA.md` |
| JMaze, VSP | `docs/PLANNING.md` |

Each guide gives the data downloads, the commands and the expected numbers.

## 6. Guides and repository structure

| Guide | Content |
|---|---|
| [`docs/INSTALL.md`](docs/INSTALL.md) | environment, downloads, use without a network connection, tested versions |
| [`docs/MODEL_ZOO.md`](docs/MODEL_ZOO.md) | released models, folder contents, inference settings, results |
| [`docs/DATA.md`](docs/DATA.md) | training data: downloads, packs, manifests, checks |
| [`docs/TRAIN.md`](docs/TRAIN.md) | all training stages, text head, task head, checkpoints |
| [`docs/EVAL_T2I.md`](docs/EVAL_T2I.md) | GenEval and DPG-Bench |
| [`docs/EVAL_VQA.md`](docs/EVAL_VQA.md) | question-answering benchmarks |
| [`docs/PLANNING.md`](docs/PLANNING.md) | JMaze and VSP: data, fine-tune, evaluation |

```
JUICE-Code/
├── juice/                  # the Python package
│   ├── model.py            # MM-RiT
│   ├── sampling.py         # samplers: text to image, question answering, joint generation
│   ├── pipeline.py         # JuicePipeline (inference API)
│   ├── representations.py  # frozen encoders: SigLIP2, WebSSL-DINO, Flan-T5-Small
│   ├── rae_decoder.py      # frozen image decoders
│   ├── heads.py            # text head and task head
│   ├── export.py           # checkpoint to model folder
│   ├── train/              # trainer, loss, data loader, head trainers
│   ├── eval/               # benchmark scoring
│   └── tasks/              # JMaze and VSP
├── scripts/
│   ├── inference/          # text_to_image.py, answer.py, plan.py
│   ├── train/              # train.py, train_text_head.py, train_task_head.py
│   ├── eval/               # benchmark generation and scoring
│   ├── data/               # training data preparation
│   └── export_planning.py  # checkpoint to planning model folder
├── configs/train/          # one YAML file for each run of the paper
├── docs/                   # guides
└── tests/                  # unit tests (CPU)
```

## 7. License

- Code: MIT License (`LICENSE`).
- Model weights: CC BY-NC 4.0.
- Third-party components keep their own licenses. See `THIRD_PARTY_LICENSES.md`.

## 8. Citation

```bibtex
<BIBTEX>
```

## 9. Acknowledgements

This code uses parts of [Scale-RAE](https://github.com/ZitengWangNYU/Scale-RAE) (image decoders)
and [MiniT2I](https://github.com/Hope7Happiness/minit2i-torch).
The frozen encoders are SigLIP2, WebSSL-DINO and Flan-T5.
We thank the authors of GenEval, DPG-Bench, lmms-eval and VSP for their benchmarks.
