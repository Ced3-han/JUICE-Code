# Installation

## 1. Requirements

- Linux and an NVIDIA GPU with CUDA.
- Python 3.10 or later.
- PyTorch 2.5 or later.

We tested the code with these versions on NVIDIA A100 GPUs with 80 GB:

| Package | Version |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.5.1 (CUDA 12.1) |
| torchvision | 0.20.1 |
| transformers | 5.13.0 |
| huggingface_hub | 1.22.0 |
| safetensors | 0.8.0 |
| Pillow | 12.2.0 |
| webdataset | 0.2.111 (training only) |
| pyarrow | 24.0.0 (question-answering benchmarks only) |

## 2. Install the package

Install PyTorch for your CUDA version first (see https://pytorch.org).
Then install JUICE:

```bash
git clone https://github.com/Ced3-han/JUICE-Code.git
cd JUICE-Code
pip install -e .
```

`pyproject.toml` limits transformers to versions earlier than 5.14 and webdataset to
versions earlier than 1.0. Transformers 5.19 does not operate with PyTorch 2.5: it needs
`torch.accelerator`, which PyTorch 2.6 added. We did not test webdataset 1.x.

Optional parts:

| Command | Adds | Use |
|---|---|---|
| `pip install -e ".[train]"` | webdataset | training (`docs/TRAIN.md`) |
| `pip install -e ".[eval]"` | pyarrow | question-answering benchmarks (`docs/EVAL_VQA.md`) |

Run the unit tests (CPU, about 20 seconds):

```bash
python -m unittest discover -s tests -t .
```

## 3. Downloads

The first run downloads the frozen encoders and decoders from Hugging Face.
The code pins each download to the revision that the paper used
(`juice/representations.py`).

| Model | Size | Used by |
|---|---|---|
| `google/siglip2-so400m-patch14-224` | 4.3 GB | SigLIP2 models |
| `nyu-visionx/siglip2_decoder` | 1.6 GB | SigLIP2 models (images only) |
| `facebook/webssl-dino300m-full2b-224` | 1.2 GB | WebSSL-DINO models |
| `nyu-visionx/webssl300m_decoder` | 1.6 GB | WebSSL-DINO models (images only) |
| `google/flan-t5-small` | 0.3 GB | all models |

A JUICE model folder is about 8.6 GB (`docs/MODEL_ZOO.md`).

To keep the downloads in a different folder, set `HF_HOME`:

```bash
export HF_HOME=/path/to/huggingface
```

## 4. Use without a network connection

1. Run each task one time with a network connection. This puts all files in the cache.
2. Then set these variables:

   ```bash
   export HF_HUB_OFFLINE=1
   export TRANSFORMERS_OFFLINE=1
   ```

3. Give the model as a local folder (`--model /path/to/JUICE-2B-SigLIP2`).

## 5. Bit-equal results

On A100 GPUs with the versions in section 1, the released models reproduce the
results of the paper runs bit for bit (images, answer states and answers).
We checked the images against the 100-step paper runs (`--steps 100`); the 25-step results of the
paper come from the generation script of this repository.
Other GPUs, other library versions or other batch sizes can change the last bits of
the calculation. The results are then slightly different, but equally valid.

We also installed the package in two new environments from PyPI on an A100 GPU:

| Environment | PyTorch | transformers | huggingface_hub | safetensors |
|---|---|---|---|---|
| latest permitted versions | 2.5.1 (PyPI build, CUDA 12.4) | 5.13.1 | 1.33.0 | 0.8.0 |
| minimum versions of `pyproject.toml` | 2.5.1 (PyPI build, CUDA 12.4) | 4.46.3 | 0.26.5 | 0.4.5 |

In both environments, the unit tests passed, and the four images of GenEval prompt 0
(100 steps) were equal bit for bit to the images of the paper runs.
