# Model zoo

We release the pretrained (Base) and the SFT models of the paper, each with the SigLIP2 and the
WebSSL-DINO image encoder. All models are MM-RiT d28 (2.1 B parameters) in float32.
The model weights are under the CC BY-NC 4.0 license.
Each folder holds one weight set: the one that we evaluated (the fast EMA, decay 0.9996).
The four models are in the Hugging Face collection https://huggingface.co/collections/CedLJH/juice.

## 1. Models

| Model | Image space | Training | Weights | Files | Use |
|---|---|---|---|---|---|
| `CedLJH/JUICE-2B-SigLIP2-Base` | SigLIP2 | pretraining 60k | EMA (decay 0.9996) at step 60,000 | 8.41 GB + text head 0.14 GB | base for fine-tuning, text-to-image |
| `CedLJH/JUICE-2B-DINO-Base` | WebSSL-DINO | pretraining 60k | EMA (decay 0.9996) at step 60,000 | 8.41 GB + text head 0.14 GB | base for fine-tuning, text-to-image |
| `CedLJH/JUICE-2B-SigLIP2` | SigLIP2 | pretraining 60k + SFT 15k | EMA (decay 0.9996) at step 15,000 | 8.41 GB + text head 0.14 GB | text-to-image, question answering |
| `CedLJH/JUICE-2B-DINO` | WebSSL-DINO | pretraining 60k + SFT 15k | EMA (decay 0.9996) at step 15,000 | 8.41 GB + text head 0.14 GB | text-to-image, question answering |

`docs/TRAIN.md` gives the training settings. The WebSSL-DINO encoder is for
non-commercial use only (`THIRD_PARTY_LICENSES.md`).
We do not release the JMaze and VSP fine-tunes; `docs/PLANNING.md` shows how to make them.

## 2. Contents of a model folder

| File | Content |
|---|---|
| `model.safetensors` | MM-RiT weights (float32) |
| `config.json` | architecture, image space, flow settings, the inference defaults of the paper, and the source checkpoint (name, step, weight set, SHA-256) |
| `text_head/` | parallel text head (the parallel decoder of the paper): text states to words for question answering (the same head in all four folders) |

A planning folder that you export with `scripts/export_planning.py` also holds `task_head/`, the
linear text head that reads the move sequence (`docs/PLANNING.md`).

The inference defaults in `config.json`:

| Task | Steps | Guidance | Other settings | Precision |
|---|---|---|---|---|
| text-to-image | 25 | 5.0 | generic caption "a photo." as the guidance reference, 4 images in each batch | FP32 |
| question answering | 16 | 3.0 | answer window of 12 slots, batches of 128 | TF32 (SigLIP2), FP32 (WebSSL-DINO) |
| planning (exported planning folders) | 16 | 1.0 | answer window of 64 slots, batches of 25 | FP32 |

## 3. Results of the released models

The numbers of the SFT models are the results of the paper (text-to-image: 25 steps, guidance 5).
VQAv2 test-dev is the score of the EvalAI server for the files of `docs/EVAL_VQA.md` section 7.
DPG-Bench uses the scoring rule of the official script (1,064 prompts, `docs/EVAL_T2I.md`).
The guides tell you how to reproduce the numbers.

### Text-to-image (`docs/EVAL_T2I.md`)

| Model | Steps | GenEval | DPG-Bench |
|---|---|---|---|
| JUICE-2B-SigLIP2 | 25 | 0.866 | 85.27 |
| JUICE-2B-DINO | 25 | 0.883 | 84.55 |

At 100 steps, the SFT models give GenEval 0.868 and 0.886 (`docs/EVAL_T2I.md`, section 7).

### Question answering (`docs/EVAL_VQA.md`)

| Model | VQAv2 test-dev | VQAv2 (1k) | OK-VQA | POPE | MME-P | GQA | MMBench | AI2D | MMMU | SQA-IMG | RWQA |
|---|---|---|---|---|---|---|---|---|---|---|---|
| JUICE-2B-SigLIP2 | 73.10 | 71.2 | 33.9 | 84.0 | 1115.7 | 58.6 | 66.0 | 34.4 | 29.6 | 80.6 | 53.6 |
| JUICE-2B-DINO | 68.75 | 66.3 | 28.7 | 82.3 | 960.2 | 55.7 | 55.8 | 31.9 | 31.5 | 73.7 | 40.5 |

The suite also gives VizWiz (50.3 and 39.5) and MME-C (244.6 and 254.6). The paper does not report them.

## 4. Weights that we do not release

Each training stage starts from the raw weights (`model`) of the stage before it.
We do not release these raw weights of the Base and SFT models, because we did not evaluate them.
If you fine-tune from a released EMA model, the result can be slightly different from the paper
(`docs/TRAIN.md`, section 1).
