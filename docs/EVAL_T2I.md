# Text to image: GenEval and DPG-Bench

This document tells you how to get the text-to-image results of the paper.
The suite has two benchmarks.

| Benchmark | Prompts | Images | Metric of the table | Scorer |
|---|---|---|---|---|
| GenEval | 553 | 4 for each prompt | overall score (mean of the 6 task scores) | Mask2Former and CLIP (official GenEval code) |
| DPG-Bench | 1,065 | one 2x2 grid of 4 images for each prompt | DPG-Bench score | mPLUG VQA (official DPG-Bench code) |

## 1. How the evaluation works

1. The generation script makes the images from the official prompt files.
2. The official scorer of each benchmark reads the images and gives the score.

This repository does not contain the prompt files or the scorers.
Section 3 tells you where to get them.

## 2. Install

Install the package:

```
pip install -e .
```

Run the commands below from the root of the repository.
To run the unit tests of the prompt, seed and layout helpers, use this command:

```
python -m unittest discover -s tests -t .
```

The scorers need their own environments (section 5 and section 6).
Do not install them in the environment of the model.

## 3. Get the prompt files

| Benchmark | Repository | File | License |
|---|---|---|---|
| GenEval | [github.com/djghosh13/geneval](https://github.com/djghosh13/geneval) | `prompts/evaluation_metadata.jsonl` | MIT |
| DPG-Bench | [github.com/TencentQQGYLab/ELLA](https://github.com/TencentQQGYLab/ELLA) | `dpg_bench/dpg_bench.csv` | Apache-2.0 (repository license) |

```
git clone https://github.com/djghosh13/geneval.git
git clone https://github.com/TencentQQGYLab/ELLA.git
```

Compare the files with these SHA-256 checksums:

| File | SHA-256 |
|---|---|
| `evaluation_metadata.jsonl` | `5c48e0813e812e3c373fa5c8ed07a8f0a483be30272b4427b0559c8048e67c13` |
| `dpg_bench.csv` | `68f507652a49aaa4f103f17bbf25e2322a9a318dc61380e6534b749fbf9317aa` |

We compared the files with these repository commits:
GenEval `af4902f24d3ca90ebbb446dd9891a59e0f82725f`, ELLA `3c228f1dc6c4d3cad0a47493816151a419f14db3`.

The paper read the DPG-Bench prompts from the Hugging Face copy `Jialuo21/DPG-Bench` (a parquet file).
We examined it: it has the same 1,065 prompts in the same order as `dpg_bench.csv`.
Thus the images are the same.

## 4. Generate the images

The script makes the images of one benchmark:

```
python scripts/eval/generate_t2i_bench.py --benchmark geneval --model <model folder or Hugging Face id> \
    --prompts geneval/prompts/evaluation_metadata.jsonl --out /data/t2i/siglip2/geneval
python scripts/eval/generate_t2i_bench.py --benchmark dpg --model <model folder or Hugging Face id> \
    --prompts ELLA/dpg_bench/dpg_bench.csv --out /data/t2i/siglip2/dpg
```

To use more GPUs, run one process for each shard:

```
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k python scripts/eval/generate_t2i_bench.py --benchmark geneval \
      --model <model> --prompts geneval/prompts/evaluation_metadata.jsonl \
      --out /data/t2i/siglip2/geneval --num-shards 4 --shard $k &
done
wait
```

The settings of the paper are the defaults. Do not change them to get the results of the paper:

- 25 sampling steps (Euler), guidance scale 5.0 (`config.json`, `defaults` / `text_to_image`).
- The generic caption "a photo." is the reference branch of the guidance (paper Sec. 3.5).
- FP32 precision (TF32 off). The T5 text encoder runs in bfloat16, as in the paper's runs.
- Seed 0. The 4 images of prompt `i` (line `i` of the prompt file, from 0) come from one batch
  with the generator seed `seed + 1000 * i`. The seed does not depend on the number of shards.
- Native 224 x 224 images. The paper did not use `--upsample`.

The script writes these files:

- GenEval: `<out>/<i:05d>/metadata.jsonl` (line `i` of the prompt file) and `<out>/<i:05d>/samples/00000.png` ... `00003.png`.
- DPG-Bench: `<out>/<item_id>.png`, a 448 x 448 grid of the 4 images (first row: images 0 and 1; second row: images 2 and 3).
- `<out>/gen_protocol.json`: the settings of the run.

If a job stops, run the same command again.
The script skips the prompts that have all their files.
It stops with an error if `gen_protocol.json` records different settings.
Use `--overwrite` to make all images again.

On an A100 GPU, one prompt (4 images, 25 steps, FP32) takes approximately 14 seconds.
Thus GenEval takes approximately 2 GPU-hours and DPG-Bench approximately 4 GPU-hours for each model.

### 4.1 Sampling steps

The paper uses 25 sampling steps (paper Sec. 5.1).
The paper also gives GenEval at 50 and 100 steps (paper Table 32). To get these, use `--steps 50` or
`--steps 100`. Section 7 gives the expected values.
We did not validate other numbers of steps.

## 5. Score GenEval

Use the official GenEval code (`evaluation/evaluate_images.py` and `evaluation/summary_scores.py`).
The detector code uses the MMDetection 2.x API.
Thus it needs its own environment.
The paper used these versions:

| Package | Version |
|---|---|
| Python | 3.10 |
| torch / torchvision | 2.0.1+cu118 / 0.15.2+cu118 |
| mmcv-full | 1.7.2 |
| mmdet | 2.28.2 |
| open_clip_torch | 3.3.0 |
| clip-benchmark | 1.6.2 |
| numpy | 1.26.4 |

Example installation:

```
conda create -n geneval python=3.10 -y && conda activate geneval
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install mmdet==2.28.2 open_clip_torch==3.3.0 clip-benchmark==1.6.2 "numpy<2" pandas tqdm
```

Get the detector weights and the detector config:

1. Run `geneval/evaluation/download_models.sh <detector folder>`.
   It downloads `mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth`.
2. The pip package of mmdet does not contain the config files.
   Get them from MMDetection tag `v2.28.2`:

```
git clone --branch v2.28.2 --depth 1 https://github.com/open-mmlab/mmdetection.git
```

The CLIP model (ViT-L-14, `openai`) downloads on the first run.

Score the images:

```
python geneval/evaluation/evaluate_images.py /data/t2i/siglip2/geneval \
    --outfile /data/t2i/siglip2/geneval_results.jsonl \
    --model-path <detector folder> \
    --model-config mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py
python geneval/evaluation/summary_scores.py /data/t2i/siglip2/geneval_results.jsonl
```

The last line of the output is the overall score, for example `Overall score (avg. over tasks): 0.86597`.
Make sure that the results file has 2,212 lines (553 prompts x 4 images).
The scorer does not stop when images are missing.

## 6. Score DPG-Bench

Use the official DPG-Bench code: `ELLA/dpg_bench/compute_dpg_bench.py`.
It uses the mPLUG VQA model `damo/mplug_visual-question-answering_coco_large_en` from ModelScope
(now also named `iic/mplug_visual-question-answering_coco_large_en`).
The paper used these versions:

| Package | Version |
|---|---|
| Python | 3.12 |
| torch / torchvision | 2.5.1+cu121 / 0.20.1+cu121 |
| modelscope | 1.38.0 |
| transformers | 4.44.2 |
| fairseq-fixed | 0.12.3.1 |
| datasets | 3.6.0 |
| numpy | 1.26.4 |

The script also needs `accelerate`.
The modelscope import needs the `decord` package, but the image VQA does not use it.

Score the grids at the native resolution (`--resolution 224`, the size of one image in the grid):

```
cd ELLA
accelerate launch --num_processes 1 dpg_bench/compute_dpg_bench.py \
    --image-root-path /data/t2i/siglip2/dpg --resolution 224 --pic-num 4 --vqa-model mplug \
    --res-path /data/t2i/siglip2/dpg_results.txt
```

The last line of the output is the DPG-Bench score.
Put `--res-path` outside the image folder.
The script tries to score each file in the image folder; it prints "Failed filename" for files that are not grids
(for example `gen_protocol.json`) and skips them.

Note about the first question.
The official script skips the first row of `dpg_bench.csv` (question 1 of `partiprompts97`).
Question 2 of that prompt depends on question 1, so the script cannot score `partiprompts97`.
It prints "Failed filename: partiprompts97.png" and gives the mean of the other 1,064 prompts.
The DPG-Bench scores of this guide use the scoring rule of the official script (1,064 prompts).

## 7. Expected results

These are the results of the default settings (25 steps, guidance scale 5, generic caption "a photo.",
seed 0, 224 x 224). These are the settings of the paper (paper Table 4 and Table 32).

| Benchmark | JUICE-2B-SigLIP2 | JUICE-2B-DINO |
|---|---|---|
| GenEval overall | 0.8660 | 0.8834 |
| single object | 0.9938 | 0.9938 |
| two object | 0.9343 | 0.9470 |
| counting | 0.7781 | 0.8063 |
| colors | 0.9096 | 0.9309 |
| position | 0.8500 | 0.8475 |
| color attribution | 0.7300 | 0.7750 |
| DPG-Bench (official scoring rule, 1,064 prompts) | 85.27 | 84.55 |

GenEval overall at other numbers of steps (paper Table 32):

| Steps | JUICE-2B-SigLIP2 | JUICE-2B-DINO |
|---|---|---|
| 25 (default) | 0.8660 | 0.8834 |
| 50 | 0.8680 | 0.8853 |
| 100 | 0.8676 | 0.8864 |

Sources:

- 25 and 50 steps: the images of this generation script (`--steps 25`, `--steps 50`), scored with the
  scorers of the paper runs. GenEval overall 0.865965 and 0.883391 at 25 steps.
- 100 steps: the images of the paper runs. GenEval overall 0.867604 and 0.886373.
- DPG-Bench: the official script (section 6), as in the paper.

We checked the generation script against the research code of the paper at 100 steps.
On an A100 GPU, the script makes PNG files that are byte-equal to the images of the paper runs.
We examined GenEval prompts 0, 1, 276 and 552 (two shards) and the DPG-Bench prompts
`partiprompts97`, `77` and `COCOval2014000000191919` (first, middle and last), for both models.
The check needs the images of the paper runs, which are not in this repository.
The number of steps changes only the time grid of the sampler.

We also scored the SigLIP2 GenEval images of the 100-step paper run with the official GenEval scripts
(commit `af4902f`) and the commands of section 5.
The overall score is 0.86760, the same as the table.
We ran the official DPG-Bench script (ELLA commit `3c228f1`, not changed) with the settings of
`dist_eval.sh` (`--mixed_precision fp16`, `--pic-num 4`, `--resolution 224`), with one process on one
A100 GPU for each model. The script scored 1,064 prompts, and all its mPLUG answers were the same as
the answers of our scorer. The fp16 setting does not change the answers, because the script does not
run the mPLUG model through `forward`. Note: `accelerate` does not accept `--multi_gpu` with one process.
The category scores of the script use the `tuple` column of `dpg_bench.csv` and only the answers for
the last image of each grid. For 23 questions, this category is different from the `category_broad`
and `category_detailed` columns.

## 8. File checksums (SHA-256)

| File | SHA-256 |
|---|---|
| GenEval `prompts/evaluation_metadata.jsonl` | `5c48e0813e812e3c373fa5c8ed07a8f0a483be30272b4427b0559c8048e67c13` |
| DPG-Bench `dpg_bench/dpg_bench.csv` | `68f507652a49aaa4f103f17bbf25e2322a9a318dc61380e6534b749fbf9317aa` |
| `mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth` | `743b7d99015f1224c6d57fd4b14d04b15cc8ec72ae7ee7831e7c71d8873b7a54` |
| mPLUG `pytorch_model.bin` | `702b1fcb760b1a2bb4ab7f7b0d6f540aad438716f9fdb602f85e121d5a371332` |
