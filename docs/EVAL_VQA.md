# Question answering: evaluation

This document tells you how to get the question-answering results of the paper.
The suite has eleven benchmarks.
The paper reports all of them except VizWiz, and for MME only the perception score (MME-P).

| Benchmark | Split | Metric of the table |
|---|---|---|
| VQAv2 n1000 | val2014, fixed subset of 1,000 questions | VQA soft accuracy |
| OK-VQA | val2014 | VQA soft accuracy |
| VizWiz | val | VQA soft accuracy |
| POPE | COCO random + popular + adversarial | accuracy (all 9,000 questions) |
| MME | test | perception score (MME-P), cognition score (MME-C) |
| GQA | testdev-balanced | accuracy |
| MMBench | dev (English) | letter accuracy |
| AI2D | test | letter accuracy |
| MMMU | validation | letter accuracy |
| ScienceQA-IMG | test (questions with an image) | letter accuracy |
| RealWorldQA | test (multiple-choice questions) | letter accuracy |

The suite also makes the VQAv2 test-dev submission file for the EvalAI server.

## 1. How the evaluation works

1. A builder makes "probe" files from the benchmark files.
   A probe item has one question, one image and the token ids of the question.
2. The model answers the items of one probe shard.
   The model gets the image and the question.
   It does not get the answer or the answer length.
3. A scorer compares the answers with the reference answers.

The text of a question is `Question: {question} Answer:`.
Short-answer questions end with "Answer the question using a single word or phrase."
Multiple-choice questions show the options as `A. ...` lines.
They end with "Answer with the option's letter from the given choices directly."

The builder drops some questions:

- a short-answer question with a reference answer of more than 8 tokens,
- a question where the tokens of the question and the answer merge,
- a question that does not fit the 128 text slots,
- cases that are specific to a benchmark (for example, a MMMU question with two images,
  or a RealWorldQA question without options).

The file `drops.json` lists each dropped question and the reason.
The table gives the accuracy on the probe items.
The metrics file also gives `full_set_accuracy`, which counts each dropped question as wrong.

## 2. Install

Install the package with the evaluation extra:

```
pip install -e ".[eval]"
```

Run the commands below from the root of the repository.
To run the unit tests of the builders and the scorers, use this command:

```
python -m unittest discover -s tests -t .
```

## 3. Get the benchmark files

Download each benchmark from the source in the table.
The Hugging Face copies are the copies that lmms-eval uses.
Obey the license of each benchmark.
Do not put the benchmark files or the images in a public location.
This repository does not contain benchmark data.

| `--bench` | Download | Give `--src` | Images |
|---|---|---|---|
| `vqav2_n1000` | [visualqa.org](https://visualqa.org/download.html): `v2_OpenEnded_mscoco_val2014_questions.json`, `v2_mscoco_val2014_annotations.json` | the folder with the two files | COCO val2014 (`--images`) and COCO val2017 (`--coco-val2017`) from [cocodataset.org](https://cocodataset.org) |
| `vqav2_testdev` | visualqa.org: `v2_OpenEnded_mscoco_test-dev2015_questions.json` (and `..._test2015_questions.json` for the submission) | the folder with the files | COCO test2015 (`--images`) |
| `okvqa` | Hugging Face [`lmms-lab/OK-VQA`](https://huggingface.co/datasets/lmms-lab/OK-VQA): `data/val2014-*.parquet` | `OK-VQA/data` | COCO val2014 (`--images`) |
| `vizwiz` | [`lmms-lab/VizWiz-VQA`](https://huggingface.co/datasets/lmms-lab/VizWiz-VQA): `data/val-*.parquet` | `VizWiz-VQA/data` | in the parquet files |
| `pope` | [github.com/RUCAIBox/POPE](https://github.com/RUCAIBox/POPE): `output/coco/coco_pope_{random,popular,adversarial}.json` | the folder with the three files | COCO val2014 (`--images`) |
| `mme` | [`lmms-lab/MME`](https://huggingface.co/datasets/lmms-lab/MME): `data/test-*.parquet` | `MME/data` | in the parquet files |
| `gqa` | [`lmms-lab/GQA`](https://huggingface.co/datasets/lmms-lab/GQA): `testdev_balanced_instructions/*.parquet`, `testdev_balanced_images/*.parquet` | `GQA` (it holds the two folders) | in the parquet files |
| `mmbench` | [`lmms-lab/MMBench`](https://huggingface.co/datasets/lmms-lab/MMBench): `en/dev-00000-of-00001.parquet` | `MMBench/en` | in the parquet file |
| `ai2d` | [`lmms-lab/ai2d`](https://huggingface.co/datasets/lmms-lab/ai2d): `data/test-*.parquet` | `ai2d/data` | in the parquet files |
| `mmmu` | [`MMMU/MMMU`](https://huggingface.co/datasets/MMMU/MMMU): `<Subject>/validation-00000-of-00001.parquet` for all 30 subjects | `MMMU` (it holds the 30 subject folders) | in the parquet files |
| `scienceqa_img` | [`lmms-lab/ScienceQA`](https://huggingface.co/datasets/lmms-lab/ScienceQA): `ScienceQA-IMG/test-00000-of-00001.parquet` | `ScienceQA/ScienceQA-IMG` | in the parquet file |
| `realworldqa` | [`lmms-lab/RealWorldQA`](https://huggingface.co/datasets/lmms-lab/RealWorldQA): `data/test-*.parquet` | `RealWorldQA/data` | in the parquet files |

Example download of one Hugging Face benchmark:

```
huggingface-cli download lmms-lab/GQA --repo-type dataset --local-dir data/GQA \
    --include "testdev_balanced_instructions/*" "testdev_balanced_images/*"
```

The question ids of AI2D, ScienceQA-IMG, RealWorldQA and MME are row numbers.
Thus the file contents must be the same as ours.
Section 9 gives the SHA-256 checksums of the files that we used.
Compare your files with these checksums.

## 4. Build the probes

Build each benchmark with one command.
Give absolute paths: the probe items keep the image paths that you give.

```
python scripts/eval/build_vqa_probes.py --bench vqav2_n1000 --src /data/vqav2 \
    --images /data/coco/val2014 --coco-val2017 /data/coco/val2017 --out /data/probes
python scripts/eval/build_vqa_probes.py --bench okvqa --src /data/OK-VQA/data --images /data/coco/val2014 --out /data/probes
python scripts/eval/build_vqa_probes.py --bench pope --src /data/pope --images /data/coco/val2014 --out /data/probes
python scripts/eval/build_vqa_probes.py --bench gqa --src /data/GQA --out /data/probes
```

Use the same form for the other benchmarks (see the table in section 3).
For VizWiz, MME, GQA and the multiple-choice benchmarks, the builder writes the images to `<out>/<bench>/images`.
Use `--images` to write them to a different folder.

The builder writes these files to `<out>/<bench>/`:

- `probe_shard0.json` ... `probe_shard3.json` (VQAv2 n1000: `probe_shard0.json` only),
- `drops.json`,
- `annotations.json` (the reference answers; not for the multiple-choice benchmarks and not for test-dev).

Make sure that the counts in the output line are the same as these counts:

| Benchmark | Questions | Probe items | Shard sizes |
|---|---|---|---|
| vqav2_n1000 | 1,000 | 1,000 | 1000 |
| okvqa | 5,046 | 5,028 | 1257 / 1257 / 1257 / 1257 |
| vizwiz | 4,319 | 4,272 | 1068 / 1068 / 1068 / 1068 |
| pope | 9,000 | 9,000 | 2250 / 2250 / 2250 / 2250 |
| mme | 2,374 | 2,374 | 594 / 594 / 593 / 593 |
| gqa | 12,578 | 12,578 | 3145 / 3145 / 3144 / 3144 |
| mmbench | 4,329 | 3,837 | 960 / 959 / 959 / 959 |
| ai2d | 3,088 | 3,084 | 771 / 771 / 771 / 771 |
| mmmu | 900 | 622 | 156 / 156 / 155 / 155 |
| scienceqa_img | 2,017 | 1,714 | 429 / 429 / 428 / 428 |
| realworldqa | 765 | 427 | 107 / 107 / 107 / 106 |
| vqav2_testdev | 107,394 | 107,394 | 26849 / 26849 / 26848 / 26848 |

## 5. Run the model

Run one process for each shard.
Use the shards as the builder made them.

```
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k python scripts/eval/run_vqa.py --model <model folder or Hugging Face id> \
      --probe /data/probes/gqa/probe_shard$k.json --out /data/answers/gqa/shard$k.json &
done
wait
```

The default settings are the settings of the paper:

- 16 sampling steps and guidance scale 3 (from `config.json`),
- batches of 128 items, in the order of the shard,
- seed 0; the batch that starts at item `start` of the shard uses the seed `0 + start`,
- the precision of `config.json` (SigLIP2 model: TF32; WebSSL-DINO model: FP32),
- the parallel text head (the parallel decoder of the paper) reads the answer up to the first end-of-sequence token.

If you change the batch size or the shards, the noise of each item changes.
The answers then change by a small quantity.
Different GPU types can also change the last bits of the results.
Our checks used NVIDIA A100 GPUs.

Use `--save-states <folder>` to keep the sampled text states of each batch.

To use more GPUs than shards, split a shard by batches. The seed of a batch depends only on
its position in the shard, so the answers do not change. For example, two processes for shard 0
of VQAv2 test-dev (210 batches):

```
python scripts/eval/run_vqa.py ... --probe <probes>/probe_shard0.json --max-batches 105 --out shard0.part0.json
python scripts/eval/run_vqa.py ... --probe <probes>/probe_shard0.json --skip-batches 105 --out shard0.part1.json
```

Give all part files to the scoring or submission script, in the order of the shards and parts.

## 6. Score

Give the answer files of all shards to the scorer:

```
python scripts/eval/score_vqa.py --bench gqa --probe-dir /data/probes/gqa \
    --answers /data/answers/gqa/shard0.json /data/answers/gqa/shard1.json \
              /data/answers/gqa/shard2.json /data/answers/gqa/shard3.json \
    --out /data/answers/gqa/metrics.json
```

The scorer prints the row of the table.
The scorer stops with an error if an item of the probe has no answer.

The scorers are:

- VQA soft accuracy: a port of the official VQA evaluation code.
  POPE and GQA have one reference answer.
  The probe keeps it ten times, so the soft accuracy is the exact-match accuracy.
- POPE: the metrics file also gives accuracy, precision, recall and F1 for each split ("yes" is positive).
- MME: for each category, score = 100 x (accuracy + accuracy+).
  Accuracy+ is the fraction of images with two correct answers.
  MME-P is the sum of 10 perception categories.
  MME-C is the sum of 4 cognition categories.
- Multiple choice: the scorer takes the first letter A-I that is an option of the item.
  If there is no letter, it compares the full answer with the option texts.
  An answer without a letter or an option text is wrong.

## 7. VQAv2 test-dev

1. Build the `vqav2_testdev` probes (section 4).
2. Run the four shards (section 5).
3. Make the submission file:

```
python scripts/eval/vqav2_testdev_submission.py --answers /data/answers/vqav2_testdev/shard*.json \
    --questions /data/vqav2 --out /data/answers/vqav2_testdev/submission.json
```

4. Send `submission.json` to the test-dev phase of the VQA challenge on EvalAI.

The file has one entry for each test2015 question (447,793).
EvalAI accepts a file only if it has results for the entire test set,
but the test-dev phase scores only the 107,394 test-dev questions.
Thus the questions that are not in test-dev get an empty answer.
The test-dev answers are not public, so this suite cannot give a local score.

The EvalAI server gave these scores for our files (test-dev phase, the settings of section 5):

| Answer type | JUICE-2B-SigLIP2 | JUICE-2B-DINO |
|---|---|---|
| overall | 73.10 | 68.75 |
| yes/no | 90.66 | 88.18 |
| number | 58.73 | 53.30 |
| other | 61.29 | 55.57 |

In these files, the model gave an empty answer for 1 (SigLIP2) and 2 (WebSSL-DINO) test-dev questions.

## 8. Expected results

These are the results of the paper (paper Table 34; parallel text head, guidance scale 3, 16 steps, seed 0).
The paper does not report VizWiz and MME-C.

| Benchmark | JUICE-2B-SigLIP2 | JUICE-2B-DINO |
|---|---|---|
| VQAv2 n1000 | 0.7121 | 0.6632 |
| OK-VQA | 0.3387 | 0.2865 |
| VizWiz (not in the paper) | 0.5027 | 0.3952 |
| POPE | 0.8397 | 0.8229 |
| MME-P | 1115.7 | 960.2 |
| MME-C (not in the paper) | 244.6 | 254.6 |
| GQA | 0.5858 | 0.5568 |
| MMBench | 0.6604 | 0.5582 |
| AI2D | 0.3440 | 0.3187 |
| MMMU | 0.2958 | 0.3151 |
| ScienceQA-IMG | 0.8057 | 0.7375 |
| RealWorldQA | 0.5363 | 0.4052 |

Notes:

- The SFT data of the models contains the ScienceQA train split.
  Thus ScienceQA-IMG is an in-domain benchmark.
- MMMU is near the accuracy of the most frequent letter (0.289).
- VQAv2 test-dev (EvalAI server): section 7.
- Accuracy with the dropped questions counted as wrong: OK-VQA 0.3375 / 0.2855, VizWiz 0.4972 / 0.3909 (SigLIP2 / DINO).

We checked the suite against the research code of the paper:

- The builders make the same probe items, in the same shard order, as the research probes (all twelve benchmarks).
- The scorers give exactly the values of this table from the answers of the paper runs.
- On an A100 GPU, `run_vqa.py` gives text states that are bit-equal to the states of the paper runs, and the same answers.
  We examined the first batch of MMBench and of GQA (SigLIP2 model) and of VizWiz (DINO model, FP32).
- The official VQA evaluation code (github.com/GT-Vision-Lab/VQA, commit `a013f00`, Python 2.7) gives
  the same values as `score_vqa.py` for each question and answer type: VQAv2 n1000 71.21 and 66.32,
  OK-VQA 33.87 and 28.65. The code does not run in Python 3 without a conversion.

Note about the OK-VQA reference answers. `score_vqa.py` uses the reference answers of
`lmms-lab/OK-VQA`. They are the `raw_answer` field of the official OK-VQA v1.1 file.
The `answer` field of that file has lemmatized answers (for example "race" for "racing").
The official code does not lemmatize the predictions. Thus, with the `answer` field, it gives
lower values (29.15 and 24.63).

## 9. File checksums (SHA-256)

| File | SHA-256 |
|---|---|
| `v2_OpenEnded_mscoco_val2014_questions.json` | `f34d9c9909dea76700d361baddd63dd47b4494945097459b542d9fe840244811` |
| `v2_mscoco_val2014_annotations.json` | `0564650e712b34c246e4ad142894692bf2e45dee8e9cc6ebb2cedbe85731c4bf` |
| `v2_OpenEnded_mscoco_test-dev2015_questions.json` | `d96a59e8ce7e6e74689dd5198ec0c482e4c5a8744874a805a3e28c101061fae7` |
| `v2_OpenEnded_mscoco_test2015_questions.json` | `69169f086e0fa9f878a7425aabe6573e999af79029726b64bbf144482000960a` |
| OK-VQA `val2014-00000-of-00002.parquet` | `2283adb5cf7030fea709892102b3180dfc6f3a7a64cc68d798e5c31916907e4f` |
| OK-VQA `val2014-00001-of-00002.parquet` | `208d970081399c0fd5e9f4c1a13839d5e77d75819dc6b5c028d5619db987440d` |
| VizWiz `val-00000-of-00005-7775fd61bc6a3d98.parquet` | `62a5fddb577d9165c3b89d6fb565cddfb4e7d41294ca16373de6263baff7b36d` |
| VizWiz `val-00001-of-00005-18e4fb673cfdd7cb.parquet` | `7618d6e6ed3e1eecc0c0b1857b6b1e79f256b21120d40f653786f8abe61f2dc2` |
| VizWiz `val-00002-of-00005-eeb6c831a97fe54e.parquet` | `f787b93c9305c4e05cbfc9bf48f69fab274f98bb57467f2aa092ad07794f40d1` |
| VizWiz `val-00003-of-00005-ff7829371634e9f2.parquet` | `468f0f70c3b492bae32c62c52fc570ef13fa0a7903f1549ff68a10d83ebe312d` |
| VizWiz `val-00004-of-00005-59be8a2af5f336e2.parquet` | `f8a27b1dfaf0f3703d0df79520e015c0b0299e4fa224879c2e106e2a459eb36c` |
| `coco_pope_random.json` | `ac25245170b975a5bdf9080b23fd431dfe6be458bc038259c1f4f09a6bef7994` |
| `coco_pope_popular.json` | `72c1a8ad45d0c13514f5f22598261df41d3b533854d29682e924db50ed8aa753` |
| `coco_pope_adversarial.json` | `420b3407db1fa9f1187a805dca41cb7b97fd91504e6c2179706188c107fb8ef8` |
| MME `test-00000-of-00004-a25dbe3b44c4fda6.parquet` | `1563fd59c8049dbbabe211a049c38b2e20a6d58da94a75c90cfc937d1f6c52e3` |
| MME `test-00001-of-00004-7d22c7f1aba6fca4.parquet` | `3623728c0c98807cc7eef9cbd10dfa18c21654321c4b02b7b19bc8e6fc86d830` |
| MME `test-00002-of-00004-594798fd3f5b029c.parquet` | `c169d85bdf84e21b005170698217e96380bef172e8bd7432fc75ddce51dcaabe` |
| MME `test-00003-of-00004-53ae1794f93b1e35.parquet` | `aec84886b2bb6dd7bffc94eb3bbca9319e0671f3ee989a08e0758bc2de6c67f6` |
| GQA `testdev_balanced_instructions/testdev-00000-of-00001.parquet` | `e967085f896deee3b99b7c7a830e74fc0662a61cdbcbd0154ffbb8aea4ab67ed` |
| GQA `testdev_balanced_images/testdev-00000-of-00001.parquet` | `ba0a57a72c483d6e4a390ecbbee05d07a55365aeac3e93f4a923ac61c9e465eb` |
| MMBench `en/dev-00000-of-00001.parquet` | `765dc6a1a800831e89645d87bdae9ed47f4c8c5fb082bbd58121cd729f34eeb8` |
| AI2D `data/test-00000-of-00002.parquet` | `450ecfa95b0c475ba214cd9a33b7ec5d1d782e7321a54fc652453d8776743702` |
| AI2D `data/test-00001-of-00002.parquet` | `13f5813136e51902cc4048bf48c53104b29dbe7ac7d31b170c0e9c36bceed06b` |
| ScienceQA `ScienceQA-IMG/test-00000-of-00001.parquet` | `1f9729818b577d70325a0e59f488f6eeda74c827caddd56b75ccf3e36ed45064` |
| RealWorldQA `data/test-00000-of-00002.parquet` | `0ed8b555586923099bd5d6ba5dd8b656b403ccfc418881facd237a6d6fe64952` |
| RealWorldQA `data/test-00001-of-00002.parquet` | `7dcb3ac3483362ca082cd4cddd0ab1389e9a276f1aefd4603fde0a2ce6bc74d0` |

The Hugging Face revisions that we used: `lmms-lab/OK-VQA` `c9f858961f4095a9aeef615057b0822cce967096`, `lmms-lab/VizWiz-VQA` `d428a2dae984f79cf1b9d99467dfa883e0c30686`, `lmms-lab/MME` `d6c9023f017b564f7b3ccccf5348166bce8fdbcd`.
