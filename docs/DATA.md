# Training data

This guide tells you how to make the training data of the paper.
We do not supply the training data.
You download the public datasets and run our scripts.
The scripts make WebDataset shards and the manifests `data/manifests/<name>.txt`.
The configs in `configs/train/` read these manifests.

## 1. Overview

| Stage | Configs | Manifests | Rows (config counts) |
|---|---|---|---|
| 1. Pretraining | `pretrain_*.yaml` | 22 | 43,192,498 |
| 2. SFT | `sft_*.yaml` | 18 | 20,589,708 |
| 3. JMaze, VSP | `jmaze_*.yaml`, `vsp_siglip2.yaml` | 1 each | see `docs/PLANNING.md` |

A manifest lists one shard (`.tar`) per line, with an absolute path.
The row counts of the configs (`train_manifest_counts`) are the counts of the paper runs.
Do not change them. The trainer uses them to give the shards to the data workers.
Thus they set the data order.

The scripts use this folder layout. Make `data` a link to a large disk if necessary.

| Folder | Content | Size |
|---|---|---|
| `data/src/` | the downloads (section 2) | approximately 6.3 TB |
| `data/wds/` | the packs and the replica links | approximately 2.1 TB |
| `data/manifests/` | the manifests | small |
| `data/logs/` | the logs of the packers | small |

The largest downloads are the three BLIP3o pretraining sets (5.3 TB).
Pretraining reads them as they are. You cannot delete them after the build.

## 2. Download

Install the Hugging Face command line tool (`pip install -U huggingface_hub`).
Run the commands from the root of the repository.
The revisions are the revisions that we used.

### 2.1 BLIP3o sets (pretraining and SFT captions)

```bash
huggingface-cli download BLIP3o/BLIP3o-Pretrain-Long-Caption --repo-type dataset \
    --revision e4d07091a466d1a1e35a9b0c61caddc78d14a059 --local-dir data/src/BLIP3o-Pretrain-Long-Caption
huggingface-cli download BLIP3o/BLIP3o-Pretrain-Short-Caption --repo-type dataset \
    --revision e84d184540a6cc4fc3b1ba5528bb45dcc6b3fb0e --local-dir data/src/BLIP3o-Pretrain-Short-Caption
huggingface-cli download BLIP3o/BLIP3o-Pretrain-JourneyDB --repo-type dataset \
    --revision 734b390be883b2021ae6d6dc1b24e6b97e7253a2 --local-dir data/src/BLIP3o-Pretrain-JourneyDB
huggingface-cli download BLIP3o/BLIP3o-60k --repo-type dataset \
    --revision f7316b0aa446338ee1707484924aa59457b4bbf3 --local-dir data/src/BLIP3o-60k
```

At these revisions the sets have 2,891, 1,831, 419 and 11 tar files.

### 2.2 Cambrian-10M (question-answer rows)

Download the jsonl file and the image archives of 25 sources:

```bash
SRCS="gqa coco lnqa vizwiz oodvqa vg dvqa ocr_vqa clevr tallyqa hfdata Q-Instruct-DB docvqa pathvqa chartqa ai2d
      scienceqa llava textvqa sam wikiart share_textvqa web-celebrity web-landmark"
huggingface-cli download nyu-visionx/Cambrian-10M --repo-type dataset \
    --revision a087b9234c59bc6c64e7e4a091a6a618cb887132 --local-dir data/src/Cambrian-10M \
    --include "jsons/Cambrian7M.jsonl" "allava.tar.gz_part*" $(for s in $SRCS; do echo "$s.tar.gz"; done)
cd data/src/Cambrian-10M
cat allava.tar.gz_part{1..12} > allava.tar.gz
mkdir -p images
for a in *.tar.gz; do tar -xzf "$a" -C images; done
cd -
```

The packer reads the images at `data/src/Cambrian-10M/images/<image field of the row>`.
Then make the input file of the packer (5,354,769 lines):

```bash
python -m scripts.data.make_cambrian_mmu_jsonl --root data/src/Cambrian-10M
sha256sum data/src/Cambrian-10M/jsons/cambrian7m_mmu.jsonl   # paper file: see section 7
```

### 2.3 COCO 2014 and 2017 images, COCO captions

```bash
mkdir -p data/src/coco && cd data/src/coco
for z in train2014 val2014 val2017; do wget http://images.cocodataset.org/zips/$z.zip && unzip -q $z.zip; done
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip && unzip -q annotations_trainval2014.zip
cd -
```

The packers need `train2014/`, `val2014/`, `val2017/` and `annotations/captions_train2014.json`.
The val folders give the image ids of section 5. No val image goes into a COCO caption row.

### 2.4 VQAv2, ScienceQA, ShareGPT-4o-Image, DALL-E 3

```bash
huggingface-cli download lmms-lab/VQAv2 --repo-type dataset \
    --revision 32665d35052eb4a6d4414851c3c829a72754915a --include "data/train-*" --local-dir data/src/VQAv2
huggingface-cli download lmms-lab/ScienceQA --repo-type dataset \
    --include "ScienceQA-IMG/train-*" --local-dir data/src/ScienceQA
huggingface-cli download FreedomIntelligence/ShareGPT-4o-Image --repo-type dataset \
    --revision f9bcb9494e562f3b6d46585bda0e7f0f0d6ec43f \
    --include "text_to_image.json" "text_to_image_part_*.tar" --local-dir data/src/ShareGPT-4o-Image
huggingface-cli download OpenDatasets/dalle-3-dataset --repo-type dataset \
    --revision 96822d5c0b1338ec4bfd04b0d5909fa070e301b6 --include "*.parquet" --local-dir data/src/dalle-3-dataset
```

VQAv2: only the 136 train parquets. The val parquets are evaluation data.
ScienceQA: only the train split of ScienceQA-IMG. The test split is an evaluation benchmark.
We did not record the ScienceQA revision. Section 7 gives the sha256 of our file.

### 2.5 JMaze and VSP

Make the data with `docs/PLANNING.md`, section 3, with `--out data/jmaze` and `--out data/vsp`.

## 3. Build the packs and the manifests

Run all steps with one command:

```bash
STEP=all JOBS=16 bash scripts/data/prepare_data.sh
```

- `JOBS` is the number of packer processes that run at the same time.
  The script starts one process for each shard of a pack (115 Cambrian processes).
- Each Cambrian process reads all of `cambrian7m_mmu.jsonl` first (approximately 5 minutes)
  and keeps the rows of its sources in memory (up to some GB).
  Then the speed depends on how fast your disk reads the images.
- A step does not run a pack again if the pack has its summary file.
  To run a pack again, remove its folder.
- At the end, the script runs `make_manifests.py --strict`.
  It writes the 41 manifests of the configs (and 12 base manifests of the replicas),
  makes the replica links and compares the row counts of the packs with the paper counts.
  A missing manifest or a different count stops the script.

You can also run one step at a time, in this order:
`cambrian_jsonl s8 valrows coco_caption journeydb blip3o60k dalle3 sharegpt4o vqav2 mcq scienceqa v2 long manifests captions`.
The `captions` step reads the manifests of the `manifests` step.

## 4. The pools

### 4.1 Sample format

| Member | Content |
|---|---|
| `<key>.jpg` (or `.png`, ...) | the image (the bytes of the source, except ShareGPT-4o: see 4.4) |
| `<key>.txt` | the caption. Question-answer rows: `Question: <question> Answer: <answer>` |
| `<key>.json` | question-answer rows only: the answer span and the source of the row |

The JSON member of a question-answer row contains these fields:

- `answer_tok_start`, `answer_tok_end`: the answer tokens in the caption
  (Flan-T5-Small content tokens, no EOS, tokenizer revision `0fc9ddf78a1e`).
- `n_content_tokens`: the number of caption tokens. With the EOS, it is not more than 128.
- `qa_window`: the answer window of the row (SFT packs only: 48 or 101).
  Rows without this field use the window of the config (12).
- The source of the row, for example `source`, `cambrian_line`, `pair_idx`, `image`, `answer`.
- `truncated`, `n_sentences_kept`, `n_sentences_total`: when the packer cut a long answer.

The trainer reads `answer_tok_start` and `qa_window`. A row without `answer_tok_start` is a caption row.

### 4.2 Question-answer rules

A question-answer row is one (human, gpt) turn pair of a source conversation.

- The text is `Question: {question} Answer: {answer}`. The `<image>` tag is removed.
- v1 prompts (pretraining): the question loses the LLaVA cue "Answer the question using a single word or phrase.".
- v2 prompts (SFT): the question keeps the cue. If the first turn of a conversation has the cue,
  all pairs of the conversation get it.
- The packer keeps a row only when the tokens of `Question: {question} Answer:` are a prefix
  of the tokens of the full text, and when the answer has the allowed number of tokens.
- Short answers: 1 to 8 tokens. Long answers: 9 to 100 tokens; the packer cuts a longer
  answer at the last sentence end that fits.
- All rows: the text and the EOS fit in 128 tokens.
- The packer removes the 1,024 Cambrian rows of our held-out probe
  (`scripts/data/cambrian7m_mmu_valprobe1024.linehash`, sha1 of each jsonl line).
- The rows are shuffled once with the seed 4242. `--shard K/N` takes rows K, K+N, K+2N, ...
  The union of the N shards is the full pack.

### 4.3 Pretraining pools

| Manifest | Source | Processing | Rows | Shards | Pack size |
|---|---|---|---|---|---|
| `blip3o_long_caption_train` | BLIP3o-Pretrain-Long-Caption | none: the first 2,875 tars in name order (the last 16 are validation tars) | 29,187,000 (*) | 2,875 | 1,374 GB (download) |
| `blip3o_short_caption_train` | BLIP3o-Pretrain-Short-Caption | none: the first 1,815 tars in name order | 4,728,000 (*) | 1,815 | 828 GB (download) |
| `journeydb_train` | BLIP3o-Pretrain-JourneyDB | `JourneyDB_000`-`410`; 7 tars cleaned by `clean_journeydb.py` (section 6) | 4,197,986 (*) | 411 | 3,132 GB (download) + 51 GB |
| `cambrian_gqa_x4` | Cambrian gqa | `pack_cambrian.py`, v1, 10 shards | 943,833 | approx. 380 | 136 GB |
| `cambrian_coco_x4` | Cambrian coco | v1, 7 shards, COCO val images removed | 634,889 | approx. 259 | 104 GB |
| `cambrian_lnqa_x4` | Cambrian lnqa | v1, 3 shards | 275,207 | approx. 111 | 30 GB |
| `cambrian_vizwiz_x4` | Cambrian vizwiz | v1, 1 shard | 20,128 | 9 | 10 GB |
| `cambrian_oodvqa_x4` | Cambrian oodvqa | v1, 1 shard | 16,630 | 7 | 1 GB |
| `cambrian_vg` | Cambrian vg | v1, 8 shards | 336,224 | 136 | 49 GB |
| `cambrian_dvqa` | Cambrian dvqa | v1, 8 shards | 774,888 | 312 | 19 GB |
| `cambrian_ocr_vqa` | Cambrian ocr_vqa | v1, 4 shards | 317,675 | 128 | 15 GB |
| `cambrian_clevr` | Cambrian clevr | v1, 4 shards | 351,828 | 144 | 68 GB |
| `cambrian_tallyqa` | Cambrian tallyqa | v1, 3 shards | 250,279 | 102 | 41 GB |
| `cambrian_hfdata` | Cambrian hfdata | v1, 4 shards | 313,014 | 128 | 20 GB |
| `cambrian_Q-Instruct-DB` | Cambrian Q-Instruct-DB | v1, 3 shards | 284,400 | 114 | 88 GB |
| `cambrian_docvqa` | Cambrian docvqa | v1, 1 shard | 35,240 | 15 | 25 GB |
| `cambrian_pathvqa` | Cambrian pathvqa | v1, 1 shard | 29,940 | 12 | 2 GB |
| `cambrian_chartqa` | Cambrian chartqa | v1, 1 shard | 28,272 | 12 | 1 GB |
| `cambrian_ai2d` | Cambrian ai2d | v1, 1 shard | 15,075 | 7 | 3 GB |
| `cambrian_scienceqa` | Cambrian scienceqa | v1, 1 shard | 5,867 | 3 | 0.1 GB |
| `cambrian_coco_valrows` | Cambrian coco | v1, 8 shards, ONLY the rows on COCO val images (section 5) | 32,010 | 16 | 6 GB |
| `coco_train2014_caption` | COCO train2014 | `pack_coco_captions.py`: 5 human captions per image, 2,000 rows per tar | 414,113 | 208 | 68 GB |

(*) These three counts are the counts of the paper runs, not exact row counts (section 6).
The `_x4` in a name has no function. All Cambrian packs have 2,500 rows per tar.
The Cambrian archives of the 25 sources are 326 GB. Extract them, then you can delete the archives.

### 4.4 SFT pools

| Manifest | Base pack and processing | Base rows | Copies | Rows | Pack size |
|---|---|---|---|---|---|
| `blip3o_ft60k_r64_x75` | `repack_blip3o60k.py`: BLIP3o-60k without `mscoco_human.tar`, 64 equal shards | 50,355 | 75 | 3,776,625 | 93 GB |
| `dalle3_t2i_x52` | `stage_dalle3.py`: dalle-3-dataset, 5 parquets per shard | 19,024 | 52 | 989,248 | 33 GB |
| `sharegpt4o_t2i_x52` | `pack_sharegpt4o.py`: text-to-image rows, JPEG, longest side 1024, 8 shards | 45,717 | 52 | 2,377,284 | 16 GB |
| `vqav2_train_v2_s13_x7` | `pack_vqav2.py` (v2 prompt, 10,000 rows per tar), tars 0-42 split in 13 by `split_shards.py` | 430,000 | 7 | 3,010,000 | 71 GB + 68 GB |
| `cambrian_v2_gqa_x2` | Cambrian gqa, v2, 8 shards | 943,833 | 2 | 1,887,666 | 255 GB (3 packs) |
| `cambrian_v2_coco_x2` | Cambrian coco, v2, 12 shards | 634,889 | 2 | 1,269,778 | (above) |
| `cambrian_v2_ocr_vqa_x2` | Cambrian ocr_vqa, v2, 4 shards | 317,675 | 2 | 635,350 | (above) |
| `cambrian_vg_x2` | the pretraining pack `cambrian_vg` | 336,224 | 2 | 672,448 | 0 |
| `cambrian_vizwiz_x2` | the pretraining pack `cambrian_vizwiz_x4` | 20,128 | 2 | 40,256 | 0 |
| `cambrian_coco_valrows` | the pretraining pack | 32,010 | 1 | 32,010 | 0 |
| `cambrian_mcq_x10` | Cambrian coco, v1, only multiple-choice rows (`--only-mcq`) | 43,357 | 10 | 433,570 | 8 GB |
| `scienceqa_train_mcq_x10` | `convert_scienceqa_mcq.py` + `pack_cambrian.py` | 5,313 | 10 | 53,130 | 0.3 GB |
| `sft_capc_coco2014_x2` | `pack_prompted_captions.py`, concise prompts, answer 1-40 tokens, window 48, from `coco_train2014_caption` | 413,803 | 2 | 827,606 | 69 GB |
| `sft_capc_blip3o_short` | concise, from `blip3o_short_caption_train`: 8 shards, first 156,250 rows of each | 1,250,000 | 1 | 1,250,000 | 217 GB |
| `sft_capd_blip3o_long` | detailed prompts, answer 30-100 tokens (cut at a sentence end), window 101, from `blip3o_long_caption_train`: 8 shards, first 187,500 rows of each | 1,500,000 | 1 | 1,500,000 | 78 GB |
| `sft_capd_blip3o60k_x4` | detailed, from `blip3o_ft60k_r64` | 14,233 | 4 | 56,932 | 27 GB |
| `sft_capd_dalle3_x4` | detailed, from `dalle3_t2i` | 14,896 | 4 | 59,584 | 27 GB |
| `sft_long_all` | Cambrian, v2, answers 9-100 tokens, cue rows dropped, window 101: allava (11 shards) 671,552; rows of the jsonl `source` "sharegpt4v*" on 11 image folders (8 shards) 1,042,950; llava (1 shard) 3,719 | 1,718,221 | 1 | 1,718,221 | 309 GB |

A replica (`_xN`) lists N hard links of each base shard (`data/wds/links/<name>/c<NN>/`).
Hard links use no disk space. They must be on the file system of the shards.
The trainer refuses a file that a manifest lists two times; the links have different paths.

## 5. COCO val images in the training data

Our evaluation uses COCO val2014 and val2017 images (VQAv2, OK-VQA, POPE, COCO captions).
The training data contains some of these images:

1. `cambrian_coco_valrows` (pretraining and SFT): 32,010 Cambrian coco rows whose image is a COCO
   val2014 or val2017 image. We added these rows on purpose. All other coco packs remove these rows.
2. The other Cambrian sources reach COCO val images through other paths. The packer does not
   remove these rows:

| Source | Rows (jsonl) | Unique COCO val images | Packs |
|---|---|---|---|
| tallyqa, `val2014/COCO_val2014_*.jpg` | 67,361 | 32,633 (80 % of val2014; equal bytes) | `cambrian_tallyqa` |
| tallyqa, Visual Genome images with a COCO val `coco_id` | 12,626 | 8,808 | `cambrian_tallyqa` |
| vg, Visual Genome images with a COCO val `coco_id` | 14,010 | 13,812 | `cambrian_vg`, `cambrian_vg_x2` |
| gqa, Visual Genome images with a COCO val `coco_id` | 12,090 (approx. 150,000 pairs) | 11,823 | `cambrian_gqa_x4`, `cambrian_v2_gqa_x2` |
| lnqa (`train_N.jpg`) | 0 of 5,000 sampled files are equal to a val2014 image | 0 found | `cambrian_lnqa_x4` |

The Visual Genome to COCO map is the `coco_id` field of Visual Genome `image_data.json`
(108,077 images; 51,498 with a `coco_id`; 17,471 in val2014 or val2017).
The long-answer pack (`sft_long_all`) removes the coco rows on COCO val images. Its other image
folders are not filtered.

The paper results on benchmarks with COCO val images (VQAv2, OK-VQA, POPE, COCO captions)
come from models that saw these images in training. GenEval, DPG-Bench, GQA test-dev and the
BLIP3o held-out captions do not use COCO val images.
The COCO caption rows (`coco_train2014_caption`) use train2014 only. The packer stops if a
train2014 id is in val2014 or val2017.

## 6. What is not equal byte for byte

- **Row counts of the three BLIP3o pools.** The configs give 29,187,000, 4,728,000 and
  4,197,986 rows. The paper runs used these values. They are not the exact row counts:
  the tars hold 29,202,492, 4,766,724 and 4,109,986 rows.
  Keep the config values: the trainer uses them to give the shards to the workers.
- **JourneyDB.** 14 images of 7 tars do not decode (12 that PIL cannot identify, 2 truncated).
  `clean_journeydb.py` writes the 7 tars without these 14 rows. The list is in the script.
- **BLIP3o-Pretrain-Short-Caption** has no tar `00842` and no tar `01812` at our revision.
  This is the upstream state, not an error.
- **DALL-E 3.** We could not check `stage_dalle3.py` against our pack: we do not have the
  parquets any more. If the upstream dataset changes, `dalle3_t2i` and `sft_capd_dalle3`
  change. Our pack has 19,024 rows in 14 shards (1,420 x 3, 1,416, 1,415 x 8, 1,466, 562).
- **ShareGPT-4o JPEG.** The packer encodes JPEG with Pillow. A different Pillow or libjpeg
  version can give different bytes. The rows and the captions do not change.
- **Upstream changes.** A different revision of a source can change the rows. A different
  `Cambrian7M.jsonl` changes the row keys and the held-out exclusion: compare the sha256 of
  section 7 first.
- **Tar headers.** Our check used Python 3.11. Another Python version can write different tar
  headers. The members do not change.
- **Data order.** The order depends on the manifests and the shard layout. The scripts make the
  layout of the paper runs. A different `--samples-per-tar` or number of shards gives the same
  rows in a different order.

## 7. Reproduction check

We packed slices of the real sources with these scripts and compared them with the shards of
the paper runs:

| Pack | Slice | Result |
|---|---|---|
| `cambrian7m_mmu.jsonl` | first 200,000 source lines | equal lines |
| `cambrian_vg` (v1) | shard 0/8, first tar | equal bytes |
| `cambrian_gqa_x4` (v1) | shard 0/10, 2,500 rows | equal samples, same order |
| `cambrian_coco_valrows` | shard 0/8, all 3,983 rows | equal bytes |
| `cambrian_v2_gqa` (v2) | shard 0/8, first tar | equal bytes |
| `cambrian_mcq` | first tar | equal bytes |
| `sft_long_all` | llava, all 3,719 rows; sharegpt4v shard 0/8, first tar | equal bytes |
| `scienceqa_train_mcq` | all 5,313 rows | equal bytes |
| `vqav2_train_v2` and `_s13` | first tar and its 13 parts | equal bytes |
| `coco_train2014_caption` | first 410 images | equal bytes (first tar) |
| `sft_capc_coco2014` | shard 0/4, first tar | equal bytes |
| `sft_capd_blip3o_long` | shard 0/8, first tar | equal bytes |
| `sharegpt4o_t2i` | shard 0/8 (3 tars, 5,714 rows) | equal bytes |
| `blip3o_ft60k_r64` | shard 0 of 64 | equal bytes |
| `JourneyDB_332_clean` | one tar (19,998 members) | equal bytes |
| all manifests except the five `_x4` Cambrian ones and `cambrian_vizwiz_x2` | written over the packs of the paper runs | same shard files in the same order as the manifests of the paper runs |

We also ran the filters without images (`--count-only`) and compared the row
counts with the configs. The counts are equal for all 17 Cambrian-S8 sources, `cambrian_coco_valrows`,
the three v2 packs, `cambrian_mcq`, the three long-answer groups, VQAv2 (443,317 rows in 45 tars),
`coco_train2014_caption`, `sft_capc_coco2014`, `sft_capd_blip3o60k` and `sft_capd_dalle3`.
`scienceqa_train_mcq` (5,313) is equal in the full pack above.

The five `_x4` Cambrian packs of the paper's pretraining runs have 2,500 rows per tar. Our local copy of
these five packs has a different tar layout, so we compared them by samples (`cambrian_gqa_x4`
above) and by row counts.

sha256 of our input files:

| File | sha256 |
|---|---|
| `Cambrian-10M/jsons/Cambrian7M.jsonl` | `a12df86fa914e7da4ce1ccac17d5812e052f9e8556a8c902520c6b7d68223e59` |
| `Cambrian-10M/jsons/cambrian7m_mmu.jsonl` | `075d7f89fdce3fc35168f14b30c01da44571096e0aa789e39dd5967a39a4c92d` |
| `scripts/data/cambrian7m_mmu_valprobe1024.linehash` | `2bdaa7f203241eb574bef7e5446e927ee53c02465353b06691daf69140514f93` |
| `ScienceQA-IMG/train-00000-of-00001.parquet` | `00c1c569dce474fee0ff6262aa48fe6005f5dc82f1c9a7a581388c9595c89baa` |
| `coco/annotations/captions_train2014.json` | `dd8c9636dc11740f956e36728866ea0c4ebe4988dcbdc5e712b7c2267f152d12` |

## 8. Licenses

We do not supply these datasets. Read the terms of each dataset before you use it.

| Dataset | License |
|---|---|
| BLIP3o-Pretrain-Long-Caption, -Short-Caption, -JourneyDB, BLIP3o-60k | the dataset cards give Apache-2.0. The images come from other sets; check the license at <https://huggingface.co/BLIP3o> |
| Cambrian-10M | the dataset card gives Apache-2.0. Each source dataset has its own terms; check the license at <https://huggingface.co/datasets/nyu-visionx/Cambrian-10M> |
| COCO images and captions | check the terms at <https://cocodataset.org/#termsofuse> |
| VQAv2 (lmms-lab/VQAv2) | check the license at <https://visualqa.org/terms.html> and <https://huggingface.co/datasets/lmms-lab/VQAv2> |
| ScienceQA (lmms-lab/ScienceQA) | check the license at <https://huggingface.co/datasets/lmms-lab/ScienceQA> and <https://scienceqa.github.io> |
| ShareGPT-4o-Image | check the license at <https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image> |
| dalle-3-dataset | check the license at <https://huggingface.co/datasets/OpenDatasets/dalle-3-dataset> |
