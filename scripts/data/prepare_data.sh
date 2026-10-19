#!/usr/bin/env bash
# Build all training packs and manifests of configs/train/*.yaml (docs/DATA.md).
# Run it from the root of the repository, after the downloads of docs/DATA.md section 2.
#
#   STEP=all JOBS=16 bash scripts/data/prepare_data.sh
#
# Env:
#   DATA   data folder (default: data). Downloads in $DATA/src, packs in $DATA/wds,
#          manifests in $DATA/manifests (the configs read data/manifests).
#   JOBS   packer processes that run at the same time (default 8).
#   STEP   one of: cambrian_jsonl s8 valrows coco_caption journeydb blip3o60k dalle3 sharegpt4o
#          vqav2 mcq scienceqa v2 long manifests captions all (default all, in this order).
#   PY     python (default python).
# A step skips each pack that already has its summary file. Logs go to $DATA/logs.
set -euo pipefail
PY=${PY:-python}; DATA=${DATA:-data}; JOBS=${JOBS:-8}; STEP=${STEP:-all}
SRC=$DATA/src; WDS=$DATA/wds; MANI=$DATA/manifests; LOGS=$DATA/logs
CAMB=$SRC/Cambrian-10M; COCO=$SRC/coco
mkdir -p "$WDS" "$MANI" "$LOGS"
export OMP_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH:-.}
JOBLIST=$(mktemp)
step() { [ "$STEP" = "$1" ] || [ "$STEP" = all ]; }
job() { # log_name summary_file command...: queue the command unless its summary exists
  local log=$1 done_file=$2; shift 2
  [ -f "$done_file" ] && { echo "skip $log (done)"; return; }
  printf '%q ' "$@" >> "$JOBLIST"; printf '> %q 2>&1\n' "$LOGS/$log.log" >> "$JOBLIST"
}
run_jobs() {
  if [ -s "$JOBLIST" ]; then
    xargs -P "$JOBS" -d '\n' -I{} bash -c '{}' < "$JOBLIST" || { echo "a packer failed; see the logs in $LOGS"; exit 1; }
  fi
  : > "$JOBLIST"
}
camb() { echo "$PY -m scripts.data.pack_cambrian --root $CAMB --coco-val2014 $COCO/val2014 --coco-val2017 $COCO/val2017"; }
LONG="--min-answer-tokens 9 --max-answer-tokens 100 --truncate-answers sentence --qa-window 101 --keep-prompt-suffix --skip-cued-rows"

if step cambrian_jsonl && [ ! -f "$CAMB/jsons/cambrian7m_mmu.jsonl" ]; then
  $PY -m scripts.data.make_cambrian_mmu_jsonl --root "$CAMB"
fi

cambrian_shards() { # group out_root prefix_stem n_shards [packer args...]
  local group=$1 root=$2 stem=$3 n=$4; shift 4
  for ((k = 0; k < n; k++)); do s=s$(printf %02d $k)
    job "${stem}_$s" "$root/$s/pack_summary.json" $(camb) --shard "$k/$n" --out "$root/$s" --tar-prefix "${stem}_$s" "$@"
  done
}
if step s8; then  # Pretraining: 17 Cambrian sources, v1 prompts, answers 1..8 tokens
  for item in gqa:10 coco:7 lnqa:3 vizwiz:1 oodvqa:1 vg:8 dvqa:8 ocr_vqa:4 clevr:4 tallyqa:3 hfdata:4 \
              Q-Instruct-DB:3 docvqa:1 pathvqa:1 chartqa:1 ai2d:1 scienceqa:1; do
    src=${item%%:*}; cambrian_shards s8 "$WDS/cambrian-s8/$src" "cambrian_$src" "${item##*:}" --sources "$src"
  done
fi
if step valrows; then  # Pretraining and SFT: the coco rows on COCO val images (docs/DATA.md section 5)
  cambrian_shards valrows "$WDS/cambrian-s8/coco_valrows" cambrian_coco_valrows 8 --sources coco --coco-val-mode only
fi
if step coco_caption; then
  job coco_caption "$WDS/coco_train2014_caption/pack_summary.json" $PY -m scripts.data.pack_coco_captions \
    --captions "$COCO/annotations/captions_train2014.json" --images-dir "$COCO/train2014" \
    --val2014-dir "$COCO/val2014" --val2017-dir "$COCO/val2017" --out "$WDS/coco_train2014_caption"
fi
if step journeydb; then
  $PY -m scripts.data.clean_journeydb --src "$SRC/BLIP3o-Pretrain-JourneyDB" --out "$WDS/journeydb_clean"
fi
if step blip3o60k; then
  job blip3o60k "$WDS/blip3o_ft60k_r64/pack_summary.json" $PY -m scripts.data.repack_blip3o60k \
    --src "$SRC/BLIP3o-60k" --out "$WDS/blip3o_ft60k_r64"
fi
if step dalle3; then
  job dalle3 "$WDS/dalle3_t2i/pack_summary.json" $PY -m scripts.data.stage_dalle3 --src "$SRC/dalle-3-dataset" --out "$WDS/dalle3_t2i"
fi
if step sharegpt4o; then
  for k in 0 1 2 3 4 5 6 7; do
    job "sharegpt4o_s$k" "$WDS/sharegpt4o_t2i/summary_s0$k.json" $PY -m scripts.data.pack_sharegpt4o \
      --src "$SRC/ShareGPT-4o-Image" --shard "$k/8" --out "$WDS/sharegpt4o_t2i"
  done
fi
if step vqav2; then
  job vqav2 "$WDS/vqav2_train_v2/pack_summary.json" $PY -m scripts.data.pack_vqav2 \
    --parquet-dir "$SRC/VQAv2/data" --out "$WDS/vqav2_train_v2"
fi
if step mcq; then
  job cambrian_mcq "$WDS/cambrian_mcq/pack_summary.json" $(camb) --sources coco --only-mcq \
    --out "$WDS/cambrian_mcq" --tar-prefix cambrian_mcq
fi
if step scienceqa; then
  [ -f "$WDS/scienceqa_mcq_src/jsons/scienceqa_train_mcq.jsonl" ] || $PY -m scripts.data.convert_scienceqa_mcq \
    --parquet "$SRC/ScienceQA/ScienceQA-IMG/train-00000-of-00001.parquet" --root "$WDS/scienceqa_mcq_src"
  job scienceqa_mcq "$WDS/scienceqa_train_mcq/pack_summary.json" $PY -m scripts.data.pack_cambrian \
    --root "$WDS/scienceqa_mcq_src" --jsonl jsons/scienceqa_train_mcq.jsonl --exclude-linehash "" \
    --sources scienceqa_mcq --coco-val-mode keep --out "$WDS/scienceqa_train_mcq" --tar-prefix scienceqa_train_mcq
fi
if step v2; then  # SFT: v2 prompts (the question keeps the LLaVA cue)
  for item in gqa:8 coco:12 ocr_vqa:4; do
    src=${item%%:*}; cambrian_shards v2 "$WDS/cambrian-v2/$src" "cambrian_v2_$src" "${item##*:}" --sources "$src" --keep-prompt-suffix
  done
fi
if step long; then  # SFT: long answers (9..100 tokens, cut at a sentence end)
  cambrian_shards long "$WDS/sft_long/allava" sft_long_allava 11 --sources allava $LONG
  cambrian_shards long "$WDS/sft_long/sharegpt4v" sft_long_sharegpt4v 8 --source-file-prefix sharegpt4v \
    --sources coco,gqa,vg,ocr_vqa,textvqa,sam,llava,wikiart,share_textvqa,web-celebrity,web-landmark $LONG
  cambrian_shards long "$WDS/sft_long/llava" sft_long_llava 1 --sources llava $LONG
fi
run_jobs
if step vqav2; then  # the paper trains on tars 00000-00042, each split into 13 parts
  for i in $(seq 0 42); do t=$(printf "vqav2_train_%05d" "$i")
    [ -f "$WDS/vqav2_train_v2_s13/${t}_12.tar" ] || $PY -m scripts.data.split_shards --parts 13 \
      --out "$WDS/vqav2_train_v2_s13" "$WDS/vqav2_train_v2/$t.tar"
  done
fi

manifests() { $PY -m scripts.data.make_manifests --src "$SRC" --wds "$WDS" --out "$MANI" \
                --jmaze "$DATA/jmaze" --vsp "$DATA/vsp" "$@"; }
if step manifests || step captions; then manifests; fi
if step captions; then  # SFT: prompted captions; the source manifests come from the step above
  cap() { # name source_manifest n_shards rows_per_shard style args...
    local name=$1 man=$2 n=$3 lim=$4; shift 4
    [ -f "$man" ] || { echo "missing $man"; exit 1; }
    for ((k = 0; k < n; k++)); do s=s$(printf %02d $k)
      job "sft_${name}_$s" "$WDS/sft_cap/$name/$s/pack_summary.json" $PY -m scripts.data.pack_prompted_captions \
        --src-manifest "$man" --shard "$k/$n" --limit-rows "$lim" --source-name "$name" \
        --out "$WDS/sft_cap/$name/$s" --tar-prefix "sft_${name}_$s" "$@"
    done
  }
  CONCISE="--style concise --qa-window 48 --max-answer-tokens 40"
  DETAILED="--style detailed --qa-window 101 --min-answer-tokens 30 --max-answer-tokens 100 --truncate sentence"
  cap capc_coco2014 "$MANI/coco_train2014_caption.txt" 4 0 $CONCISE
  cap capc_blip3o_short "$MANI/blip3o_short_caption_train.txt" 8 156250 $CONCISE
  cap capd_blip3o_long "$MANI/blip3o_long_caption_train.txt" 8 187500 $DETAILED
  cap capd_blip3o60k "$MANI/blip3o_ft60k_r64.txt" 1 0 $DETAILED
  cap capd_dalle3 "$MANI/dalle3_t2i.txt" 1 0 $DETAILED
  run_jobs
  manifests --strict
fi
rm -f "$JOBLIST"
