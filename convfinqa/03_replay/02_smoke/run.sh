#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp
cd "$ROOT/ConvFinQA"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
python 03_finqa_sft_replay/train_replay.py \
  --variant transfer --smoke --smoke-steps 5 \
  --grad-accum 2 --smoke-train-rows 32 --smoke-eval-rows 8 \
  --max-seq-length 4096 \
  --output-dir "$ROOT/project/convfinqa_posttrain/experiments/convfinqa_sft_v2/03_finqa_sft_replay/smoke" "$@"
