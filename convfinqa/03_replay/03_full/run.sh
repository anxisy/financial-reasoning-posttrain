#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp
cd "$ROOT/ConvFinQA"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
python 03_finqa_sft_replay/train_replay.py \
  --variant transfer --epochs 1 --learning-rate 5e-5 \
  --batch-size 1 --grad-accum 8 --save-steps 250 --max-seq-length 4096 \
  --output-dir "$ROOT/project/convfinqa_posttrain/experiments/convfinqa_sft_v2/03_finqa_sft_replay/full" "$@"
