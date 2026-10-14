#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/ConvFinQA
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
python 00_common/train_sft.py --variant transfer --smoke --smoke-steps 5 --grad-accum 2 --smoke-train-rows 32 --smoke-eval-rows 8 \
  --output-dir /root/autodl-tmp/project/convfinqa_posttrain/experiments/convfinqa_sft_v2/01_finqa_sft_adapter/smoke "$@"
