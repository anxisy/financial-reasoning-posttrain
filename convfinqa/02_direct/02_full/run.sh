#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/ConvFinQA
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
python 00_common/train_sft.py --variant direct --epochs 1 --learning-rate 5e-5 --grad-accum 8 --save-steps 250 \
  --output-dir /root/autodl-tmp/project/convfinqa_posttrain/experiments/convfinqa_sft_v2/02_qwen3_base/full "$@"
