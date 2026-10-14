#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/ConvFinQA/00_common
export VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
python eval_sft_vllm.py --variant transfer \
  --adapter /root/autodl-tmp/project/convfinqa_posttrain/experiments/convfinqa_sft_v2/01_finqa_sft_adapter/full/final_adapter \
  --output-dir /root/autodl-tmp/project/convfinqa_posttrain/experiments/convfinqa_sft_v2/01_finqa_sft_adapter/eval_dev "$@"
