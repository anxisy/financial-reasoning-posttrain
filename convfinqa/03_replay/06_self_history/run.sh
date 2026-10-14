#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp
export VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
python "$ROOT/ConvFinQA/03_finqa_sft_replay/06_self_history/run_replay.py" "$@"