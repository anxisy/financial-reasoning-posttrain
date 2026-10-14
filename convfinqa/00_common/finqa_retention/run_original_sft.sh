#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp
bash "$ROOT/ConvFinQA/00_common/finqa_retention/run_eval.sh" original_sft \
  "$ROOT/project/finqa_posttrain/experiments/sft_qwen3_1.7b/full_v1/final_adapter" "$@"
