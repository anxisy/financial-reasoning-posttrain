#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp
bash "$ROOT/ConvFinQA/00_common/finqa_retention/run_eval.sh" direct \
  "$ROOT/project/convfinqa_posttrain/experiments/convfinqa_sft_v2/02_qwen3_base/full/final_adapter" "$@"
