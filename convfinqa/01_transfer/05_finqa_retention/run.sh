#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp
bash "$ROOT/ConvFinQA/00_common/finqa_retention/run_eval.sh" transfer \
  "$ROOT/project/convfinqa_posttrain/experiments/convfinqa_sft_v2/01_finqa_sft_adapter/full/final_adapter" "$@"
