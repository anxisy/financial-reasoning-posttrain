#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/ConvFinQA
python 00_common/finqa_retention/preflight.py
bash 00_common/finqa_retention/run_original_sft.sh
bash 01_finqa_sft_adapter/05_finqa_retention/run.sh
bash 02_qwen3_base/05_finqa_retention/run.sh
if [[ "${FINQA_MAX_SAMPLES:-0}" == "0" ]]; then
  python 00_common/finqa_retention/compare.py --bootstrap 5000
else
  echo "SMOKE MODE: do not analyze as final results; unset FINQA_MAX_SAMPLES and rerun"
fi
