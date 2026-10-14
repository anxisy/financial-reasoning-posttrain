#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp
cd "$ROOT/ConvFinQA/00_common"
export VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
ADAPTER="$ROOT/project/convfinqa_posttrain/experiments/convfinqa_sft_v2/03_finqa_sft_replay/full/final_adapter"
N="${CONV_MAX_SAMPLES:-0}"
if [[ ! "$N" =~ ^[0-9]+$ ]]; then echo 'CONV_MAX_SAMPLES must be a nonnegative integer' >&2; exit 2; fi
SUFFIX=""
if [[ "$N" != '0' ]]; then SUFFIX="_smoke${N}"; fi
OUT="$ROOT/project/convfinqa_posttrain/experiments/convfinqa_sft_v2/03_finqa_sft_replay/eval_convfinqa_oracle${SUFFIX}"
[[ -f "$ADAPTER/adapter_config.json" ]] || { echo "Missing $ADAPTER" >&2; exit 1; }
[[ ! -f "$OUT/metrics.json" ]] || { echo "Existing evaluation output; refusing overwrite: $OUT" >&2; exit 1; }
# The frozen script labels the experimental variant "transfer"; the adapter and output
# path identify this distinct REPLAY run in run_config.json.
python eval_sft_vllm.py --variant transfer --adapter "$ADAPTER" \
  --output-dir "$OUT" --max-samples "$N" "$@"
