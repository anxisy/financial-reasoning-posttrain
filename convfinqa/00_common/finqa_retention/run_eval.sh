#!/usr/bin/env bash
set -euo pipefail
# usage: run_eval.sh original_sft|transfer|direct PATH_TO_ADAPTER [additional bench_vllm.py args]
if [[ $# -lt 2 ]]; then echo "Usage: $0 VARIANT ADAPTER [extra args]" >&2; exit 2; fi
VARIANT="$1"; ADAPTER="$2"; shift 2
case "$VARIANT" in original_sft|transfer|direct) ;; *) echo "Bad variant: $VARIANT" >&2; exit 2;; esac
ROOT=/root/autodl-tmp
BENCH="$ROOT/open-r1/20_finqa_vllm_benchmark_v2"
BASE="$ROOT/models/Qwen3-1.7B-Base"
DEV="$ROOT/project/finqa_posttrain/data/processed_v1_1_1/eval/finqa_eval_dev.jsonl"
OUT_ROOT="$ROOT/project/convfinqa_posttrain/experiments/convfinqa_sft_v2/finqa_retention_v1"
N="${FINQA_MAX_SAMPLES:-0}"
if [[ ! "$N" =~ ^[0-9]+$ ]]; then echo "FINQA_MAX_SAMPLES must be a nonnegative integer" >&2; exit 2; fi
if [[ "$N" == 0 ]]; then LABEL=full_dev; else LABEL="smoke_${N}"; fi
OUT="$OUT_ROOT/$VARIANT/$LABEL"
for f in "$BENCH/bench_vllm.py" "$ADAPTER/adapter_config.json" "$DEV"; do
  if [[ ! -f "$f" ]]; then echo "Missing: $f" >&2; exit 1; fi
done
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"
echo "Variant=$VARIANT Samples=$N Adapter=$ADAPTER Output=$OUT"
python "$BENCH/bench_vllm.py" \
  --base "$BASE" \
  --adapter "$ADAPTER" \
  --input "$DEV" \
  --output-dir "$OUT" \
  --max-samples "$N" \
  --request-batch-size 0 \
  --warmup-samples 4 \
  --max-new-tokens 384 \
  --max-model-len 4096 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.70 \
  --seed 42 \
  "$@"
