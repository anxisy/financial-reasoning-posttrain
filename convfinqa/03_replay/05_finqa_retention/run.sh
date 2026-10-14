#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp
ADAPTER="$ROOT/project/convfinqa_posttrain/experiments/convfinqa_sft_v2/03_finqa_sft_replay/full/final_adapter"
BENCH="$ROOT/open-r1/20_finqa_vllm_benchmark_v2/bench_vllm.py"
DEV="$ROOT/project/finqa_posttrain/data/processed_v1_1_1/eval/finqa_eval_dev.jsonl"
N="${FINQA_MAX_SAMPLES:-0}"
if [[ ! "$N" =~ ^[0-9]+$ ]]; then echo 'FINQA_MAX_SAMPLES must be a nonnegative integer' >&2; exit 2; fi
if [[ "$N" == '0' ]]; then SUB='full_dev'; else SUB="smoke_${N}"; fi
OUT="$ROOT/project/convfinqa_posttrain/experiments/convfinqa_sft_v2/03_finqa_sft_replay/eval_finqa/$SUB"
for f in "$ADAPTER/adapter_config.json" "$BENCH" "$DEV"; do [[ -f "$f" ]] || { echo "Missing $f" >&2; exit 1; }; done
[[ ! -f "$OUT/metrics.json" ]] || { echo "Existing evaluation output; refusing overwrite: $OUT" >&2; exit 1; }
mkdir -p "$OUT"
export VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
python "$BENCH" --base "$ROOT/models/Qwen3-1.7B-Base" --adapter "$ADAPTER" \
  --input "$DEV" --output-dir "$OUT" \
  --max-samples "$N" --request-batch-size 0 --warmup-samples 4 \
  --max-new-tokens 384 --max-model-len 4096 --max-num-seqs 32 \
  --gpu-memory-utilization 0.70 --seed 42 "$@"
