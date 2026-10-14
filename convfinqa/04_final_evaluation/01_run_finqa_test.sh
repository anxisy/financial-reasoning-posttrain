#!/usr/bin/env bash
# Four frozen FinQA Test evaluations, each using the original frozen vLLM scorer.
set -euo pipefail
ROOT="${AUTODL_ROOT:-/root/autodl-tmp}"
PROJECT="$ROOT/ConvFinQA"
EXP="$ROOT/project/convfinqa_posttrain/experiments/convfinqa_sft_v2"
BENCH="$ROOT/open-r1/20_finqa_vllm_benchmark_v2/bench_vllm.py"
TEST="${FINQA_TEST_INPUT:-$ROOT/project/finqa_posttrain/data/processed_v1_1_1/eval/finqa_eval_test.jsonl}"
OUT="$EXP/04_final_evaluation_v1/finqa_test"
BASE="$ROOT/models/Qwen3-1.7B-Base"
VARIANTS=(original_sft transfer direct replay)
if [[ ${1:-all} != all ]]; then
  case "$1" in original_sft|transfer|direct|replay) VARIANTS=("$1");; *) echo 'Usage: 01_run_finqa_test.sh [all|original_sft|transfer|direct|replay]' >&2; exit 2;; esac
fi
export VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
# Preflight must have been written and validated prior to Test.
python "$PROJECT/04_final_evaluation/00_preflight.py" --write
for v in "${VARIANTS[@]}"; do
  case "$v" in
    original_sft) ADAPTER="$ROOT/project/finqa_posttrain/experiments/sft_qwen3_1.7b/full_v1/final_adapter";;
    transfer) ADAPTER="$EXP/01_finqa_sft_adapter/full/final_adapter";;
    direct) ADAPTER="$EXP/02_qwen3_base/full/final_adapter";;
    replay) ADAPTER="$EXP/03_finqa_sft_replay/full/final_adapter";;
  esac
  DEST="$OUT/$v/full_test"
  if [[ -e "$DEST" ]]; then
    # Never overwrite, and do not silently accept partial directories.
    if [[ ! -f "$DEST/predictions.jsonl" || ! -f "$DEST/metrics.json" || ! -f "$DEST/benchmark.json" ]]; then
      echo "PARTIAL output detected; STOP. Inspect and back it up before retrying: $DEST" >&2; exit 1
    fi
    python "$PROJECT/04_final_evaluation/verify_one_test.py" "$DEST" "$v"
    echo "[SKIP verified existing] $v: $DEST"
    continue
  fi
  mkdir -p "$OUT/$v"
  TMP="$(mktemp -d "$OUT/$v/.running_full_test_XXXXXXXX")"
  echo "======== FinQA Test $v (1147 rows) ========"
  echo "Temporary output: $TMP"
  # Use same prompt, numeric evaluator, sampling and batch settings as FinQA Dev retention.
  set +e
  python "$BENCH" \
    --base "$BASE" --adapter "$ADAPTER" --input "$TEST" --output-dir "$TMP" \
    --max-samples 0 --request-batch-size 0 --warmup-samples 4 \
    --max-new-tokens 384 --max-model-len 4096 --max-num-seqs 32 \
    --gpu-memory-utilization 0.70 --seed 42 > "$TMP/inference.log" 2>&1
  status=$?
  set -e
  tail -n 22 "$TMP/inference.log" || true
  if [[ $status -ne 0 ]]; then
    echo "FAILED $v (code $status); kept temporary directory for debugging: $TMP" >&2; exit "$status"
  fi
  python "$PROJECT/04_final_evaluation/verify_one_test.py" "$TMP" "$v"
  if [[ -e "$DEST" ]]; then echo "Race: destination appeared: $DEST" >&2; exit 1; fi
  mv -T "$TMP" "$DEST"
  echo "[SAVED] $DEST"
done
