# Final-eval patch v1.1 — FinQA metric schema correction

Bug: The original `20_finqa_vllm_benchmark_v2/metrics.py` emits
`metrics.overall` with `n` and `accuracy` but *no* `correct` field.
The first final-eval `04_final_evaluation/common.py` incorrectly required
`overall.correct`, rejecting all successful FinQA Test runs after inference.

Patch modifies ONLY:
- `common.py`: verify computed count and numeric accuracy from saved predictions,
  plus strict/format/answer-present/max-token rates and frozen vLLM metadata;
  tolerate absent `overall.correct` (but verify it if present).
- `recover_finqa_test.py`: recover already completed temporary Test runs after
  running the full validator (same prompt fingerprint, weights, scoring).

Run:

```bash
cd /root/autodl-tmp
unzip -o ConvFinQA_final_eval_fix_v1_1.zip
cd /root/autodl-tmp/ConvFinQA
python 04_final_evaluation/recover_finqa_test.py original_sft
bash 04_final_evaluation/01_run_finqa_test.sh all
```

Do not delete the `.running_full_test_...` output. Do not change
`bench_vllm.py`, `metrics.py`, the scoring rule, or any Adapter. This fix
changes *verification only*, not the actual benchmark numbers.
