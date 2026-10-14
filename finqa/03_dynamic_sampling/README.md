# FinQA Qwen3-1.7B GRPO + Capped Dynamic Sampling v1

**Purpose:** independent scale-up comparison with Qwen3-1.7B Vanilla GRPO. Starts from the SAME **original 1.7B Full SFT adapter** as Vanilla, **never** from the Vanilla final adapter.

## Hard constraints

- Base: `/root/autodl-tmp/models/Qwen3-1.7B-Base`
- Init SFT: `/root/autodl-tmp/project/finqa_posttrain/experiments/sft_qwen3_1.7b/full_v1/final_adapter`
- Frozen FinQA Train 6251: `/root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1/grpo/finqa_grpo_train.jsonl`
- Preflight: `/root/autodl-tmp/project/finqa_posttrain/experiments/grpo_dynamic_qwen3_1.7b/preflight_v1`
- Full: `/root/autodl-tmp/project/finqa_posttrain/experiments/grpo_dynamic_qwen3_1.7b/full_v1`
- TRL==0.18.0, torch==2.6.0+cu124, transformers==4.52.3, PEFT==0.21.0 (same environment)
- one RTX 4090 (24GiB); explicit frozen initial SFT reference policy; no vLLM
- G=4, max 4 candidate groups per optimizer update; accepts mixed correctness, fallback fourth zero-variance group
- LR=5e-6, warmup=0.1, beta=0.04, epsilon=0.2, temperature=0.8, top_p=0.95
- batch=4, grad_acc=1, prompt_max=1792, completion_max=256, 1 epoch, seed=42
- group-standardized GRPO advantage, reward = correctness 1.0 + format 0.1
- max_grad_norm=1.0 explicitly pinned to frozen Transformers Trainer default

**The only algorithmic difference relative to 1.7B Vanilla is Capped Dynamic Sampling.**  We use the proven 0.6B implementation; no DAPO clip-higher, no reward shaping, no DAA, no Learnability, no evidence prompt, no Agent.

## 1. Install / unpack

Upload this zip to `/root/autodl-tmp/open-r1/`, then:

```bash
cd /root/autodl-tmp/open-r1
mkdir -p 18_dynamic_1p7b
cd 18_dynamic_1p7b
unzip ../finqa_qwen3_1p7b_dynamic_sampling_v1.zip
cd finqa_qwen3_1p7b_dynamic_sampling_v1
```

## 2. Preflight

```bash
unset OMP_NUM_THREADS
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
python env_check.py
python self_test.py
python train_dynamic_1p7b.py --preflight
```

Preflight uses a stratified 64-prompt loader and 12 optimizer steps, G=4, max attempts 4. It tests actual CUDA generation, gradient, reference KL, LoRA updates, filtered candidates, and final adapter save. It is NOT a performance evaluation.

Confirm: `DYNAMIC SAMPLING PREFLIGHT COMPLETED`, no OOM/NaN/Inf, `Reference mode = explicit_frozen_sft_copy`, `accepted_mixed_groups > 0`, `total_candidate_groups > optimizer_steps`, `changed_trainable_tensors > 0`, finite KL/grad norms.

If the 1.7B Preflight OOMs, **DO NOT start Full or silently change G, max prompt, beta, model precision, or reference policy**. Diagnose first.

## 3. Full run

After Preflight passes:

```bash
python train_dynamic_1p7b.py
```

This starts from **the original 1.7B SFT**, not from Vanilla GRPO. Normal behavior: due to capped rejection sampling, one optimizer step uses 1-4 candidate groups; rollout cost can be ~2x-3x Vanilla. On a single 4090, budget considerably more time than the prior 1.7B Vanilla GRPO run; no reliable runtime estimate until full has warmed up.

Output:

`/root/autodl-tmp/project/finqa_posttrain/experiments/grpo_dynamic_qwen3_1.7b/full_v1`

Files:

`run_config.json`, `dynamic_sampling_events.jsonl`, `group_diagnostics.jsonl`, `rollout_diagnostics.jsonl`, `dynamic_sampling_summary.json`, `diagnostics_summary.json`, `trainer_log_history.json`, `full_report.json`, `final_adapter/`, `checkpoint-*`.

## 4. Resume from checkpoint (IMPORTANT)

If AutoDL stops at optimizer step > last checkpoint, custom JSONL logs include future events; trim them before resuming to avoid duplicated diagnostics.

```bash
OUT=/root/autodl-tmp/project/finqa_posttrain/experiments/grpo_dynamic_qwen3_1.7b/full_v1
python prepare_dynamic_resume.py --output-dir "$OUT" --checkpoint latest
python train_dynamic_1p7b.py --resume-from-checkpoint latest
```

This trims `dynamic_sampling_events.jsonl`, `group_diagnostics.jsonl`, `rollout_diagnostics.jsonl` to the last checkpoint (backs up originals). Do not use `--overwrite-output-dir` when resuming. The helper is idempotent for an already trimmed checkpoint.

## 5. Reload

```bash
python reload_dynamic_1p7b.py
```

## 6. Frozen Dev evaluation (not Test)

```bash
python /root/autodl-tmp/open-r1/07_sft_eval/evaluate_finqa_peft.py \
  --base-model-path /root/autodl-tmp/models/Qwen3-1.7B-Base \
  --adapter-path /root/autodl-tmp/project/finqa_posttrain/experiments/grpo_dynamic_qwen3_1.7b/full_v1/final_adapter \
  --input /root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1/eval/finqa_eval_dev.jsonl \
  --output-dir /root/autodl-tmp/project/finqa_posttrain/experiments/grpo_dynamic_qwen3_1.7b/eval_dev_v1 \
  --batch-size 2 \
  --max-new-tokens 384 \
  --prompt-format manual
```

**Known Dev baselines:** 1.7B Base 35.67%, SFT 55.49%, Vanilla GRPO 58.78%.

## 7. Compare Dev

```bash
python compare_metrics.py \
 --before /root/autodl-tmp/project/finqa_posttrain/experiments/grpo_qwen3_1.7b/eval_dev_v1/metrics.json \
 --after /root/autodl-tmp/project/finqa_posttrain/experiments/grpo_dynamic_qwen3_1.7b/eval_dev_v1/metrics.json \
 --before-name Vanilla-GRPO-1.7B \
 --after-name Dynamic-Sampling-1.7B
```

## 8. Post-hoc KL/grad_norm outliers

```bash
python analyze_stability.py --output-dir /root/autodl-tmp/project/finqa_posttrain/experiments/grpo_dynamic_qwen3_1.7b/full_v1
```

**Do not interpret higher effective update ratio alone as superior overall efficiency**: additional candidate rollouts incur time/memory cost. Compare Dev accuracy, filtered/fallback counts, candidate rollout multiplier, GPU hours, and KL/grad outliers with the Vanilla baseline. Test split remains frozen until method choice.
