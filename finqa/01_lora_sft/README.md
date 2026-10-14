# FinQA Qwen3-1.7B LoRA-SFT v1

This package scales the already-validated Qwen3-0.6B FinQA SFT recipe to Qwen3-1.7B while keeping the recipe controlled.

## Frozen recipe

- Base: `/root/autodl-tmp/models/Qwen3-1.7B-Base`
- Train: `/root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1/sft/finqa_sft_train.jsonl`
- Dev: `/root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1/sft/finqa_sft_dev.jsonl`
- Assistant-only loss
- Prompt ends exactly in `Assistant:`; target starts with one newline
- max_seq_length=2048, response reserve=384, prompt budget=1664
- LoRA r=8, alpha=16, dropout=0.05
- q/k/v/o + gate/up/down
- batch=1, gradient accumulation=8, effective batch=8
- lr=2e-4, cosine scheduler, warmup=0.05
- epochs=2
- BF16 + gradient checkpointing
- seed=42

## 0. Fix the OMP warning before Python starts

```bash
unset OMP_NUM_THREADS
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

## 1. Environment and serialization checks

```bash
python env_check.py
python self_test.py
```

## 2. Preflight

```bash
python train_sft_1p7b.py --preflight
```

Preflight output:

`/root/autodl-tmp/project/finqa_posttrain/experiments/sft_qwen3_1.7b/preflight_v1`

Then reload it:

```bash
python reload_adapter.py \
  --adapter-path /root/autodl-tmp/project/finqa_posttrain/experiments/sft_qwen3_1.7b/preflight_v1/final_adapter
```

The preflight uses only 128 train rows, 64 dev rows and 12 optimizer steps. It is only a memory/backward/reload check, not an accuracy experiment.

## 3. Full SFT

```bash
python train_sft_1p7b.py
```

Final adapter:

`/root/autodl-tmp/project/finqa_posttrain/experiments/sft_qwen3_1.7b/full_v1/final_adapter`

Training report:

`/root/autodl-tmp/project/finqa_posttrain/experiments/sft_qwen3_1.7b/full_v1/training_report.json`

## 4. Resume if interrupted

```bash
python train_sft_1p7b.py --resume-from-checkpoint latest
```

## 5. Reload final adapter

```bash
python reload_adapter.py
```

## 6. Frozen FinQA Dev evaluation

```bash
python /root/autodl-tmp/open-r1/07_sft_eval/evaluate_finqa_peft.py \
  --base-model-path /root/autodl-tmp/models/Qwen3-1.7B-Base \
  --adapter-path /root/autodl-tmp/project/finqa_posttrain/experiments/sft_qwen3_1.7b/full_v1/final_adapter \
  --input /root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1/eval/finqa_eval_dev.jsonl \
  --output-dir /root/autodl-tmp/project/finqa_posttrain/experiments/sft_qwen3_1.7b/eval_dev_v1 \
  --batch-size 2 \
  --max-new-tokens 384 \
  --prompt-format manual
```

Do not run Test yet. Freeze the 1.7B recipe on Dev first.

## Notes

This intentionally does **not** change LoRA rank, LR, epochs, prompt format, dataset, or evaluator relative to the validated 0.6B SFT recipe. This keeps the scale-up comparison clean.
