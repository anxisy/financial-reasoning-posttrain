# Experiment 03 — 10% FinQA Experience Replay (Qwen3-1.7B)

Goal: test whether replaying old FinQA Train samples during continued ConvFinQA SFT mitigates forgetting while keeping ConvFinQA accuracy. This is the only new training run; no GRPO.

## Experimental contract (frozen v1)

| Axis | Exp 01 (existing Transfer) | Exp 03 (Replay) |
| --- | --- | --- |
| Initialization | original FinQA SFT Adapter | **same original FinQA SFT Adapter**, not experiment 01 |
| Data examples | 11,104 ConvFinQA Train | 9,994 ConvFinQA Train + 1,110 FinQA Train |
| Sampling | full ConvFinQA 1 epoch | fixed-seed no-replacement mixture 1 epoch |
| Micro-batch / accum | 1 / 8 | 1 / 8 |
| Optimizer updates | 1388 | 1388 |
| LR / schedule | 5e-5, cosine, 3% warmup | same |
| Max context / bf16 | 4096 / yes | same |
| LoRA | frozen original FinQA SFT initialized r8 | same, trainable |
| SFT loss | Assistant only | same |
| Dev/Test in training | no | no |

The new run deliberately replaces ~10% of ConvFinQA training *instances* with FinQA samples, so the data composition differs. This tests a fixed-budget rehearsal configuration. It does **not** test adding more examples while keeping all ConvFinQA samples.

FinQA originals retain their original single-turn prompts and SFT targets. ConvFinQA originals retain their multi-turn Oracle-history prompts and type-aware v2 targets. No future turns or current gold labels are injected into the prompts. No runtime truncation: any oversized example fails loudly.

**GPU:** one RTX 4090 24GB, previous 01 peak was 22.56 GiB. Shut down other GPU jobs. Same configuration does not guarantee no OOM: smoke first.

## Install

Upload `ConvFinQA_03_replay_v1.zip` to `/root/autodl-tmp`.

```bash
cd /root/autodl-tmp
unzip -o ConvFinQA_03_replay_v1.zip
conda activate openr1
cd /root/autodl-tmp/ConvFinQA
python 03_finqa_sft_replay/self_test.py
```

Only **`ConvFinQA/03_finqa_sft_replay/`** is added; 00_common, 01, 02 and their existing outputs are not touched.

## Stage 1: generate, freeze and verify training mix (CPU)

```bash
python 03_finqa_sft_replay/01_data/build_replay.py
python 03_finqa_sft_replay/01_data/verify_replay.py
```

Paths:
- ConvFinQA SFT: `/root/autodl-tmp/project/convfinqa_posttrain/data/processed_v2/sft/train.jsonl`
- FinQA SFT: `/root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1/sft/finqa_sft_train.jsonl`
- Mixed training file: `/root/autodl-tmp/project/convfinqa_posttrain/experiments/convfinqa_sft_v2/03_finqa_sft_replay/data/replay_train.jsonl`
- `mix_manifest.json`: sampling seed, counts, original file SHA256, output SHA256.

The builder refuses to overwrite the frozen mixture by default. Do not change the seed or rebuild the mixture after training. **No Dev or Test** examples are sampled.

Expected: 11,104 rows = 9,994 ConvFinQA + 1,110 FinQA (9.996%); 1,388 optimizer updates with batch=1/accum=8.

`verify_replay.py` uses the original Base tokenizer to inspect *all* 11,104 complete sequences and fails if any exceeds 4096 tokens. It also verifies Assistant-only loss labels and correct mixture proportions.

## Stage 2: GPU smoke

```bash
bash 03_finqa_sft_replay/02_smoke/run.sh
```

Training from the *original* FinQA SFT Adapter. Smoke: 32 examples, both sources, 5 optimizer steps, explicit longest sampled-example backward test and 8 ConvFinQA Dev examples. Verify finite train/eval loss, no OOM, trainable LoRA, adapter output.

Smoke Adapter stays under `.../03_finqa_sft_replay/smoke/final_adapter` and **must not be used for official evaluation**.

## Stage 3: full SFT (GPU, conda openr1)

```bash
mkdir -p /root/autodl-tmp/ConvFinQA/logs
set -o pipefail
bash 03_finqa_sft_replay/03_full/run.sh 2>&1 | tee logs/03_replay_full.log
```

Expected 1 epoch, **1388 optimizer steps**. The training code rejects inconsistent budgets and does not overwrite a finished run. It saves Trainer checkpoints every 250 steps to `.../03_finqa_sft_replay/full/` for recovery and the final adapter to `full/final_adapter`. If interrupted, after investigating the issue run `bash 03_finqa_sft_replay/03_full/run.sh --resume latest`; do not use this on an already finished run.

## Stage 4: evaluate with frozen scorers (vLLM environment)

```bash
conda deactivate
source /root/autodl-tmp/open-r1/19_vllm_benchmark/.venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd /root/autodl-tmp/ConvFinQA
```

Optional smoke inference (will save to separate smoke output directories):

```bash
CONV_MAX_SAMPLES=20 bash 03_finqa_sft_replay/04_convfinqa_oracle/run.sh
FINQA_MAX_SAMPLES=20 bash 03_finqa_sft_replay/05_finqa_retention/run.sh
bash 03_finqa_sft_replay/06_self_history/run.sh --max-conversations 5
```

Full Oracle ConvFinQA Dev (1490 turns), full FinQA Dev (883 questions), full model-answer-history ConvFinQA Dev (421 conversations):

```bash
bash 03_finqa_sft_replay/04_convfinqa_oracle/run.sh
bash 03_finqa_sft_replay/05_finqa_retention/run.sh
bash 03_finqa_sft_replay/06_self_history/run.sh
```

The self-history wrapper reuses the **previously installed** `00_common/eval_self_history_v1.py` and patches only the Adapter map in memory. It sends **one** variant, `replay`; no modifications to the old evaluator. In self-history, each previous *model-generated final answer* is supplied to subsequent turns; not entire response text.

Input fingerprint safeguards are implemented in the post-eval comparison step. Original FinQA Dev SHA of encoded prompt+IDs is `752ba9302ba1153bbdcae95d0d9e743b1f00576ab75761939e821dbd678a58af`; ConvFinQA frozen Oracle Dev is `4fb737022d435bb472c316bb1b5de7776b8238a623bbc1efe900767eab09952f`.

**Do not evaluate private Test**, and do not tune Replay fraction from results on Dev without clearly marking the analysis exploratory.

## Stage 5: compare results (CPU, after both full Oracle & FinQA evaluations)

```bash
python 03_finqa_sft_replay/07_analysis/compare_replay.py --bootstrap 5000 --seed 42
```

This requires the previously saved original FinQA SFT / Transfer / Direct FinQA Dev predictions and 01/02 ConvFinQA Oracle predictions. It checks the exact prompt fingerprints, ID alignment, gold labels and bootstraps over source PDF in FinQA and conversation IDs in ConvFinQA. Outputs a report to `.../03_finqa_sft_replay/analysis/replay_comparison.json`.

**Interpretation:** Replay should improve FinQA Dev retention relative to 38.96% (prior Transfer), while preserving as much as possible of the 81.48% ConvFinQA Oracle Dev and 73.42% model-answer-history Dev. A tradeoff is possible; none of these gains is guaranteed. Original pre-Conv FinQA SFT Dev is 55.83% and serves as the forgetting reference. Comparisons within an eval task are meaningful; FinQA and ConvFinQA raw accuracies are not directly comparable because their task/conditions differ.

The full **Dev** history and repeated analysis mean these are development results, not untouched private Test or proof of production reliability.
