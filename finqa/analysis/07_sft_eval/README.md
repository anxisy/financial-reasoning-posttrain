# FinQA SFT Evaluation v1

这套代码只在冻结的 Baseline evaluator 基础上增加 PEFT/LoRA adapter 加载。

固定不变：

```text
manual prompt
prompt 结尾 = Assistant:
greedy decoding
max_new_tokens = 384
seed = 42
same answer parser
same numeric / percent tolerances
same processed_v1_1_1 eval data
```

## 1. 解压

```bash
cd /root/autodl-tmp/open-r1
mkdir -p 07_sft_eval
cd 07_sft_eval

unzip finqa_sft_eval_v1.zip
cd finqa_sft_eval_v1
```

## 2. 自检

```bash
python self_test.py
```

必须出现：

```text
SELF TEST PASSED
```

## 3. 先跑 SFT Dev（883 条）

```bash
python evaluate_finqa_peft.py \
  --base-model-path /root/autodl-tmp/models/Qwen3-0.6B-Base \
  --adapter-path /root/autodl-tmp/project/finqa_posttrain/experiments/sft_qwen3_0.6b/full_v1/final_adapter \
  --input /root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1/eval/finqa_eval_dev.jsonl \
  --output-dir /root/autodl-tmp/project/finqa_posttrain/experiments/sft_qwen3_0.6b/eval_dev_v1 \
  --batch-size 2 \
  --max-new-tokens 384 \
  --prompt-format manual
```

中断后可加：

```bash
--resume
```

## 4. 比较 Base Dev vs SFT Dev

```bash
python compare_base_vs_sft.py \
  --base /root/autodl-tmp/project/finqa_posttrain/experiments/base_qwen3_0.6b/dev_manual_384/metrics.json \
  --sft /root/autodl-tmp/project/finqa_posttrain/experiments/sft_qwen3_0.6b/eval_dev_v1/metrics.json
```

先检查 Dev，确认正常后再跑 Test。

## 5. SFT Test

```bash
python evaluate_finqa_peft.py \
  --base-model-path /root/autodl-tmp/models/Qwen3-0.6B-Base \
  --adapter-path /root/autodl-tmp/project/finqa_posttrain/experiments/sft_qwen3_0.6b/full_v1/final_adapter \
  --input /root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1/eval/finqa_eval_test.jsonl \
  --output-dir /root/autodl-tmp/project/finqa_posttrain/experiments/sft_qwen3_0.6b/eval_test_v1 \
  --batch-size 2 \
  --max-new-tokens 384 \
  --prompt-format manual
```

Test 跑完：

```bash
python compare_base_vs_sft.py \
  --base /root/autodl-tmp/project/finqa_posttrain/experiments/base_qwen3_0.6b/test_manual_384/metrics.json \
  --sft /root/autodl-tmp/project/finqa_posttrain/experiments/sft_qwen3_0.6b/eval_test_v1/metrics.json
```

## 6. Dev 跑完后发回来

请发：

```text
eval_dev_v1/metrics.json
eval_dev_v1/run_config.json
```

再执行：

```bash
head -n 10 \
/root/autodl-tmp/project/finqa_posttrain/experiments/sft_qwen3_0.6b/eval_dev_v1/predictions.jsonl
```

把输出发回来。

最后把：

```bash
python compare_base_vs_sft.py ...
```

的终端输出也发回来。
