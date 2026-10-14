# FinQA Baseline Evaluation v1.1

这是对你当前 Baseline smoke-test 的修正版。

## 本次修复

### 1. 修复 `<answer>...</answer>` 误判

旧版会把模型复述提示词中的：

```text
<answer>...</answer>
```

当成真实答案。

v1.1 会把 `...` 等占位符排除，并按以下顺序抽取答案：

```text
真实 <answer>...</answer>
→ \boxed{...}
→ Final answer / Answer: ...
→ 最后一行
```

`format_ok=True` 只代表模型真的生成了有效 `<answer>...</answer>`。

### 2. 新增生成诊断

metrics.json 新增：

- answer_found_rate
- max_token_hit_rate
- role_repetition_rate
- by_extraction_method

这样可以判断 Base 的问题到底是：
- 不会算
- 没输出完整
- role token 重复
- 不遵循格式
- evaluator 抽取失败

## 第一步：备份旧版

建议保留旧实验，不要覆盖：

```text
dev_smoke/
```

新代码放到单独目录即可。

## 第二步：自检

```bash
python self_test.py
```

应看到：

```text
SELF TEST PASSED
```

## 第三步：重新跑相同 20 条，chat-template

```bash
python baseline_eval.py \
  --model-path /root/autodl-tmp/models/Qwen3-0.6B-Base \
  --input /root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1/eval/finqa_eval_dev.jsonl \
  --output-dir /root/autodl-tmp/project/finqa_posttrain/experiments/base_qwen3_0.6b/dev_smoke_v1_1_chat \
  --max-samples 20 \
  --batch-size 2 \
  --max-new-tokens 256 \
  --prompt-format chat
```

## 第四步：同样 20 条，manual prompt

```bash
python baseline_eval.py \
  --model-path /root/autodl-tmp/models/Qwen3-0.6B-Base \
  --input /root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1/eval/finqa_eval_dev.jsonl \
  --output-dir /root/autodl-tmp/project/finqa_posttrain/experiments/base_qwen3_0.6b/dev_smoke_v1_1_manual \
  --max-samples 20 \
  --batch-size 2 \
  --max-new-tokens 256 \
  --prompt-format manual
```

两次必须保持：
- 相同 20 条
- batch-size 相同
- max-new-tokens 相同
- greedy decoding 相同

唯一变量是 Prompt formatting。

## 第五步：比较

```bash
python compare_prompt_runs.py \
  --a /root/autodl-tmp/project/finqa_posttrain/experiments/base_qwen3_0.6b/dev_smoke_v1_1_chat/metrics.json \
  --b /root/autodl-tmp/project/finqa_posttrain/experiments/base_qwen3_0.6b/dev_smoke_v1_1_manual/metrics.json \
  --name-a chat \
  --name-b manual
```

## 第六步：发给我

请发：

```text
dev_smoke_v1_1_chat/metrics.json
dev_smoke_v1_1_manual/metrics.json
```

以及：

```bash
head -n 10 .../dev_smoke_v1_1_chat/predictions.jsonl
head -n 10 .../dev_smoke_v1_1_manual/predictions.jsonl
```

先不要跑完整 883 Dev。

20 条只是工程 smoke test，不是正式 Baseline 结果。
