# FinQA Baseline Evaluation v1

用于冻结后的 FinQA `processed_v1_1_1` 数据，先在 Qwen3-0.6B-Base 上验证完整评测链路，后续只替换 `--model-path` 即可复用到最终约 1.5B 模型。

## 输出指标

- Overall Answer Accuracy
- Strict Accuracy：答案正确且存在 `<answer>...</answer>`
- Format Accuracy
- Average Response Tokens
- Easy / Medium / Hard Accuracy
- Operator Accuracy
- Context-length Accuracy
- Annotation Quality slice
- Semantic Status slice
- Truncated / Not-truncated slice

## 0. 为什么先跑 Dev

先用 Dev 验证 Prompt / Generation / Answer Parser / Metrics。

在这些代码和参数被冻结之前，不建议反复查看 Test 指标并据此调 Prompt，否则会逐渐把 Test 变成调参集。

推荐顺序：

```text
20条 Dev smoke
→ 完整 Dev
→ 确认输出正常
→ 冻结 baseline 设置
→ 最后跑 Test
```

## 1. 自检

```bash
python self_test.py
```

最后应看到：

```text
SELF TEST PASSED
```

## 2. 20 条 Dev smoke test

建议在 GPU 实例上运行。0.6B 用 4090 24GB 非常宽裕。

```bash
python baseline_eval.py \
  --model-path /root/autodl-tmp/models/Qwen3-0.6B-Base \
  --input /root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1/eval/finqa_eval_dev.jsonl \
  --output-dir /root/autodl-tmp/project/finqa_posttrain/experiments/base_qwen3_0.6b/dev_smoke \
  --max-samples 20 \
  --batch-size 2 \
  --max-new-tokens 256 \
  --prompt-format auto
```

## 3. 检查 20 条输出

```bash
cat /root/autodl-tmp/project/finqa_posttrain/experiments/base_qwen3_0.6b/dev_smoke/metrics.json
```

看前 5 条预测：

```bash
head -n 5 /root/autodl-tmp/project/finqa_posttrain/experiments/base_qwen3_0.6b/dev_smoke/predictions.jsonl
```

重点确认：

1. 模型真的产生了 completion，而不是空字符串。
2. `prediction` 没有把输入 Prompt 整段重复出来。
3. `prediction_answer` 提取结果合理。
4. `gold_answer` 正确。
5. `prompt_format_used` 是 `chat_template` 还是 `manual`。
6. Accuracy 不是因为 parser 错误而全部为 0。

## 4. 完整 Dev

smoke 没问题后：

```bash
python baseline_eval.py \
  --model-path /root/autodl-tmp/models/Qwen3-0.6B-Base \
  --input /root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1/eval/finqa_eval_dev.jsonl \
  --output-dir /root/autodl-tmp/project/finqa_posttrain/experiments/base_qwen3_0.6b/dev \
  --batch-size 2 \
  --max-new-tokens 256 \
  --prompt-format auto
```

## 5. 完整 Test

只有 Dev 格式和 evaluator 都确认无误后，再运行：

```bash
python baseline_eval.py \
  --model-path /root/autodl-tmp/models/Qwen3-0.6B-Base \
  --input /root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1/eval/finqa_eval_test.jsonl \
  --output-dir /root/autodl-tmp/project/finqa_posttrain/experiments/base_qwen3_0.6b/test \
  --batch-size 2 \
  --max-new-tokens 256 \
  --prompt-format auto
```

## 6. 中断后继续

脚本逐条写 `predictions.jsonl`。如果中断，可以在原命令最后加：

```bash
--resume
```

已完成 ID 会跳过。

## 7. 输出文件

每个 output 目录包含：

```text
predictions.jsonl
metrics.json
errors.jsonl
run_config.json
```

- `predictions.jsonl`：所有模型输出和评分
- `metrics.json`：总指标 + 分层指标
- `errors.jsonl`：只保存答错的样本，方便错误分析
- `run_config.json`：记录本次模型/生成参数，保证实验可复现

## 8. Prompt format

默认：

```bash
--prompt-format auto
```

逻辑：

```text
tokenizer 有 chat_template
→ 用 chat_template

否则
→ 用稳定的 System/User/Assistant 手工格式
```

第一次 smoke 后务必查看 `run_config.json` 中：

```text
actual_prompt_format
```

后续 SFT / GRPO 要保持同一套 Prompt 格式，保证公平比较。

## 9. 当前阶段不要调 sampling

Baseline 默认是：

```text
do_sample = False
greedy decoding
max_new_tokens = 256
seed = 42
```

这样结果稳定、可复现。

如果 Base 输出很长或格式差，先记录这个现象，不要立即根据 Test 调 Prompt。
