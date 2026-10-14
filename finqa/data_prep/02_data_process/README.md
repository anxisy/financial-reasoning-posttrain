# FinQA Post-Training Data Pipeline

这是当前 FinQA 后训练项目的 **正式数据预处理 Pipeline v1**。它只负责把原始 FinQA 转成统一的 Canonical / SFT / GRPO / Eval 数据，还不会启动模型训练。

## 文件作用

- `config.py`：路径、2048 序列长度、SFT/GRPO prompt budget、数值容差等统一配置。
- `clean_context.py`：清理 PDF 抽取造成的空行、纯句点、重复空白等噪声，保留数字/货币/百分比。
- `answer_normalizer.py`：选择 canonical gold，并统一 number / percent / yes-no 比较逻辑。
- `reasoning_builder.py`：把 `qa["steps"]` 转成稳定的 SFT reasoning target。
- `prompt_builder.py`：构建 full-context prompt；Question 和 Table 永远保留，超长时只裁 report text。
- `preprocess.py`：主入口；输出 Canonical、SFT、GRPO、Eval 和数据质量报告。
- `evaluator.py`：Base/SFT 评测以及后续 GRPO 的 correctness/format reward 共用逻辑。
- `self_test.py`：不需要 GPU 和 FinQA 数据的基础逻辑自检。

## 为什么这样设计

当前分析显示：
- `answer` 与 `exe_ans` 大量存在百分比尺度/舍入差异，因此不能做简单字符串匹配。
- `explanation` 经常为空，而 `steps` 更稳定，因此 SFT 主要使用 `steps`。
- full-context prompt 的 P95 大约 1600 token，所以第一版以 2048 总序列长度作为低成本基线是合理的。
- `gold_inds` / `text_retrieved` / `table_retrieved` 不进入模型 prompt，避免 gold evidence 泄漏。

## Gold 选择

默认：

```text
qa["answer"] 非空
      ↓ 否则
qa["steps"][-1]["res"]
      ↓ 否则
qa["exe_ans"]
```

不会用 `exe_ans` 覆盖一个非空的人工 `answer`。

## 难度标签

```text
easy   = 1 step
medium = 2 steps
hard   = 3+ steps
```

现在只是 metadata。第一版 SFT 仍然随机 shuffle；Curriculum Learning 留到后续消融。

## 训练/评测 split

```text
FinQA Train
├── SFT Train View
└── GRPO Train View

FinQA Dev
└── Validation / Eval

FinQA Test
└── Final Eval
```

SFT 和 GRPO 可以共享 Train prompt，因为它们是连续训练阶段；Dev/Test 不能进入优化。

## 运行顺序

### 1. 自检

```bash
python self_test.py
```

正常最后应看到：

```text
SELF TEST PASSED
```

### 2. 正式预处理

```bash
python preprocess.py \
  --data-dir /root/autodl-tmp/project/finqa_posttrain/data/raw/FinQA/dataset \
  --output-dir /root/autodl-tmp/project/finqa_posttrain/data/processed \
  --model-path /root/autodl-tmp/models/Qwen3-0.6B-Base
```

不需要 GPU。

### 3. 输出目录

```text
processed/
├── canonical/
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
├── sft/
│   ├── finqa_sft_train.jsonl
│   └── finqa_sft_dev.jsonl
├── grpo/
│   └── finqa_grpo_train.jsonl
├── eval/
│   ├── finqa_eval_dev.jsonl
│   └── finqa_eval_test.jsonl
└── reports/
    └── preprocess_report.json
```

### 4. 重点检查

```bash
cat /root/autodl-tmp/project/finqa_posttrain/data/processed/reports/preprocess_report.json
```

并看一条 SFT：

```bash
head -n 1 /root/autodl-tmp/project/finqa_posttrain/data/processed/sft/finqa_sft_train.jsonl
```

再看一条 GRPO：

```bash
head -n 1 /root/autodl-tmp/project/finqa_posttrain/data/processed/grpo/finqa_grpo_train.jsonl
```

确认：
1. Question / Table 在 prompt 里。
2. `gold_inds` 没进 prompt。
3. SFT 有 assistant reasoning + `<answer>`.
4. GRPO prompt 里没有 gold answer。
5. GRPO 的 `gold_answer` 只作为 reward metadata 单独保存。
6. report 中查看 ID overlap / filename overlap / quality / difficulty / truncation 情况。

## 下一阶段

数据验证通过后，顺序是：

```text
Base Evaluation
→ 正式 LoRA-SFT
→ SFT Evaluation
→ Vanilla GRPO
→ Training Diagnosis
→ Dynamic Sampling
→ Curriculum / Reward Shaping 消融
```
