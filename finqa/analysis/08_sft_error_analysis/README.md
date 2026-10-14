# FinQA SFT Error Analysis v1

只分析 **SFT Dev**，不要用 Test 错误来设计 GRPO。

## 1. 解压

```bash
cd /root/autodl-tmp/open-r1
mkdir -p 08_sft_error_analysis
cd 08_sft_error_analysis
unzip finqa_sft_error_analysis_v1.zip
cd finqa_sft_error_analysis_v1
```

## 2. 自检

```bash
python self_test.py
```

应看到：

```text
SELF TEST PASSED
```

## 3. 正式分析 Dev

默认路径已按当前项目写好，直接：

```bash
python error_analysis.py
```

完整写法：

```bash
python error_analysis.py \
  --predictions /root/autodl-tmp/project/finqa_posttrain/experiments/sft_qwen3_0.6b/eval_dev_v1/predictions.jsonl \
  --eval-file /root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1/eval/finqa_eval_dev.jsonl \
  --output-dir /root/autodl-tmp/project/finqa_posttrain/analysis/sft_dev_error_v1 \
  --review-per-category 8 \
  --seed 42
```

## 4. 输出

```text
sft_dev_error_v1/
├── error_analysis.jsonl
├── category_summary.json
├── priority_slices.json
├── review_sample.jsonl
└── report.md
```

自动类别是**启发式候选**，不是人工真值。尤其：

```text
operator_mismatch_candidate
operand_selection_candidate
```

只能用于筛选人工复核样本。

## 5. 查看具体错误

```bash
python inspect_errors.py \
  --analysis /root/autodl-tmp/project/finqa_posttrain/analysis/sft_dev_error_v1/error_analysis.jsonl \
  --category near_numeric_miss \
  --limit 10
```

也可把 category 换成：

```text
percentage_scale_error
operator_mismatch_candidate
operand_selection_candidate
multi_step_failure
annotation_quality_flag
other_numeric_reasoning_error
```

## 6. 跑完发给我

请发：

```text
category_summary.json
report.md
review_sample.jsonl
```

另外把 `python error_analysis.py` 最后的终端统计一起发过来。
