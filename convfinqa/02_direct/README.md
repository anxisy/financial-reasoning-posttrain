# 02_qwen3_base

**目标**：从原始 Qwen3-1.7B-Base 开始新的 LoRA-SFT，使用相同 ConvFinQA v2 Train 数据，作为严格对照。

三个阶段都在本实验目录里：

```bash
# openr1 conda 环境
bash 02_qwen3_base/01_smoke/run.sh
bash 02_qwen3_base/02_full/run.sh
# 以下在独立 vLLM uv 环境
bash 02_qwen3_base/03_eval/run.sh
```

训练实际输出：`/root/autodl-tmp/project/convfinqa_posttrain/experiments/convfinqa_sft_v2/02_qwen3_base/smoke` 和 `full`；评估输出 `eval_dev`。
这些结果在顶层 `experiments/convfinqa_sft_v2/02_qwen3_base/` 软链接中可见。
