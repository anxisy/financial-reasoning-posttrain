# 01_finqa_sft_adapter

**目标**：从已训练 FinQA SFT Adapter 继续 LoRA-SFT，使用相同 ConvFinQA v2 Train 数据，作为严格对照。

三个阶段都在本实验目录里：

```bash
# openr1 conda 环境
bash 01_finqa_sft_adapter/01_smoke/run.sh
bash 01_finqa_sft_adapter/02_full/run.sh
# 以下在独立 vLLM uv 环境
bash 01_finqa_sft_adapter/03_eval/run.sh
```

训练实际输出：`/root/autodl-tmp/project/convfinqa_posttrain/experiments/convfinqa_sft_v2/01_finqa_sft_adapter/smoke` 和 `full`；评估输出 `eval_dev`。
这些结果在顶层 `experiments/convfinqa_sft_v2/01_finqa_sft_adapter/` 软链接中可见。
