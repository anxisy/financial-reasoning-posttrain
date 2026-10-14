# ConvFinQA 单轮→多轮迁移：按**实验模型**编号（v2）

你要的目录：**一个根目录放 ConvFinQA 相关代码与可见的数据/结果**，每个初始化模型的 `smoke / full / eval` 都在自己的独立编号目录，而不是以训练阶段混排。

```text
/root/autodl-tmp/
└── ConvFinQA/
    ├── 00_common/                  # 公共训练实现、环境/数据检查、vLLM 评估及对照
    │   ├── train_sft.py
    │   ├── eval_sft_vllm.py
    │   ├── check_env.py
    │   ├── check_data.py
    │   └── compare_sft.py
    ├── 01_finqa_sft_adapter/        # B：原 FinQA SFT Adapter → ConvFinQA SFT
    │   ├── 01_smoke/run.sh
    │   ├── 02_full/run.sh
    │   └── 03_eval/run.sh
    ├── 02_qwen3_base/               # A：Qwen3 Base → ConvFinQA SFT
    │   ├── 01_smoke/run.sh
    │   ├── 02_full/run.sh
    │   └── 03_eval/run.sh
    ├── data/                       # 链接原有项目 data，零复制、原样保留
    ├── experiments/                # 链接原有项目 experiments
    ├── analysis/                   # 链接原有项目 analysis
    ├── scripts/                    # 链接原有数据预处理脚本
    ├── setup_links.sh
    └── README.md
```

**注意：`data`、`experiments` 等为软链接**，实际数据和训练结果仍保存在之前已经存在的 `/root/autodl-tmp/project/convfinqa_posttrain/`，避免复制/移动、占满磁盘或破坏旧 Baseline。VS Code 中可以从 `ConvFinQA` 一个总目录查看这些内容。旧 FinQA 的 `/root/autodl-tmp/open-r1/`、已训练 Adapter 一律不动。

## 1. 解压和建立已有数据的快捷入口

把 **ConvFinQA_by_experiment_v2.zip** 上传到 `/root/autodl-tmp/`：

```bash
cd /root/autodl-tmp
unzip -o ConvFinQA_by_experiment_v2.zip
cd /root/autodl-tmp/ConvFinQA
bash setup_links.sh
```

如果之前已把旧版本 `convfinqa-r1` 解压过，不必删除；新包使用全新根目录 `ConvFinQA`，不会覆盖旧代码。

## 2. 先做环境检查（无需大规模训练）

```bash
conda activate openr1
cd /root/autodl-tmp/ConvFinQA
python 00_common/self_test.py
python 00_common/check_env.py
python 00_common/check_data.py
```

`check_env.py` 检查旧 FinQA SFT Adapter 路径、CUDA/BF16 以及 v2 数据。`check_data.py` 检查 SFT 训练标签和样本量。检查目录时使用**实际已有数据路径**，不依赖软链接是否显示在编辑器中。

## 3. 实验 01：从 FinQA SFT Adapter 继续微调（先 Smoke）

```bash
bash 01_finqa_sft_adapter/01_smoke/run.sh
```

它检查已有 LoRA r8 为可训练状态，确认 Assistant-only Loss 掩码，进行最长样本前后向检查与 5 个 optimizer steps；保存路径为
`/root/autodl-tmp/project/convfinqa_posttrain/experiments/convfinqa_sft_v2/01_finqa_sft_adapter/smoke/`。

如果失败，请保存完整日志。**不要直接运行正式训练。** 如果 Smoke 成功并评估显存可接受，再执行：

```bash
bash 01_finqa_sft_adapter/02_full/run.sh
# 中断后：bash 01_finqa_sft_adapter/02_full/run.sh --resume latest
```

## 4. 实验 02：从 Base 直接微调（对照）

```bash
bash 02_qwen3_base/01_smoke/run.sh
# Smoke 成功以后才执行：
bash 02_qwen3_base/02_full/run.sh
```

两组共用 **相同 SFT 数据、Tokenizer、LoRA r8、训练配置、Assistant-only Loss 掩码**，唯一主要区别是初始化 Adapter。

## 5. vLLM 评估（训练完成以后）

```bash
# 若仍在 openr1 conda 环境，先退出
conda deactivate
source /root/autodl-tmp/open-r1/19_vllm_benchmark/.venv/bin/activate
cd /root/autodl-tmp/ConvFinQA
bash 01_finqa_sft_adapter/03_eval/run.sh
bash 02_qwen3_base/03_eval/run.sh
python 00_common/compare_sft.py
```

使用 ConvFinQA v2 冻结 Dev 1490 条 Oracle-history 数据、同一 vLLM 输入序列化和数值评分器。会检查 Base、之前 FinQA SFT、之前 Dynamic 与这两组 SFT 的可配对性。

运行结果分别保存在：

```
/root/autodl-tmp/project/convfinqa_posttrain/experiments/convfinqa_sft_v2/01_finqa_sft_adapter/{smoke,full,eval_dev}
/root/autodl-tmp/project/convfinqa_posttrain/experiments/convfinqa_sft_v2/02_qwen3_base/{smoke,full,eval_dev}
/root/autodl-tmp/project/convfinqa_posttrain/experiments/convfinqa_sft_v2/summary.json
```

## 6. 注意事项

- 这里的 4096 最大长度是**不截断**的训练预算。24 GB RTX 4090 上能否容纳完整训练，先通过 Smoke 验证。
- 样本历史回答使用 Oracle Gold，评估不是自主连聊；不能夸大为 free-running 结果。
- 当前答案评分仍保留冻结 FinQA numeric 版本，对百分号 / 比例等价及少量字符串的单位问题要另做预先定义的审计。
- 训练环境使用 `openr1`（Torch 2.6.0）；推理使用已有独立 vLLM `uv .venv`，不要混装。
- 目前只交付 **01 和 02**，后续新模型/GRPO 创建新编号目录，见 `03_future_experiments.md`。
