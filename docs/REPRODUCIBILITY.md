# 复现实验与路径说明

## 1. 来源与状态

此仓库由 AutoDL 工作空间中**已完成实验的源码副本**整理而来。构建器只复制文件，不改动原脚本内容，因此部分脚本依赖原始工作空间布局和硬编码绝对路径；**本仓库不是一个即插即用的新训练框架**。

原始工作空间关键路径：

```text
/root/autodl-tmp/
├── open-r1/
│   ├── 16_sft_1p7b/
│   ├── 17_grpo_1p7b/
│   ├── 18_dynamic_1p7b/
│   ├── 19_vllm_benchmark/
│   └── 20_finqa_vllm_benchmark_v2/
├── ConvFinQA/
│   ├── 00_common/
│   ├── 01_finqa_sft_adapter/
│   ├── 02_qwen3_base/
│   ├── 03_finqa_sft_replay/
│   └── 04_final_evaluation/
├── project/finqa_posttrain/{data,experiments,...}
├── project/convfinqa_posttrain/{data,experiments,...}
└── models/Qwen3-1.7B-Base/
```

**注意**：ConvFinQA 原始目录中的 `data/`、`experiments/`、`scripts/`、`analysis/` 等可能是指向 `project/convfinqa_posttrain/` 的软链接。导出工具不会递归复制软链接目标下的数据、权重和日志；预处理脚本会从真实 `project/.../scripts/` 路径另外选取。

## 2. 环境隔离

- 训练环境 `openr1`：已观察 `torch==2.6.0+cu124`、`transformers==4.52.3`、`peft==0.21.0`、`trl==0.18.0`；其他包以你的服务器实际 `pip freeze` 为准。
- vLLM 环境：`/root/autodl-tmp/open-r1/19_vllm_benchmark/.venv`，已观察 `vllm==0.31.0`，不同 PyTorch/CUDA 版本；`VLLM_USE_FLASHINFER_SAMPLER=0`。
- 单卡 RTX 4090（24GB），LoRA rank 8、BF16；完整 ConvFinQA Transfer 训练 1 epoch 1388 optimizer steps，显存峰值约 22.56 GiB。

请在原机器**分别**记录环境快照（不是混成一个 requirements 文件）：

```bash
conda activate openr1
python -m pip freeze > /root/autodl-tmp/train_pip_freeze.txt
conda deactivate
source /root/autodl-tmp/open-r1/19_vllm_benchmark/.venv/bin/activate
python -m pip freeze > /root/autodl-tmp/vllm_pip_freeze.txt
```

审查生成文件是否包含私有镜像地址或凭证后，才考虑加入 GitHub。

## 3. 数据与提示词协议

- FinQA 预处理版本：`processed_v1_1_1`；FinQA Dev 883 条，Test 1147 条。
- ConvFinQA `processed_v2`：Train 11104 轮，Dev 1490 轮（421 段对话）。
- `number_turn` 输出最终答案，`program_turn` 输出步骤和最终答案；监督 Loss 只计算 Assistant 标签，Prompt 标签为 `-100`。
- ConvFinQA Oracle History 只包含先前轮次的标准答案；当前/未来标准答案不泄漏。
- Self-History 仅把先前**模型最终答案**回填当前 Prompt。
- Replay：在共 11104 条训练预算内抽取 9994 ConvFinQA + 1110 FinQA，约 9:1，同一 FinQA SFT Adapter 初始化；不是在 Transfer Adapter 上补训。

## 4. 实验入口

主代码镜像在 [`finqa/`](../finqa) 和 [`convfinqa/`](../convfinqa)。原代码多数入口假设自己在原工作空间目录执行，**克隆后的路径被重新编排，因此不能保证 `bash convfinqa/01_transfer/02_full/run.sh` 可直接运行**。实际执行时应恢复源码的原路径结构，或显式适配各脚本中的路径和相对 imports；请先通过 Smoke Test 再运行正式训练。

原环境中经过验证的主要命令：

```bash
conda activate openr1
cd /root/autodl-tmp/ConvFinQA
python 00_common/self_test.py
bash 01_finqa_sft_adapter/01_smoke/run.sh
bash 01_finqa_sft_adapter/02_full/run.sh
bash 02_qwen3_base/01_smoke/run.sh
bash 02_qwen3_base/02_full/run.sh
python 03_finqa_sft_replay/01_data/build_replay.py
python 03_finqa_sft_replay/01_data/verify_replay.py
bash 03_finqa_sft_replay/02_smoke/run.sh
bash 03_finqa_sft_replay/03_full/run.sh
```

推理/复评需切换到独立 vLLM 环境，使用已冻结的数值评分器及相同 Prompt。所有已完成评估的精确指纹、模型配置、文件哈希以外部冻结目录 `frozen_release_v1` 为权威来源；本仓库只保存精选摘要及原始 Manifest 副本。

## 5. 哈希与资产备份

原 AutoDL 上的 `04_final_evaluation/04_freeze_results.py` 已生成归档；`05_verify_frozen.py --check-sources` 核验通过 **209 个已复制证据文件**和 **221 个源引用**。GitHub 目录不是冻结归档的代用品：它**不包含**完整预测数据和权重。请单独保存：

1. 四个最终 Adapter 及所需基座模型配置/来源。
2. 冻结 `frozen_release_v1` 的完整压缩包。
3. 原始数据获取方式及预处理脚本、训练配置、环境快照。
4. 包含每次运行种子/评估模型名/文件哈希的不可变实验元数据。

## 6. 公平比较注意事项

- FinQA Test 已在早期阶段被观察过；目前重新跑的四模型同设置测试并不是完全未接触的盲测。
- 旧 Dev 分数（不同历史推理实现）与本次统一 vLLM Test 不可直接计算训练增益。
- ConvFinQA 的 matched Oracle 是在中性 Prompt 标题下与 Self 直接配对的；最初的 `eval_dev` Oracle 和后续 matched Oracle 并不完全相同。
- Replay FinQA 的恢复具有统计证据；Self-History 性能略降，其区间包括 0，但不支持“完全没有损失”的等价性判断。
