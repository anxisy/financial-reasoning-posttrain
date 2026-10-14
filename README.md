# Financial Reasoning Post-Training & Continual Learning

**Qwen3-1.7B-Base · FinQA / ConvFinQA · LoRA-SFT · GRPO · Dynamic Sampling · Experience Replay · vLLM**

基于单卡 RTX 4090 的金融数值推理后训练项目：先开展 FinQA 单轮 SFT 与 GRPO，再研究 ConvFinQA 多轮适配、历史错误传播以及持续微调的旧任务遗忘，并通过 Experience Replay 寻求两任务的性能平衡。

> **评估口径说明：** FinQA 最终测试成绩来自同一 1,147 题、同一 vLLM/数值评分器的固定配置复评；ConvFinQA 在 421 段对话（1,490 轮）上评估。Oracle History 使用历史标准答案；Self-History 回填模型上轮**最终答案**，不是把完整推理文本回填。ConvFinQA 是 Dev 结果，不是公开 Test 成绩。两个数据集源文档存在大量重叠；此前曾查看过 FinQA Test 结果，因此不宣称严格未接触盲测。

## 结果速览

### FinQA 单轮：Base → SFT → GRPO → Dynamic Sampling

以下是**原 FinQA 单轮后训练阶段**的历史对照成绩；这是你关心的基础 1.7B 模型、SFT、Vanilla GRPO 和 Dynamic Sampling 的完整结果，不能与后面的 ConvFinQA 遗忘检查混为同一次评估。

| 训练阶段 | FinQA Dev 准确率 | FinQA Test 准确率 |
|:--|--:|--:|
| Qwen3-1.7B-Base（未微调） | 35.67% | / |
| LoRA-SFT | 55.49% | 53.62% |
| Vanilla GRPO | 58.78% | 56.58% |
| GRPO + Dynamic Sampling | **60.14%** | **57.45%** |

- **LoRA-SFT**：Dev 相比 Base **+19.82pp**。
- **Vanilla GRPO**：Test 相比 SFT **+2.96pp**。
- **Dynamic Sampling**：Test 相比 Vanilla GRPO **+0.87pp**，相比 SFT **+3.83pp**。对 GRPO 的小幅增益不能单凭此表断言统计显著。
- **推理加速**：FinQA Dev（883 题），单卡 RTX 4090，vLLM **23.681 samples/s**，所测最佳 Transformers batch=32 **3.088 samples/s**，吞吐约 **7.67×**。

上述是先前阶段的历史评估记录；最终统一的四模型 vLLM **FinQA Test 保持能力**评估则使用同一 Prompt、数值评分器和 1,147 道题：

| 后续多轮训练路线 | 最终统一 FinQA Test | 正确题数 |
|:--|--:|--:|
| 原始 FinQA SFT | **54.14%** | 621 / 1147 |
| Transfer | 37.75% | 433 / 1147 |
| Direct | 19.88% | 228 / 1147 |
| Transfer + 10% Replay | **52.83%** | 606 / 1147 |

**为什么 SFT 有 53.62% 和 54.14% 两个数字？** 前者是此前单轮 GRPO 阶段的历史 Test 结果；后者是本次采用统一 vLLM 评估配置的最终复评。两者属于不同评估过程，**不能把 0.52pp 差距解释为训练带来的增益**。详细来源说明和能力保持统计见 [`results/FINQA_RESULTS.md`](results/FINQA_RESULTS.md)、[`results/finqa_test_paired.json`](results/finqa_test_paired.json)。

### ConvFinQA 多轮

| 模型 | 原始 Oracle Dev | 匹配中性标题 Oracle Dev | Self-History Dev |
|:--|--:|--:|--:|
| Transfer | 81.48% | 82.08% | 73.42% |
| Direct | 80.54% | 81.01% | 73.62% |
| Replay | 80.60% | 80.60% | 71.95% |

Oracle 与 Self-History 的直接差异使用**同一中性标题 Prompt**的匹配评估：Transfer −8.66pp，Direct −7.38pp，Replay −8.66pp。Replay 对比 Transfer 的 Self-History 为 **−1.48pp**，95% cluster CI **[−3.28, +0.20]pp**；包含 0 不意味着已证明无性能损失。

结论：10% 旧任务样本回放大幅缓解 FinQA 任务遗忘，同时让 ConvFinQA 多轮性能保持在接近无回放的水平；这是**能力平衡**而非“两项性能都提高”。

## 主要流程

```text
Qwen3-1.7B-Base
    ├─ FinQA → LoRA-SFT → Vanilla GRPO → Dynamic Sampling
    └─ ConvFinQA 对照
         ├─ Direct:   Base → ConvFinQA SFT
         ├─ Transfer: FinQA SFT → ConvFinQA SFT
         └─ Replay:   FinQA SFT → (约90% ConvFinQA + 10% FinQA) SFT
                                   ↓
                       FinQA retention / ConvFinQA Oracle & Self
                                   ↓
                     Paired clustered bootstrap & frozen release
```

## 仓库导航

- [`finqa/01_lora_sft`](finqa/01_lora_sft)：最终 1.7B FinQA LoRA-SFT。
- [`finqa/02_vanilla_grpo`](finqa/02_vanilla_grpo)：Vanilla GRPO、奖励和训练脚本。
- [`finqa/03_dynamic_sampling`](finqa/03_dynamic_sampling)：Dynamic Sampling GRPO。
- [`finqa/04_inference_benchmark`](finqa/04_inference_benchmark)：vLLM 性能实验相关脚本；已排除内嵌 `.venv`。
- [`finqa/05_frozen_vllm_evaluator`](finqa/05_frozen_vllm_evaluator)：FinQA 冻结评分器、vLLM/Transformers 评估。
- [`convfinqa/00_common`](convfinqa/00_common)：ConvFinQA 共用 SFT、评估代码。
- [`convfinqa/01_transfer`](convfinqa/01_transfer)、[`02_direct`](convfinqa/02_direct)、[`03_replay`](convfinqa/03_replay)：三种训练路线。
- [`convfinqa/04_final_evaluation`](convfinqa/04_final_evaluation)：结果核验、Bootstrap 与冻结代码。
- [`results/`](results/)：经过选择的小型结果汇总、统计 JSON 和原冻结源哈希清单；不包含完整预测文件或模型权重。
- [`docs/`](docs/)：实验协议、运行环境、项目局限与发布说明。

## 环境与复现


**完整配置步骤：**参见 [环境配置指南](docs/ENVIRONMENT_SETUP.md)，包含训练 Conda 与推理 uv/vLLM 两套隔离环境的建立命令、依赖版本核验、环境快照导出、数据目录要求与 Smoke Test 注意事项。

本仓库由**原 AutoDL 工作目录的实际有效源码**原样整理，原训练脚本可能仍包含 `/root/autodl-tmp/...` 等绝对路径，并且对数据/Adapter 有外部依赖。**代码发布不等于可在任意机器上开箱即跑**；当前准确复现需具备与原实验一致的模型权重、数据预处理和路径布局。详见 [复现与路径说明](docs/REPRODUCIBILITY.md)。

先使用已有的两个隔离环境：

```bash
# 训练（旧环境）
conda activate openr1
# vLLM 推理（另一个环境；不要混装 PyTorch）
source /root/autodl-tmp/open-r1/19_vllm_benchmark/.venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
```

本项目在已有 AutoDL 环境下执行过的关键入口包括：

```bash
cd /root/autodl-tmp/ConvFinQA
# 注意：以下是原 AutoDL 工作目录命令，不是从本 GitHub 根目录直接运行的命令。
bash 01_finqa_sft_adapter/02_full/run.sh
bash 02_qwen3_base/02_full/run.sh
bash 03_finqa_sft_replay/03_full/run.sh
python 04_final_evaluation/05_verify_frozen.py --check-sources
```

完整复现前请阅读 [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) 并准备原始数据与 Adapter。仓库中保留了数据整理脚本，但**不重新分发 FinQA/ConvFinQA 原始样本或模型权重**。

## 不确定性与限制

- 使用同一 Dev 多次诊断构建模型，Dev 比较为探索性结果；独立盲测证据有限。
- FinQA 与 ConvFinQA 具有源财务文档重叠，不能解读为跨领域知识迁移。
- Self-History 回填最终答案，仍非复杂的工具增强、自由文本完整对话环境。
- 两种任务的指标不能简单相加或按相同难度比较；报告置信区间，避免误用“显著无下降”。
- 未在本仓库实现完整 DAPO、其他未完成 GRPO 改进，不做相关主张。

## 许可证和数据

原始数据集和基座模型受其各自的使用与分发条款约束。此仓库暂不自动指定代码开源许可证：公开前由项目所有者确认版权与许可选择。参见 [docs/PUBLISHING_CHECKLIST.md](docs/PUBLISHING_CHECKLIST.md)。


## 从 GitHub 仓库直接运行：统一入口

新增的 [`docs/PORTABLE_REPRODUCTION.md`](docs/PORTABLE_REPRODUCTION.md) 提供不依赖历史 `open-r1/`、`ConvFinQA/` 代码目录的命令入口：`tools/portable_runner.py`。它可配置模型、数据、Adapter、输出目录，并保持原始训练/评估算法源码与已冻结结果不变。注意这不是已完成新机器 GPU 端到端验证的声明。
