# FinQA 单轮推理实验结果（Qwen3-1.7B-Base）

本表归纳项目此前已经完成的 FinQA 训练对照，属于**历史评估结果**，应与 2026-10-14 四模型统一 vLLM FinQA Test 的最终复评区分，不混合计算差值。

## 1. 原 FinQA 后训练对照

| 训练阶段 | FinQA Dev 准确率 | FinQA Test 准确率 |
|:--|--:|--:|
| Qwen3-1.7B-Base（未微调） | **35.67%** | 未提供一致的历史 Test 结果 |
| LoRA-SFT | 55.49% | 53.62% |
| Vanilla GRPO | 58.78% | 56.58% |
| GRPO + Dynamic Sampling | **60.14%** | **57.45%** |

- SFT 相比 Base：FinQA Dev **+19.82pp**。
- Vanilla GRPO 相比 SFT：FinQA Dev **+3.29pp**，Test **+2.96pp**。
- Dynamic Sampling 相比 Vanilla GRPO：FinQA Dev **+1.36pp**，Test **+0.87pp**。增幅有限，未以此证明 Dynamic Sampling 显著优于 Vanilla GRPO。
- Dynamic Sampling 相比 SFT：FinQA Test **+3.83pp**。

以上为此前实验输出的历史结果记录。**公开仓库前，应对照原始实验日志、`metrics.json` 和对应 Adapter/评分器确认来源**；本发布整理程序不会自动重新评估这些模型。

## 2. 最终统一复评：FinQA Test（2026-10-14）

以下为不同问题的统一对照：比较 ConvFinQA SFT 之后**单轮 FinQA 能力保持情况**，而不是比较单轮 GRPO 的算法收益。

| 模型/路线 | FinQA Test 准确率 | 正确题数 |
|:--|--:|--:|
| 原始 FinQA SFT | 54.14% | 621 / 1147 |
| FinQA SFT → ConvFinQA SFT (Transfer) | 37.75% | 433 / 1147 |
| Base → ConvFinQA SFT (Direct) | 19.88% | 228 / 1147 |
| FinQA SFT → 90% ConvFinQA + 10% FinQA Replay | **52.83%** | 606 / 1147 |

固定评估：全部 1,147 道题，统一 vLLM、Prompt 与数值评分器。Replay 比 Transfer 提升 **+15.08pp**，按来源财务 PDF 聚类的配对 Bootstrap 95% CI **[+11.81, +18.35]pp**。

**为什么有两个不同的 SFT Test 数值？** 历史 GRPO 阶段报告的 SFT Test 为 53.62%，最终统一四模型复评为 54.14%，来自不同时间的评估过程/生成实现；应分别保留，不把二者误认为同一次评估或算法带来的变化。

## 3. 推理吞吐基准

在此前单卡 RTX 4090 的 FinQA Dev 883 条任务对比中，vLLM **23.681 samples/s**，Transformers（测得的最佳 batch=32 配置）**3.088 samples/s**，约 **7.67×** 吞吐提升。

这是特定模型、显卡和 batch/并发条件下的基准，不代表所有请求长度及服务条件都可以达到相同加速。

## 4. 数据与统计边界

- 原始 FinQA Train/Dev/Test 为 6,251 / 883 / 1,147 条。
- FinQA Test 在项目早期曾被查看，不将最终四模型复评称为未接触盲测。
- FinQA/ConvFinQA 来源财务文档有大量重叠；不能把这里的结果称为跨领域泛化。
- 本仓库默认仅存精选统计摘要，不公开完整评估预测、数据集或 Adapter 权重。完整源文件依据外部已冻结的 `frozen_release_v1` 哈希档案核验。
