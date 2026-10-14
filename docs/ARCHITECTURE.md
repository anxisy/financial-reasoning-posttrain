# 技术路线与代码模块

## FinQA 单轮后训练

- **数据处理与错误诊断**：见 `finqa/data_prep/`，输入财务文本/表格和数值标准答案；用固定数值评分规则评估。
- **LoRA-SFT**：见 `finqa/01_lora_sft/`，Qwen3-1.7B-Base 上的适配器训练与助手回答监督。
- **Vanilla GRPO**：见 `finqa/02_vanilla_grpo/`，依照生成候选组、数值奖励和组内相对优势执行训练。
- **Dynamic Sampling**：见 `finqa/03_dynamic_sampling/`，过滤/处理低信息量采样组，并和 Vanilla GRPO 比较。
- **吞吐评估**：见 `finqa/04_inference_benchmark/` 与 `05_frozen_vllm_evaluator/`，同 GPU 的 Transformers/vLLM 对照；以前测得约 3.088 vs 23.681 samples/s（约 7.67 倍），取决于推理批处理配置。

## ConvFinQA 多轮持续学习

- **公共训练、评估**：见 `convfinqa/00_common/`。
- **Transfer**：见 `convfinqa/01_transfer/`，从原始 FinQA SFT Adapter 初始化 ConvFinQA LoRA SFT。
- **Direct**：见 `convfinqa/02_direct/`，从 Qwen3 Base 初始化同训练数据和预算。
- **Replay**：见 `convfinqa/03_replay/`，9:1 新旧数据混合，仍从原始 FinQA SFT Adapter 初始化。
- **结果冻结/核验**：见 `convfinqa/04_final_evaluation/`，四模型 FinQA Test、Self-History paired bootstrap、归档及 SHA-256 校验。

## 核心结论

ConvFinQA 任务内性能可通过 SFT 获得大幅改善，但会牺牲原 FinQA 单轮能力。10% FinQA 样本回放可恢复显著部分旧任务准确率；Self-History 较 Oracle 仍存在约 7–9 个百分点下降，表明历史错误传播并未被 Replay 解决。

## 代码发布注意

这是**源码快照整理**，不是一个从零编写并与上游库完全解耦的新框架。保留原始实验入口和内部依赖是为了真实反映当时的训练过程；复现时需按原目录结构部署或重构路径。不要将部分 `GRPOTrainer` 使用表述为完全手写全套 RL trainer。
