# 04 — Final FinQA Test Evaluation & Results Freeze (v1)

本步骤 **没有任何训练**；复用原始的 `20_finqa_vllm_benchmark_v2/bench_vllm.py`、冻结数字匹配规则，测试四个已经训练好的 Adapter。每个完整输出存在时 **绝不覆盖**，中断残留保存在独立 `.running_full_test_...` 目录里。

> **重要**：历史项目在更早阶段已经检查过 FinQA Test，因此最终 Test 仍不能称为完全从未接触的 blind set。现在只做一次固定配置复评，**不根据 Test 再修改超参数、采样比例或挑选检查点**。

## 目录和数据契约

- 基础模型：`/root/autodl-tmp/models/Qwen3-1.7B-Base`
- FinQA Dev：`/root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1/eval/finqa_eval_dev.jsonl` (883)
- **FinQA Test**：`/root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1/eval/finqa_eval_test.jsonl` (1147)，路径不同用 `export FINQA_TEST_INPUT=/实际路径/...jsonl`。
- 评估代码：`/root/autodl-tmp/open-r1/20_finqa_vllm_benchmark_v2/bench_vllm.py`。
- Test 统一使用 BF16 / 贪心解码 / max_new_tokens=384 / max_model_len=4096 / seed=42 / vLLM / 数值答案原评分器 / max_num_seqs=32。
- Test Prompt SHA 在服务器上通过 Preflight 实际计算并存储；四模型必须一致。预处理数据不得包含 Test 训练样本。
- FinQA Test 结果：`/root/autodl-tmp/project/convfinqa_posttrain/experiments/convfinqa_sft_v2/04_final_evaluation_v1/finqa_test/{variant}/full_test/`。
- Self-history 只读原来的三个模型 1490 条预测；不会重新推理。

## 0. 安装并检查（CPU/Tokenizer）

```bash
cd /root/autodl-tmp
unzip -o ConvFinQA_final_eval_freeze_v1.zip
source /root/autodl-tmp/open-r1/19_vllm_benchmark/.venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd /root/autodl-tmp/ConvFinQA
python 04_final_evaluation/self_test.py
python 04_final_evaluation/00_preflight.py --write
```

**STOP** 如果 Test 数据找不到、数量不是 1147、FinQA Dev 883 的已冻结指纹不匹配、模型 Adapter 不存在、Prompt+生成超出 4096。不应通过修改评估集或规则来绕过这些检查。

## 1. GPU：四模型正式 FinQA Test

```bash
bash 04_final_evaluation/01_run_finqa_test.sh all
```

依次执行 `original_sft, transfer, direct, replay`。已经完整评估的模型自动验证并跳过，不重复花费 GPU 时间；若中断，检查 `.running_full_test_...` 目录后重试即可（失败中途输出不会覆盖最终 `full_test`）。可分别运行某个模型，如 `bash 04_final_evaluation/01_run_finqa_test.sh replay`。

## 2. CPU：FinQA Test 配对统计

```bash
python 04_final_evaluation/02_compare_finqa_test.py --bootstrap 5000 --seed 42
```

自动核对 4×1147 个 ID、gold、评分结果、输入 Token SHA 和生成设置，使用财务 PDF 聚类 Bootstrap（若 PDF 信息缺失，报告按题目回退数量），保存到 `.../04_final_evaluation_v1/analysis/finqa_test_paired.json`。

## 3. CPU：ConvFinQA Self-History 配对统计

```bash
python 04_final_evaluation/03_compare_self_history.py --bootstrap 5000 --seed 42
```

读取已存在的 `self_history_v1/{transfer,direct}` 和 `03_finqa_sft_replay/eval_self_history/replay`，校验 1490 题/421 对话、gold、一致的 matched-oracle 中性 Prompt 指纹。**Self 历史来自各模型自己的预测，三模型生成不同历史，所以 Self Prompt 指纹不需要相同。** 保存 `self_history_paired.json`。

## 4. CPU：结果冻结 & SHA-256 校验

```bash
python 04_final_evaluation/04_freeze_results.py
python 04_final_evaluation/05_verify_frozen.py --check-sources
```

输出 `.../04_final_evaluation_v1/frozen_release_v1`：`RESULTS.md`、`MANIFEST.json`、`MANIFEST.sha256` 与评估预测/统计/运行配置副本 `evidence/`。**会尽力一并收录已有 FinQA SFT/GRPO/Dynamic Sampling 的小型历史指标或配置文件（若原路径中存在），但不会猜测缺失指标；不复制大型模型或原始数据集**，而是在清单中记录它们和源码的 SHA-256。请额外备份四个 Adapter 和所使用的原始数据，否则后续无法仅用哈希恢复模型。冻结操作会拒绝覆盖已有版本。

## 全流程一键执行

确认前置条件满足后：

```bash
bash 04_final_evaluation/run_all.sh
```

**建议第一次分步执行**，这样 Test 完成后可核查数值和异常再冻结。`run_all.sh` 不适合已经生成全部分析产物后反复执行，因为分析文件和冻结清单采用一次性写入保护。

## 结果解释边界

1. FinQA Test 衡量单轮数据集上的泛化；并非保证真实金融任务推理可靠。
2. ConvFinQA Self-History 是答案回填（非完整自由对话），且这部分只在 Dev 上评测。
3. 95% Cluster Bootstrap CI 含零 **不等于** 模型完全等价。
4. 这是一组 LoRA 单随机种子实验。提高推断可靠性还需多种子或新独立评估；本收尾阶段不继续调参。
