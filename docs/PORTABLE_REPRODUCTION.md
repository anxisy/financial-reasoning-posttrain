# 从整理后的 GitHub 仓库运行：可配置入口 v1

## 适用范围和不可夸大的部分

本补丁为**已运行过的原算法源码增加独立启动入口**。为保留经过核验的 183 个源码快照，**不修改原 FinQA/ConvFinQA 训练与评分器文件**。因此历史 README、Shell 脚本和部分辅助/冻结脚本仍包含旧 AutoDL 路径；如果直接运行那些旧 Shell 文件，依然需要原工作目录。本指南使用**新的统一入口**，由其向训练/评估 Python 模块显式传入数据、模型、Adapter、输出位置。

已用离线模拟数据验证命令生成、路径检查、非覆盖行为与 Shell/Python 语法；**未在新机器或真实 GPU 上验证完整重训练，不能宣称一键复现最终分数**。首次使用时请依次运行检查、小样本和正式任务。`03_replay` 的 10% 样本比例与 seed=42 仅复现构建规则；若源 JSONL 字节不同，训练与历史冻结结果也可能不同。最终评估与冻结脚本是一份保留的**历史来源证明**，不由新入口重写或复现已有 SHA-256。

## 1. 安装环境

请先阅读 [`docs/ENVIRONMENT_SETUP.md`](ENVIRONMENT_SETUP.md)，分别配置训练 `openr1`（PyTorch 2.6 + cu124、Transformers 4.52.3、TRL 0.18.0、PEFT 0.21.0）和推理 vLLM（独立 uv/.venv）。本入口不自动安装 Python 包，也不在两套环境中混装 PyTorch。

## 2. 配置本地路径

在你的 AutoDL 现有项目上执行：

```bash
cd /root/autodl-tmp/financial-reasoning-posttrain-release
python tools/portable_runner.py init --workspace /root/autodl-tmp
```

生成本地文件 `configs/paths.local.json`（已加入 `.gitignore`）。本文件由默认的已有 AutoDL 布局推导，不会迁移或复制任何模型/数据/实验结果。**必须打开检查路径；从其他计算机克隆时可参考 `configs/paths.example.json` 手动填写。**

默认结果保存到 **`/root/autodl-tmp/financial_portable_runs_v1/`**，与原始冻结实验结果完全分离。该路径必须不包含你的 GitHub 仓库。已有同名输出会被拒绝，避免重复运行覆盖。

## 3. 查看所有支持的任务并先检查路径

```bash
python tools/portable_runner.py list
python tools/portable_runner.py check finqa-sft-smoke
python tools/portable_runner.py plan finqa-grpo-smoke
python tools/portable_runner.py check conv-transfer-smoke
python tools/portable_runner.py check replay-build
```

`check`：只检验当前任务源码与必需路径是否存在；`plan`：输出准确工作目录及 Python argv，**不会加载模型或 GPU**；`run`：才会真正执行。若不想看到全部模型路径，可只查看 `PASS` 与 `MISSING` 提示。可以用 `--config /ABS/PATH.json` 指定其他本地配置。运行 `check` 对不存在的 Adapter 会失败，只有先完成相应训练或恢复权重才可能通过，这是预期行为。

## 4. 单轮 FinQA：SFT → GRPO → Dynamic Sampling

切换到**训练** Conda 环境：

```bash
conda activate openr1
cd /root/autodl-tmp/financial-reasoning-posttrain-release

# 先跑 smoke / preflight；这两个命令需要 GPU
python tools/portable_runner.py run finqa-sft-smoke
# 根据显存和任务情况决定是否全量训练
python tools/portable_runner.py run finqa-sft-full

python tools/portable_runner.py run finqa-grpo-smoke
python tools/portable_runner.py run finqa-grpo-full
python tools/portable_runner.py run finqa-dynamic-smoke
python tools/portable_runner.py run finqa-dynamic-full
```

**重要：**GRPO 和 Dynamic Sampling 默认读取 `configs/paths.local.json` 的 `original_sft_adapter`，即已经训练好的 FinQA SFT Adapter；**不会自动切换到刚才新训练产生的目录**。如果确实要串联新的实验结果，请先修改这项配置并运行 `check`。输出始终写入一个**全新的** `output_root` 下，不会覆盖历史的 `full_v1`。

## 5. ConvFinQA：Direct / Transfer / Experience Replay

仍在训练环境中：

```bash
python tools/portable_runner.py run conv-transfer-smoke
python tools/portable_runner.py run conv-transfer-full
python tools/portable_runner.py run conv-direct-smoke
python tools/portable_runner.py run conv-direct-full

python tools/portable_runner.py run replay-build
python tools/portable_runner.py run replay-verify
python tools/portable_runner.py run replay-smoke
python tools/portable_runner.py run replay-full
```

Replay 在 11,104 条预算内随机选择 9,994 条 ConvFinQA 和 1,110 条 FinQA（seed=42），输出混合数据及哈希清单。**新建的 Replay 模型不会自动成为配置中的 `replay_adapter`**，需要用户核对结果后改配置；Transfer 和 Direct 也一样。避免把新模型与历史模型或历史结果误认为是同一次实验。

## 6. vLLM 评估：单轮与多轮

先激活**独立的 vLLM** 环境（以下路径仅是历史 AutoDL 实例示例）：

```bash
conda deactivate
source /root/autodl-tmp/open-r1/19_vllm_benchmark/.venv/bin/activate
cd /root/autodl-tmp/financial-reasoning-posttrain-release

# FinQA: 先在 20 条上做小样本，再运行正式 Dev（两种输出独立）
python tools/portable_runner.py run finqa-replay-dev --limit 20
python tools/portable_runner.py run finqa-replay-dev

# ConvFinQA Oracle: 先在 10 条上做小样本
python tools/portable_runner.py run conv-replay-oracle --limit 10
python tools/portable_runner.py run conv-replay-oracle

# ConvFinQA Self-History: --limit 指 *完整对话数*，不是轮次数
python tools/portable_runner.py run conv-replay-self --limit 5
python tools/portable_runner.py run conv-replay-self
```

另外可以运行 `finqa-original-test`、`finqa-transfer-test`、`finqa-direct-test`、`finqa-replay-test`，**但若正在复现训练，请不要根据 Test 结果再次调参**。FinQA 历史结果与这次新运行结果不共享输出目录。

## 7. 查看运行结果及保持历史证据

- 新代码只允许写到 `output_root`，默认 `/root/autodl-tmp/financial_portable_runs_v1/`，从而和原始冻结档案彻底隔离。
- `python tools/verify_source_snapshot.py --root .` 应继续显示 183 个源文件均匹配。
- `python tools/release_audit.py --root .` 仍应无安全错误。新写入的 `configs/paths.local.json` 受 `.gitignore` 保护，**但 `release_audit.py` 依旧会检查它，所以原路径字段可能被本地环境审查工具读取；请不要上传原本地配置**。
- `python tools/path_audit.py --root .` 仍会列出历史绝对路径，这是有意的源码保留；要运行项目请使用 `tools/portable_runner.py` 的新入口。
- 本工具不直接承担最终冻结报告的生成；当历史源码、输入、模型版本或 Prompt 改变，旧 SHA-256 与 Accuracy **不再可以照搬**。

## 8. 现阶段仍然存在的局限

1. 尚未在新服务器上完成 GPU 端到端回归，受 CUDA 与 vLLM 环境限制，实测结果可能存在差异。
2. 数据预处理使用历史版本脚本，需要官方数据及对应源目录。请参考 `finqa/data_prep` 和 `convfinqa/data_prep`，不同数据版本的 Prompt/答案可能不相同。
3. 原项目的 183 份已导出源码一字未动，其他**辅助入口**（例如历史 `run_full.sh`、`setup_links.sh`、最终冻结脚本）并不自动获得新路径支持。本补丁只支持上文列出的核心统一入口。
4. Replay 与 Self-History 的评估依赖已安装的 Transformers、vLLM 和 CUDA。请避免多个任务并行占用单卡显存。
5. 不应再公开携带本地凭证的 `pip freeze` 或任何模型权重文件；发布前检查 Git 暂存区和许可证要求。
