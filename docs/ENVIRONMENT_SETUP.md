# 环境配置与运行指南（AutoDL / Linux）

此指南对应项目已实际运行的 **Qwen3-1.7B-Base + 单张 RTX 4090 24 GiB** 环境。训练与 vLLM 推理需要分开创建环境，**不要在同一个环境里同时强制安装两套不同的 PyTorch**。

> **复现级别说明**：下面包含已观察到的关键软件版本与建议的重建命令，但不等同于从原机器导出的完整锁文件。想要严格复现，请先按第 4 节导出两个环境的实际 `pip freeze`，并检查 wheel 来源与 CUDA Driver。建议先在原服务器继续使用已经验证可工作的环境，不必无故重装。

## 1. 系统与项目目录

- GPU：NVIDIA RTX 4090（24 GiB），建议驱动满足对应 CUDA 构建的最低要求。
- 原实验训练环境：`conda activate openr1`，Python 3.11（原 FlashAttention wheel 对应 `cp311`；请以 `python -V` 实际核对）。
- 原实验训练主要版本：PyTorch `2.6.0+cu124`、Transformers `4.52.3`、TRL `0.18.0`、PEFT `0.21.0`。
- 原 vLLM 环境：`/root/autodl-tmp/open-r1/19_vllm_benchmark/.venv`，vLLM `0.31.0`；**不与训练环境共享 PyTorch**。

原始实验的关键目录（原始脚本中的部分路径仍然是绝对路径）：

```text
/root/autodl-tmp/
├── models/Qwen3-1.7B-Base/                 # 基座权重（需自行准备）
├── project/finqa_posttrain/
│   ├── data/processed_v1_1_1/
│   └── experiments/
├── project/convfinqa_posttrain/
│   ├── data/processed_v2/
│   └── experiments/
├── open-r1/                                # 原 FinQA 实验工作目录
└── ConvFinQA/                              # 原 ConvFinQA 实验工作目录
```

**重要**：本 GitHub 仓库是源码的整理快照，并未自动重写所有原始脚本里的绝对路径。对于不同服务器，应先修改路径配置，检查脚本引用并运行 Smoke Test。不能保证直接在 `financial-reasoning-posttrain-release/` 根目录里运行旧 shell 脚本就能训练。

## 2. 训练环境：Conda `openr1`

### 2.1 原 AutoDL 服务器（推荐）

```bash
conda activate openr1
python -V
python - <<'PY'
import torch, transformers, peft, trl
print('torch=', torch.__version__, 'torch_cuda=', torch.version.cuda)
print('transformers=', transformers.__version__)
print('peft=', peft.__version__, 'trl=', trl.__version__)
print('GPU=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')
PY
```

### 2.2 新建训练环境（重建参考）

已安装 Conda 且驱动兼容时，可以采用：

```bash
conda create -n openr1 python=3.11 -y
conda activate openr1
python -m pip install --upgrade pip
# CUDA 12.4 版本 PyTorch；不需要另装系统级 CUDA toolkit 才能运行普通 torch CUDA wheel
python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
# 其他核心训练依赖；完整传递依赖以原环境 freeze 为准
python -m pip install -r requirements/train_core.txt
python -m pip check
```

该 `requirements/train_core.txt` **故意不包含 PyTorch**，因为它从 CUDA 专属索引单独安装。若采用另一显卡和 Driver，需先选择兼容的 PyTorch wheel；不能照搬 `cu124`。

执行 `python tools/check_runtime.py --mode train` 检查训练核心依赖。`flash-attn` 并非必须；之前曾遇到手动安装 FlashAttention wheel 导致 Qwen3 加载失败，所以**不建议仅为复现而强行安装 FlashAttention**。先以训练成功的原始配置为准。

## 3. 推理环境：uv + vLLM（与训练隔离）

### 3.1 继续使用已跑通的环境（推荐）

```bash
# 不要在训练环境中叠加 vLLM
conda deactivate
source /root/autodl-tmp/open-r1/19_vllm_benchmark/.venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

python - <<'PY'
import torch, vllm
print('torch=', torch.__version__, 'torch_cuda=', torch.version.cuda)
print('vllm=', vllm.__version__)
PY
```

此前使用 `VLLM_USE_FLASHINFER_SAMPLER=0` 是为规避 FlashInfer JIT 兼容问题；不是要求关闭 FlashAttention 注意力后端。

### 3.2 新建 uv 环境（重建参考）

下述命令以本机已有 `uv` 为前提；若没有，可以参照 uv 官方安装文档安装并确认 `uv --version`：

```bash
cd /root/autodl-tmp
uv venv --python 3.11 /root/autodl-tmp/finqa-vllm-env
uv pip install --python /root/autodl-tmp/finqa-vllm-env/bin/python -r \
  financial-reasoning-posttrain-release/requirements/vllm_core.txt
source /root/autodl-tmp/finqa-vllm-env/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
python -m pip check    # 若环境没有 pip，可以改为 uv pip check
```

> **兼容性提醒**：vLLM 通常会选择自己兼容的 PyTorch/CUDA 轮子；重建后版本可能与最初服务器不同。请用 `python tools/check_runtime.py --mode inference` 检查，并先用少量题目做数值结果对照。GPU/驱动、Python 和 Linux wheel 是否匹配需要在目标机器验证。不要把训练 Conda 的 torch 版本强制覆盖到 vLLM 环境。

## 4. 导出真实环境版本（发布前建议做）

在原 AutoDL 上分别运行，**不要将两个 freeze 合并**：

```bash
mkdir -p /root/autodl-tmp/finqa_env_snapshots
conda activate openr1
python -m pip freeze > /root/autodl-tmp/finqa_env_snapshots/train_pip_freeze.txt
conda list --explicit > /root/autodl-tmp/finqa_env_snapshots/train_conda_explicit.txt
conda deactivate
source /root/autodl-tmp/open-r1/19_vllm_benchmark/.venv/bin/activate
python -m pip freeze > /root/autodl-tmp/finqa_env_snapshots/vllm_pip_freeze.txt
```

**审查后再公开**。`pip freeze` 可能含 `file:///...`、私有仓库链接、个人镜像地址、本地绝对路径或不可迁移的包引用。必要时只公布经过人工审查的版本清单，而不是原样提交整个环境。

## 5. 数据准备与模型依赖

- FinQA 原始划分：Train 6,251、Dev 883、Test 1,147；本项目采用 `processed_v1_1_1` 转换后文件。
- ConvFinQA 原始训练 3,037 段对话，经逐轮展开为 11,104 条；Dev 421 段对话，1,490 个轮次；本项目采用 `processed_v2`。
- 模型：Qwen3-1.7B-Base 本地目录；LoRA 训练输出与最终结果仓库**不含权重**，需要自行恢复已保存的 Adapter。
- 原训练与推理入口参见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)；此 GitHub 仓库保留实际实验源码，但无法凭此文档自动生成外部原始数据与 Adapter。

## 6. 先验证再复现

```bash
# 在训练 Conda 环境里执行
python tools/check_runtime.py --mode train
# 在 vLLM 环境里执行
python tools/check_runtime.py --mode inference
# 代码静态检查
python tools/release_audit.py --root .
python tools/verify_source_snapshot.py --root .
python tools/path_audit.py --root . --limit 20
```

正常情况下，静态检查可以通过；`path_audit.py` 仍会报告原实验中硬编码的 AutoDL 路径。此结果是有意保留的源码事实，不代表路径已实现自动移植。**先核对路径与数据，再运行小规模 Smoke Test，最后才运行全量训练或最终评估**。


## 7. 可配置的 GitHub 运行入口

上文描述的是历史工作目录的实际配置。**如从新的 GitHub 发布目录运行核心实验**，请先查阅 [`PORTABLE_REPRODUCTION.md`](PORTABLE_REPRODUCTION.md)，使用 `python tools/portable_runner.py init --workspace /root/autodl-tmp` 建立本地路径配置、`check` 预检查和 `run` 启动；不要直接执行历史脚本中依赖 `cd /root/autodl-tmp/ConvFinQA` 的旧 Shell 入口。
