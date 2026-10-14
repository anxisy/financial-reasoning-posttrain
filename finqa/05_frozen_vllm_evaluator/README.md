# FinQA Qwen3-1.7B: Transformers 与 vLLM 0.31.0 离线推理基准（v2）

**目标**：在 RTX 4090 24GB，加载**同一个** `Qwen3-1.7B-Base + Dynamic Sampling LoRA r=8`，对比原 Transformers/PEFT 与 vLLM 的完整 FinQA Dev 准确率、离线吞吐与峰值 GPU 显存。**只做推理，不训练、不重新选超参数、不触碰 Test。**

代码用的是你已经安装成功并通过单题 Smoke Test 的环境：

- **Transformers**：原 Conda `openr1`（torch 2.6.0 + transformers 4.52.3 等），保持冻结。
- **vLLM**：`/root/autodl-tmp/open-r1/19_vllm_benchmark/.venv`（uv / Python 3.12 / vLLM 0.31.0 / torch 2.13.0+cu132）。
- `env_check.py` 和 `self_test.py` 只做环境及数据评分自检，不加载 GPU 大模型。

## 设计保障

1. `manual_serialize` **逐字符对齐**之前的冻结评估器：`System:\n...\n\nUser:\n...\n\nAssistant:`，末尾 **没有**换行；**不调用 chat template**。
2. 两种引擎都使用 Base Model 的 tokenizer，并传入同样的完整 token ID 列表。通过 `prompt_token_ids_sha256` 校验输入一致性；不一致时比较脚本拒绝给出对照结果。
3. 统一 BF16（4090 支持），`do_sample=False` / `temperature=0`，每题最多生成 384 token；评分原封不动地复制旧版 `answer_normalizer.py`、`metrics.py`（数字相对容差 1e-3、绝对容差 1e-2、百分比容差 0.1）。
4. **排除**模型加载、warmup、整个数据集的 tokenization；**计入**逐批输入准备、生成、文本解码与评分的总墙钟时间。分别报告 `samples/sec`、`output tokens/sec` 和 GPU NVML 全设备峰值显存。GPU 必须空闲，两个引擎不能同时跑。
5. 两个引擎使用不同内核与调度，输出 **可能不逐字相同**。速度比较必须同时报告准确率和输出长度。`benchmark.json` 保存版本、参数、时间和 token fingerprint。

## 固定的 AutoDL 路径

```text
Base     /root/autodl-tmp/models/Qwen3-1.7B-Base
Adapter  /root/autodl-tmp/project/finqa_posttrain/experiments/grpo_dynamic_qwen3_1.7b/full_v1/final_adapter
Dev      /root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1/eval/finqa_eval_dev.jsonl
Output   /root/autodl-tmp/project/finqa_posttrain/experiments/vllm_benchmark_1p7b/
```

路径如果不同，在两个 benchmark 中同时设置 `--base --adapter --input`。

## 0. 上传 ZIP，解压

把 `finqa_vllm_benchmark_v2.zip` 上传到 `/root/autodl-tmp/open-r1/`，然后：

```bash
cd /root/autodl-tmp/open-r1
unzip finqa_vllm_benchmark_v2.zip
cd finqa_vllm_benchmark_v2
unset OMP_NUM_THREADS
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
python self_test.py
```

## 1. Transformers 小样本正确性检查（原 `openr1` 环境）

```bash
conda activate openr1
cd /root/autodl-tmp/open-r1/finqa_vllm_benchmark_v2
python -m pip install nvidia-ml-py
python env_check.py
python self_test.py
python bench_transformers.py \
  --max-samples 20 --batch-size 2 --warmup-samples 2 \
  --output-dir /root/autodl-tmp/project/finqa_posttrain/experiments/vllm_benchmark_1p7b/transformers_smoke20
```

20 道题的准确率**不代表整体**，只用于排查推理链路。

## 2. vLLM 小样本正确性检查（已建好的 uv 环境）

不需要重装 torch/vllm，也**不需要**再创建 `.venv`。

```bash
conda deactivate  # 若不在 conda 环境可忽略
source /root/autodl-tmp/open-r1/19_vllm_benchmark/.venv/bin/activate
cd /root/autodl-tmp/open-r1/finqa_vllm_benchmark_v2
uv pip install nvidia-ml-py
python env_check.py
python self_test.py
python bench_vllm.py \
  --max-samples 20 --request-batch-size 2 --warmup-samples 2 \
  --max-model-len 4096 --max-num-seqs 8 --gpu-memory-utilization 0.70 \
  --enforce-eager \
  --output-dir /root/autodl-tmp/project/finqa_posttrain/experiments/vllm_benchmark_1p7b/vllm_smoke20
```

`--enforce-eager` 只用于第一次批量检查，减少 CUDA Graph 编译/显存干扰。正式优化测试去掉此参数。

比较小样本：

```bash
ROOT=/root/autodl-tmp/project/finqa_posttrain/experiments/vllm_benchmark_1p7b
python compare_runs.py \
  --transformers "$ROOT/transformers_smoke20" \
  --vllm "$ROOT/vllm_smoke20" \
  --output-dir "$ROOT/compare_smoke20"
```

只有在小样本双引擎均成功生成且 `inputs_identical_token_ids=True` 后，继续正式评测。

## 3. 正式 883 道 FinQA Dev 评测

### Transformers 控制组（原 openr1）

```bash
deactivate  # 仅在当前已激活 uv .venv 时运行
conda activate openr1
cd /root/autodl-tmp/open-r1/finqa_vllm_benchmark_v2
ROOT=/root/autodl-tmp/project/finqa_posttrain/experiments/vllm_benchmark_1p7b
python bench_transformers.py \
  --max-samples 0 --batch-size 2 --warmup-samples 4 \
  --output-dir "$ROOT/transformers_full_b2"
```

完成后，核对其准确率是否与你之前冻结的 **60.1359% (531/883)** 相同，并用如下命令进行逐题核对：

```bash
python verify_frozen_eval.py \
  --benchmark-dir "$ROOT/transformers_full_b2"
```

如果准确率变化明显，先排查，不要立刻比较加速比。

### vLLM 等提交批量基准：2 个 Prompt / 调用（切换到 uv）

```bash
conda deactivate
source /root/autodl-tmp/open-r1/19_vllm_benchmark/.venv/bin/activate
cd /root/autodl-tmp/open-r1/finqa_vllm_benchmark_v2
ROOT=/root/autodl-tmp/project/finqa_posttrain/experiments/vllm_benchmark_1p7b
python bench_vllm.py \
  --max-samples 0 --request-batch-size 2 --warmup-samples 4 \
  --max-model-len 4096 --max-num-seqs 8 --gpu-memory-utilization 0.70 \
  --output-dir "$ROOT/vllm_full_b2"
```

**备注**：vLLM 虽然这里每次提交 2 道题，但内部还是使用自己的调度。这个场景可供理解小批量加速率。

### vLLM 高吞吐基准：把全部 883 道题一次提交调度

上一步稳定后再执行：

```bash
python bench_vllm.py \
  --max-samples 0 --request-batch-size 0 --warmup-samples 4 \
  --max-model-len 4096 --max-num-seqs 32 --gpu-memory-utilization 0.70 \
  --output-dir "$ROOT/vllm_full_queue"
```

`--request-batch-size 0` 是全部一次提交，不等于显存中同时存在 883 个序列。由 vLLM 用 `max_num_seqs=32` 管理同时执行数。相比 Transformers `batch_size=2`，这是**生产优化吞吐**对照，不是相同批大小的纯框架比较。

比较：

```bash
python compare_runs.py \
  --transformers "$ROOT/transformers_full_b2" \
  --vllm "$ROOT/vllm_full_b2" \
  --output-dir "$ROOT/compare_full_b2"
python compare_runs.py \
  --transformers "$ROOT/transformers_full_b2" \
  --vllm "$ROOT/vllm_full_queue" \
  --output-dir "$ROOT/compare_full_queue"
```

## 4. 最终结果文件

每个推理引擎的输出目录中：

```text
benchmark.json    # 加载、预热、正式推理耗时、GPU 显存、token SHA 等
metrics.json      # FinQA 原始统计口径，包括难度/运算类别
predictions.jsonl # 每题实际生成、数值答案、正确性
errors.jsonl      # 错题
```

比较结果：

```text
comparison.json   # 双引擎准确率、吞吐和实际加速倍数
changed_outputs.csv  # 两引擎输出发生变化的样本
```

最终简历**只填写实测结果**，并注明批量提交模式。例如：“在单卡 RTX 4090 上，使用 vLLM 对 1.7B LoRA 金融问答模型进行离线批量推理，固定 883 道 Dev 问题，验证相同输入 token 与评分逻辑，在高吞吐模式下实现 X 倍 samples/sec 加速（实际结果）”。

## 5. 异常处理

* `ModuleNotFoundError`: 检查当前 `which python` 是否指向对应 env。
* `torch.cuda.OutOfMemoryError`：先停其他 GPU 任务，把 `--max-num-seqs 32` 改为 `8`，`--gpu-memory-utilization` 降至 `0.65`，必要时添加 `--enforce-eager`；**不要改模型、Prompt、评分容差**。
* `max_model_len` 太小：如出现提示，核对最大 prompt 长度与预留的 384 tokens，再将 `--max-model-len` 调大，并记录配置。
* Prompt SHA 不同：说明 tokenizer / 模板 / tokenization 不一致，停止比较，先定位差异。
* 准确率明显变差：排查 `changed_outputs.csv` 里的回答与两边 token ID；不能以准确率损失为代价只宣传速度。
* vLLM 的单条推理首次运行速度慢：本实验会预热并比较整批数据，不能用单题 Smoke Test 的 4.45 tokens/s 评价整套吞吐。
* 峰值显存是**整张 GPU 的 NVML 已用显存**，不是只测模型显存；必须保证同一张 GPU 没有其他任务。
