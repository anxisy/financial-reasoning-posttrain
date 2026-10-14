# Qwen3-1.7B FinQA Vanilla GRPO v1

This package is the 1.7B scale-up of the already-validated 0.6B Vanilla GRPO implementation.
Only model/adapter/output/run-name paths were changed. The GRPO recipe stays frozen:

- start from Qwen3-1.7B Base + the new 1.7B Full SFT adapter
- FinQA Train 6251, 1 epoch
- G=4
- max_prompt_length=1792
- max_completion_length=256
- temperature=0.8, top_p=0.95
- batch=4, grad_acc=1
- lr=5e-6, warmup=0.1
- beta=0.04, epsilon=0.2
- correctness reward=1.0, format reward=0.1
- explicit frozen 1.7B SFT reference policy for KL
- TRL 0.18.0, no vLLM

Current 1.7B Dev baseline before GRPO:
- Base: 35.67%
- LoRA-SFT: 55.49%

IMPORTANT: run preflight first. On a 24GB RTX 4090, the explicit train policy + frozen reference is the main memory risk.
Do not start Full if preflight OOMs.


# Patch 1：Preflight 梯度流验证

第一版 KL Preflight 虽然成功完成并出现 `kl`，但 PyTorch 打印了：

```text
None of the inputs have requires_grad=True. Gradients will be None
```

因此在正式 Full GRPO 前，本补丁增加三层保护：

1. `gradient_checkpointing_kwargs={"use_reentrant": False}`
2. Trainer 初始化后再次调用 `enable_input_require_grads()`
3. Preflight 对所有 trainable LoRA 参数做训练前/训练后的快照比较

此外，Preflight 强制 `logging_steps=1`，因此可以逐 step 看到：

```text
reward_std
grad_norm
kl
```

如果出现 mixed-reward group，但 LoRA 参数完全没有变化，脚本会直接报错并阻止把 Preflight 当作成功。

---

# FinQA Full Vanilla GRPO v1

这是 1.7B 工程验证阶段的正式 Vanilla GRPO baseline。

## 固定配置

```text
Initial policy:
Qwen3-1.7B-Base + ORIGINAL Full SFT final_adapter

Train:
FinQA Train 全部 6251 prompts
1 epoch

GRPO:
num_generations = 4
max_prompt_length = 1792
max_completion_length = 256
temperature = 0.8
top_p = 0.95

per_device_train_batch_size = 4
gradient_accumulation_steps = 1

learning_rate = 5e-6
warmup_ratio = 0.1

loss_type = grpo
scale_rewards = group
epsilon = 0.2

Reward:
correctness = 1.0
format = 0.1

KL:
beta = 0.04
reference = 冻结的原始 SFT policy
```

---

# 重要：为什么正式版对 KL 做了额外处理？

当前环境是：

```text
TRL 0.18.0
PEFT 0.21.0
```

在 TRL 0.18.0 中，如果直接把已经训练好的 `PeftModel` 交给
`GRPOTrainer` 并设置 `beta > 0`，Trainer 的 PEFT reference 路径会通过
**disable adapter** 得到 reference。

对于我们的项目，这会变成：

```text
reference = Qwen3-1.7B Base
```

而不是我们真正希望的：

```text
reference = GRPO 开始前的 Full SFT policy
```

因此正式代码会额外加载：

```text
Qwen3-1.7B-Base + 同一个 Full SFT adapter
```

作为**冻结 reference policy**，然后显式安装到：

```text
trainer.ref_model
```

所以正式的 KL 是：

```text
GRPO current policy
vs
initial Full SFT policy
```

这比直接对 bare Base 做 KL 更符合本项目的阶段设计。

---

# 1. 解压

```bash
cd /root/autodl-tmp/open-r1

mkdir -p 10_grpo_full
cd 10_grpo_full

unzip finqa_vanilla_grpo_full_v1.zip
cd finqa_vanilla_grpo_full_v1
```

如果你把文件直接解压到了 `10_grpo_full`，就不需要最后一个 `cd`。

---

# 2. 环境检查

```bash
python env_check.py
```

应继续看到：

```text
trl         : 0.18.0
CUDA        : True
GPU         : NVIDIA GeForce RTX 4090
BF16        : True
```

---

# 3. Self test

```bash
python self_test.py
```

必须出现：

```text
SELF TEST PASSED
```

---

# 4. 强烈建议先跑 KL Preflight

正式版和 Smoke 最大的新变化是：

```text
beta 0.0 -> 0.04
+
explicit frozen SFT reference
```

因此不要直接开始 6+ 小时 Full run。

先运行：

```bash
python train_grpo_1p7b.py --preflight
```

默认会：

```text
32 prompts
8 optimizer steps
G=4
beta=0.04
```

并写入独立目录：

```text
/root/autodl-tmp/project/finqa_posttrain/experiments/
grpo_qwen3_1.7b/preflight_kl_v1/
```

它绝对不会作为 Full GRPO 的初始化 checkpoint。

Preflight 最重要的通过条件：

```text
1. 没有 OOM / Traceback
2. reference_mode = explicit_frozen_sft_copy
3. trainer_log_history.json 中出现 kl
4. 能保存 final_adapter
5. reward / diagnostics 正常生成
```

如果碰巧 8 个 group 全是 zero-variance，`grad_norm=0` 也不一定是工程失败；
Smoke 已经验证过 mixed group 时确实有梯度。

---

# 5. 正式 Full Vanilla GRPO

Preflight 通过后，正式 run 必须重新从原始 Full SFT adapter 开始。

直接：

```bash
python train_grpo_1p7b.py
```

默认输出：

```text
/root/autodl-tmp/project/finqa_posttrain/experiments/
grpo_qwen3_1.7b/full_v1/
```

主要文件：

```text
checkpoint-*/
final_adapter/

run_config.json
rollout_diagnostics.jsonl
group_diagnostics.jsonl
diagnostics_summary.json
trainer_log_history.json
full_report.json
```

---

# 6. 中断续训

每 250 optimizer steps 保存一次 checkpoint，最多保留最近两个。

查看：

```bash
ls -d \
/root/autodl-tmp/project/finqa_posttrain/experiments/grpo_qwen3_1.7b/full_v1/checkpoint-*
```

自动从最新 checkpoint 续训：

```bash
python train_grpo_1p7b.py \
  --resume-from-checkpoint latest
```

或者指定：

```bash
python train_grpo_1p7b.py \
  --resume-from-checkpoint \
  /root/autodl-tmp/project/finqa_posttrain/experiments/grpo_qwen3_1.7b/full_v1/checkpoint-2500
```

注意：续训时不要加 `--overwrite-output-dir`。

---

# 7. 不要从 Smoke adapter 开始

正确：

```text
Full SFT final_adapter
├── Smoke GRPO
└── Full GRPO
```

错误：

```text
Full SFT
↓
Smoke GRPO
↓
Full GRPO
```

脚本默认会阻止看起来像 GRPO/Smoke adapter 的初始化路径。

---

# 8. Full 完成后先 Reload Check

```bash
python reload_grpo_1p7b.py
```

它会用官方评测仍然使用的：

```text
Manual Prompt
greedy
max_new_tokens = 384
```

生成 3 条 Dev。

这只是加载检查，不是正式 Accuracy。

---

# 9. 然后先跑 GRPO Dev Evaluation

继续复用之前已经冻结的 evaluator：

```bash
python /root/autodl-tmp/open-r1/07_sft_eval/evaluate_finqa_peft.py \
  --base-model-path /root/autodl-tmp/models/Qwen3-1.7B-Base \
  --adapter-path /root/autodl-tmp/project/finqa_posttrain/experiments/grpo_qwen3_1.7b/full_v1/final_adapter \
  --input /root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1/eval/finqa_eval_dev.jsonl \
  --output-dir /root/autodl-tmp/project/finqa_posttrain/experiments/grpo_qwen3_1.7b/eval_dev_v1 \
  --batch-size 2 \
  --max-new-tokens 384 \
  --prompt-format manual
```

如果你的 `evaluate_finqa_peft.py` 实际不在这个路径，就把命令第一行替换为它的真实路径。

---

# 10. 比较 SFT vs GRPO Dev

```bash
python compare_metrics.py \
  --before /root/autodl-tmp/project/finqa_posttrain/experiments/sft_qwen3_1.7b/eval_dev_v1/metrics.json \
  --after /root/autodl-tmp/project/finqa_posttrain/experiments/grpo_qwen3_1.7b/eval_dev_v1/metrics.json \
  --before-name SFT \
  --after-name GRPO
```

先看 Dev，不要先根据 Test 调 GRPO。

---

# 11. 训练期间重点看哪些指标？

`diagnostics_summary.json`：

```text
correctness_reward_mean

zero_variance_group_ratio
all_correct_group_ratio
all_wrong_group_ratio
mixed_group_ratio

group_reward_variance_mean

unique_completion_ratio_mean

response_length_mean
response_length_p95
max_length_hit_rate

by_difficulty
by_operator
```

`trainer_log_history.json`：

```text
kl
grad_norm
reward
reward_std
frac_reward_zero_std
clip_ratio/region_mean
```

---

# 12. Full 跑完后先发什么？

请优先发：

```text
full_report.json
diagnostics_summary.json
run_config.json
trainer_log_history.json
```

以及：

```bash
python reload_grpo_1p7b.py
```

的输出。

如果你已经顺便跑完 Dev Evaluation，再额外发：

```text
grpo_qwen3_1.7b/eval_dev_v1/metrics.json
```

以及：

```text
compare_metrics.py
```

终端输出。

我们会据此决定：

```text
Vanilla GRPO 是否有效
↓
zero-variance 到底有多严重
↓
Dynamic Sampling v1 应该如何正式实现
```
