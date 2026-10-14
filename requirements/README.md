# 训练 / 推理环境说明

项目原环境相互不兼容，**不要将两套依赖装进同一个 Python 环境**。

- **训练 Conda `openr1`**：torch `2.6.0+cu124`、transformers `4.52.3`、trl `0.18.0`、peft `0.21.0`，其他依赖需依据原环境冻结快照。
- **vLLM `.venv`**：vLLM `0.31.0`，采用与训练 Conda 不同的 PyTorch/CUDA 构建；`VLLM_USE_FLASHINFER_SAMPLER=0`。

这不是通用、自动可安装的锁定依赖文件；为防止凭空推断包版本而产生不可复现环境，建议从原 AutoDL 环境分别导出 `pip freeze` 并检查私人地址后再提交。
