#!/usr/bin/env bash
set -euo pipefail
unset OMP_NUM_THREADS || true
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
python env_check.py
python self_test.py
python train_grpo_1p7b.py --preflight
