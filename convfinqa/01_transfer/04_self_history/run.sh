#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/ConvFinQA
python 00_common/eval_self_history_v1.py --variants transfer "$@"
