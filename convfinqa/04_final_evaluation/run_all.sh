#!/usr/bin/env bash
set -euo pipefail
cd "${AUTODL_ROOT:-/root/autodl-tmp}/ConvFinQA"
python 04_final_evaluation/self_test.py
python 04_final_evaluation/00_preflight.py --write
bash 04_final_evaluation/01_run_finqa_test.sh all
python 04_final_evaluation/02_compare_finqa_test.py --bootstrap 5000 --seed 42
python 04_final_evaluation/03_compare_self_history.py --bootstrap 5000 --seed 42
python 04_final_evaluation/04_freeze_results.py
python 04_final_evaluation/05_verify_frozen.py --check-sources
