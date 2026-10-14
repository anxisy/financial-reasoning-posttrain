#!/usr/bin/env python3
"""Read-only preflight for the original frozen FinQA Dev benchmark. No GPU required."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path('/root/autodl-tmp')
PROJECT = ROOT/'project/convfinqa_posttrain/experiments/convfinqa_sft_v2'
BENCH = ROOT/'open-r1/20_finqa_vllm_benchmark_v2'
BASE = ROOT/'models/Qwen3-1.7B-Base'
DEV = ROOT/'project/finqa_posttrain/data/processed_v1_1_1/eval/finqa_eval_dev.jsonl'
ADAPTERS = {
  'original_sft': ROOT/'project/finqa_posttrain/experiments/sft_qwen3_1.7b/full_v1/final_adapter',
  'transfer': PROJECT/'01_finqa_sft_adapter/full/final_adapter',
  'direct': PROJECT/'02_qwen3_base/full/final_adapter',
}
FROZEN_SHA = '752ba9302ba1153bbdcae95d0d9e743b1f00576ab75761939e821dbd678a58af'

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-sha', default=FROZEN_SHA)
    a = p.parse_args()
    checks = [BASE/'config.json', DEV, BENCH/'bench_vllm.py', BENCH/'common.py', BENCH/'metrics.py', BENCH/'answer_normalizer.py']
    checks.extend(adapter/'adapter_config.json' for adapter in ADAPTERS.values())
    for path in checks:
        ok = path.is_file()
        print(f'{"OK" if ok else "MISSING"}: {path}')
    missing = [str(x) for x in checks if not x.is_file()]
    if missing:
        raise SystemExit('Missing required files; do not run evaluation:\n' + '\n'.join(missing))
    sys.path.insert(0, str(BENCH))
    from common import read_rows, prepare
    from transformers import AutoTokenizer
    rows = read_rows(str(DEV), 0)
    if len(rows) != 883:
        raise SystemExit(f'Expected 883 FinQA Dev items, got {len(rows)}')
    tok = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True, trust_remote_code=True)
    _, token_ids, fingerprint = prepare(rows, tok)
    print('Rows:', len(rows))
    print('Maximum prompt tokens:', max(map(len, token_ids)))
    print('FinQA Dev prompt fingerprint:', fingerprint)
    print('Expected frozen fingerprint:', a.expected_sha)
    if fingerprint != a.expected_sha:
        raise SystemExit('STOP: Prompt tokenizer/serialization fingerprint differs from frozen FinQA benchmark.')
    print('PASS: all 3 adapter paths, original FinQA Dev, and frozen prompt fingerprint.')

if __name__ == '__main__': main()
