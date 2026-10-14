#!/usr/bin/env python3
"""Call unchanged ConvFinQA neutral-header evaluator with an explicit adapter."""
from __future__ import annotations
import argparse
import importlib
import sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=['transfer','direct','replay'], required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--base', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--max-conversations', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    a=p.parse_args()
    if not (a.adapter/'adapter_config.json').is_file(): raise FileNotFoundError(a.adapter)
    if a.output_dir.exists(): raise FileExistsError(f'Existing evaluation: {a.output_dir}')
    common=Path(__file__).resolve().parent.parent/'convfinqa'/'00_common'
    sys.path.insert(0,str(common))
    evaluator=importlib.import_module('eval_self_history_v1')
    evaluator.ADAPTERS[a.variant]=a.adapter.resolve()
    sys.argv=[str(common/'eval_self_history_v1.py'), '--variants',a.variant,
              '--input',a.input, '--base',a.base,'--output-dir',str(a.output_dir),
              '--max-conversations',str(a.max_conversations),'--seed',str(a.seed)]
    evaluator.main()

if __name__=='__main__': main()
