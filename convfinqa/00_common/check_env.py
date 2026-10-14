#!/usr/bin/env python3
from pathlib import Path
import importlib
import sys
import torch
roots={
'Base model':'/root/autodl-tmp/models/Qwen3-1.7B-Base',
'FinQA SFT adapter':'/root/autodl-tmp/project/finqa_posttrain/experiments/sft_qwen3_1.7b/full_v1/final_adapter/adapter_config.json',
'ConvFinQA v2 train':'/root/autodl-tmp/project/convfinqa_posttrain/data/processed_v2/sft/train.jsonl',
'ConvFinQA v2 dev':'/root/autodl-tmp/project/convfinqa_posttrain/data/processed_v2/sft/dev.jsonl',
'ConvFinQA v2 eval':'/root/autodl-tmp/project/convfinqa_posttrain/data/processed_v2/eval/dev.jsonl',
}
print('Python:',sys.version,'CUDA:',torch.cuda.is_available(),'BF16:',torch.cuda.is_bf16_supported() if torch.cuda.is_available() else 'N/A')
for mod in ['torch','transformers','peft','accelerate']:
    p=importlib.import_module(mod)
    print(mod, getattr(p,'__version__','unknown'))
missing=[]
for label,path in roots.items():
    ok=Path(path).exists(); print('[OK]' if ok else '[MISSING]',label,path)
    if not ok: missing.append(label)
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or missing:
    raise SystemExit('ENV CHECK FAILED. Missing: '+', '.join(missing))
print('ENV CHECK PASSED')
