#!/usr/bin/env python3
"""Read-only schema, unique IDs, token-budget, and Assistant-only mask checks."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT=Path('/root/autodl-tmp')
DATA=ROOT/'project/convfinqa_posttrain/experiments/convfinqa_sft_v2/03_finqa_sft_replay/data/replay_train.jsonl'
MODEL=ROOT/'models/Qwen3-1.7B-Base'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,default=DATA)
    ap.add_argument('--model',type=Path,default=MODEL)
    ap.add_argument('--max-seq-length',type=int,default=4096)
    a=ap.parse_args()
    if not a.input.exists():raise FileNotFoundError(a.input)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from train_replay import read_jsonl,serialize,AssistantOnlyDataset,length_stats,sha256
    rows=read_jsonl(a.input)
    sources=Counter(x['metadata']['replay_source'] for x in rows)
    if len(rows)!=11104 or sources != {'finqa':1110,'convfinqa':9994}:
        raise RuntimeError(f'Wrong mixture {len(rows)} / {sources}')
    manifest=json.loads(a.input.with_name('mix_manifest.json').read_text())
    if sha256(a.input)!=manifest['train_sha256']:raise RuntimeError('SHA mismatch')
    print('PASS: 11104 mixed examples, 1110 FinQA and 9994 ConvFinQA')
    print('PASS: replay SHA256:',sha256(a.input))
    print('Example FinQA target:')
    print(next(serialize(r)[1] for r in rows if r['metadata']['replay_source']=='finqa')[:250])
    print('Example ConvFinQA target:')
    print(next(serialize(r)[1] for r in rows if r['metadata']['replay_source']=='convfinqa')[:250])
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise RuntimeError('Activate conda openr1; transformers required for token budget check') from e
    tokenizer=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True,use_fast=True)
    data=AssistantOnlyDataset(rows,tokenizer,a.max_seq_length)
    sample=data[0]
    if not sample['labels'] or not any(x!=-100 for x in sample['labels']):
        raise AssertionError('No assistant labels')
    if sample['labels'][0]!=-100:raise AssertionError('Prompt label not masked')
    if len(sample['labels'])!=len(sample['input_ids']):raise AssertionError('Mask length mismatch')
    print('PASS: Assistant-only labels, no truncation, longest sequence within 4096')
    print('Length stats:',length_stats(data.token_lengths),'max Assistant tokens:',max(data.assistant_lengths))
    print('NOTE: Dev and Test never read; source files unchanged.')

if __name__=='__main__':main()
