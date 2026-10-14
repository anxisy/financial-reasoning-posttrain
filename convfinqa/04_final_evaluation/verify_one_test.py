#!/usr/bin/env python3
"""Strict validator for each FinQA Test output; no GPU and no writes."""
import argparse
from pathlib import Path
from common import OUT, FINQA_TEST, ADAPTERS, BASE, BENCH, N_TEST, MODEL_ORDER, jsonread, predictions, validate_metrics, sha256

def verify(folder, variant):
    f=Path(folder)
    preds=predictions(f/'predictions.jsonl',N_TEST)
    fingerprint=validate_metrics(f,preds,kind='finqa')
    pf=jsonread(OUT/'preflight.json')
    adapter_info=pf['adapters'][variant]
    for filename, expected_hash in adapter_info['weights'].items():
        if sha256(ADAPTERS[variant]/filename)!=expected_hash:
            raise ValueError(f'{variant}: Adapter weights changed after preflight')
    if pf['test_input_sha256']!=sha256(FINQA_TEST):
        raise ValueError('Test input changed since preflight')
    if fingerprint!=pf['test_prompt_token_ids_sha256']:
        raise ValueError(f'{variant}: prompt fingerprint differs from preflight: {fingerprint}')
    benchmark=jsonread(f/'benchmark.json')
    run=benchmark['run']
    for key, expected in {'adapter':str(ADAPTERS[variant]),'input':str(FINQA_TEST),
                          'base':str(BASE),'seed':42,'request_batch_size':0,'max_num_seqs':32}.items():
        if run.get(key)!=expected: raise ValueError(f'{variant}: {key}: {run.get(key)!r} != {expected!r}')
    acc=sum(r['correct'] for r in preds.values())/N_TEST
    print(f'PASS {variant}: {N_TEST} Test predictions, acc={acc:.2%}, same frozen prompt/scorer')
    return preds

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('folder',type=Path)
    ap.add_argument('variant',choices=MODEL_ORDER)
    a=ap.parse_args()
    verify(a.folder,a.variant)

if __name__=='__main__': main()
