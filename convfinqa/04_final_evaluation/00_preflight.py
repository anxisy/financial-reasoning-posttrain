#!/usr/bin/env python3
"""CPU-side contract checks and immutable prompt fingerprint generation for FinQA Test."""
import argparse, importlib, json, os, platform, runpy, sys
from pathlib import Path
from common import (ROOT, OUT, BASE, BENCH, ADAPTERS, FINQA_TEST, FINQA_DEV,
                    FINQA_DEV_SHA, N_TEST, N_DEV, sha256, jsonread, jsonwrite)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--write',action='store_true',help='Save preflight JSON, once only')
    args=p.parse_args()
    expected=[FINQA_TEST,FINQA_DEV,BASE/'config.json',BENCH/'bench_vllm.py',BENCH/'common.py',BENCH/'metrics.py',BENCH/'answer_normalizer.py']
    expected.extend(p/'adapter_config.json' for p in ADAPTERS.values())
    for f in expected:
        print(('OK      ' if f.is_file() else 'MISSING ')+str(f))
    missing=[f for f in expected if not f.is_file()]
    if missing: raise SystemExit('STOP: missing input files: '+str(missing))
    sys.path.insert(0,str(BENCH))
    # Load the ORIGINAL frozen benchmark helper under a distinct module namespace.
    # Importing "common" normally here would incorrectly return this project's common.py.
    benchmark_common=runpy.run_path(str(BENCH/'common.py'))
    read_rows=benchmark_common['read_rows']
    prepare=benchmark_common['prepare']
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(BASE),local_files_only=True,trust_remote_code=True)
    dev=read_rows(FINQA_DEV,0)
    test=read_rows(FINQA_TEST,0)
    if len(dev)!=N_DEV or len(test)!=N_TEST:
        raise ValueError(f'Wrong FinQA split size dev={len(dev)} test={len(test)}')
    if set(str(x['id']) for x in dev) & set(str(x['id']) for x in test):
        raise ValueError('STOP: duplicate question ID in FinQA Test and Dev')
    for item in test:
        if 'gold_answer' not in item: raise ValueError('Test gold_answer missing for '+str(item['id']))
        split=str((item.get('metadata') or {}).get('source_split','')).lower()
        if split and split!='test': raise ValueError('Test row mislabeled split='+split)
    _,dev_ids,dev_sha=prepare(dev,tok)
    if dev_sha!=FINQA_DEV_SHA:
        raise ValueError(f'STOP: FinQA Dev fingerprint drift {dev_sha} != {FINQA_DEV_SHA}')
    _,test_ids,test_sha=prepare(test,tok)
    max_token=max(len(z) for z in test_ids)
    if max_token + 384 > 4096:
        raise ValueError(f'Test max prompt {max_token} +384 >4096; do not silently truncate')
    print(f'PASS: Dev={len(dev)}, Test={len(test)}, test prompt max={max_token}')
    print('FinQA TEST prompt token ID fingerprint:',test_sha)
    print('FinQA TEST input file SHA-256:',sha256(FINQA_TEST))
    adapter_fingerprints={}
    for name, folder in ADAPTERS.items():
        weights=sorted(list(folder.glob('*.safetensors'))+list(folder.glob('*.bin')))
        if not weights:raise FileNotFoundError(f'No weight file for {name} at {folder}')
        adapter_fingerprints[name]={
            'path':str(folder), 'config_sha256':sha256(folder/'adapter_config.json'),
            'weights':{w.name:sha256(w) for w in weights}}
    output={'status':'PASS','dataset':'FinQA','split':'test','count':len(test),'dev_count':len(dev),
            'test_input':str(FINQA_TEST),'test_input_sha256':sha256(FINQA_TEST),
            'dev_input_sha256':sha256(FINQA_DEV),'test_prompt_token_ids_sha256':test_sha,
            'dev_prompt_token_ids_sha256':dev_sha,'max_prompt_tokens':max_token,
            'max_new_tokens':384,'max_model_len':4096,'seed':42,
            'base':str(BASE),'base_config_sha256':sha256(BASE/'config.json'),
            'bench':str(BENCH/'bench_vllm.py'),'bench_sha256':sha256(BENCH/'bench_vllm.py'),
            'adapters':adapter_fingerprints,
            'environment':{'python':sys.version.split()[0],'platform':platform.platform()}}
    if args.write:
        target=OUT/'preflight.json'
        if target.exists():
            prior=jsonread(target)
            # Environment info may change; compare only frozen input/model contracts
            for key in ('test_input_sha256','test_prompt_token_ids_sha256','dev_prompt_token_ids_sha256','bench_sha256','base_config_sha256','adapters'):
                if prior[key]!=output[key]:raise ValueError(f'STOP: previous preflight changed: {key}')
            print('Existing preflight matches; skipping write:',target)
        else:
            jsonwrite(target,output)
            print('Saved preflight:',target)

if __name__=='__main__':main()
