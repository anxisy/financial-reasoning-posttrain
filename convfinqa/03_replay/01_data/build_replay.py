#!/usr/bin/env python3
"""Build a frozen 10% FinQA / 90% ConvFinQA replay SFT mixture.

Deterministic, with no replacement, preserves original system/user/assistant text.
No Dev/Test used; output is under new experiment only.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path('/root/autodl-tmp')
CONV = ROOT/'project/convfinqa_posttrain/data/processed_v2/sft/train.jsonl'
FINQA = ROOT/'project/finqa_posttrain/data/processed_v1_1_1/sft/finqa_sft_train.jsonl'
OUT = ROOT/'project/convfinqa_posttrain/experiments/convfinqa_sft_v2/03_finqa_sft_replay/data'


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def load(path):
    rows=[]; ids=set()
    with Path(path).open(encoding='utf8') as f:
        for n,line in enumerate(f,1):
            if not line.strip():continue
            r=json.loads(line)
            ident=str(r.get('id',''))
            if not ident or ident in ids:raise ValueError(f'duplicate/missing ID {ident} at {path}:{n}')
            ids.add(ident)
            msgs=r.get('messages')
            if not isinstance(msgs,list) or [m.get('role') for m in msgs] != ['system','user','assistant']:
                raise ValueError(f'bad message schema {path}:{n}')
            if not all(isinstance(m.get('content'),str) and m['content'].strip() for m in msgs):
                raise ValueError(f'empty content {path}:{n}')
            answer=msgs[-1]['content']
            if answer.count('<answer>')!=1 or answer.count('</answer>')!=1:
                raise ValueError(f'missing or multiple answer tags {path}:{n}')
            rows.append(r)
    if not rows: raise ValueError('empty input file: '+str(path))
    return rows


def build(conv,finqa,finqa_n,seed):
    if not 0 < finqa_n < len(conv):raise ValueError('0 < finqa-count < ConvFinQA size required')
    if finqa_n>len(finqa):raise ValueError('FinQA count exceeds unique FinQA training records')
    rng=random.Random(seed)
    nconv=len(conv)-finqa_n
    # Stratify number/program to retain the original ConvFinQA question-type balance.
    groups={t:[r for r in conv if r['metadata'].get('cur_type')==t] for t in ('number_turn','program_turn')}
    if sum(map(len,groups.values()))!=len(conv):raise ValueError('unexpected ConvFinQA cur_type')
    nnum=round(nconv*len(groups['number_turn'])/len(conv))
    chosen_conv=rng.sample(groups['number_turn'],nnum)+rng.sample(groups['program_turn'],nconv-nnum)
    chosen_finqa=rng.sample(finqa,finqa_n)
    selected=[]
    for row in chosen_conv:
        r=copy.deepcopy(row)
        original=str(r['id'])
        meta=r.setdefault('metadata',{})
        if meta.get('history_mode')!='oracle_gold_answers':raise ValueError(f'wrong oracle history {original}')
        r['id']='convfinqa::'+original
        meta['replay_source']='convfinqa'
        meta['replay_original_id']=original
        selected.append(r)
    for row in chosen_finqa:
        r=copy.deepcopy(row)
        original=str(r['id'])
        r['id']='finqa::'+original
        meta=r.setdefault('metadata',{})
        # A FinQA original single-turn prompt must not be mislabelled as oracle history.
        if meta.get('history_mode')=='oracle_gold_answers':
            raise ValueError('FinQA input unexpectedly marked as oracle history: '+original)
        meta['replay_source']='finqa'
        meta['replay_original_id']=original
        selected.append(r)
    rng.shuffle(selected)
    if len(selected)!=len(conv) or len({r['id'] for r in selected})!=len(selected):
        raise AssertionError('wrong size/duplicate IDs')
    return selected


def write(path, rows):
    with path.open('w',encoding='utf8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--conv',type=Path,default=CONV)
    ap.add_argument('--finqa',type=Path,default=FINQA)
    ap.add_argument('--output-dir',type=Path,default=OUT)
    ap.add_argument('--finqa-count',type=int,default=1110)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--overwrite',action='store_true',help='Only if rebuilding intentionally; never change seed after training.')
    a=ap.parse_args()
    for p in (a.conv,a.finqa):
        if not p.is_file():raise FileNotFoundError(p)
    out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    dest=out/'replay_train.jsonl'; manifest=out/'mix_manifest.json'
    if (dest.exists() or manifest.exists()) and not a.overwrite:
        raise FileExistsError('Frozen replay dataset exists. No overwrite: '+str(out))
    conv=load(a.conv); finqa=load(a.finqa)
    if len(conv)!=11104 or len(finqa)!=6251:
        raise ValueError(f'Unexpected source size conv={len(conv)} finqa={len(finqa)}; inspect frozen datasets')
    rows=build(conv,finqa,a.finqa_count,a.seed)
    write(dest,rows)
    c=Counter(r['metadata']['replay_source'] for r in rows)
    types=Counter(r['metadata'].get('cur_type','unspecified') for r in rows if r['metadata']['replay_source']=='convfinqa')
    expected_steps=len(rows)//8
    rep={'schema':'convfinqa_finqa_replay_v1', 'seed':a.seed,
      'source_paths':{'convfinqa':str(a.conv),'finqa':str(a.finqa)},
      'source_sha256':{'convfinqa':digest(a.conv),'finqa':digest(a.finqa)},
      'train_file':str(dest),'train_sha256':digest(dest),'total_rows':len(rows),
      'finqa_rows':c['finqa'],'convfinqa_rows':c['convfinqa'],
      'finqa_fraction':c['finqa']/len(rows),'convfinqa_type_counts':dict(types),
      'batch_size':1,'gradient_accumulation_steps':8,'expected_optimizer_steps':expected_steps,
      'notes':['No Dev or Test used','No sample repeated in mix','Sample budget matches experiment 01',
               'About 10% of original ConvFinQA samples replaced by FinQA Train samples',
               'Both datasets retain their own original prompts and targets','FinQA single-turn training rows are not oracle multi-turn history']}
    manifest.write_text(json.dumps(rep,ensure_ascii=False,indent=2),encoding='utf8')
    print('=== REPLAY MIX CREATED ===')
    print('total=',len(rows),'convfinqa=',c['convfinqa'],'finqa=',c['finqa'],
          f'FinQA fraction={100*c["finqa"]/len(rows):.3f}%',flush=True)
    print('ConvFinQA types:',dict(types),'optimizer steps:',expected_steps)
    print('Dataset:',dest,'\nManifest:',manifest)

if __name__=='__main__':main()
