#!/usr/bin/env python3
"""Paired by-example/document and by-conversation replay vs prior frozen models."""
from __future__ import annotations
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT=Path('/root/autodl-tmp/project/convfinqa_posttrain/experiments/convfinqa_sft_v2')
FROZEN_FINQA='752ba9302ba1153bbdcae95d0d9e743b1f00576ab75761939e821dbd678a58af'
FROZEN_CONV='4fb737022d435bb472c316bb1b5de7776b8238a623bbc1efe900767eab09952f'
PDF_RE=re.compile(r'([a-z0-9_\-.]+/\d{4}/page_\d+\.pdf)',re.I)

def load_pred(path):
    if not path.is_file():raise FileNotFoundError('Required evaluation not run: '+str(path))
    index={}
    for line in path.read_text(encoding='utf8').splitlines():
        if not line.strip():continue
        x=json.loads(line)
        rid=str(x['id'])
        if rid in index: raise ValueError('Duplicate ID '+rid)
        index[rid]=x
    return index

def load_run(base,expected_n,expected_sha,mode):
    pred=load_pred(base/'predictions.jsonl')
    if len(pred)!=expected_n:raise ValueError(f'Expected {expected_n} in {base}, got {len(pred)}')
    info_path=base/('benchmark.json' if mode=='finqa' else 'metrics.json')
    if not info_path.is_file():raise FileNotFoundError(info_path)
    info=json.loads(info_path.read_text(encoding='utf8'))
    actual_sha=info.get('prompt_token_ids_sha256') if mode=='finqa' else info.get('input_fingerprint')
    if actual_sha!=expected_sha:
        raise ValueError(f'Fingerprint mismatch for {base}: {actual_sha} vs {expected_sha}')
    return pred

def cluster_key(x,mode):
    md=x.get('metadata') or {}
    if mode=='conv':
        key=md.get('conversation_id')
        if not key:raise ValueError('Missing conversation ID: '+str(x['id']))
        return str(key)
    for val in (md.get('source_filename'),md.get('filename'),md.get('document_id'),str(x['id'])):
        m=PDF_RE.search(str(val or ''))
        if m:return m.group(1).lower()
    return 'item:'+str(x['id'])

def quantile(sorted_v,p):return sorted_v[min(len(sorted_v)-1,int((len(sorted_v)-1)*p))]

def compare(reference,replay,mode,seed,boots):
    if set(reference)!=set(replay):raise ValueError('ID mismatch between reference and replay')
    wins=losses=0
    clusters=defaultdict(list)
    mismatched=[]
    for rid in sorted(reference):
        a,b=reference[rid],replay[rid]
        if str(a.get('gold_answer'))!=str(b.get('gold_answer')):
            raise ValueError('Gold mismatch '+rid)
        ac=int(bool(a['correct'])); bc=int(bool(b['correct']))
        wins+=(not ac and bc)
        losses+=(ac and not bc)
        clusters[cluster_key(a,mode)].append((ac,bc))
        if ac!=bc and len(mismatched)<20:
            mismatched.append({'id':rid,'reference_correct':bool(ac),'replay_correct':bool(bc),
              'reference_answer':a.get('prediction_answer'),'replay_answer':b.get('prediction_answer'),
              'gold_answer':a.get('gold_answer')})
    groups=list(clusters.values());rng=random.Random(seed)
    vals=[]
    for _ in range(boots):
        pick=rng.choices(groups,k=len(groups))
        n=sum(len(g) for g in pick)
        vals.append(100*sum(b-a for g in pick for a,b in g)/n)
    vals.sort()
    n=len(reference)
    return {'n':n,'clusters':len(groups),'reference_correct':sum(bool(r['correct']) for r in reference.values()),
     'replay_correct':sum(bool(r['correct']) for r in replay.values()),
     'reference_accuracy':sum(bool(r['correct']) for r in reference.values())/n,
     'replay_accuracy':sum(bool(r['correct']) for r in replay.values())/n,
     'delta_pp':100*(wins-losses)/n,'replay_wins':wins,'replay_losses':losses,
     'cluster_bootstrap_95pct_ci_pp':[quantile(vals,.025),quantile(vals,.975)],
     'examples_preview':mismatched}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--bootstrap',type=int,default=5000)
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()
    if a.bootstrap<200:ap.error('bootstrap >= 200')
    finqa=a.root/'finqa_retention_v1'
    originals={label:load_run(finqa/label/'full_dev',883,FROZEN_FINQA,'finqa')
               for label in ('original_sft','transfer','direct')}
    originals['replay']=load_run(a.root/'03_finqa_sft_replay/eval_finqa/full_dev',883,FROZEN_FINQA,'finqa')
    conv={'transfer':load_run(a.root/'01_finqa_sft_adapter/eval_dev',1490,FROZEN_CONV,'conv'),
          'direct':load_run(a.root/'02_qwen3_base/eval_dev',1490,FROZEN_CONV,'conv'),
          'replay':load_run(a.root/'03_finqa_sft_replay/eval_convfinqa_oracle',1490,FROZEN_CONV,'conv')}
    result={'note':'FinQA Dev and ConvFinQA oracle Dev are different tasks; compare within dataset, not raw percentage across datasets',
            'bootstrap_replicates':a.bootstrap,'seed':a.seed,'finqa':{},'convfinqa_oracle':{}}
    print('\n=== FinQA Dev (883 questions) ===')
    for name in originals:
        acc=sum(x['correct'] for x in originals[name].values())/883
        print(f'  {name:13s}: {acc:.2%}')
    for ref in ('original_sft','transfer','direct'):
        stats=compare(originals[ref],originals['replay'],'finqa',a.seed,a.bootstrap)
        result['finqa'][ref+'_vs_replay']=stats
        print(f'  {ref:13s} -> replay: {stats["delta_pp"]:+.2f} pp CI={stats["cluster_bootstrap_95pct_ci_pp"]} wins/losses={stats["replay_wins"]}/{stats["replay_losses"]}')
    print('\n=== ConvFinQA Oracle Dev (1490 turns; 421 conversation clusters) ===')
    for name in conv:
        acc=sum(x['correct'] for x in conv[name].values())/1490
        print(f'  {name:13s}: {acc:.2%}')
    for ref in ('transfer','direct'):
        stats=compare(conv[ref],conv['replay'],'conv',a.seed,a.bootstrap)
        result['convfinqa_oracle'][ref+'_vs_replay']=stats
        print(f'  {ref:13s} -> replay: {stats["delta_pp"]:+.2f} pp CI={stats["cluster_bootstrap_95pct_ci_pp"]} wins/losses={stats["replay_wins"]}/{stats["replay_losses"]}')
    out=a.root/'03_finqa_sft_replay/analysis/replay_comparison.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print('\nSaved:',out)
    print('Reminder: no repeated tuning on the same Dev set; report confidence intervals as exploratory.')

if __name__=='__main__':main()
