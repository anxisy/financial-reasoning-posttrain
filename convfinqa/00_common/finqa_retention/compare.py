#!/usr/bin/env python3
"""Paired 883-item FinQA Dev retention test; vLLM/frozen scorer for all 3 adapters."""
from __future__ import annotations
import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path('/root/autodl-tmp/project/convfinqa_posttrain/experiments/convfinqa_sft_v2/finqa_retention_v1')
FROZEN_SHA = '752ba9302ba1153bbdcae95d0d9e743b1f00576ab75761939e821dbd678a58af'
VARIANTS = ('original_sft', 'transfer', 'direct')
DOC_PATTERN = re.compile(r'(?i)([a-z0-9_.-]+/\d{4}/page_\d+\.pdf)')

def read_file(path):
    with path.open(encoding='utf-8') as f:
        return json.load(f)

def read_predictions(path):
    data=[]
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
    index={str(x['id']):x for x in data}
    if len(index)!=len(data): raise ValueError(f'Duplicate IDs in {path}')
    return index

def document_key(row):
    m=row.get('metadata') or {}
    ident=str(row['id'])
    for value in (m.get('source_filename'),m.get('filename'),m.get('source_document'),m.get('document_id'),ident):
        if value:
            hit=DOC_PATTERN.search(str(value))
            if hit: return re.sub(r'^(?:single|double)_', '', hit.group(1), flags=re.I).lower()
    return 'item:'+ident

def percentile(vals,q):
    vs=sorted(vals)
    return vs[min(len(vs)-1,max(0,int((len(vs)-1)*q)))]

def exact_mcnemar_p(wins, losses):
    n=wins+losses
    if n==0:return 1.0
    smaller=min(wins,losses)
    return min(1.0, 2.0 * sum(math.comb(n,k) for k in range(smaller+1)) / (2**n))

def paired(a,b,seed,bootstrap):
    ids=sorted(a)
    rows=[];clusters=defaultdict(list)
    for rid in ids:
        ca,cb=a[rid],b[rid]
        ia=int(bool(ca['correct'])); ib=int(bool(cb['correct']))
        if str(ca.get('gold_answer')) != str(cb.get('gold_answer')):
            raise ValueError('Gold answer mismatch for '+rid)
        clusters[document_key(ca)].append((ia,ib))
        if ia!=ib:
            rows.append({'id':rid,'source_document':document_key(ca),
                         'gold_answer':ca.get('gold_answer'),
                         'a_correct':bool(ia),'b_correct':bool(ib),
                         'a_answer':ca.get('prediction_answer'), 'b_answer':cb.get('prediction_answer'),
                         'a_text':ca.get('prediction','')[:400], 'b_text':cb.get('prediction','')[:400]})
    wins=sum(1 for x in rows if x['b_correct'])
    losses=len(rows)-wins
    n=len(ids)
    rng=random.Random(seed)
    groups=list(clusters.values())
    changes=[]
    for _ in range(bootstrap):
        sampled=rng.choices(groups,k=len(groups))
        den=sum(len(g) for g in sampled)
        num=sum(y-x for g in sampled for x,y in g)
        changes.append(100*num/den)
    result={
      'n_questions':n,'n_document_clusters':len(groups),
      'a_correct':sum(bool(v['correct']) for v in a.values()),
      'b_correct':sum(bool(v['correct']) for v in b.values()),
      'a_accuracy':sum(bool(v['correct']) for v in a.values())/n,
      'b_accuracy':sum(bool(v['correct']) for v in b.values())/n,
      'delta_pp':100*(wins-losses)/n,
      'b_wins_a_losses':wins,'a_wins_b_losses':losses,
      'document_cluster_bootstrap_ci95_pp':[percentile(changes,.025),percentile(changes,.975)],
      'mcnemar_exact_p_exploratory':exact_mcnemar_p(wins,losses),
      'notes': 'Document-cluster bootstrap if document IDs parseable, otherwise item clustering. McNemar assumes paired independent items; report as exploratory when documents repeat.'
    }
    return result,rows

def summarize_operator_variant(preds):
    op = defaultdict(list)
    for r in preds.values():
        md=r.get('metadata') or {}
        for o in set(map(str,md.get('operators',[]) or [])):
            op[o].append(r)
    return {o:{'n':len(vals),'accuracy':sum(bool(x['correct']) for x in vals)/len(vals)}
            for o,vals in sorted(op.items())}

def run(root, seed=42,bootstrap=5000,expected_sha=FROZEN_SHA):
    preds={}; metrics={}; fingerprints={}
    for v in VARIANTS:
        p=root/v/'full_dev'
        for name in ('predictions.jsonl','metrics.json','benchmark.json'):
            if not (p/name).exists(): raise FileNotFoundError(f'Missing {p/name} (run the three eval commands first)')
        preds[v]=read_predictions(p/'predictions.jsonl')
        metrics[v]=read_file(p/'metrics.json')
        meta=read_file(p/'benchmark.json')
        fingerprints[v]=meta.get('prompt_token_ids_sha256')
        if len(preds[v])!=883 or metrics[v]['overall']['n']!=883:
            raise ValueError(f'{v}: expected 883 Dev rows, got {len(preds[v])}')
        if fingerprints[v]!=expected_sha:
            raise ValueError(f'{v}: frozen FinQA prompt fingerprint mismatch {fingerprints[v]} vs {expected_sha}')
    idsets=[set(preds[v]) for v in VARIANTS]
    if not all(s==idsets[0] for s in idsets):
        raise ValueError('Different sample IDs across three models; STOP.')
    if any(fingerprints[v]!=fingerprints[VARIANTS[0]] for v in VARIANTS):
        raise ValueError('Different prompt fingerprints; STOP.')
    print(f'PASS: 883 identical FinQA Dev question IDs; fingerprint={expected_sha}')
    print('\n=== FINQA DEV (same evaluator, vLLM, no retraining) ===')
    print(f'{"Variant":<16}{"Correct":>10}{"Acc":>10}{"Format":>10}{"Strict":>10}')
    for v in VARIANTS:
        m=metrics[v]['overall']
        print(f'{v:<16}{sum(bool(r["correct"]) for r in preds[v].values()):>7}/883'
              f'{100*m["accuracy"]:>9.2f}%{100*m["format_accuracy"]:>9.2f}%'
              f'{100*m["strict_accuracy"]:>9.2f}%')
    comparisons={}; errors=[]
    for a,b in [('original_sft','transfer'),('direct','transfer'),('original_sft','direct')]:
        stat,rows=paired(preds[a],preds[b],seed,bootstrap)
        label=f'{a}_vs_{b}'
        comparisons[label]=stat
        for row in rows: errors.append({'comparison':label,**row})
        print(f'\n{a} -> {b}: {stat["delta_pp"]:+.2f} pp; wins/losses={stat["b_wins_a_losses"]}/{stat["a_wins_b_losses"]}; '
              f'document clusters={stat["n_document_clusters"]}; '
              f'95% CI={[round(x,2) for x in stat["document_cluster_bootstrap_ci95_pp"]]}')
    out=root/'analysis'
    out.mkdir(parents=True,exist_ok=True)
    result={'dataset':'FinQA Dev','n':883,'prompt_token_ids_sha256':expected_sha,
            'models':{v:metrics[v]['overall'] for v in VARIANTS},
            'by_operator':{v:summarize_operator_variant(preds[v]) for v in VARIANTS},
            'pairs':comparisons,
            'interpretation_cautions':[
              'ConvFinQA/FinQA Dev has been used during model selection; do not claim unseen independent test generalization.',
              'Transfer above Direct does not by itself prove retention; compare Transfer against original FinQA SFT.',
              'Confidence intervals including 0 do not establish a reliable difference.',
              'FinQA vs ConvFinQA accuracies refer to different tasks and should not be directly compared.',
              'This is a one-seed adapter comparison; multiple seeds needed for robust claims.'
            ]}
    (out/'retention_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    with (out/'discordant_cases.csv').open('w',encoding='utf8',newline='') as f:
        keys=['comparison','id','source_document','gold_answer','a_correct','b_correct','a_answer','b_answer','a_text','b_text']
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(errors)
    print('\nSaved:',out/'retention_summary.json')
    print('Saved:',out/'discordant_cases.csv')
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--bootstrap',type=int,default=5000)
    ap.add_argument('--expected-sha',default=FROZEN_SHA)
    a=ap.parse_args()
    if a.bootstrap<10:ap.error('--bootstrap must >= 10')
    run(a.root,a.seed,a.bootstrap,a.expected_sha)

if __name__=='__main__':main()
