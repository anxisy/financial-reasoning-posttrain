# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, math, random, re
from collections import Counter, defaultdict
from pathlib import Path

OPS=['add','subtract','multiply','divide','greater','average','sum','max','min','exp']
OP_RE=re.compile(r'\b('+'|'.join(OPS)+r')\s*\(', re.I)
CALL_RE=re.compile(r'\b('+'|'.join(OPS)+r')\s*\(([^)]*)\)', re.I)
NUM_RE=re.compile(r'[-+]?(?:\d[\d,]*\.?\d*|\.\d+)')

def read_jsonl(path):
    with open(path,'r',encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]

def write_jsonl(path, rows):
    with open(path,'w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\\n')

def parse_num(v):
    if v is None: return None, False
    s=str(v).strip(); m=NUM_RE.search(s)
    if not m: return None, '%' in s
    try: return float(m.group(0).replace(',','')), '%' in s
    except: return None, '%' in s

def norm_yesno(v):
    s='' if v is None else str(v).strip().lower()
    if s in {'yes','true'}: return 'yes'
    if s in {'no','false'}: return 'no'
    return None

def prompt_text(row):
    p=row.get('prompt',[])
    if isinstance(p,list): return '\\n'.join(str(m.get('content','')) for m in p if isinstance(m,dict))
    return str(p or '')

def pred_ops(text): return [m.group(1).lower() for m in OP_RE.finditer(text or '')]
def gold_ops(meta): return [str(x).lower() for x in (meta.get('operators',[]) or [])]

def first_call_operands(text):
    m=CALL_RE.search(text or '')
    if not m: return []
    return [x.strip() for x in m.group(2).split(',') if x.strip()]

def prompt_nums(text):
    out=[]
    for m in NUM_RE.finditer(text or ''):
        try: out.append(float(m.group(0).replace(',','')))
        except: pass
    return out

def operand_candidate(prediction, ptext):
    ops=first_call_operands(prediction); vals=prompt_nums(ptext)
    if not ops or not vals: return False
    for x in ops:
        if x.startswith('#'): continue
        v,_=parse_num(x)
        if v is None: continue
        if not any(math.isclose(v,p,rel_tol=1e-9,abs_tol=1e-9) for p in vals): return True
    return False

def close_ratio(r, targets, rel=0.02):
    return r is not None and any(math.isclose(abs(r),t,rel_tol=rel,abs_tol=1e-12) for t in targets)

def classify(pred, src):
    meta=pred.get('metadata',{}) or {}
    p_ans=pred.get('prediction_answer',''); g_ans=pred.get('gold_answer','')
    pn,pp=parse_num(p_ans); gn,gp=parse_num(g_ans)
    pyn,gyn=norm_yesno(p_ans),norm_yesno(g_ans)
    pops=pred_ops(pred.get('prediction','')); gops=gold_ops(meta)
    ptxt=prompt_text(src)
    cats=[]; ev={}

    if meta.get('annotation_quality')=='medium': cats.append('annotation_quality_flag')
    if meta.get('semantic_status')=='suspicious': cats.append('semantic_suspicious_flag')
    if gyn is not None:
        cats.append('yes_no_error'); ev['gold_yesno']=gyn; ev['pred_yesno']=pyn
    if gops and pops and set(pops)!=set(gops):
        cats.append('operator_mismatch_candidate'); ev['pred_ops']=pops; ev['gold_ops']=gops
    if operand_candidate(pred.get('prediction',''), ptxt): cats.append('operand_selection_candidate')
    steps=int(meta.get('num_steps',0) or 0)
    if steps>=2: cats.append('multi_step_failure'); ev['num_steps']=steps

    if pn is not None and gn is not None:
        diff=abs(pn-gn); rel=diff/max(abs(gn),1e-12)
        ev.update({'pred_numeric':pn,'gold_numeric':gn,'abs_error':diff,'rel_error':rel})
        ratios=[]
        if gn!=0: ratios.append(pn/gn)
        if pn!=0: ratios.append(gn/pn)
        if any(close_ratio(r,[100.0,0.01]) for r in ratios): cats.append('percentage_scale_error')
        if any(close_ratio(r,[1e3,1e-3,1e6,1e-6,1e9,1e-9]) for r in ratios): cats.append('unit_scale_error')
        if gp or pp:
            if diff<=0.5: cats.append('near_numeric_miss')
        else:
            if diff<=0.05 or rel<=0.02: cats.append('near_numeric_miss')

    if not cats:
        cats.append('other_numeric_reasoning_error' if pn is not None and gn is not None else 'unparsed_or_other')

    order=['semantic_suspicious_flag','annotation_quality_flag','yes_no_error','percentage_scale_error','unit_scale_error','near_numeric_miss','operator_mismatch_candidate','operand_selection_candidate','multi_step_failure','other_numeric_reasoning_error','unparsed_or_other']
    primary=next((x for x in order if x in cats), cats[0])
    return {'primary_category':primary,'categories':sorted(set(cats)),'evidence':ev,'pred_ops':pops,'gold_ops':gops,'prompt_text':ptxt}

def accuracy_group(rows, fn):
    g=defaultdict(list)
    for r in rows: g[str(fn(r))].append(r)
    out={}
    for k,v in sorted(g.items()):
        n=len(v); c=sum(bool(x.get('correct')) for x in v)
        out[k]={'n':n,'accuracy':c/n if n else 0.0,'errors':n-c}
    return out

def priority_slices(preds):
    by_diff=accuracy_group(preds,lambda r:r.get('metadata',{}).get('difficulty','unknown'))
    by_ann=accuracy_group(preds,lambda r:r.get('metadata',{}).get('annotation_quality','unknown'))
    og=defaultdict(list)
    for r in preds:
        for op in set(r.get('metadata',{}).get('operators',[]) or []): og[str(op)].append(r)
    by_op={}
    for op,v in sorted(og.items()):
        n=len(v); c=sum(bool(x.get('correct')) for x in v)
        by_op[op]={'n':n,'accuracy':c/n if n else 0.0,'errors':n-c}
    return {'by_difficulty':by_diff,'by_operator':by_op,'by_annotation_quality':by_ann}

def report_text(summary):
    n=summary['n_predictions']; e=summary['n_errors']; acc=summary['accuracy']
    L=['# FinQA SFT Dev Error Analysis v1','', '> Automatic categories are heuristic candidates, not ground-truth diagnoses.','', '## Overall','',f'- Dev samples: **{n}**',f'- Errors: **{e}**',f'- Accuracy: **{acc*100:.2f}%**','', '## Primary categories','', '| Category | Count | % of errors |','|---|---:|---:|']
    for k,v in summary['primary_category_counts'].items(): L.append(f'| {k} | {v} | {100*v/e if e else 0:.2f}% |')
    L += ['', '## Diagnostic flags','', '| Flag | Count | % of errors |','|---|---:|---:|']
    for k,v in summary['category_flag_counts'].items(): L.append(f'| {k} | {v} | {100*v/e if e else 0:.2f}% |')
    L += ['', '## Notes','', '- `operator_mismatch_candidate` and `operand_selection_candidate` are low-confidence heuristics.', '- Keep evaluator/tolerances frozen.', '- Use only Dev analysis to motivate GRPO design; do not tune on Test.']
    return '\\n'.join(L)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',default='/root/autodl-tmp/project/finqa_posttrain/experiments/sft_qwen3_0.6b/eval_dev_v1/predictions.jsonl')
    p.add_argument('--eval-file',default='/root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1/eval/finqa_eval_dev.jsonl')
    p.add_argument('--output-dir',default='/root/autodl-tmp/project/finqa_posttrain/analysis/sft_dev_error_v1')
    p.add_argument('--review-per-category',type=int,default=8)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    preds=read_jsonl(a.predictions); srcs=read_jsonl(a.eval_file); src_by={x.get('id'):x for x in srcs}
    errors=[r for r in preds if not r.get('correct',False)]
    analyzed=[]
    for r in errors:
        d=classify(r,src_by.get(r.get('id'),{}))
        analyzed.append({'id':r.get('id'),'prediction':r.get('prediction',''),'prediction_answer':r.get('prediction_answer',''),'gold_answer':r.get('gold_answer',''),'match_reason':r.get('match_reason'),'metadata':r.get('metadata',{}),**d})
    pc=Counter(x['primary_category'] for x in analyzed); fc=Counter(c for x in analyzed for c in x['categories'])
    pri=priority_slices(preds)
    summary={'version':'sft_dev_error_analysis_v1','predictions_file':a.predictions,'eval_file':a.eval_file,'n_predictions':len(preds),'n_errors':len(errors),'accuracy':sum(bool(x.get('correct')) for x in preds)/len(preds) if preds else 0.0,'primary_category_counts':dict(pc.most_common()),'category_flag_counts':dict(fc.most_common()),'priority_slices':pri,'notes':{'automatic_labels_are_heuristic':True,'test_split_used_for_design':False}}
    write_jsonl(out/'error_analysis.jsonl',analyzed)
    (out/'category_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'priority_slices.json').write_text(json.dumps(pri,ensure_ascii=False,indent=2),encoding='utf-8')
    rng=random.Random(a.seed); g=defaultdict(list)
    for r in analyzed: g[r['primary_category']].append(r)
    review=[]
    for cat in sorted(g):
        pool=list(g[cat]); rng.shuffle(pool)
        review += [{'review_category':cat,**r} for r in pool[:a.review_per_category]]
    write_jsonl(out/'review_sample.jsonl',review)
    (out/'report.md').write_text(report_text(summary),encoding='utf-8')
    print('='*90); print('FinQA SFT Dev Error Analysis v1'); print('='*90)
    print('Predictions:',len(preds)); print('Errors:',len(errors)); print(f"Accuracy: {summary['accuracy']*100:.2f}%")
    print('\\nPrimary categories:')
    for k,v in pc.most_common(): print(f'  {k:30s} {v:4d} ({100*v/len(errors) if errors else 0:6.2f}%)')
    print('\\nDiagnostic flags:')
    for k,v in fc.most_common(): print(f'  {k:30s} {v:4d} ({100*v/len(errors) if errors else 0:6.2f}%)')
    print('\\nSaved to:',out)

if __name__=='__main__': main()
