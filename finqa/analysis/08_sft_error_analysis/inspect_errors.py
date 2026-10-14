# -*- coding: utf-8 -*-
import argparse, json

def read_jsonl(path):
    with open(path,'r',encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--analysis',required=True); p.add_argument('--category',required=True); p.add_argument('--limit',type=int,default=10); a=p.parse_args()
    rows=read_jsonl(a.analysis); m=[r for r in rows if a.category==r.get('primary_category') or a.category in (r.get('categories') or [])]
    print(f'Matched {len(m)} rows for category={a.category}')
    for i,r in enumerate(m[:a.limit],1):
        print('\\n'+'='*90); print(f'[{i}] id={r.get("id")}'); print('primary:',r.get('primary_category')); print('flags:',r.get('categories')); print('gold:',r.get('gold_answer')); print('pred:',r.get('prediction_answer')); print('gold ops:',r.get('gold_ops')); print('pred ops:',r.get('pred_ops')); print('evidence:',r.get('evidence')); print('\\nMODEL RESPONSE:'); print(r.get('prediction',''))
if __name__=='__main__':main()
