#!/usr/bin/env python3
"""Check the frozen ConvFinQA v2 SFT interface and its causal-label format. No GPU needed."""
import json,sys
from collections import Counter
from pathlib import Path
root=Path('/root/autodl-tmp/project/convfinqa_posttrain/data/processed_v2')
for split,nwant in [('train',11104),('dev',1490)]:
    path=root/'sft'/f'{split}.jsonl'
    seen=set(); types=Counter(); bad=[]
    with path.open(encoding='utf8') as f:
        for lineno,line in enumerate(f,1):
            r=json.loads(line); ident=r['id']
            if ident in seen: bad.append(f'{ident}: duplicate id')
            seen.add(ident)
            ms=r['messages']; types[r['metadata']['cur_type']]+=1
            if [m['role'] for m in ms]!=['system','user','assistant']: bad.append(f'{ident}: bad roles')
            t=ms[-1]['content']
            if t.count('<answer>')!=1 or t.count('</answer>')!=1: bad.append(f'{ident}: target bad answer tag')
            if r['metadata']['cur_type']=='number_turn' and 'Step ' in t: bad.append(f'{ident}: unexpected number steps')
            if r['metadata'].get('history_mode')!='oracle_gold_answers': bad.append(f'{ident}: history')
    print(split,'n=',len(seen),'types=',dict(types),'issues=',len(bad))
    if len(seen)!=nwant or bad: raise RuntimeError(f'{split} failed: {bad[:5]}')
print('DATA CHECK PASSED (token length and loss masking checked by training script)')
