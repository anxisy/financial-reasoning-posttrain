"""CPU-only deterministic paired comparison and scoring/cluster smoke test."""
import json
import tempfile
from pathlib import Path
from compare import exact_mcnemar_p,document_key,paired,run,FROZEN_SHA

assert abs(exact_mcnemar_p(1,0)-1)<1e-9
assert abs(exact_mcnemar_p(3,0)-.25)<1e-9
assert document_key({'id':'Single_ABC/2011/page_3.pdf-1'})=='abc/2011/page_3.pdf'
ids=['Single_ABC/2011/page_3.pdf-0','Single_ABC/2011/page_3.pdf-1',
     'Single_DEF/2012/page_4.pdf-0']
a={k:{'id':k,'correct':i==0,'gold_answer':'1','metadata':{'operators':['add']}} for i,k in enumerate(ids)}
b={k:{'id':k,'correct':i<2,'gold_answer':'1','metadata':{'operators':['add']}} for i,k in enumerate(ids)}
s,rows=paired(a,b,42,150)
assert s['n_document_clusters']==2
assert s['delta_pp']>33.3 and s['delta_pp']<33.4
assert s['b_wins_a_losses']==1 and s['a_wins_b_losses']==0
assert len(rows)==1
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)
    for name,results in [('original_sft',a),('transfer',b),('direct',a)]:
        p=root/name/'full_dev';p.mkdir(parents=True)
        # full run() guard should reject incomplete rows, not write a report
        (p/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results.values()))
        (p/'metrics.json').write_text(json.dumps({'overall':{'n':3}}))
        (p/'benchmark.json').write_text(json.dumps({'prompt_token_ids_sha256':FROZEN_SHA}))
    try:run(root,42,150)
    except ValueError as e:assert '883' in str(e)
    else:raise AssertionError('missing full-set guard')
print('PASS: paired wins/losses, document grouping, bootstrap, McNemar, incomplete-data guard')
