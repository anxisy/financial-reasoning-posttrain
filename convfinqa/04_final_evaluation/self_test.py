#!/usr/bin/env python3
"""Pure-Python offline unit tests for pairing, clusters and snapshot integrity."""
import json, tempfile
from pathlib import Path
from common import (ci_paired, assert_aligned, document_cluster, predictions,
                    jsonwrite, sha256, validate_metrics)

def main():
    a={};b={}
    for ci in range(7):
      for j in range(3):
        rid=f'c{ci}_turn{j}'
        src=f'company/{2000+ci}/page_1.pdf'
        ac=(ci+j)%3!=0;bc=ac if j else not ac
        meta={'conversation_id':f'c{ci}','turn_ind':j,'source_filename':src}
        a[rid]={'id':rid,'correct':ac,'gold_answer':'123','metadata':meta}
        b[rid]={**a[rid],'correct':bc}
    assert_aligned(a,b,'test')
    r=ci_paired(a,b,lambda row:row['metadata']['conversation_id'],bootstrap:=250,seed=42)
    assert r['cluster_count']==7 and r['candidate_wins']+r['candidate_losses']==7
    assert len(r['cluster_bootstrap_ci95_pp'])==2
    assert document_cluster(next(iter(a.values())))[1]
    with tempfile.TemporaryDirectory() as tmp:
      p=Path(tmp)/'rows.jsonl'
      p.write_text(''.join(json.dumps(v)+'\n' for v in a.values()),encoding='utf-8')
      assert len(predictions(p,21))==21
      out=Path(tmp)/'result.json'
      jsonwrite(out,{'hash':sha256(p)})
      assert json.loads(out.read_text())['hash']==sha256(p)
      try:
        predictions(p,22)
        raise AssertionError('Missing strict record count')
      except ValueError:pass
    print('PASS: paired CI, doc/conv clusters, duplicate/row checks, deterministic hashes')

if __name__=='__main__':main()
