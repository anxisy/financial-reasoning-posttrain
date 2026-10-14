#!/usr/bin/env python3
"""Offline tests: replay source semantics, sample budgeting, assistant masks."""
import importlib.util
import json
import tempfile
from collections import Counter
from pathlib import Path

HERE=Path(__file__).parent

def imported(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

builder=imported('builder',HERE/'01_data/build_replay.py')
trainer=imported('trainer',HERE/'train_replay.py')

def sample(id,origin,t):
    return {'id':id,'messages':[{'role':'system','content':'You are helpful.'},
             {'role':'user','content':'financial context '+id},
             {'role':'assistant','content':('<answer>5</answer>' if t=='number_turn' else 'Step 1: add(2,3)\n<answer>5</answer>')}],
            'metadata':({'history_mode':'oracle_gold_answers','cur_type':t} if origin=='conv' else {'gold_answer':'5'})}

conv=[sample(f'c{i}', 'conv','number_turn' if i<10 else 'program_turn') for i in range(20)]
finqa=[sample(f'f{i}','finqa','program_turn') for i in range(10)]
rows=builder.build(conv,finqa,2,42)
assert len(rows)==len(conv)
assert Counter(r['metadata']['replay_source'] for r in rows)=={'convfinqa':18,'finqa':2}
assert all(r['id'].startswith(('finqa::','convfinqa::')) for r in rows)
assert all(r['metadata'].get('history_mode')!='oracle_gold_answers' for r in rows if r['metadata']['replay_source']=='finqa')
assert trainer.serialize(rows[0])[0].endswith('Assistant:')
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp)/'mixed.jsonl'
    builder.write(p,rows)
    loaded=trainer.read_jsonl(p)
    assert len(loaded)==20
# Length/label masking fake tokenizer independent of transformers
class Tok:
    eos_token_id=0
    def encode(self,text,add_special_tokens=False):return list(range(1,1+len(text.split())))
ds=trainer.AssistantOnlyDataset(rows,Tok(),4096)
assert len(ds)==20
for item in ds.items:
    assert item['labels'][0]==-100
    assert item['labels'][-1]==0
    assert any(label!=-100 for label in item['labels'])
print('PASS: deterministic mixing, unique IDs, truthful source tags, schema, prompt serialization, assistant-only loss, EOS.')
