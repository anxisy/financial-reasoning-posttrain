#!/usr/bin/env python3
"""Pure-Python offline self-test; no torch or model required."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_sft import AssistantOnlyDataset,Collator,serialize,smoke_select,length_stats
class FakeTok:
    eos_token_id=99
    def encode(self,s,add_special_tokens=False): return [ord(c)%67+1 for c in s]
def mk(t,typ):
    return {'id':str(t),'messages':[{'role':'system','content':'SYS'},
          {'role':'user','content':'Q'+str(t)},
          {'role':'assistant','content':('<answer>3</answer>' if typ=='number_turn' else 'Step 1: subtract(4, 1)\n\n<answer>3</answer>')}],
          'metadata':{'cur_type':typ,'history_mode':'oracle_gold_answers'}}
rows=[mk(0,'number_turn'),mk(1,'program_turn')]
a,b=serialize(rows[0]);assert a.endswith('Assistant:') and b.startswith('\n<answer>')
ds=AssistantOnlyDataset(rows,FakeTok(),4096)
assert len(ds)==2
for it in ds:
    start=it['labels'].index(next(x for x in it['labels'] if x!=-100))
    assert all(v==-100 for v in it['labels'][:start])
    assert all(v!=-100 for v in it['labels'][start:])
assert len(smoke_select(rows,32,42))==2
try:
    AssistantOnlyDataset(rows,FakeTok(),32)
except ValueError as e:
    assert 'NO TRUNCATION' in str(e)
else: raise AssertionError('Missing overlength rejection')
print('PASS: assistant-only label mask; serialize boundary; no silent truncation; smoke subset')
