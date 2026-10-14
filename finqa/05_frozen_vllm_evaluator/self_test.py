"""Run without torch/vllm using only Python standard library."""
import json
import tempfile
from pathlib import Path
from common import manual_serialize,make_prediction,read_rows,save_results,perf_dict
from answer_normalizer import compare_answer_strings
from compare_runs import compare
from argparse import Namespace

class FakeTok:
    def __call__(self,strings,**_):
        return {'input_ids':[[ord(c) for c in s] for s in strings]}

def test():
    from common import prepare
    ex={'id':'test-1','prompt':[{'role':'system','content':'Calculate.'},{'role':'user','content':'1+2?'}],
        'gold_answer':'3','metadata':{'difficulty':'easy','operators':['add']}}
    p=manual_serialize(ex['prompt'])
    assert p=='System:\nCalculate.\n\nUser:\n1+2?\n\nAssistant:'
    a=make_prediction(ex,'\nStep 1: add(1,2)=3\n<answer>3</answer>',20,384,'peft_adapter')
    assert a['correct'] and a['format_ok'] and a['prediction_answer']=='3'
    assert compare_answer_strings('16.4%','16.4%')[0]
    assert compare_answer_strings('0.164','16.4%')[0]
    with tempfile.TemporaryDirectory() as td:
        src=Path(td)/'input.jsonl';src.write_text(json.dumps(ex)+'\n')
        rows=read_rows(src)
        _,token_ids,fingerprint=prepare(rows,FakeTok())
        assert len(token_ids)==1 and len(fingerprint)==64
        perf=perf_dict('transformers',1,20,2.,.2,.1,.01,fingerprint,{'engine':'transformers','max_new_tokens':384,'input':str(src),'base':'x','adapter':'y','prompt_format':'manual','do_sample':False},5000.)
        perf2=dict(perf,engine='vllm',samples_per_second=1.,generated_tokens_per_second=10.,inference_seconds=1.)
        perf2['run']=dict(perf['run'],engine='vllm')
        t=Path(td)/'torch';v=Path(td)/'vllm'
        save_results(t,[a],perf['run'],perf)
        save_results(v,[a],perf2['run'],perf2)
        result=compare(Namespace(transformers=str(t),vllm=str(v),output_dir=str(Path(td)/'comparison')))
        assert result['samples_per_second_speedup']==2.0
        assert result['accuracy_difference_pp']==0.0
    print('\nSELF TEST PASSED')

if __name__=='__main__':test()
