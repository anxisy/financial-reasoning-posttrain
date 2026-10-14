"""Strict accuracy + throughput comparison between matched input token IDs."""
import argparse
import csv
import json
from pathlib import Path


def loadpred(p):
    rows={}
    with Path(p).open(encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line); key=str(r.get('id'))
            if key in rows:raise ValueError(f'Duplicate prediction ID {key}')
            rows[key]=r
    return rows


def compare(args):
    ta=Path(args.transformers);vb=Path(args.vllm)
    a=json.loads((ta/'benchmark.json').read_text())
    b=json.loads((vb/'benchmark.json').read_text())
    if a['engine']!='transformers' or b['engine']!='vllm':raise ValueError('Expected --transformers then --vllm output directories')
    if a['prompt_token_ids_sha256']!=b['prompt_token_ids_sha256']:
        raise ValueError('Prompt IDs/tokenization mismatch; do not compare speed/accuracy as a matched benchmark.')
    for field in ['max_new_tokens','input','base','adapter','prompt_format','do_sample']:
        if a['run'].get(field)!=b['run'].get(field):
            raise ValueError(f'Mismatched {field}: {a["run"].get(field)!r} vs {b["run"].get(field)!r}')
    xa=loadpred(ta/'predictions.jsonl')
    xb=loadpred(vb/'predictions.jsonl')
    if set(xa)!=set(xb): raise ValueError('Different sample ID sets')
    wins=losses=agree=identical=0
    changed=[]
    for k in sorted(xa):
        q=xa[k]; r=xb[k]
        if str(q.get('gold_answer'))!=str(r.get('gold_answer')):raise ValueError(f'Gold mismatch on {k}')
        ac,bc=bool(q['correct']),bool(r['correct'])
        wins+=int(bc and not ac);losses+=int(ac and not bc)
        agree+=int(ac==bc)
        identical+=int(q['prediction']==r['prediction'])
        if q['prediction']!=r['prediction']:
            changed.append({'id':k,'torch_correct':ac,'vllm_correct':bc,
                            'torch_answer':q.get('prediction_answer'),
                            'vllm_answer':r.get('prediction_answer'),
                            'torch_text':q['prediction'],'vllm_text':r['prediction']})
    ma=json.loads((ta/'metrics.json').read_text())['overall']
    mb=json.loads((vb/'metrics.json').read_text())['overall']
    n=len(xa)
    comparison={
        'n':n,'inputs_identical_token_ids':True,
        'prompt_token_ids_sha256':a['prompt_token_ids_sha256'],
        'transformers_accuracy':ma['accuracy'],'vllm_accuracy':mb['accuracy'],
        'accuracy_difference_pp':(mb['accuracy']-ma['accuracy'])*100,
        'both_correctness_agree':agree,'identical_generated_text':identical,
        'vllm_right_transformers_wrong':wins,'vllm_wrong_transformers_right':losses,
        'transformers_samples_per_second':a['samples_per_second'],
        'vllm_samples_per_second':b['samples_per_second'],
        'samples_per_second_speedup':b['samples_per_second']/a['samples_per_second'],
        'transformers_generated_tokens_per_second':a['generated_tokens_per_second'],
        'vllm_generated_tokens_per_second':b['generated_tokens_per_second'],
        'tokens_per_second_speedup':b['generated_tokens_per_second']/a['generated_tokens_per_second'],
        'transformers_inference_seconds':a['inference_seconds'],
        'vllm_inference_seconds':b['inference_seconds'],
        'transformers_peak_gpu_used_mib':a['peak_gpu_used_mib'],
        'vllm_peak_gpu_used_mib':b['peak_gpu_used_mib'],
        'comparison_note': 'Same prompts/token IDs/adapters/greedy budget. Different GPU kernel and scheduling may change outputs. samples/s compares end-to-end inference with the respective configured request batch sizes.'
    }
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'changed_outputs.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(changed[0]) if changed else ['id','torch_correct','vllm_correct','torch_answer','vllm_answer','torch_text','vllm_text']);w.writeheader();w.writerows(changed)
    print('\n=== MATCHED FINQA BENCHMARK ===')
    for k,v in comparison.items():
        if k.endswith('_sha256') or k=='comparison_note':continue
        print(f'{k:42s} : {v}')
    print('Saved: '+str(out/'comparison.json'))
    return comparison


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--transformers',required=True)
    p.add_argument('--vllm',required=True)
    p.add_argument('--output-dir',required=True)
    a=p.parse_args()
    compare(a)

if __name__=='__main__':main()
