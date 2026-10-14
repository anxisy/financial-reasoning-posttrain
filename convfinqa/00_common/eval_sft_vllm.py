#!/usr/bin/env python3
"""ConvFinQA Oracle-history Dev evaluation of frozen direct or transfer SFT LoRA.
Reuse SAME vLLM prompt serialization, strict matcher, and metrics as frozen Base/SFT baseline.
"""
from __future__ import annotations
import argparse,hashlib,importlib.metadata,json,os,time
from pathlib import Path
from frozen_baseline import read_examples,get_adapter_rank,serialize_manual,score_prediction,summarize_predictions

BASE='/root/autodl-tmp/models/Qwen3-1.7B-Base'
DEV='/root/autodl-tmp/project/convfinqa_posttrain/data/processed_v2/eval/dev.jsonl'
ROOT=Path('/root/autodl-tmp/project/convfinqa_posttrain/experiments/convfinqa_sft_v2')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant',choices=['direct','transfer'],required=True)
    ap.add_argument('--base',default=BASE)
    ap.add_argument('--input',default=DEV)
    ap.add_argument('--adapter',default=None)
    ap.add_argument('--output-dir',default=None)
    ap.add_argument('--max-samples',type=int,default=0)
    ap.add_argument('--max-new-tokens',type=int,default=384)
    ap.add_argument('--max-model-len',type=int,default=4096)
    ap.add_argument('--max-num-seqs',type=int,default=32)
    ap.add_argument('--gpu-memory-utilization',type=float,default=.70)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    adapter=Path(args.adapter or (ROOT/f'{args.variant}_full/final_adapter'))
    if not (adapter/'adapter_config.json').exists(): raise FileNotFoundError(f'Expected trained adapter: {adapter}')
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    tok=AutoTokenizer.from_pretrained(args.base,local_files_only=True,trust_remote_code=True)
    examples=read_examples(args.input,args.max_samples,1490)
    prompts=[serialize_manual(x['prompt']) for x in examples]
    ids=tok(prompts,truncation=False,padding=False,add_special_tokens=True)['input_ids']
    too_long=[(r['id'],len(s)) for r,s in zip(examples,ids) if len(s)+args.max_new_tokens>args.max_model_len]
    if too_long: raise ValueError(f'Prompt + generation exceeds context for {len(too_long)} rows: {too_long[:5]}')
    fingerprint=hashlib.sha256(json.dumps([(r['id'], seq) for r,seq in zip(examples,ids)],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    print('Input fingerprint:',fingerprint,'Rows:',len(examples),'max prompt tokens:',max(map(len,ids)),flush=True)
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    llm=LLM(model=args.base,tokenizer=args.base,dtype='bfloat16',gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,max_num_seqs=args.max_num_seqs,trust_remote_code=True,
            seed=args.seed,enable_lora=True,max_lora_rank=max(8,get_adapter_rank(adapter)),max_loras=1)
    sampler=SamplingParams(temperature=0.,top_p=1.,max_tokens=args.max_new_tokens,
                           seed=args.seed,skip_special_tokens=True,stop_token_ids=[tok.eos_token_id])
    req=LoRARequest(f'convfinqa_{args.variant}_sft',1,str(adapter))
    def generate(chunks):
        res=llm.generate([{'prompt_token_ids':x} for x in chunks],sampling_params=sampler,lora_request=req,use_tqdm=False)
        if len(res)!=len(chunks): raise RuntimeError('Incomplete outputs')
        return [(r.outputs[0].text.strip(),len(r.outputs[0].token_ids)) for r in res]
    generate(ids[:min(4,len(ids))])
    t0=time.perf_counter()
    # One queue. vLLM schedules according to max_num_seqs.
    completions=generate(ids)
    elapsed=time.perf_counter()-t0
    predictions=[score_prediction(row,text,n,args.max_new_tokens,args.variant) for row,(text,n) in zip(examples,completions)]
    metrics=summarize_predictions(predictions)
    metrics.update(engine='vllm',variant=args.variant,input_fingerprint=fingerprint,
                   num_rows=len(predictions),elapsed_inference_seconds=elapsed,
                   samples_per_second=len(predictions)/elapsed,
                   generated_tokens_per_second=sum(n for _,n in completions)/elapsed)
    output=Path(args.output_dir or (ROOT/f'{args.variant}_eval_dev'))
    output.mkdir(parents=True,exist_ok=True)
    with (output/'predictions.jsonl').open('w',encoding='utf-8') as f:
        for p in predictions: f.write(json.dumps(p,ensure_ascii=False)+'\n')
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf-8')
    (output/'run_config.json').write_text(json.dumps({'variant':args.variant,'input':args.input,
        'adapter':str(adapter),'input_sha256':hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        'max_new_tokens':args.max_new_tokens,'max_model_len':args.max_model_len,
        'max_num_seqs':args.max_num_seqs,'gpu_memory_utilization':args.gpu_memory_utilization,
        'seed':args.seed,'prompt_token_ids_sha256':fingerprint,'vllm_version':importlib.metadata.version('vllm'),
        'history':'oracle_gold_answers','same_frozen_scoring_as_baseline':True},indent=2),encoding='utf-8')
    print(f"{args.variant}: acc={metrics['overall']['accuracy']:.2%}, "
          f"number={metrics['by_cur_type']['number_turn']['accuracy']:.2%}, "
          f"program={metrics['by_cur_type']['program_turn']['accuracy']:.2%}",flush=True)
    print('Saved:',output,flush=True)

if __name__=='__main__':main()
