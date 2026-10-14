"""Offline vLLM+LoRA benchmark using exactly the same prompt token IDs and scoring."""
import argparse
import json
import os
import time
from pathlib import Path
from common import (DEFAULT_ADAPTER,DEFAULT_BASE,DEFAULT_DEV,make_prediction,
                    perf_dict,pkg_version,prepare,read_rows,save_results)
from gpu_monitor import VramMonitor
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("OMP_NUM_THREADS", "8")

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',default=DEFAULT_BASE)
    p.add_argument('--adapter',default=DEFAULT_ADAPTER)
    p.add_argument('--input',default=DEFAULT_DEV)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--max-samples',type=int,default=20)
    p.add_argument('--max-new-tokens',type=int,default=384)
    p.add_argument('--request-batch-size',type=int,default=32,
                   help='Number of prompts submitted to each vLLM generate() call; 0=submit all at once')
    p.add_argument('--warmup-samples',type=int,default=2)
    p.add_argument('--max-model-len',type=int,default=4096)
    p.add_argument('--max-num-seqs',type=int,default=32)
    p.add_argument('--gpu-memory-utilization',type=float,default=.70)
    p.add_argument('--enforce-eager',action='store_true',help='Safer when CUDA graph memory is tight; slower')
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if a.max_new_tokens<1 or a.max_num_seqs<1 or a.request_batch_size<0:
        p.error('invalid batch / generation parameters')
    # Import lazily, so self-test and Transformers benchmark do not require vLLM.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    if not Path(a.adapter,'adapter_config.json').exists():
        raise FileNotFoundError('Missing LoRA adapter_config.json in '+a.adapter)
    cfg=json.loads(Path(a.adapter,'adapter_config.json').read_text())
    rank=int(cfg.get('r',8))
    pattern=cfg.get('rank_pattern') or {}
    if pattern: rank=max(rank,*(int(v) for v in pattern.values()))
    rows=read_rows(a.input,a.max_samples)
    with VramMonitor() as mem:
        t0=time.perf_counter()
        tok=AutoTokenizer.from_pretrained(a.base,local_files_only=True,trust_remote_code=True)
        tok.padding_side='left'
        if tok.pad_token_id is None: tok.pad_token=tok.eos_token
        tokenize_t0=time.perf_counter()
        _,ids,fingerprint=prepare(rows,tok)
        tokenize_secs=time.perf_counter()-tokenize_t0
        too_long=[len(x) for x in ids if len(x)+a.max_new_tokens>a.max_model_len]
        if too_long:
            raise ValueError(f'{len(too_long)} prompts exceed max-model-len={a.max_model_len} '
                             f'with max_new_tokens={a.max_new_tokens}. Largest actual prompt tokens={max(map(len,ids))}; '
                             'raise --max-model-len or use a smaller max-new-tokens.')
        os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
        engine=LLM(model=a.base,tokenizer=a.base,enable_lora=True,
                   max_lora_rank=rank,max_loras=1,dtype='bfloat16',
                   tensor_parallel_size=1,trust_remote_code=True,
                   gpu_memory_utilization=a.gpu_memory_utilization,
                   max_model_len=a.max_model_len,max_num_seqs=a.max_num_seqs,
                   enforce_eager=a.enforce_eager,seed=a.seed)
        load_secs=time.perf_counter()-t0-tokenize_secs
        sampler=SamplingParams(temperature=0.0,top_p=1.0,
                               max_tokens=a.max_new_tokens,seed=a.seed,
                               skip_special_tokens=True,
                               stop_token_ids=[tok.eos_token_id])
        req=LoRARequest('finqa_dynamic',1,a.adapter)

        def generate_chunk(chunk):
            inputs=[{'prompt_token_ids':seq} for seq in chunk]
            results=engine.generate(inputs,sampling_params=sampler,lora_request=req,use_tqdm=False)
            if len(results)!=len(chunk):
                raise RuntimeError(f'vLLM returned {len(results)} for {len(chunk)} prompts')
            return [(x.outputs[0].text.strip(),len(x.outputs[0].token_ids)) for x in results]

        w=min(a.warmup_samples,len(ids))
        t0=time.perf_counter()
        if w: generate_chunk(ids[:w])
        warmup_secs=time.perf_counter()-t0
        bs=a.request_batch_size or len(ids)
        preds=[]; count=0
        t0=time.perf_counter()
        for start in range(0,len(ids),bs):
            outputs=generate_chunk(ids[start:start+bs])
            for row,(text,rt) in zip(rows[start:start+bs],outputs):
                preds.append(make_prediction(row,text,rt,a.max_new_tokens,'peft_adapter'))
                count+=rt
            print(f'Progress: {len(preds)}/{len(rows)}',flush=True)
        gen_secs=time.perf_counter()-t0
    run={'engine':'vllm','base':a.base,'adapter':a.adapter,'input':a.input,
         'max_new_tokens':a.max_new_tokens,'request_batch_size':a.request_batch_size,
         'max_num_seqs':a.max_num_seqs,'max_model_len':a.max_model_len,
         'gpu_memory_utilization':a.gpu_memory_utilization,'enforce_eager':a.enforce_eager,
         'seed':a.seed,'do_sample':False,'prompt_format':'manual',
         'vllm_version':pkg_version('vllm'),'transformers_version':pkg_version('transformers'),
         'dtype':'bfloat16','lora_rank':rank}
    perf=perf_dict('vllm',len(preds),count,gen_secs,load_secs,warmup_secs,
                   tokenize_secs,fingerprint,run,mem.peak)
    save_results(a.output_dir,preds,run,perf)

if __name__=='__main__':main()
