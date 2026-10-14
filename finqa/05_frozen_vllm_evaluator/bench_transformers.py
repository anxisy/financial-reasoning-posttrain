"""Transformers+PEFT reference benchmark aligned with the frozen FinQA evaluator."""
import argparse
import random
import time
from pathlib import Path
from common import (DEFAULT_ADAPTER,DEFAULT_BASE,DEFAULT_DEV,make_prediction,perf_dict,
                    pkg_version,prepare,read_rows,save_results)
from gpu_monitor import VramMonitor


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',default=DEFAULT_BASE)
    p.add_argument('--adapter',default=DEFAULT_ADAPTER)
    p.add_argument('--input',default=DEFAULT_DEV)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--max-samples',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--max-new-tokens',type=int,default=384)
    p.add_argument('--warmup-samples',type=int,default=2)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if a.batch_size<1 or a.max_new_tokens<1: p.error('batch-size and max-new-tokens must be positive')
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    rows=read_rows(a.input,a.max_samples)
    with VramMonitor() as mem:
        t0=time.perf_counter()
        tok=AutoTokenizer.from_pretrained(a.base,local_files_only=True,trust_remote_code=True)
        tok.padding_side='left'
        if tok.pad_token_id is None: tok.pad_token=tok.eos_token
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        base=AutoModelForCausalLM.from_pretrained(
            a.base,local_files_only=True,trust_remote_code=True,
            torch_dtype=dtype,device_map='auto')
        model=PeftModel.from_pretrained(base,a.adapter,is_trainable=False)
        if hasattr(model.config,'use_cache'): model.config.use_cache=True
        model.eval()
        torch.cuda.synchronize()
        load_secs=time.perf_counter()-t0
        t0=time.perf_counter()
        _,ids,fingerprint=prepare(rows,tok)
        tokenize_secs=time.perf_counter()-t0
        device=next(model.parameters()).device

        def generate_chunk(token_ids):
            enc=tok.pad({'input_ids':token_ids},return_tensors='pt',padding=True)
            enc={k:v.to(device) for k,v in enc.items()}
            with torch.inference_mode():
                gen=model.generate(**enc,max_new_tokens=a.max_new_tokens,do_sample=False,
                                   use_cache=True,pad_token_id=tok.pad_token_id,
                                   eos_token_id=tok.eos_token_id)
            comp=gen[:,enc['input_ids'].shape[1]:]
            result=[]
            for i in range(len(token_ids)):
                text=tok.decode(comp[i],skip_special_tokens=True).strip()
                count=int((comp[i]!=tok.pad_token_id).sum().item())
                result.append((text,count))
            return result

        w=min(a.warmup_samples,len(ids))
        t0=time.perf_counter()
        if w: generate_chunk(ids[:w]); torch.cuda.synchronize()
        warmup_secs=time.perf_counter()-t0
        preds=[]; count=0
        t0=time.perf_counter()
        for s in range(0,len(ids),a.batch_size):
            output=generate_chunk(ids[s:s+a.batch_size])
            for r,(text,rt) in zip(rows[s:s+a.batch_size],output):
                preds.append(make_prediction(r,text,rt,a.max_new_tokens,'peft_adapter'))
                count+=rt
            if (s//a.batch_size+1)%20==0:
                print(f'Progress: {len(preds)}/{len(rows)}',flush=True)
        torch.cuda.synchronize()
        gen_secs=time.perf_counter()-t0
    run={'engine':'transformers','base':a.base,'adapter':a.adapter,'input':a.input,
         'max_new_tokens':a.max_new_tokens,'batch_size':a.batch_size,
         'seed':a.seed,'do_sample':False,'prompt_format':'manual',
         'torch_version':torch.__version__, 'transformers_version':pkg_version('transformers'),
         'peft_version':pkg_version('peft'), 'dtype':str(dtype)}
    perf=perf_dict('transformers',len(preds),count,gen_secs,load_secs,warmup_secs,
                   tokenize_secs,fingerprint,run,mem.peak)
    save_results(a.output_dir,preds,run,perf)

if __name__=='__main__':main()
