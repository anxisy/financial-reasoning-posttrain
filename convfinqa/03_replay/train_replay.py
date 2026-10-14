#!/usr/bin/env python3
"""FinQA Adapter -> mixed ConvFinQA/FinQA SFT with 10% experience replay.
Cloned from frozen experiment-01 trainer to preserve hyperparameter parity.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random
import sys
import time
from pathlib import Path

BASE = '/root/autodl-tmp/models/Qwen3-1.7B-Base'
FINQA_ADAPTER = '/root/autodl-tmp/project/finqa_posttrain/experiments/sft_qwen3_1.7b/full_v1/final_adapter'
PROJECT = Path('/root/autodl-tmp/project/convfinqa_posttrain')
TRAIN = str(PROJECT/'experiments/convfinqa_sft_v2/03_finqa_sft_replay/data/replay_train.jsonl')
DEV = str(PROJECT/'data/processed_v2/sft/dev.jsonl')
EXPERIMENTS = PROJECT/'experiments/convfinqa_sft_v2/03_finqa_sft_replay'
# Normal invocation passes an explicit experiment-specific --output-dir.
LORA_TARGETS = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda: f.read(1<<20),b''): h.update(part)
    return h.hexdigest()


def read_jsonl(path):
    rows=[]; seen=set()
    with open(path,encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            if not line.strip(): continue
            r=json.loads(line); rid=str(r.get('id',''))
            if not rid or rid in seen: raise ValueError(f'duplicate/missing id {rid!r} in {path}:{i}')
            seen.add(rid)
            msgs=r.get('messages',[])
            if [m.get('role') for m in msgs]!=['system','user','assistant']:
                raise ValueError(f'bad message roles for {rid}')
            meta=r.get('metadata',{})
            origin=meta.get('replay_source')
            if rid.startswith('convfinqa::') and origin=='convfinqa':
                if meta.get('history_mode')!='oracle_gold_answers':
                    raise ValueError(f'non-oracle ConvFinQA row {rid}')
                if meta.get('cur_type') not in ('number_turn','program_turn'):
                    raise ValueError(f'unknown cur_type {rid}')
            elif rid.startswith('finqa::') and origin=='finqa':
                if meta.get('history_mode')=='oracle_gold_answers':
                    raise ValueError(f'FinQA single-turn row falsely marked oracle-history {rid}')
            elif origin is None and meta.get('history_mode')=='oracle_gold_answers' and not rid.startswith(('finqa::','convfinqa::')):
                # Unmixed, original ConvFinQA Dev set is valid for smoke evaluation.
                if meta.get('cur_type') not in ('number_turn','program_turn'):
                    raise ValueError(f'unknown Dev type {rid}')
            else:
                raise ValueError(f'invalid origin / sample id pairing {rid}: {origin}')
            target=msgs[-1].get('content','')
            if target.count('<answer>')!=1 or target.count('</answer>')!=1: raise ValueError(f'bad target answer tag {rid}')
            if meta.get('cur_type')=='number_turn' and 'Step ' in target: raise ValueError(f'number turn has steps {rid}')
            rows.append(r)
    if not rows: raise ValueError(f'empty dataset {path}')
    return rows


def serialize(row):
    m=row['messages']
    prompt=f"System:\n{m[0]['content'].strip()}\n\nUser:\n{m[1]['content'].strip()}\n\nAssistant:"
    target='\n'+m[2]['content'].strip()
    return prompt,target


def smoke_select(rows, n, seed):
    """Select both sources and one longest row from each in the smoke set."""
    if len(rows) <= n:return rows
    rng=random.Random(seed)
    by_origin={src:[r for r in rows if r['metadata'].get('replay_source')==src]
               for src in ('convfinqa','finqa')}
    if not all(by_origin.values()): raise ValueError('Smoke needs BOTH datasets')
    selected=[]
    for src in ('convfinqa','finqa'):
        longest=max(by_origin[src],key=lambda r:len(serialize(r)[0]))
        selected.append(longest)
    finqa_target=max(4,round(.1*n))
    selected.extend(rng.sample([r for r in by_origin['finqa'] if r not in selected],finqa_target-1))
    selected.extend(rng.sample([r for r in by_origin['convfinqa'] if r not in selected],n-len(selected)))
    rng.shuffle(selected)
    if len(selected)!=n: raise AssertionError('smoke selection count mismatch')
    return selected


class AssistantOnlyDataset:
    def __init__(self, rows, tokenizer, max_seq_length):
        self.items=[]; self.token_lengths=[]; self.assistant_lengths=[]
        eos=tokenizer.eos_token_id
        if eos is None: raise ValueError('tokenizer eos_token_id is None')
        for row in rows:
            prompt,target=serialize(row)
            pids=tokenizer.encode(prompt,add_special_tokens=False)
            tids=tokenizer.encode(target,add_special_tokens=False)
            total=len(pids)+len(tids)+1
            if total>max_seq_length:
                raise ValueError(f"NO TRUNCATION: id={row['id']} tokens={total}, max={max_seq_length}. Increase max_seq_length or revise a NEW data version.")
            if not tids: raise ValueError(f'empty assistant target id={row["id"]}')
            self.items.append({'input_ids':pids+tids+[eos],
                               'attention_mask':[1]*total,
                               'labels':[-100]*len(pids)+tids+[eos]})
            self.token_lengths.append(total)
            self.assistant_lengths.append(len(tids)+1)

    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]


class Collator:
    def __init__(self,pad_id): self.pad_id=pad_id
    def __call__(self, features):
        import torch
        m=max(len(r['input_ids']) for r in features)
        m=math.ceil(m/8)*8
        return {key:torch.tensor([r[key]+[(-100 if key=='labels' else 0 if key=='attention_mask' else self.pad_id)]*(m-len(r[key])) for r in features],dtype=torch.long)
                for key in ('input_ids','attention_mask','labels')}


def length_stats(vals):
    vals=sorted(vals)
    return {'min':vals[0],'median':vals[len(vals)//2],'p95':vals[min(len(vals)-1,math.ceil(.95*len(vals))-1)],'max':vals[-1]}


def training_args(**kwargs):
    from transformers import TrainingArguments
    sig=inspect.signature(TrainingArguments.__init__)
    strategy=kwargs.pop('evaluation_strategy_setting')
    kwargs['eval_strategy' if 'eval_strategy' in sig.parameters else 'evaluation_strategy']=strategy
    return TrainingArguments(**kwargs)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant', choices=['transfer'],default='transfer', help='Replay always continues original FinQA SFT LoRA')
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--base',default=BASE)
    ap.add_argument('--finqa-adapter',default=FINQA_ADAPTER)
    ap.add_argument('--train',default=TRAIN)
    ap.add_argument('--dev',default=DEV)
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--max-seq-length',type=int,default=4096)
    ap.add_argument('--learning-rate',type=float,default=5e-5)
    ap.add_argument('--epochs',type=float,default=1.0)
    ap.add_argument('--batch-size',type=int,default=1)
    ap.add_argument('--grad-accum',type=int,default=None)
    ap.add_argument('--smoke-steps',type=int,default=5)
    ap.add_argument('--smoke-train-rows',type=int,default=32)
    ap.add_argument('--smoke-eval-rows',type=int,default=8)
    ap.add_argument('--save-steps',type=int,default=250)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--resume',default=None,help='full only: Trainer checkpoint directory or latest')
    args=ap.parse_args()
    if args.grad_accum is None: args.grad_accum=2 if args.smoke else 8
    if args.max_seq_length<512 or args.grad_accum<1 or args.smoke_steps<1: ap.error('invalid numeric args')
    if args.resume and args.smoke: ap.error('smoke may not resume')
    if not args.smoke and args.epochs<=0: ap.error('--epochs must be positive')
    for p in (args.base,args.train,args.dev):
        if not Path(p).exists(): raise FileNotFoundError(p)
    if not (Path(args.finqa_adapter)/'adapter_config.json').exists():
        raise FileNotFoundError(f'FinQA SFT adapter missing: {args.finqa_adapter}')
    exp_dir=EXPERIMENTS/('smoke' if args.smoke else 'full')
    out=Path(args.output_dir or exp_dir).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'training_report.json').exists() and not args.resume:
        raise RuntimeError(f'completed output exists: {out}. Do not overwrite frozen experiments.')
    # Delay heavyweight import until all path/schema checks pass.
    import torch,transformers,peft
    from transformers import AutoModelForCausalLM,AutoTokenizer,Trainer
    from peft import LoraConfig,PeftModel,get_peft_model
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    if not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 support required')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    os.environ.setdefault('WANDB_DISABLED','true')
    print(f'Variant={args.variant} smoke={args.smoke} output={out}',flush=True)
    print(f'Versions torch={torch.__version__}, transformers={transformers.__version__}, peft={peft.__version__}',flush=True)
    tok=AutoTokenizer.from_pretrained(args.base,local_files_only=True,use_fast=True)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    tok.padding_side='right'
    train_rows=read_jsonl(args.train); dev_rows=read_jsonl(args.dev)
    mix_manifest=Path(args.train).with_name('mix_manifest.json')
    if not mix_manifest.is_file():
        raise FileNotFoundError('Replay manifest missing: '+str(mix_manifest))
    mx=json.loads(mix_manifest.read_text(encoding='utf8'))
    if mx['train_sha256']!=sha256(args.train) or mx['total_rows']!=len(train_rows):
        raise RuntimeError('Replay dataset differs from frozen mix manifest')
    sources={name:sum(r['metadata'].get('replay_source')==name for r in train_rows)
             for name in ('finqa','convfinqa')}
    if sources!={'finqa':1110,'convfinqa':9994} or len(train_rows)!=11104:
        raise RuntimeError(f'Unexpected replay mixture: {sources}')
    if not args.smoke and (args.epochs!=1.0 or args.batch_size!=1 or args.grad_accum!=8):
        raise RuntimeError('Fair replay v1 requires epochs=1, batch_size=1, grad_accum=8 (1388 steps)')
    print('Validated frozen replay mix:',sources,'sha256:',sha256(args.train),flush=True)
    if args.smoke:
        train_rows=smoke_select(train_rows,args.smoke_train_rows,args.seed)
        dev_rows=dev_rows[:args.smoke_eval_rows]
    ds=AssistantOnlyDataset(train_rows,tok,args.max_seq_length)
    ev=AssistantOnlyDataset(dev_rows,tok,args.max_seq_length)
    print('Train rows:',len(ds),'lengths:',length_stats(ds.token_lengths),flush=True)
    print('Dev rows:',len(ev),'lengths:',length_stats(ev.token_lengths),flush=True)
    sample_p,sample_t=serialize(train_rows[0])
    (out/'serialization_preview.txt').write_text('=== PROMPT ===\n'+sample_p+'\n=== TARGET ===\n'+sample_t,encoding='utf-8')
    model=AutoModelForCausalLM.from_pretrained(args.base,torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True)
    model.config.use_cache=False
    model=PeftModel.from_pretrained(model,args.finqa_adapter,is_trainable=True)
    cfg=model.peft_config[model.active_adapter]
    if int(cfg.r)!=8: raise RuntimeError(f'Unexpected FinQA SFT rank {cfg.r}; inspect adapter')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    total=sum(p.numel() for p in model.parameters())
    if trainable<=0: raise RuntimeError('No trainable parameters; adapter was loaded frozen')
    print(f'LoRA trainable params = {trainable:,}/{total:,} ({100*trainable/total:.4f}%)',flush=True)
    model.print_trainable_parameters()
    manifest={
        'variant':'transfer_replay_10pct','smoke':args.smoke,'model':args.base,'finqa_adapter':args.finqa_adapter if args.variant=='transfer' else None,
        'output':str(out),'train':args.train,'dev':args.dev,
        'train_sha256':sha256(args.train),'dev_sha256':sha256(args.dev),
        'n_train_this_run':len(ds),'n_dev_this_run':len(ev),
        'max_seq_length':args.max_seq_length,'no_truncation':True,'assistant_only_loss':True,
        'prompt_ends':'Assistant:','target_begins_newline':True,
        'lr':args.learning_rate,'epochs':args.epochs,'grad_accum':args.grad_accum,'batch_size':args.batch_size,
        'smoke_steps':args.smoke_steps if args.smoke else None,'seed':args.seed,
        'train_lengths':length_stats(ds.token_lengths),'dev_lengths':length_stats(ev.token_lengths),
        'trainable_parameters':trainable,'total_parameters':total,
        'versions':{'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__},
    }
    (out/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    kwargs=dict(output_dir=str(out),overwrite_output_dir=False,per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,lr_scheduler_type='cosine',warmup_ratio=.03,weight_decay=.01,
        max_grad_norm=1.0,logging_strategy='steps',logging_steps=1 if args.smoke else 20,
        report_to=[],seed=args.seed,data_seed=args.seed,bf16=True,fp16=False,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        remove_unused_columns=False,dataloader_num_workers=0,optim='adamw_torch',
        save_safetensors=True,save_total_limit=2,dataloader_pin_memory=True,
        evaluation_strategy_setting='steps' if args.smoke else 'no')
    if args.smoke:
        kwargs.update(max_steps=args.smoke_steps,eval_steps=args.smoke_steps,
                      save_strategy='no')
    else:
        kwargs.update(num_train_epochs=args.epochs,save_strategy='steps',save_steps=args.save_steps)
    targs=training_args(**kwargs)
    trainer=Trainer(model=model,args=targs,train_dataset=ds,eval_dataset=ev if args.smoke else None,
                    data_collator=Collator(tok.pad_token_id))
    if args.smoke:
        # Explicitly test the longest selected training sequence with a backward pass.
        # A shuffled five-step smoke run alone could miss the expensive examples.
        index=max(range(len(ds)), key=lambda i: ds.token_lengths[i])
        stress=Collator(tok.pad_token_id)([ds[index]])
        stress={key:value.to('cuda') for key,value in stress.items()}
        print(f'LONGEST-SAMPLE BACKWARD CHECK: tokens={ds.token_lengths[index]}',flush=True)
        model.train()
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            stress_loss=model(**stress).loss
        if not torch.isfinite(stress_loss).item():
            raise RuntimeError('Longest sample stress loss is not finite')
        stress_loss.backward()
        torch.cuda.synchronize()
        print('LONGEST-SAMPLE BACKWARD PASS OK; loss=',float(stress_loss.detach().cpu()),flush=True)
        model.zero_grad(set_to_none=True)
        del stress, stress_loss
        torch.cuda.empty_cache()
    resume=None
    if args.resume:
        if args.resume=='latest':
            checkpoints=sorted(((int(p.name.split('-')[-1]),p) for p in out.glob('checkpoint-*') if p.name.split('-')[-1].isdigit()),reverse=True)
            if not checkpoints: raise FileNotFoundError(f'no checkpoint-* in {out}')
            resume=str(checkpoints[0][1])
        else:
            resume=str(Path(args.resume).resolve())
            if not Path(resume).exists(): raise FileNotFoundError(resume)
    print('TRAIN START:',time.ctime(),flush=True)
    start=time.perf_counter()
    train_result=trainer.train(resume_from_checkpoint=resume)
    elapsed=time.perf_counter()-start
    if not math.isfinite(train_result.training_loss): raise RuntimeError('training_loss is non-finite')
    eval_metrics=trainer.evaluate() if args.smoke else None
    final_adapter=out/'final_adapter'
    if final_adapter.exists() and not args.resume:
        raise RuntimeError(f'Final adapter already exists: {final_adapter}')
    final_adapter.mkdir(exist_ok=True,parents=True)
    trainer.model.save_pretrained(str(final_adapter),safe_serialization=True)
    tok.save_pretrained(str(final_adapter))
    report={**manifest,'elapsed_seconds':elapsed,'global_step':trainer.state.global_step,
            'train_metrics':train_result.metrics,'eval_metrics':eval_metrics,
            'peak_cuda_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'peak_cuda_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
            'final_adapter':str(final_adapter)}
    (out/'training_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print('TRAIN COMPLETED / ADAPTER:',final_adapter,flush=True)
    print('TRAIN LOSS:',train_result.training_loss,'EVAL:',eval_metrics,'PEAK CUDA GIB:',report['peak_cuda_reserved_gib'],flush=True)

if __name__=='__main__': main()
