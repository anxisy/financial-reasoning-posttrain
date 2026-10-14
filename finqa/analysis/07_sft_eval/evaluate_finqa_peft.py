# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, random, re, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from answer_normalizer import compare_answer_strings, extract_answer_with_method
from metrics import compute_metrics

def read_jsonl(path):
    with open(path,"r",encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def load_existing(path):
    if not path.exists(): return [],set()
    rows=[]; ids=set()
    with path.open("r",encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line); rows.append(r)
            if r.get("id") is not None: ids.add(r["id"])
    return rows,ids

def manual_serialize(messages):
    parts=[]
    for msg in messages:
        role=str(msg.get("role","")).lower()
        c=str(msg.get("content","")).strip()
        if role=="system": parts.append(f"System:\n{c}")
        elif role=="user": parts.append(f"User:\n{c}")
        elif role=="assistant": parts.append(f"Assistant:\n{c}")
        else: parts.append(f"{role.title()}:\n{c}")
    parts.append("Assistant:")
    p="\n\n".join(parts)
    assert p.endswith("Assistant:") and not p.endswith("Assistant:\n")
    return p

def dtype_of(name):
    if name=="float32": return torch.float32
    if name=="float16": return torch.float16
    if name=="bfloat16": return torch.bfloat16
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

def role_rep(text):
    return len(re.findall(r"\bAssistant\b",text or "",flags=re.I))>=5

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base-model-path",default="/root/autodl-tmp/models/Qwen3-0.6B-Base")
    p.add_argument("--adapter-path",default=None)
    p.add_argument("--input",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--max-new-tokens",type=int,default=384)
    p.add_argument("--max-samples",type=int,default=0)
    p.add_argument("--prompt-format",choices=["manual"],default="manual")
    p.add_argument("--dtype",choices=["auto","float32","float16","bfloat16"],default="auto")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--print-every",type=int,default=20)
    p.add_argument("--numeric-rel-tol",type=float,default=1e-3)
    p.add_argument("--numeric-abs-tol",type=float,default=1e-2)
    p.add_argument("--percent-abs-tol",type=float,default=0.10)
    a=p.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)

    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    pred_path=out/"predictions.jsonl"; metrics_path=out/"metrics.json"
    errors_path=out/"errors.jsonl"; config_path=out/"run_config.json"

    rows=read_jsonl(a.input)
    if a.max_samples>0: rows=rows[:a.max_samples]
    _,done=load_existing(pred_path) if a.resume else ([],set())
    todo=[r for r in rows if r.get("id") not in done]

    tok=AutoTokenizer.from_pretrained(a.base_model_path,local_files_only=True,trust_remote_code=True)
    tok.padding_side="left"
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    dtype=dtype_of(a.dtype)

    base=AutoModelForCausalLM.from_pretrained(
        a.base_model_path,local_files_only=True,trust_remote_code=True,
        torch_dtype=dtype,device_map="auto" if torch.cuda.is_available() else None
    )
    kind="base"
    model=base
    if a.adapter_path:
        from peft import PeftModel
        model=PeftModel.from_pretrained(base,a.adapter_path,is_trainable=False)
        kind="peft_adapter"
    if hasattr(model.config,"use_cache"): model.config.use_cache=True
    if not torch.cuda.is_available(): model=model.to("cpu")
    model.eval()

    cfg={
        "version":"finqa_peft_eval_v1",
        "model_kind":kind,
        "base_model_path":a.base_model_path,
        "adapter_path":a.adapter_path,
        "input":a.input,
        "num_examples_requested":len(rows),
        "batch_size":a.batch_size,
        "max_new_tokens":a.max_new_tokens,
        "prompt_format":"manual",
        "prompt_ends_exactly_with":"Assistant:",
        "dtype":str(dtype),
        "seed":a.seed,
        "do_sample":False,
        "numeric_rel_tol":a.numeric_rel_tol,
        "numeric_abs_tol":a.numeric_abs_tol,
        "percent_abs_tol":a.percent_abs_tol,
    }
    config_path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding="utf-8")

    mode="a" if a.resume and pred_path.exists() else "w"
    n_done=0; start=time.time()
    with pred_path.open(mode,encoding="utf-8") as pf:
        for s in range(0,len(todo),a.batch_size):
            batch=todo[s:s+a.batch_size]
            prompts=[manual_serialize(r["prompt"]) for r in batch]
            enc=tok(prompts,return_tensors="pt",padding=True,truncation=False,add_special_tokens=True)
            device=next(model.parameters()).device
            enc={k:v.to(device) for k,v in enc.items()}
            with torch.inference_mode():
                gen=model.generate(**enc,max_new_tokens=a.max_new_tokens,do_sample=False,use_cache=True,
                                   pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
            width=enc["input_ids"].shape[1]
            comp=gen[:,width:]
            for i,r in enumerate(batch):
                text=tok.decode(comp[i],skip_special_tokens=True).strip()
                rt=int((comp[i]!=tok.pad_token_id).sum().item())
                ans,fmt,method=extract_answer_with_method(text,True)
                ok,reason=compare_answer_strings(ans,r.get("gold_answer",""),
                    numeric_rel_tol=a.numeric_rel_tol,numeric_abs_tol=a.numeric_abs_tol,
                    percent_abs_tol=a.percent_abs_tol)
                row={
                    "id":r.get("id"),"prediction":text,"prediction_answer":ans,
                    "gold_answer":r.get("gold_answer",""),"correct":bool(ok),
                    "format_ok":bool(fmt),"strict_correct":bool(ok and fmt),
                    "match_reason":reason,"extraction_method":method,
                    "response_tokens":rt,"hit_max_new_tokens":rt>=a.max_new_tokens,
                    "role_repetition_flag":role_rep(text),"prompt_format_used":"manual",
                    "model_kind":kind,"metadata":r.get("metadata",{})
                }
                pf.write(json.dumps(row,ensure_ascii=False)+"\n"); pf.flush(); n_done+=1
            if n_done and (n_done%a.print_every==0 or n_done==len(todo)):
                rate=n_done/max(time.time()-start,1e-9)
                print(f"Generated {n_done}/{len(todo)} ({rate:.2f} examples/s)",flush=True)

    all_rows,_=load_existing(pred_path)
    m=compute_metrics(all_rows); m["run"]=cfg; m["num_predictions"]=len(all_rows)
    metrics_path.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding="utf-8")
    with errors_path.open("w",encoding="utf-8") as f:
        for r in all_rows:
            if not r.get("correct"): f.write(json.dumps(r,ensure_ascii=False)+"\n")

    o=m["overall"]
    print("\nRESULT")
    for k in ["n","accuracy","strict_accuracy","format_accuracy","answer_found_rate",
              "max_token_hit_rate","role_repetition_rate","avg_response_tokens"]:
        print(f"{k:22s}: {o[k]}")

if __name__=="__main__":
    main()
