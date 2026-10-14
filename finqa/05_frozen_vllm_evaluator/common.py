"""Frozen FinQA serialization, scoring and benchmark report utilities.

`manual_serialize`, `role_rep`, evaluation row fields and metric routines are
aligned to finqa_sft_eval_v1/evaluate_finqa_peft.py (the established evaluator).
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import re
import time
from pathlib import Path
from answer_normalizer import extract_answer_with_method, compare_answer_strings
from metrics import compute_metrics

DEFAULT_BASE = "/root/autodl-tmp/models/Qwen3-1.7B-Base"
DEFAULT_ADAPTER = "/root/autodl-tmp/project/finqa_posttrain/experiments/grpo_dynamic_qwen3_1.7b/full_v1/final_adapter"
DEFAULT_DEV = "/root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1/eval/finqa_eval_dev.jsonl"


def manual_serialize(messages):
    parts=[]
    for msg in messages:
        role=str(msg.get("role", "")).lower()
        c=str(msg.get("content", "")).strip()
        if role=="system": parts.append(f"System:\n{c}")
        elif role=="user": parts.append(f"User:\n{c}")
        elif role=="assistant": parts.append(f"Assistant:\n{c}")
        else: parts.append(f"{role.title()}:\n{c}")
    parts.append("Assistant:")
    p="\n\n".join(parts)
    assert p.endswith("Assistant:") and not p.endswith("Assistant:\n")
    return p


def role_rep(text):
    return len(re.findall(r"\bAssistant\b", text or "", flags=re.I)) >= 5


def read_rows(path, limit=0):
    out=[]; seen=set()
    with Path(path).open(encoding="utf-8") as f:
        for n,line in enumerate(f,1):
            if not line.strip(): continue
            r=json.loads(line)
            if "prompt" not in r or not isinstance(r["prompt"], list):
                raise ValueError(f"{path}:{n}: missing list 'prompt'")
            rid=r.get("id")
            if rid is None or rid in seen:
                raise ValueError(f"{path}:{n}: missing/duplicate ID: {rid!r}")
            seen.add(rid)
            out.append(r)
            if limit and len(out)>=limit: break
    if not out: raise ValueError(f"No rows in {path}")
    return out


def prepare(rows, tok):
    prompts=[manual_serialize(r['prompt']) for r in rows]
    # Frozen evaluator uses add_special_tokens=True, no truncation.
    encoded=tok(prompts, padding=False, truncation=False, add_special_tokens=True)
    ids=encoded['input_ids']
    if any(not s for s in ids): raise ValueError('Empty token IDs')
    payload=json.dumps([(str(r['id']), s) for r,s in zip(rows,ids)],
                       ensure_ascii=False,separators=(',',':')).encode()
    fingerprint=hashlib.sha256(payload).hexdigest()
    return prompts,ids,fingerprint


def make_prediction(row,text,token_count,max_new_tokens,kind):
    text=text.strip()
    ans,fmt,method=extract_answer_with_method(text, True)
    ok,reason=compare_answer_strings(ans,row.get('gold_answer',''),
                                     numeric_rel_tol=1e-3, numeric_abs_tol=1e-2,
                                     percent_abs_tol=.10)
    return {
        'id':row.get('id'),'prediction':text,'prediction_answer':ans,
        'gold_answer':row.get('gold_answer',''),'correct':bool(ok),
        'format_ok':bool(fmt),'strict_correct':bool(ok and fmt),
        'match_reason':reason,'extraction_method':method,
        'response_tokens':int(token_count),'hit_max_new_tokens':int(token_count)>=max_new_tokens,
        'role_repetition_flag':role_rep(text),'prompt_format_used':'manual',
        'model_kind':kind,'metadata':row.get('metadata',{})
    }


def pkg_version(name):
    try: return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: return None


def save_results(out, preds, run, perf):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    # Fresh writes only, to avoid accidentally comparing resumed/duplicate predictions.
    with (out/'predictions.jsonl').open('w',encoding='utf-8') as f:
        for p in preds: f.write(json.dumps(p,ensure_ascii=False)+'\n')
    met=compute_metrics(preds)
    met['run']=run; met['num_predictions']=len(preds)
    (out/'metrics.json').write_text(json.dumps(met,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'benchmark.json').write_text(json.dumps(perf,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'errors.jsonl').open('w',encoding='utf-8') as f:
        for p in preds:
            if not p['correct']: f.write(json.dumps(p,ensure_ascii=False)+'\n')
    o=met['overall']
    print('\n=== RESULT ===')
    for k in ['n','accuracy','strict_accuracy','format_accuracy','answer_found_rate',
              'max_token_hit_rate','role_repetition_rate','avg_response_tokens']:
        print(f'{k:24s}: {o[k]}')
    print(f'Inference wall-time      : {perf["inference_seconds"]:.3f} s')
    print(f'Samples / s              : {perf["samples_per_second"]:.4f}')
    print(f'Generated tokens / s     : {perf["generated_tokens_per_second"]:.4f}')
    print(f'Peak GPU memory used MiB : {perf.get("peak_gpu_used_mib")}')
    print(f'Output                   : {out}')


def perf_dict(engine, n, total_tokens, generation_seconds, load_seconds, warmup_seconds,
              tokenize_seconds, fingerprint, kwargs, peak):
    return {
        'engine':engine, 'examples':n, 'generated_tokens':total_tokens,
        'inference_seconds':generation_seconds,
        'samples_per_second':n/max(generation_seconds,1e-10),
        'generated_tokens_per_second':total_tokens/max(generation_seconds,1e-10),
        'model_load_seconds':load_seconds,'warmup_seconds':warmup_seconds,
        'tokenization_seconds_excluded':tokenize_seconds,
        'prompt_token_ids_sha256':fingerprint,'peak_gpu_used_mib':peak,
        'gpu_memory_note':'NVML total device memory (includes other processes); engine stays loaded through inference',
        'timing_note':'Excludes load, warmup, whole-dataset tokenization; includes batch preparation, inference, decoding, and result extraction',
        'run':kwargs, 'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    }
