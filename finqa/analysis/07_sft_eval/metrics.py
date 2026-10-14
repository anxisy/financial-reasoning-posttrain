# -*- coding: utf-8 -*-
from collections import defaultdict

def _sum(rows):
    n=len(rows)
    if not n:
        return {"n":0,"accuracy":0.0,"strict_accuracy":0.0,"format_accuracy":0.0,
                "answer_found_rate":0.0,"max_token_hit_rate":0.0,
                "role_repetition_rate":0.0,"avg_response_tokens":0.0}
    return {
        "n":n,
        "accuracy":sum(bool(x.get("correct")) for x in rows)/n,
        "strict_accuracy":sum(bool(x.get("correct")) and bool(x.get("format_ok")) for x in rows)/n,
        "format_accuracy":sum(bool(x.get("format_ok")) for x in rows)/n,
        "answer_found_rate":sum(bool(x.get("prediction_answer")) for x in rows)/n,
        "max_token_hit_rate":sum(bool(x.get("hit_max_new_tokens")) for x in rows)/n,
        "role_repetition_rate":sum(bool(x.get("role_repetition_flag")) for x in rows)/n,
        "avg_response_tokens":sum(int(x.get("response_tokens",0) or 0) for x in rows)/n,
    }

def _group(rows, fn):
    g=defaultdict(list)
    for r in rows: g[str(fn(r))].append(r)
    return {k:_sum(v) for k,v in sorted(g.items())}

def _ctx(n):
    n=int(n or 0)
    if n<=1024:return "<=1024"
    if n<=1536:return "1025-1536"
    if n<=2048:return "1537-2048"
    return ">2048"

def compute_metrics(rows):
    m={"overall":_sum(rows)}
    m["by_difficulty"]=_group(rows,lambda x:x.get("metadata",{}).get("difficulty","unknown"))
    m["by_context_length"]=_group(rows,lambda x:_ctx(x.get("metadata",{}).get("original_prompt_tokens",0)))
    m["by_annotation_quality"]=_group(rows,lambda x:x.get("metadata",{}).get("annotation_quality","unknown"))
    m["by_semantic_status"]=_group(rows,lambda x:x.get("metadata",{}).get("semantic_status","unknown"))
    m["by_truncation"]=_group(rows,lambda x:"truncated" if int(x.get("metadata",{}).get("was_truncated",0)) else "not_truncated")
    ops=defaultdict(list)
    for r in rows:
        for op in sorted(set(map(str,r.get("metadata",{}).get("operators",[]) or []))):
            ops[op].append(r)
    m["by_operator"]={k:_sum(v) for k,v in sorted(ops.items())}
    m["by_extraction_method"]=_group(rows,lambda x:x.get("extraction_method","unknown"))
    return m
