# -*- coding: utf-8 -*-
from __future__ import annotations
import json, random
from collections import defaultdict
from pathlib import Path

CANDIDATES = [
    "/root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1/grpo/finqa_grpo_train.jsonl",
    "/root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1/grpo_train.jsonl",
    "/root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1/finqa_grpo_train.jsonl",
]

def discover_train_file(explicit=None):
    if explicit:
        p = Path(explicit)
        if p.exists(): return str(p)
        raise FileNotFoundError(f"GRPO train file not found: {p}")
    for x in CANDIDATES:
        if Path(x).exists(): return x
    raise FileNotFoundError("Could not auto-find GRPO train JSONL. Tried:\n" + "\n".join(CANDIDATES))

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def manual_prompt(messages):
    if not isinstance(messages, list):
        s = str(messages)
        if not s.rstrip().endswith("Assistant:"):
            return s.rstrip() + "\n\nAssistant:"
        return s.rstrip()
    parts=[]
    for msg in messages:
        role=str(msg.get("role","")).lower(); c=str(msg.get("content","")).strip()
        if role=="system": parts.append(f"System:\n{c}")
        elif role=="user": parts.append(f"User:\n{c}")
        elif role=="assistant": parts.append(f"Assistant:\n{c}")
        else: parts.append(f"{role.title()}:\n{c}")
    # GRPO data should normally contain only system/user. Guard against accidental assistant target leakage.
    if messages and str(messages[-1].get("role","")).lower() == "assistant":
        raise ValueError("GRPO prompt contains an assistant target; prompt-only data required.")
    parts.append("Assistant:")
    out="\n\n".join(parts)
    assert out.endswith("Assistant:") and not out.endswith("Assistant:\n")
    return out

def normalize_row(row, idx):
    prompt_obj = row.get("prompt", row.get("messages"))
    if prompt_obj is None:
        raise KeyError("GRPO row needs `prompt` or `messages`.")
    gold = row.get("gold_answer", row.get("answer"))
    if gold is None:
        raise KeyError("GRPO row needs `gold_answer` (or `answer`).")
    meta = row.get("metadata", {}) or {}
    return {
        "prompt": manual_prompt(prompt_obj),
        "gold_answer": str(gold),
        "example_id": str(row.get("id", idx)),
        "difficulty": str(meta.get("difficulty", row.get("difficulty", "unknown"))),
        "operators_json": json.dumps(meta.get("operators", row.get("operators", [])) or [], ensure_ascii=False),
    }

def stratified_smoke(rows, n=64, seed=42):
    if n >= len(rows): return list(rows)
    groups=defaultdict(list)
    for r in rows: groups[r.get("difficulty","unknown")].append(r)
    rng=random.Random(seed)
    for vals in groups.values(): rng.shuffle(vals)
    # FinQA train is roughly 59.5% easy, 32.2% medium, 8.3% hard.
    targets={"easy": round(n*0.595), "medium": round(n*0.322), "hard": n-round(n*0.595)-round(n*0.322)}
    out=[]
    for k,t in targets.items(): out.extend(groups.get(k,[])[:t])
    if len(out)<n:
        used={r["example_id"] for r in out}
        rest=[r for r in rows if r["example_id"] not in used]
        rng.shuffle(rest); out.extend(rest[:n-len(out)])
    rng.shuffle(out)
    return out[:n]
