#!/usr/bin/env python3
import json
from transformers import AutoTokenizer
MODEL = "/root/autodl-tmp/models/Qwen3-1.7B-Base"
TRAIN = "/root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1/sft/finqa_sft_train.jsonl"
def get_message(row, role):
    for m in row["messages"]:
        if m["role"] == role:
            return m["content"]
    raise KeyError(role)
with open(TRAIN, "r", encoding="utf-8") as f:
    row = json.loads(next(f))
system = get_message(row, "system").strip()
user = get_message(row, "user").strip()
assistant = get_message(row, "assistant").strip()
prompt = f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:"
target = "\n" + assistant
assert prompt.endswith("Assistant:") and not prompt.endswith("Assistant:\n")
assert target.startswith("\n") and "<answer>" in target and "</answer>" in target
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
pids = tok(prompt, add_special_tokens=False)["input_ids"]
tids = tok(target, add_special_tokens=False)["input_ids"]
print("id               :", row.get("id"))
print("prompt ends      :", repr(prompt[-20:]))
print("target begins    :", repr(target[:30]))
print("prompt tokens    :", len(pids))
print("target tokens    :", len(tids))
print("combined + EOS   :", len(pids) + len(tids) + 1)
print("prompt <= 1664   :", len(pids) <= 1664)
print("sequence <= 2048 :", len(pids) + len(tids) + 1 <= 2048)
print("SELF TEST PASSED")
