#!/usr/bin/env python3
import argparse, json
p = argparse.ArgumentParser()
p.add_argument("--before", required=True)
p.add_argument("--after", required=True)
p.add_argument("--before-name", default="Before")
p.add_argument("--after-name", default="After")
a = p.parse_args()
with open(a.before, "r", encoding="utf-8") as f: x = json.load(f)
with open(a.after, "r", encoding="utf-8") as f: y = json.load(f)
keys = ["accuracy","strict_accuracy","format_accuracy","answer_found_rate","max_token_hit_rate","role_repetition_rate","avg_response_tokens"]
print("="*88)
print(f"{'metric':28s}{a.before_name:>20s}{a.after_name:>20s}{'delta':>20s}")
print("="*88)
for k in keys:
    if k not in x or k not in y: continue
    xv, yv = x[k], y[k]
    if "tokens" in k:
        print(f"{k:28s}{xv:20.2f}{yv:20.2f}{yv-xv:20.2f}")
    else:
        print(f"{k:28s}{xv*100:19.2f}%{yv*100:19.2f}%{(yv-xv)*100:19.2f}pp")
