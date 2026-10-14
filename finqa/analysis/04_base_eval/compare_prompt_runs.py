# -*- coding: utf-8 -*-
"""
Compare two smoke-test runs, typically chat-template vs manual prompt.

Example:
python compare_prompt_runs.py \
  --a /path/to/chat/metrics.json \
  --b /path/to/manual/metrics.json
"""

import argparse
import json
from pathlib import Path


KEYS = [
    "accuracy",
    "strict_accuracy",
    "format_accuracy",
    "answer_found_rate",
    "max_token_hit_rate",
    "role_repetition_rate",
    "avg_response_tokens",
]


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--name-a", default="A")
    p.add_argument("--name-b", default="B")
    args = p.parse_args()

    a = load(args.a)
    b = load(args.b)

    oa = a["overall"]
    ob = b["overall"]

    print("=" * 78)
    print(f"{'metric':26s} {args.name_a:>20s} {args.name_b:>20s}")
    print("=" * 78)

    for key in KEYS:
        va = oa.get(key, 0)
        vb = ob.get(key, 0)
        if isinstance(va, float):
            print(f"{key:26s} {va:20.4f} {vb:20.4f}")
        else:
            print(f"{key:26s} {str(va):>20s} {str(vb):>20s}")

    print("\nPrompt formats:")
    print(f"  {args.name_a}: {a.get('run', {}).get('actual_prompt_format')}")
    print(f"  {args.name_b}: {b.get('run', {}).get('actual_prompt_format')}")

    print("\nInterpretation:")
    print("- Higher answer accuracy is better.")
    print("- Higher answer-found rate is better.")
    print("- Lower max-token-hit rate is better.")
    print("- Lower role-repetition rate is better.")
    print("- Format accuracy is diagnostic for Base and may be low before SFT.")
    print("- With only 20 samples, do NOT treat small accuracy differences as final results.")


if __name__ == "__main__":
    main()
