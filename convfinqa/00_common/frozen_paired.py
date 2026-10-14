#!/usr/bin/env python3
"""Pairwise ConvFinQA baseline analysis with conversation-clustered bootstrap CIs."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

DEFAULT_ROOT = "/root/autodl-tmp/project/convfinqa_posttrain/experiments/oracle_baselines_v1/full_dev"
LABELS = {"base": "qwen3_base", "sft": "finqa_sft", "dynamic": "finqa_dynamic"}


def read_predictions(path):
    p = Path(path)
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    d = {str(r["id"]): r for r in rows}
    if len(d) != len(rows):
        raise ValueError("Duplicate IDs in " + str(p))
    return d


def comparison(a, b, seed=42, bootstrap_replicates=2000):
    if set(a) != set(b):
        raise ValueError("Different prediction IDs; cannot compare")
    ids = sorted(a)
    wins = [k for k in ids if not a[k]["correct"] and b[k]["correct"]]
    losses = [k for k in ids if a[k]["correct"] and not b[k]["correct"]]
    conv_data = defaultdict(list)
    for ident in ids:
        ca = a[ident]["metadata"]["conversation_id"]
        cb = b[ident]["metadata"]["conversation_id"]
        if ca != cb:
            raise ValueError(f"Conversation mismatch: {ident}")
        conv_data[ca].append((int(a[ident]["correct"]), int(b[ident]["correct"])))
    rng = random.Random(seed)
    cid = sorted(conv_data)
    differences = []
    for _ in range(bootstrap_replicates):
        chosen = rng.choices(cid, k=len(cid))
        n = 0
        delta = 0
        for key in chosen:
            pairs = conv_data[key]
            n += len(pairs)
            delta += sum(b - a for a, b in pairs)
        differences.append(100.0 * delta / n)
    differences.sort()
    lo = differences[int(.025 * (bootstrap_replicates - 1))]
    hi = differences[int(.975 * (bootstrap_replicates - 1))]
    return {
        "samples": len(ids), "conversations": len(cid),
        "a_correct": sum(x["correct"] for x in a.values()),
        "b_correct": sum(x["correct"] for x in b.values()),
        "a_wrong_b_right": len(wins), "a_right_b_wrong": len(losses),
        "delta_accuracy_pp": 100 * (len(wins) - len(losses)) / len(ids),
        "cluster_bootstrap_95pct_ci_pp": [lo, hi],
        "bootstrap_replicates": bootstrap_replicates,
        "note": "Cluster bootstrap samples conversation IDs, not independent turns. Descriptive exploratory interval; avoid causal or statistical-significance claims based only on these data.",
        "wins_preview": wins[:10], "losses_preview": losses[:10],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", default=DEFAULT_ROOT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = ap.parse_args()
    root = Path(args.input_root)
    preds = {}
    print("\nConvFinQA Oracle-history Dev results (same engine / prompt / scoring)")
    print(f"{'Variant':<17}{'N':>7}{'Accuracy':>11}{'Format':>10}{'Conv Exact':>13}")
    print("-" * 60)
    for variant, dirname in LABELS.items():
        folder = root / dirname
        if not (folder / "metrics.json").exists():
            print(f"{variant:<17} -- not run --")
            continue
        m = json.loads((folder / "metrics.json").read_text())
        print(f"{variant:<17}{m['overall']['n']:>7}{m['overall']['accuracy']*100:>10.2f}%"
              f"{m['overall']['format_accuracy']*100:>9.2f}%"
              f"{m['conversation_exact_match']['accuracy']*100:>12.2f}%")
        preds[variant] = read_predictions(folder / "predictions.jsonl")
    results = {}
    for v1, v2 in [("base", "sft"), ("base", "dynamic"), ("sft", "dynamic")]:
        if v1 not in preds or v2 not in preds:
            continue
        result = comparison(preds[v1], preds[v2], args.seed, args.bootstrap_replicates)
        results[f"{v1}_vs_{v2}"] = result
        print(f"\n{v1} -> {v2}: {result['delta_accuracy_pp']:+.2f}pp "
              f"(wins={result['a_wrong_b_right']}, losses={result['a_right_b_wrong']}) "
              f"cluster 95% CI = {result['cluster_bootstrap_95pct_ci_pp']}")
    if results:
        dest = root / "paired_summary.json"
        dest.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {dest}")


if __name__ == "__main__":
    main()
