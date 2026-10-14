#!/usr/bin/env python3
"""Paired per-question comparison for frozen FinQA Dev/Test evaluation JSONL.

Dependencies: Python standard library only.
The two inputs must be predictions.jsonl from the SAME split with identical IDs.
"""
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

DEFAULT_ROOT = Path('/root/autodl-tmp/project/finqa_posttrain/experiments')


def load_predictions(path):
    rows = {}
    with Path(path).open(encoding='utf-8') as fh:
        for line_num, line in enumerate(fh, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            key = str(item.get('id'))
            if key in ('', 'None'):
                raise ValueError(f'{path}:{line_num}: missing id')
            if key in rows:
                raise ValueError(f'{path}:{line_num}: duplicated id {key!r}; check eval resume')
            rows[key] = item
    if not rows:
        raise ValueError(f'No rows in {path}')
    return rows


def quantile(data, q):
    a = sorted(data)
    pos = (len(a) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return a[lo] + (a[hi] - a[lo]) * (pos - lo)


def exact_mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    small = min(b, c)
    # Exact two-sided sign test for discordant paired predictions.
    numerator = sum(math.comb(n, i) for i in range(small + 1))
    return min(1.0, 2.0 * (numerator / (2 ** n)))


def bool_correct(row):
    if 'correct' not in row:
        raise ValueError('Missing "correct" key in prediction row')
    return bool(row['correct'])


def metrics_from_pairs(pairs):
    total = len(pairs)
    vanilla_correct = sum(bool_correct(v) for v, d in pairs)
    dynamic_correct = sum(bool_correct(d) for v, d in pairs)
    won = sum(not bool_correct(v) and bool_correct(d) for v, d in pairs)
    lost = sum(bool_correct(v) and not bool_correct(d) for v, d in pairs)
    return {
        'n': total,
        'vanilla_correct': vanilla_correct,
        'dynamic_correct': dynamic_correct,
        'vanilla_accuracy': vanilla_correct / total if total else None,
        'dynamic_accuracy': dynamic_correct / total if total else None,
        'delta_percentage_points': (dynamic_correct - vanilla_correct) * 100 / total if total else None,
        'vanilla_wrong_dynamic_right': won,
        'vanilla_right_dynamic_wrong': lost,
        'both_correct': sum(bool_correct(v) and bool_correct(d) for v, d in pairs),
        'both_wrong': sum(not bool_correct(v) and not bool_correct(d) for v, d in pairs),
    }


def compare(args):
    vanilla = load_predictions(args.vanilla)
    dynamic = load_predictions(args.dynamic)
    if set(vanilla) != set(dynamic):
        a_only = sorted(set(vanilla) - set(dynamic))
        b_only = sorted(set(dynamic) - set(vanilla))
        raise ValueError(f'Sample IDs differ. Vanilla-only count={len(a_only)} first={a_only[:5]}; DS-only count={len(b_only)} first={b_only[:5]}')
    ids = sorted(vanilla)
    pairs = []
    for sid in ids:
        va = vanilla[sid]
        ds = dynamic[sid]
        if str(va.get('gold_answer')) != str(ds.get('gold_answer')):
            raise ValueError(f'Gold answer mismatch for ID: {sid}')
        pairs.append((va, ds))

    basic = metrics_from_pairs(pairs)
    if args.expected_n and basic['n'] != args.expected_n:
        raise ValueError(f'Expected {args.expected_n} examples, found {basic["n"]}')

    b = basic['vanilla_wrong_dynamic_right']
    c = basic['vanilla_right_dynamic_wrong']
    basic['exact_mcnemar_p_two_sided'] = exact_mcnemar_p(b, c)

    # Paired bootstrap CI: resample entire sample pairs, preserving correlation.
    rng = random.Random(args.seed)
    indicators = [int(bool_correct(d)) - int(bool_correct(v)) for v, d in pairs]
    n = len(indicators)
    bootstrap_delta = [
        100 * sum(indicators[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(args.bootstrap)
    ]
    basic['paired_bootstrap_95pct_CI_pp'] = [quantile(bootstrap_delta, 0.025), quantile(bootstrap_delta, 0.975)]
    basic['bootstrap_replicates'] = args.bootstrap
    basic['bootstrap_seed'] = args.seed

    difficulty = defaultdict(list)
    operator = defaultdict(list)
    for v, d in pairs:
        meta = v.get('metadata') or {}
        difficulty[str(meta.get('difficulty', 'unknown'))].append((v, d))
        for op in set(str(x) for x in (meta.get('operators') or [])):
            operator[op].append((v, d))

    report = {
        'source': {'vanilla': str(args.vanilla), 'dynamic': str(args.dynamic)},
        'overall': basic,
        'by_difficulty': {key: metrics_from_pairs(value) for key, value in sorted(difficulty.items())},
        'by_operator': {key: metrics_from_pairs(value) for key, value in sorted(operator.items())},
        'caveat': 'Operator subgroup counts overlap for multi-operator items. Test should not be used for hyperparameter selection.',
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / 'paired_analysis.json'
    out_csv = out_dir / 'changed_predictions.csv'
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    with out_csv.open('w', newline='', encoding='utf-8-sig') as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            'id', 'change', 'difficulty', 'operators', 'gold_answer',
            'vanilla_answer', 'dynamic_answer', 'vanilla_prediction', 'dynamic_prediction'
        ])
        writer.writeheader()
        for v, d in pairs:
            vc, dc = bool_correct(v), bool_correct(d)
            if vc == dc:
                continue
            meta = v.get('metadata') or {}
            writer.writerow({
                'id': v['id'],
                'change': 'improved' if dc else 'regressed',
                'difficulty': meta.get('difficulty', ''),
                'operators': ','.join(map(str, meta.get('operators', []) or [])),
                'gold_answer': v.get('gold_answer', ''),
                'vanilla_answer': v.get('prediction_answer', ''),
                'dynamic_answer': d.get('prediction_answer', ''),
                'vanilla_prediction': v.get('prediction', ''),
                'dynamic_prediction': d.get('prediction', ''),
            })

    print('\n=== PAIRED FINQA COMPARISON ===')
    for k, value in basic.items():
        print(f'{k:39s}: {value}')
    print('\nBy difficulty:')
    for key, val in report['by_difficulty'].items():
        print(f'{key:12s} n={val["n"]:4d} delta={val["delta_percentage_points"]:+.2f}pp  wins={val["vanilla_wrong_dynamic_right"]} losses={val["vanilla_right_dynamic_wrong"]}')
    print('\nBy operator (categories overlap):')
    for key, val in report['by_operator'].items():
        print(f'{key:12s} n={val["n"]:4d} delta={val["delta_percentage_points"]:+.2f}pp  wins={val["vanilla_wrong_dynamic_right"]} losses={val["vanilla_right_dynamic_wrong"]}')
    print(f'\nSaved: {out_json}\nSaved: {out_csv}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vanilla', default=str(DEFAULT_ROOT / 'grpo_qwen3_1.7b/eval_dev_v1/predictions.jsonl'))
    p.add_argument('--dynamic', default=str(DEFAULT_ROOT / 'grpo_dynamic_qwen3_1.7b/eval_dev_v1/predictions.jsonl'))
    p.add_argument('--output-dir', default=str(DEFAULT_ROOT / 'analysis_1.7b/paired_dev_v1'))
    p.add_argument('--expected-n', type=int, default=883, help='set to 1147 for test')
    p.add_argument('--bootstrap', type=int, default=5000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.bootstrap < 1:
        p.error('--bootstrap must be >= 1')
    compare(args)


if __name__ == '__main__':
    main()
