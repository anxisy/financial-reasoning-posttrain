#!/usr/bin/env python3
"""Paired ConvFinQA Oracle-history Dev evaluation: direct SFT vs transfer SFT.

Stdlib only. Reads FROZEN predictions, does not load models, and never modifies them.
Resamples full conversation IDs for cluster bootstrap to account for correlated turns.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/root/autodl-tmp/project/convfinqa_posttrain/experiments/convfinqa_sft_v2')
DIRECT = ROOT / '02_qwen3_base/eval_dev'
TRANSFER = ROOT / '01_finqa_sft_adapter/eval_dev'
DEFAULT_OUT = ROOT / 'paired_transfer_vs_direct_v1'
REQUIRED_FIELDS = ('id', 'correct', 'gold_answer', 'metadata')
CONSISTENCY_ARGS = (
    'input_sha256', 'max_new_tokens', 'max_model_len', 'seed',
    'prompt_token_ids_sha256', 'history', 'same_frozen_scoring_as_baseline'
)


def load_dataset(folder: Path, name: str):
    path = folder / 'predictions.jsonl'
    if not path.is_file():
        raise FileNotFoundError(f'Missing {name} predictions: {path}')
    rows = {}
    with path.open(encoding='utf-8') as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            missing = [key for key in REQUIRED_FIELDS if key not in entry]
            if missing:
                raise ValueError(f'{name} line {lineno}: missing fields {missing}')
            rid = str(entry['id'])
            if not rid or rid in rows:
                raise ValueError(f'{name} line {lineno}: missing or repeated id {rid}')
            if type(entry['correct']) is not bool:
                raise ValueError(f'{name} {rid}: correct must be a JSON boolean')
            md = entry['metadata']
            if not isinstance(md, dict) or not md.get('conversation_id'):
                raise ValueError(f'{name} {rid}: missing conversation_id')
            if str(md.get('cur_type')) not in ('number_turn', 'program_turn'):
                raise ValueError(f'{name} {rid}: unexpected cur_type={md.get("cur_type")}')
            if type(md.get('turn_ind')) is not int or md['turn_ind'] < 0:
                raise ValueError(f'{name} {rid}: invalid turn_ind')
            rows[rid] = entry
    if not rows:
        raise ValueError(f'No rows in {path}')

    cfg_path = folder / 'run_config.json'
    if not cfg_path.is_file():
        raise FileNotFoundError(f'Missing {name} configuration: {cfg_path}')
    config = json.loads(cfg_path.read_text(encoding='utf-8'))
    if not (folder / 'metrics.json').is_file():
        raise FileNotFoundError(f'Missing {name} metrics.json')
    metrics = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
    if len(rows) != metrics['overall']['n'] or sum(r['correct'] for r in rows.values()) != metrics['overall']['correct']:
        raise ValueError(f'{name} prediction rows disagree with metrics.json')
    return rows, config, metrics


def check_pair(direct, transfer, dc, tc, expected: int):
    if set(direct) != set(transfer):
        print('Missing in transfer:', list(sorted(set(direct)-set(transfer)))[:5])
        print('Missing in direct:', list(sorted(set(transfer)-set(direct)))[:5])
        raise ValueError('Different question IDs; pairing is invalid')
    if expected and len(direct) != expected:
        raise ValueError(f'Expected {expected} examples; received {len(direct)}')

    for key in CONSISTENCY_ARGS:
        if key not in dc or key not in tc or dc[key] != tc[key]:
            raise ValueError(f'Run configuration mismatch: {key}: {dc.get(key)!r} != {tc.get(key)!r}')
    if dc['history'] != 'oracle_gold_answers' or dc['same_frozen_scoring_as_baseline'] is not True:
        raise ValueError('These are not frozen Oracle-history scoring configurations')
    if dc.get('variant') != 'direct' or tc.get('variant') != 'transfer':
        raise ValueError('Expected run_config variants direct and transfer')

    for rid in direct:
        a, b = direct[rid], transfer[rid]
        for k in ('conversation_id', 'cur_type', 'turn_ind', 'operators'):
            if a['metadata'].get(k) != b['metadata'].get(k):
                raise ValueError(f'{rid}: metadata {k} differs between runs')
        if json.dumps(a['gold_answer'], sort_keys=True) != json.dumps(b['gold_answer'], sort_keys=True):
            raise ValueError(f'{rid}: gold labels disagree')


def build_pairs(direct, transfer):
    pairs = []
    for rid in sorted(direct):
        a, b = direct[rid], transfer[rid]
        pairs.append({
            'id': rid, 'conv': str(a['metadata']['conversation_id']),
            'turn': int(a['metadata']['turn_ind']),
            'type': str(a['metadata']['cur_type']),
            'operators': list(a['metadata'].get('operators') or []),
            'gold': a['gold_answer'],
            'd_answer': a.get('prediction_answer'),
            't_answer': b.get('prediction_answer'),
            'd_correct': a['correct'], 't_correct': b['correct'],
            'd_prediction': a.get('prediction', ''),
            't_prediction': b.get('prediction', ''),
        })
    return pairs


def percentile(sorted_values, q):
    x = (len(sorted_values)-1)*q
    low = int(x)
    high = min(len(sorted_values)-1, low+1)
    t = x-low
    return sorted_values[low]*(1-t) + sorted_values[high]*t


def cluster_ci(rows, all_cids, bootstrap, seed, conv_exact=False):
    # Conversation IDs (not individual turns) are the resampling units.
    if conv_exact:
        grouped = {c: (1, int(all(r['d_correct'] for r in v)), int(all(r['t_correct'] for r in v)))
                   for c, v in rows.items()}
    else:
        grouped = {c: (len(v), sum(r['d_correct'] for r in v), sum(r['t_correct'] for r in v))
                   for c, v in rows.items()}
    pool = [grouped.get(c, (0, 0, 0)) for c in all_cids]
    if sum(x[0] for x in pool) == 0:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(bootstrap):
        n = 0
        diff = 0
        for _ in all_cids:
            count, d, t = pool[rng.randrange(len(pool))]
            n += count
            diff += t-d
        if n:
            values.append(100 * diff / n)
    if len(values) < .99*bootstrap:
        return None
    values.sort()
    return [percentile(values, 0.025), percentile(values, 0.975)]


def summarize(group, all_cids, bootstrap, seed, conv_exact=False):
    if conv_exact:
        by_conv = defaultdict(list)
        for row in group:
            by_conv[row['conv']].append(row)
        records = [(all(r['d_correct'] for r in rs), all(r['t_correct'] for r in rs)) for rs in by_conv.values()]
        n = len(records)
        # Confidence interval for a difference in conversation-level all-turn correctness.
        ci = cluster_ci(by_conv, all_cids, bootstrap, seed, conv_exact=True)
        convs = n
    else:
        records = [(r['d_correct'], r['t_correct']) for r in group]
        n = len(records)
        by_conv = defaultdict(list)
        for row in group:
            by_conv[row['conv']].append(row)
        convs = len(by_conv)
        # Avoid presenting unstable CIs on very small late-turn subgroups.
        ci = cluster_ci(by_conv, all_cids, bootstrap, seed) if n >= 30 and convs >= 20 else None

    dcorrect = sum(d for d, t in records)
    tcorrect = sum(t for d, t in records)
    wins = sum(not d and t for d, t in records)
    losses = sum(d and not t for d, t in records)
    return {
        'n': n, 'conversations_represented': convs,
        'direct_correct': dcorrect, 'transfer_correct': tcorrect,
        'direct_accuracy_pct': 100*dcorrect/n if n else None,
        'transfer_accuracy_pct': 100*tcorrect/n if n else None,
        'delta_transfer_minus_direct_pp': 100*(tcorrect-dcorrect)/n if n else None,
        'direct_wrong_transfer_right': wins,
        'direct_right_transfer_wrong': losses,
        'both_correct': sum(d and t for d, t in records),
        'both_wrong': sum(not d and not t for d, t in records),
        'cluster_bootstrap_95pct_ci_pp': ci,
    }


def print_result(name, x):
    d, t = x['direct_accuracy_pct'], x['transfer_accuracy_pct']
    delta = x['delta_transfer_minus_direct_pp']
    ci = x['cluster_bootstrap_95pct_ci_pp']
    ci_text = f'[{ci[0]:+.2f}, {ci[1]:+.2f}] pp' if ci is not None else 'N/A (small subgroup)'
    print(f'{name:<20} n={x["n"]:4d}  direct={d:6.2f}%  transfer={t:6.2f}%  '
          f'delta={delta:+6.2f}pp  wins/losses={x["direct_wrong_transfer_right"]}/{x["direct_right_transfer_wrong"]}  CI={ci_text}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--direct', type=Path, default=DIRECT, help='Base→ConvFinQA SFT eval_dev folder')
    ap.add_argument('--transfer', type=Path, default=TRANSFER, help='FinQA SFT→ConvFinQA SFT eval_dev folder')
    ap.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--bootstrap', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--expected-samples', type=int, default=1490)
    args = ap.parse_args()
    if args.bootstrap < 100:
        ap.error('--bootstrap should be at least 100')
    if args.expected_samples < 0:
        ap.error('--expected-samples must be nonnegative')

    direct, dc, dm = load_dataset(args.direct, 'direct')
    transfer, tc, tm = load_dataset(args.transfer, 'transfer')
    check_pair(direct, transfer, dc, tc, args.expected_samples)
    pairs = build_pairs(direct, transfer)
    all_cids = sorted({r['conv'] for r in pairs})
    print(f'PASS: {len(pairs)} identical example IDs, {len(all_cids)} conversations; '
          f'prompt fingerprint {dc["prompt_token_ids_sha256"]}')
    if args.expected_samples == 1490 and len(all_cids) != 421:
        raise ValueError(f'Expected 421 conversations, found {len(all_cids)}')
    if dm['overall']['n'] != tm['overall']['n']:
        raise ValueError('Unequal evaluation sample counts')

    results = {
        'comparison': 'Base→ConvFinQA SFT (direct) vs FinQA SFT→ConvFinQA SFT (transfer)',
        'delta_direction': 'transfer_minus_direct',
        'setting': 'ConvFinQA Dev, oracle_gold_answers, frozen scoring',
        'n': len(pairs), 'conversations': len(all_cids),
        'input_fingerprint': dc['prompt_token_ids_sha256'],
        'bootstrap_replicates': args.bootstrap, 'bootstrap_seed': args.seed,
        'overall': summarize(pairs, all_cids, args.bootstrap, args.seed),
        'conversation_exact_match': summarize(pairs, all_cids, args.bootstrap, args.seed+1, conv_exact=True),
        'by_type': {}, 'by_turn': {}, 'by_operator': {},
        'limitations': [
            'Cluster bootstrap resamples conversations; turns within a conversation are dependent.',
            'Exploratory percentile CIs are not a substitute for independent runs across training seeds.',
            'Oracle-history performance is not free-running end-to-end dialogue success.',
            'ConvFinQA Dev was used in model development; do not call this blind test generalization.',
            'Operator subsets overlap; their counts must not be summed.',
        ],
    }
    for typ in ('number_turn', 'program_turn'):
        seq = [r for r in pairs if r['type'] == typ]
        results['by_type'][typ] = summarize(seq, all_cids, args.bootstrap, args.seed+100)
    for turn in sorted({r['turn'] for r in pairs}):
        seq = [r for r in pairs if r['turn'] == turn]
        results['by_turn'][str(turn)] = summarize(seq, all_cids, args.bootstrap, args.seed+turn+200)
    for op in sorted({o for r in pairs for o in r['operators']}):
        seq = [r for r in pairs if op in r['operators']]
        results['by_operator'][op] = summarize(seq, all_cids, args.bootstrap, args.seed+300)

    print('\n=== PAIRED ORACLE DEV RESULTS: TRANSFER - DIRECT ===')
    print_result('Overall', results['overall'])
    print_result('Conv all-turn', results['conversation_exact_match'])
    for k, x in results['by_type'].items():
        print_result(k, x)
    print('\n=== BY TURN ===')
    for k, x in results['by_turn'].items():
        print_result('Turn ' + k, x)
    print('\n=== BY OPERATOR (overlapping subsets) ===')
    for k, x in results['by_operator'].items():
        print_result(k, x)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outfile = args.output_dir/'paired_summary.json'
    outfile.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    detail = args.output_dir/'discordant_cases.csv'
    discordant = [r for r in pairs if r['d_correct'] != r['t_correct']]
    with detail.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id','conversation_id','turn_ind','cur_type','winner','gold_answer',
                    'direct_answer','transfer_answer','direct_output_preview','transfer_output_preview'])
        for r in discordant:
            winner = 'transfer' if r['t_correct'] else 'direct'
            w.writerow([r['id'], r['conv'], r['turn'], r['type'], winner, r['gold'],
                        r['d_answer'],r['t_answer'],
                        str(r['d_prediction'])[:350].replace('\n','\\n'),
                        str(r['t_prediction'])[:350].replace('\n','\\n')])
    print(f'\nSaved summary: {outfile}')
    print(f'Saved {len(discordant)} discordant examples: {detail}')
    ci = results['overall']['cluster_bootstrap_95pct_ci_pp']
    if ci is not None:
        if ci[0] <= 0 <= ci[1]:
            print('INTERPRETATION: 95% cluster CI includes 0. The observed improvement is not robustly distinguishable from zero here.')
        else:
            print('INTERPRETATION: 95% cluster CI excludes 0 in this exploratory analysis; validate across seeds and independent evaluation.')


if __name__ == '__main__':
    main()
