#!/usr/bin/env python3
"""Read-only ConvFinQA 3-model baseline diagnostic. Stdlib only.

Never alters model/data outputs and never relabels correct/incorrect responses.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

FOLDERS = {'base': 'qwen3_base', 'sft': 'finqa_sft', 'dynamic': 'finqa_dynamic'}


def read_rows(path):
    with path.open(encoding='utf-8') as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    ids = [str(x['id']) for x in records]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate sample ids: ' + str(path))
    return {str(x['id']): x for x in records}


def aggregate(seq):
    size = len(seq)
    if not size:
        return {'n': 0}
    c = lambda key: sum(bool(x.get(key)) for x in seq)
    correct = c('correct')
    fmt = c('format_ok')
    strict = sum(bool(x.get('correct') and x.get('format_ok')) for x in seq)
    fallback = sum(bool(x.get('correct') and not x.get('format_ok')) for x in seq)
    return {
        'n': size, 'correct': correct, 'accuracy_pct': 100 * correct / size,
        'format_ok': fmt, 'format_pct': 100 * fmt / size,
        'strict_correct': strict, 'strict_accuracy_pct': 100 * strict / size,
        'correct_without_answer_tag': fallback,
        'extraction_method': dict(Counter(str(x.get('extraction_method', 'unknown')) for x in seq)),
        'wrong_match_reason': dict(Counter(str(x.get('match_reason', 'unknown')) for x in seq if not x.get('correct'))),
    }


def describe(r):
    return {
        'correct': bool(r['correct']),
        'format_ok': bool(r['format_ok']),
        'extracted': r.get('prediction_answer'),
        'reason': r.get('match_reason'),
        'method': r.get('extraction_method'),
        'text_preview': str(r.get('prediction', ''))[:700],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-root', required=True, help='Baseline full_dev output directory')
    ap.add_argument('--examples', type=int, default=8)
    args = ap.parse_args()
    root = Path(args.input_root)
    raw = {}
    fingerprint = {}
    for variant, folder in FOLDERS.items():
        raw[variant] = read_rows(root / folder / 'predictions.jsonl')
        m = json.loads((root / folder / 'metrics.json').read_text(encoding='utf-8'))
        fingerprint[variant] = m.get('input_fingerprint')
    ref_ids = set(raw['base'])
    for variant in FOLDERS:
        if set(raw[variant]) != ref_ids:
            raise ValueError(f'IDs mismatch for {variant}')
    if len(set(fingerprint.values())) != 1:
        raise ValueError('Prompt fingerprint differs across models: ' + str(fingerprint))
    if len(ref_ids) != 1490:
        print(f'WARNING: {len(ref_ids)} rows, not full Dev 1490')

    print('=== INPUT CONSISTENCY ===')
    print('n=', len(ref_ids), 'identical_prompt_fingerprint=', next(iter(fingerprint.values())))
    summary = {'n': len(ref_ids), 'prompt_fingerprint': next(iter(fingerprint.values())),
               'by_model': {}, 'base_wins_on_number': [], 'sft_wins_on_program': [],
               'dynamic_wins_on_program': []}
    for variant, rows in raw.items():
        group = {'all': aggregate(list(rows.values()))}
        for typ in ['number_turn', 'program_turn']:
            group[typ] = aggregate([r for r in rows.values() if r.get('metadata', {}).get('cur_type') == typ])
        summary['by_model'][variant] = group
        print('\n===', variant.upper(), '===')
        for typ in ['all', 'number_turn', 'program_turn']:
            d = group[typ]
            print(f"{typ:13s} n={d['n']:4d} relaxed={d['accuracy_pct']:6.2f}% "
                  f"strict={d['strict_accuracy_pct']:6.2f}% "
                  f"format={d['format_pct']:6.2f}% "
                  f"correct_no_tag={d['correct_without_answer_tag']}")
            print('  wrong reasons:', Counter(d['wrong_match_reason']).most_common(4))
            print('  extraction:', Counter(d['extraction_method']).most_common(4))

    cases = [
        ('base_wins_on_number', 'base', 'sft', 'number_turn'),
        ('sft_wins_on_program', 'sft', 'base', 'program_turn'),
        ('dynamic_wins_on_program', 'dynamic', 'base', 'program_turn'),
    ]
    for title, winner, loser, kind in cases:
        examples = []
        for rid in sorted(ref_ids):
            rw, rl = raw[winner][rid], raw[loser][rid]
            if (rw.get('metadata', {}).get('cur_type') == kind and
                    rw.get('correct') and not rl.get('correct')):
                examples.append({
                    'id': rid, 'turn': rw['metadata'].get('turn_ind'),
                    'type': kind, 'gold': rw['gold_answer'],
                    winner: describe(rw), loser: describe(rl),
                })
            if len(examples) >= args.examples:
                break
        summary[title] = examples
        print('\n===', title, '===')
        for x in examples:
            w, l = x[winner], x[loser]
            print(f"ID={x['id']} turn={x['turn']} gold={x['gold']!r}")
            print(f"  {winner:7s}: extracted={w['extracted']!r} format={w['format_ok']} ",
                  f"reason={w['reason']} text={w['text_preview'][:240]!r}")
            print(f"  {loser:7s}: extracted={l['extracted']!r} format={l['format_ok']} ",
                  f"reason={l['reason']} text={l['text_preview'][:240]!r}")

    output = root / 'diagnosis_v1.json'
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nSaved:', output)
    print('READ ONLY: no predictions or evaluations were modified')

if __name__ == '__main__':
    main()
