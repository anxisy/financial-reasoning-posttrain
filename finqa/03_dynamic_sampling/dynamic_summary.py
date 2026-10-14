# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


def _read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open('r', encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _slice_events(events):
    if not events:
        return {
            'candidate_groups': 0,
            'mixed_candidate_ratio': 0.0,
            'all_correct_candidate_ratio': 0.0,
            'all_wrong_candidate_ratio': 0.0,
        }
    return {
        'candidate_groups': len(events),
        'mixed_candidate_ratio': _mean([1.0 if x['mixed_correctness'] else 0.0 for x in events]),
        'all_correct_candidate_ratio': _mean([1.0 if x['all_correct'] else 0.0 for x in events]),
        'all_wrong_candidate_ratio': _mean([1.0 if x['all_wrong'] else 0.0 for x in events]),
    }


def summarize_dynamic_sampling(output_dir, num_generations=4, vanilla_group_budget=6251):
    out = Path(output_dir)
    events = _read_jsonl(out / 'dynamic_sampling_events.jsonl')
    group_rows = _read_jsonl(out / 'group_diagnostics.jsonl')
    reward_call_to_group = {int(x['reward_call']): x for x in group_rows}

    accepted_events = [x for x in events if x['accepted_for_update']]
    filtered_events = [x for x in events if not x['accepted_for_update']]
    mixed_accepted = [x for x in accepted_events if x['mixed_correctness']]
    fallback = [x for x in accepted_events if x['fallback_zero_variance']]

    accepted_group_rows = []
    for event in accepted_events:
        rc = int(event['reward_call'])
        if rc in reward_call_to_group:
            accepted_group_rows.append(reward_call_to_group[rc])

    attempt_hist = Counter(int(x['attempt']) for x in accepted_events)

    by_diff = defaultdict(list)
    by_op = defaultdict(list)
    for event in events:
        by_diff[str(event.get('difficulty', 'unknown'))].append(event)
        raw = event.get('operators_json', '[]')
        try:
            ops = json.loads(raw) if isinstance(raw, str) else list(raw or [])
        except Exception:
            ops = []
        for op in sorted(set(map(str, ops))):
            by_op[op].append(event)

    total_candidates = len(events)
    optimizer_steps = len(accepted_events)
    candidate_rollouts = total_candidates * int(num_generations)
    vanilla_rollouts = int(vanilla_group_budget) * int(num_generations)

    summary = {
        'optimizer_steps': optimizer_steps,
        'total_candidate_groups': total_candidates,
        'total_candidate_rollouts': candidate_rollouts,
        'filtered_zero_variance_groups': len(filtered_events),
        'accepted_mixed_groups': len(mixed_accepted),
        'fallback_zero_variance_groups': len(fallback),
        'effective_update_ratio': len(mixed_accepted) / optimizer_steps if optimizer_steps else 0.0,
        'fallback_ratio': len(fallback) / optimizer_steps if optimizer_steps else 0.0,
        'candidate_mixed_ratio': sum(x['mixed_correctness'] for x in events) / total_candidates if total_candidates else 0.0,
        'candidate_zero_variance_ratio': sum(not x['mixed_correctness'] for x in events) / total_candidates if total_candidates else 0.0,
        'filtered_group_ratio_over_candidates': len(filtered_events) / total_candidates if total_candidates else 0.0,
        'sampling_group_multiplier': total_candidates / optimizer_steps if optimizer_steps else 0.0,
        'candidate_rollout_multiplier_vs_vanilla_6251xG': candidate_rollouts / vanilla_rollouts if vanilla_rollouts else 0.0,
        'accepted_on_attempt_histogram': {str(k): v for k, v in sorted(attempt_hist.items())},
        'accepted_group_reward_variance_mean': _mean([float(x['reward_variance']) for x in accepted_group_rows]),
        'accepted_group_correctness_mean': _mean([float(x['correctness_mean']) for x in accepted_group_rows]),
        'accepted_unique_completion_ratio_mean': _mean([float(x['unique_completion_ratio']) for x in accepted_group_rows]),
        'by_candidate_difficulty': {k: _slice_events(v) for k, v in sorted(by_diff.items())},
        'by_candidate_operator': {k: _slice_events(v) for k, v in sorted(by_op.items())},
    }

    (out / 'dynamic_sampling_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return summary
