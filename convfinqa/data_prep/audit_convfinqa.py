#!/usr/bin/env python3
"""Read-only ConvFinQA turn-level dataset audit (stdlib only)."""
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_ROOT = Path('/root/autodl-tmp/project/convfinqa_posttrain/data/raw/ConvFinQA/data_extracted/data')
DEFAULT_OUT = Path('/root/autodl-tmp/project/convfinqa_posttrain/analysis/data_audit_v1')
EXPECTED = {'train.json': 3037, 'dev.json': 421, 'train_turn.json': 11104, 'dev_turn.json': 1490, 'test_private.json': 434, 'test_turn_private.json': 1521}


def load_json(p):
    with p.open(encoding='utf-8') as f:
        d = json.load(f)
    if not isinstance(d, list):
        raise ValueError(f'{p}: expected a JSON list, got {type(d).__name__}')
    return d


def number_or_none(x):
    if isinstance(x, bool) or x is None:
        return None
    try:
        value = float(str(x).strip().replace(',', '').rstrip('%'))
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def same_ans(a, b):
    an, bn = number_or_none(a), number_or_none(b)
    if an is not None and bn is not None:
        return math.isclose(an, bn, rel_tol=1e-7, abs_tol=1e-7)
    return str(a).strip() == str(b).strip()


def summarize_split(rows, split, conv_rows, sample_limit):
    checks = Counter()
    warnings = []
    issues_by_type = Counter()
    types, ans_types = Counter(), Counter()
    turn_indices, max_turn = Counter(), -1
    seen = set()
    by_conv = defaultdict(set)
    conv_ids = {str(r.get('id')) for r in conv_rows}
    sample = None

    def flag(kind, index, row_id, detail):
        checks['issue_rows'] += 1  # occurrences, not distinct rows
        issues_by_type[kind] += 1
        if len(warnings) < sample_limit:
            warnings.append({'kind': kind, 'row_number': index, 'id': row_id, 'detail': detail})

    for i, row in enumerate(rows):
        checks['total'] += 1
        row_id = str(row.get('id', ''))
        if row_id in seen:
            flag('duplicate_id', i, row_id, 'duplicate turn ID')
        seen.add(row_id)
        m = re.match(r'^(.*)_(\d+)$', row_id)
        if not m:
            flag('unexpected_turn_id', i, row_id, 'expected <conversation_id>_<turn_index>')
            conv_id = row_id
            suffix_ind = None
        else:
            conv_id = m.group(1)
            suffix_ind = int(m.group(2))
            by_conv[conv_id].add(suffix_ind)
            if conv_id not in conv_ids:
                flag('missing_conv_id', i, row_id, f'no matching conversation in {split}.json')
        ann = row.get('annotation', {})
        if not isinstance(ann, dict):
            flag('invalid_annotation', i, row_id, 'not an object')
            continue
        idx = ann.get('turn_ind')
        cur = ann.get('cur_dial')
        all_q = ann.get('dialogue_break')
        gold_list = ann.get('exe_ans_list')
        prog_list = ann.get('turn_program')
        types[str(ann.get('cur_type', 'MISSING'))] += 1
        ans = ann.get('exe_ans')
        ans_types[type(ans).__name__] += 1
        if not isinstance(idx, int) or idx < 0:
            flag('invalid_turn_ind', i, row_id, f'turn_ind={idx!r}')
            continue
        turn_indices[idx] += 1
        max_turn = max(max_turn, idx)
        if suffix_ind is not None and suffix_ind != idx:
            flag('id_turn_mismatch', i, row_id, f'suffix {suffix_ind}, turn_ind {idx}')
        if not isinstance(cur, list) or not all(isinstance(q, str) for q in cur):
            flag('invalid_cur_dial', i, row_id, 'cur_dial should be list[str]')
        elif len(cur) != idx + 1:
            flag('cur_dial_length_mismatch', i, row_id, f'len(cur_dial)={len(cur)}, turn_ind={idx}')
        if not isinstance(all_q, list):
            flag('invalid_dialogue_break', i, row_id, 'dialogue_break should be list')
        elif idx >= len(all_q):
            flag('turn_out_of_dialogue_range', i, row_id, f'len(dialogue_break)={len(all_q)}, index={idx}')
        elif isinstance(cur, list) and cur != all_q[:idx+1]:
            flag('question_prefix_mismatch', i, row_id, 'cur_dial does not equal dialogue_break[:turn_ind+1]')
        if not isinstance(gold_list, list) or idx >= len(gold_list):
            flag('missing_current_gold', i, row_id, f'len(exe_ans_list)={len(gold_list) if isinstance(gold_list, list) else "invalid"}')
        elif not same_ans(ans, gold_list[idx]):
            flag('gold_answer_mismatch', i, row_id, f'exe_ans={ans!r}, exe_ans_list[idx]={gold_list[idx]!r}')
        if not isinstance(prog_list, list) or idx >= len(prog_list):
            flag('missing_current_program', i, row_id, f'len(turn_program)={len(prog_list) if isinstance(prog_list,list) else "invalid"}')
        elif str(ann.get('cur_program', '')).strip() != str(prog_list[idx]).strip():
            flag('program_mismatch', i, row_id, 'cur_program != turn_program[turn_ind] (may be a formatting difference)')
        if sample is None and isinstance(cur, list) and isinstance(gold_list, list) and idx >= 2:
            sample = {'id': row_id, 'turn_ind': idx, 'current_question': cur[idx],
                      'allowed_history_questions': cur[:idx], 'allowed_history_gold_answers': gold_list[:idx],
                      'current_gold_answer_LABEL_ONLY': ans, 'current_gold_program_LABEL_ONLY': ann.get('cur_program'),
                      'DO_NOT_USE_IN_PROMPT': ['exe_ans', 'cur_program', 'exe_ans_list[turn_ind:]', 'dialogue_break[turn_ind+1:]', 'turn_program[turn_ind:]']}
    missing_convs = len(conv_ids - set(by_conv))
    extra_convs = len(set(by_conv) - conv_ids)
    gaps = []
    for conv_id, ix in by_conv.items():
        if ix and ix != set(range(max(ix)+1)) and len(gaps) < sample_limit:
            gaps.append({'id': conv_id, 'observed_turns': sorted(ix)})
    return {
        'turn_rows': len(rows), 'full_conversations': len(conv_rows),
        'unique_turn_ids': len(seen), 'unique_conv_ids_in_turns': len(by_conv),
        'conversations_missing_turn_records': missing_convs,
        'turns_without_full_conversation': extra_convs,
        'turn_ind_distribution': dict(sorted(turn_indices.items())), 'max_turn_ind': max_turn,
        'cur_type_distribution': dict(types), 'exe_ans_python_type_distribution': dict(ans_types),
        'issue_counts': dict(issues_by_type), 'issue_examples': warnings,
        'conversation_turn_index_gaps_examples': gaps,
        'safe_preview': sample,
    }


def main():
    ap = argparse.ArgumentParser(description='Audit ConvFinQA splits and turn-level annotations without modifying source data.')
    ap.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    ap.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--max-examples', type=int, default=8)
    args = ap.parse_args()
    if not args.data_root.is_dir():
        ap.error(f'Data directory does not exist: {args.data_root}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {'data_root': str(args.data_root), 'files': {}, 'splits': {},
              'IMPORTANT': 'Full annotation includes future dialogue turns; NEVER place future questions, programs, or gold answers into model prompts.'}
    print('=== DATASET FILES ===')
    loaded = {}
    for name, exp in EXPECTED.items():
        path = args.data_root / name
        if not path.exists():
            report['files'][name] = {'status': 'MISSING', 'expected': exp}
            print(f'[MISSING] {name}')
            continue
        data = load_json(path)
        loaded[name] = data
        report['files'][name] = {'status': 'OK' if len(data)==exp else 'COUNT_DIFF', 'actual':len(data), 'expected':exp}
        print(f'[{report["files"][name]["status"]}] {name}: {len(data)} (expected {exp})')
    for split in ['train','dev']:
        if split+'.json' not in loaded or split+'_turn.json' not in loaded:
            continue
        print(f'\n=== {split.upper()} TURN AUDIT ===')
        stats = summarize_split(loaded[split+'_turn.json'], split, loaded[split+'.json'], args.max_examples)
        report['splits'][split]=stats
        for k in ['turn_rows','full_conversations','unique_turn_ids','unique_conv_ids_in_turns','max_turn_ind']:
            print(f'{k}: {stats[k]}')
        print('turn_ind_distribution:', stats['turn_ind_distribution'])
        print('cur_type_distribution:', stats['cur_type_distribution'])
        print('answer_types:', stats['exe_ans_python_type_distribution'])
        print('issue_counts:', stats['issue_counts'])
        if stats['issue_examples']:
            print('First few issue examples:', json.dumps(stats['issue_examples'][:4],ensure_ascii=False))
        if stats['conversation_turn_index_gaps_examples']:
            print('Turn index gaps (sample):',stats['conversation_turn_index_gaps_examples'][:2])
    out = args.output_dir / 'audit_summary.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),encoding='utf-8')
    print('\nSaved:',out)
    print('This is a read-only audit. Counts or warnings must be reviewed before creating training prompts.')

if __name__ == '__main__':
    main()
