#!/usr/bin/env python3
"""Compare processed_v1 and processed_v2; fail on altered prompts or other unintended changes."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

DEFAULT_ROOT = Path('/root/autodl-tmp/project/convfinqa_posttrain/data')


def rows(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    old, new = args.data_root / 'processed_v1', args.data_root / 'processed_v2'
    for split, expected_counts in [('train', (11104, 3911, 7193)),
                                   ('dev', (1490, 487, 1003))]:
        prev = rows(old / 'sft' / f'{split}.jsonl')
        curr = rows(new / 'sft' / f'{split}.jsonl')
        assert len(prev) == len(curr) == expected_counts[0], (split, len(prev), len(curr))
        counters = Counter()
        for a, b in zip(prev, curr):
            rid = a['id']
            assert rid == b['id']
            assert a['messages'][:2] == b['messages'][:2], (rid, 'Prompt changed!')
            assert a['metadata'] == b['metadata'], (rid, 'Metadata changed!')
            old_target = a['messages'][-1]['content']
            new_target = b['messages'][-1]['content']
            typ = b['metadata']['cur_type']
            if typ == 'number_turn':
                match = re.search(r'<answer>(.*?)</answer>\s*$', old_target, re.S)
                assert match, (rid, 'Missing old final answer')
                assert new_target == f'<answer>{match.group(1)}</answer>', (rid, 'Wrong direct target')
                assert '\n' not in new_target and 'Step' not in new_target
                counters['number_turn'] += 1
            elif typ == 'program_turn':
                assert old_target == new_target, (rid, 'Program target changed!')
                counters['program_turn'] += 1
            else:
                raise ValueError(f'{rid} unknown type {typ}')
        assert (counters['number_turn'], counters['program_turn']) == expected_counts[1:], (split, counters)
        print(f'[PASS] {split}: {len(prev)} rows; direct={counters["number_turn"]}; program={counters["program_turn"]}; prompts unchanged')

    for rel in [Path('grpo/train.jsonl'), Path('eval/dev.jsonl')]:
        old_bytes, new_bytes = (old / rel).read_bytes(), (new / rel).read_bytes()
        assert old_bytes == new_bytes, f'{rel} changed! Existing oracle baselines may not be comparable.'
        print(f'[PASS] {rel}: byte-for-byte unchanged')
    print('ALL CHECKS PASSED. processed_v1 and Baseline outputs remain valid.')


if __name__ == '__main__':
    main()
