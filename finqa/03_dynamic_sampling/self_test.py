# -*- coding: utf-8 -*-
import json
import tempfile
from pathlib import Path

from grpo_data import manual_prompt
from reward_utils import GRPORewardWithDiagnostics
from dynamic_summary import summarize_dynamic_sampling


class Tok:
    def __call__(self, text, add_special_tokens=False, truncation=False):
        return {'input_ids': list(range(len(text.split())))}


def main():
    prompt = manual_prompt([
        {'role': 'system', 'content': 'SYS'},
        {'role': 'user', 'content': 'Q'},
    ])
    assert prompt == 'System:\nSYS\n\nUser:\nQ\n\nAssistant:'
    assert prompt.endswith('Assistant:')
    assert not prompt.endswith('Assistant:\n')

    with tempfile.TemporaryDirectory() as d:
        reward = GRPORewardWithDiagnostics(
            Tok(), d, num_generations=4, max_completion_length=256
        )

        r1 = reward(
            completions=[
                '<answer>10</answer>', '<answer>10</answer>',
                '<answer>10</answer>', '<answer>10</answer>',
            ],
            gold_answer=['10'], example_id=['a'], difficulty=['easy'],
            operators_json=['["divide"]'],
        )
        assert reward.last_group_rows[0]['mixed_correctness'] is False
        assert reward.last_group_rows[0]['all_correct'] is True

        r2 = reward(
            completions=[
                '<answer>10</answer>', '<answer>9</answer>',
                '<answer>10</answer>', '<answer>8</answer>',
            ],
            gold_answer=['10'], example_id=['b'], difficulty=['medium'],
            operators_json=['["divide"]'],
        )
        assert reward.last_group_rows[0]['mixed_correctness'] is True

        events = [
            {
                'optimizer_step': 1, 'attempt': 1, 'max_resample_attempts': 4,
                'source': 'dataloader', 'original_example_id': 'a',
                'candidate_example_id': 'a', 'reward_call': 1,
                'difficulty': 'easy', 'operators_json': '["divide"]',
                'correctness': [1,1,1,1], 'all_correct': True,
                'all_wrong': False, 'mixed_correctness': False,
                'accepted_for_update': False, 'fallback_zero_variance': False,
            },
            {
                'optimizer_step': 1, 'attempt': 2, 'max_resample_attempts': 4,
                'source': 'replacement', 'original_example_id': 'a',
                'candidate_example_id': 'b', 'reward_call': 2,
                'difficulty': 'medium', 'operators_json': '["divide"]',
                'correctness': [1,0,1,0], 'all_correct': False,
                'all_wrong': False, 'mixed_correctness': True,
                'accepted_for_update': True, 'fallback_zero_variance': False,
            },
        ]
        p = Path(d) / 'dynamic_sampling_events.jsonl'
        with p.open('w', encoding='utf-8') as f:
            for row in events:
                f.write(json.dumps(row) + '\n')

        summary = summarize_dynamic_sampling(d, num_generations=4, vanilla_group_budget=1)
        assert summary['optimizer_steps'] == 1
        assert summary['total_candidate_groups'] == 2
        assert summary['accepted_mixed_groups'] == 1
        assert summary['effective_update_ratio'] == 1.0
        assert summary['sampling_group_multiplier'] == 2.0

    print('Manual prompt:', repr(prompt))
    print('Zero-var rewards:', r1)
    print('Mixed rewards:', r2)
    print('Dynamic summary self-test: PASSED')
    print('SELF TEST PASSED')


if __name__ == '__main__':
    main()
