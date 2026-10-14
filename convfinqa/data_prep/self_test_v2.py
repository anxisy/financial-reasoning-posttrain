#!/usr/bin/env python3
"""Self-contained format and anti-future-leakage tests (stdlib only)."""
from prepare_convfinqa_v2 import convert_row, program_steps, manual_prompt

def fake(idx):
    questions = ['2009 value?', '2008 value?', 'Difference?', 'Percentage change?']
    answers = [206588.0, 181001.0, 25587.0, 0.14136]
    programs = ['206588', '181001', 'subtract(206588, 181001)',
                'subtract(206588, 181001), divide(#0, 181001)']
    return {
        'id': f'Single_JKHY/2009/page_28.pdf-3_{idx}',
        'filename': 'Single_JKHY/2009/page_28.pdf',
        'pre_text': ['Financial statement example'],
        'post_text': ['Notes'],
        'table': [['year', '2009', '2008'], ['cash flow', '206588', '181001']],
        'qa': {'leak': 'DO_NOT_INCLUDE_QA'},
        'annotation': {
            'turn_ind': idx, 'cur_dial': questions[:idx+1],
            'dialogue_break': questions, 'exe_ans': answers[idx],
            'exe_ans_list': answers, 'turn_program': programs,
            'cur_program': programs[idx], 'cur_type': 'program_turn' if idx >= 2 else 'number_turn',
        },
    }

def main():
    conv_id = 'Single_JKHY/2009/page_28.pdf-3'
    assert program_steps('subtract(206588, 181001), divide(#0, 181001)') == [
        'subtract(206588, 181001)', 'divide(#0, 181001)']
    for i in range(4):
        sft, grpo, ev, user, target = convert_row(fake(i), {conv_id}, 'train')
        assert sft['messages'][2]['role'] == 'assistant'
        assert len(grpo['prompt']) == 2
        assert '<answer>' not in user if i == 0 else True
        assert 'DO_NOT_INCLUDE_QA' not in user
        assert 'Percentage change?' not in user if i < 3 else True
        assert f'Current question:\n{fake(i)["annotation"]["cur_dial"][-1]}' in user
        assert target.endswith(f'<answer>{ev["gold_answer"]}</answer>')
        if i < 2:
            assert target == f'<answer>{ev["gold_answer"]}</answer>'
            assert 'Step' not in target
        else:
            assert target.startswith('Step 1: ')
        assert manual_prompt(sft['messages'][0]['content'], user).endswith('Assistant:')
        assert '\nStep 2: divide(#0, 181001)' in '\n'+target if i == 3 else True
        assert sft['metadata']['history_turns'] == i
    print('PASS v2: type-aware direct answers, program steps, no future leakage, manual prompt')

if __name__ == '__main__':
    main()
