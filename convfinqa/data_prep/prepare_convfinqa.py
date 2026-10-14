#!/usr/bin/env python3
"""Create leakage-aware oracle-history ConvFinQA SFT/GRPO/Dev JSONL.

Raw data are read-only; no source records are modified. Stdlib only unless
--tokenizer is supplied (requires the installed transformers environment).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

DEFAULT_INPUT = Path('/root/autodl-tmp/project/convfinqa_posttrain/data/raw/ConvFinQA/data_extracted/data')
DEFAULT_OUTPUT = Path('/root/autodl-tmp/project/convfinqa_posttrain/data/processed_v1')
EXPECTED = {'train': 11104, 'dev': 1490}
SYSTEM = (
    'You are a financial numerical reasoning assistant. Use the supplied financial text, '
    'table and conversation history to answer the CURRENT question. '
    'When calculation is needed, show concise steps. '
    'End your reply with exactly one final answer using <answer>...</answer>. '
    'Keep units and numerical values precise. Do not answer future questions.'
)


def load_list(path: Path):
    with path.open(encoding='utf-8') as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise TypeError(f'{path}: expected list')
    return obj


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for blob in iter(lambda: f.read(1 << 20), b''):
            h.update(blob)
    return h.hexdigest()


def scalar_text(x):
    if isinstance(x, bool) or x is None:
        raise ValueError(f'invalid gold answer: {x!r}')
    if isinstance(x, float) and not math.isfinite(x):
        raise ValueError(f'non-finite gold answer: {x!r}')
    if isinstance(x, (int, float, str)):
        y = str(x).strip()
        if not y:
            raise ValueError('empty gold answer')
        return y
    raise TypeError(f'unsupported gold answer type: {type(x).__name__}')


def text_lines(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        lines = []
        for elem in value:
            if isinstance(elem, (dict, list)):
                lines.append(json.dumps(elem, ensure_ascii=False))
            elif elem is not None:
                lines.append(str(elem).strip())
        return [s for s in lines if s]
    return [str(value)]


def render_table(table):
    if table is None:
        return '(empty table)'
    if isinstance(table, list):
        lines = []
        for row in table:
            if isinstance(row, list):
                lines.append(' | '.join(str(c).replace('\n', ' ').strip() for c in row))
            else:
                lines.append(str(row))
        return '\n'.join(lines) if lines else '(empty table)'
    return str(table)


def make_user(row, questions, prev_answers):
    # Only financial context and the CURRENT-or-EARLIER dialogue are allowed.
    # Never serialize row['qa'] or entire row['annotation'] into a prompt.
    if len(questions) != len(prev_answers) + 1:
        raise ValueError('Question/answer history misalignment')
    pre = '\n'.join(text_lines(row.get('pre_text'))) or '(none)'
    post = '\n'.join(text_lines(row.get('post_text'))) or '(none)'
    table = render_table(row.get('table'))
    prev = []
    for i, (q, a) in enumerate(zip(questions[:-1], prev_answers), 1):
        prev += [f'Question {i}: {q}', f'Answer {i}: <answer>{a}</answer>']
    hist = '\n'.join(prev) if prev else '(This is the first turn; there is no previous dialogue.)'
    return (
        'Financial context:\n'
        f'Pre-text:\n{pre}\n\n'
        f'Table:\n{table}\n\n'
        f'Post-text:\n{post}\n\n'
        f'Previous conversation (oracle answers):\n{hist}\n\n'
        f'Current question:\n{questions[-1]}'
    )


def program_steps(program):
    # Split commas only at depth zero. This preserves subtract(a, b), divide(#0, c).
    expr = str(program).strip()
    if not expr:
        raise ValueError('empty gold program')
    pieces, depth, start = [], 0, 0
    for i, char in enumerate(expr):
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            if depth < 0:
                raise ValueError(f'unbalanced program: {expr}')
        elif char == ',' and depth == 0:
            part = expr[start:i].strip()
            if part:
                pieces.append(part)
            start = i + 1
    if depth != 0:
        raise ValueError(f'unbalanced program: {expr}')
    last = expr[start:].strip()
    if last:
        pieces.append(last)
    return pieces


def make_target(program, gold):
    steps = program_steps(program)
    # Do not hallucinate numerical values of intermediate steps (#0, #1...).
    lines = [f'Step {i}: {step}' for i, step in enumerate(steps, 1)]
    return '\n'.join(lines) + f'\n\n<answer>{gold}</answer>'


def manual_prompt(system: str, user: str):
    # Exactly the format used in the previous Qwen3 FinQA training evaluator.
    return f'System:\n{system.strip()}\n\nUser:\n{user.strip()}\n\nAssistant:'


def convert_row(row, conv_ids, split):
    rid = str(row['id'])
    match = re.fullmatch(r'(.*)_(\d+)', rid)
    if not match:
        raise ValueError(f'Unexpected ID {rid}')
    conv_id, suffix = match.group(1), int(match.group(2))
    if conv_id not in conv_ids:
        raise ValueError(f'{rid}: parent conversation not present in {split}.json')
    ann = row['annotation']
    idx = ann['turn_ind']
    questions = ann['cur_dial']
    all_questions = ann['dialogue_break']
    answers = ann['exe_ans_list']
    programs = ann['turn_program']
    if not isinstance(idx, int) or idx < 0 or suffix != idx:
        raise ValueError(f'{rid}: invalid turn number')
    if (not isinstance(questions, list) or len(questions) != idx + 1
            or questions != all_questions[:idx+1]):
        raise ValueError(f'{rid}: current dialogue is not prefix of conversation')
    if len(answers) <= idx or len(programs) <= idx:
        raise ValueError(f'{rid}: missing answer or program')
    gold = scalar_text(ann['exe_ans'])
    listed = scalar_text(answers[idx])
    if gold != listed:
        try:
            same = math.isclose(float(gold.replace(',', '').rstrip('%')), float(listed.replace(',', '').rstrip('%')),
                                rel_tol=1e-7, abs_tol=1e-7)
        except ValueError:
            same = False
        if not same:
            raise ValueError(f'{rid}: gold does not match exe_ans_list at current index')
    prog = str(ann['cur_program']).strip()
    if prog != str(programs[idx]).strip():
        raise ValueError(f'{rid}: program mismatch')
    previous = [scalar_text(v) for v in answers[:idx]]
    user = make_user(row, questions, previous)
    target = make_target(prog, gold)
    operators = sorted(set(re.findall(r'([a-zA-Z_][a-zA-Z_0-9]*)\s*\(', prog)))
    meta = {
        'conversation_id': conv_id,
        'turn_ind': idx,
        'cur_type': str(ann.get('cur_type', 'unknown')),
        'history_mode': 'oracle_gold_answers',
        'history_turns': idx,
        'operators': operators,
        'gold_program': prog,
        'source_split': split,
        'source_filename': str(row.get('filename', '')),
        # There is no ConvFinQA difficulty annotation equivalent to FinQA's.
        'difficulty': 'unknown',
    }
    msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}]
    sft = {'id': rid, 'messages': msgs + [{'role': 'assistant', 'content': target}], 'metadata': dict(meta)}
    grpo = {'id': rid, 'prompt': msgs, 'gold_answer': gold, 'metadata': dict(meta)}
    ev = {'id': rid, 'prompt': msgs, 'gold_answer': gold, 'gold_program': prog, 'metadata': dict(meta)}
    return sft, grpo, ev, user, target


def percentiles(a):
    if not a:
        return {}
    a = sorted(a)
    return {'min': a[0], 'median': median(a), 'p90': a[math.ceil(len(a)*0.90)-1],
            'p95': a[math.ceil(len(a)*0.95)-1], 'p99': a[math.ceil(len(a)*0.99)-1], 'max': a[-1]}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def run(input_dir: Path, output_dir: Path, tokenizer_path: str | None,
        prompt_budget: int, total_budget: int):
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tok = None
    if tokenizer_path:
        from transformers import AutoTokenizer
        print(f'Loading tokenizer: {tokenizer_path}', flush=True)
        tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=True)
    manifest = {
        'schema': 'convfinqa_oracle_history_v1', 'input_dir': str(input_dir),
        'output_dir': str(output_dir), 'history_mode': 'oracle_gold_answers',
        'system': SYSTEM, 'input_sha256': {}, 'splits': {},
        'tokenizer_path': tokenizer_path, 'prompt_budget': prompt_budget, 'total_budget': total_budget,
        'scope': 'Train and Dev only; private Test is not processed or used for training.',
        'cautions': [
            'No automatic truncation; long prompts must be handled explicitly before training.',
            'Prompt uses ONLY previous gold answers; free-running evaluation is a separate task.',
            'Program and gold_answer appear only in assistant target/metadata, never in the user prompt.',
            'Percent or ratio values are kept in original scale; do not automatically multiply by 100.',
            'Generated files are a data interface; train/eval scripts require context budget and reward verification.'
        ],
    }
    counts = {}
    split_convs = {}
    split_filenames = {}
    for split in ('train', 'dev'):
        turn_src = input_dir / f'{split}_turn.json'
        conv_src = input_dir / f'{split}.json'
        manifest['input_sha256'][turn_src.name] = sha256(turn_src)
        manifest['input_sha256'][conv_src.name] = sha256(conv_src)
        convs = load_list(conv_src)
        turns = load_list(turn_src)
        if len(turns) != EXPECTED[split]:
            raise ValueError(f'{split} row count mismatch: {len(turns)} != {EXPECTED[split]}')
        conv_ids = {str(item['id']) for item in convs}
        split_convs[split] = conv_ids
        split_filenames[split] = {str(item.get('filename','')) for item in convs if item.get('filename')}
        if len(conv_ids) != len(convs):
            raise ValueError(f'{split}: duplicate conversation ID')
        sft_rows, grpo_rows, eval_rows = [], [], []
        lengths, targets = [], []
        prompt_lengths, total_lengths = [], []
        over_prompt, over_total = [], []
        types, turn_counts, answers_types = Counter(), Counter(), Counter()
        seen = set()
        preview = None
        for row in turns:
            sft, grpo, ev, user, target = convert_row(row, conv_ids, split)
            rid = sft['id']
            if rid in seen:
                raise ValueError(f'{split}: duplicate turn ID {rid}')
            seen.add(rid)
            sft_rows.append(sft)
            grpo_rows.append(grpo)
            eval_rows.append(ev)
            idx = sft['metadata']['turn_ind']
            types[sft['metadata']['cur_type']] += 1
            turn_counts[str(idx)] += 1
            answers_types[type(row['annotation']['exe_ans']).__name__] += 1
            lengths.append(len(user))
            targets.append(len(target))
            if idx >= 2 and preview is None:
                # Prompt preview contains financial context and oracle previous answers.
                preview = {'id': rid, 'prompt': manual_prompt(SYSTEM, user), 'assistant_target': '\n'+target}
            if tok is not None:
                prompt = manual_prompt(SYSTEM, user)
                a = len(tok.encode(prompt, add_special_tokens=False))
                b = len(tok.encode('\n'+target, add_special_tokens=False))
                prompt_lengths.append(a)
                total_lengths.append(a+b+1)
                if a > prompt_budget and len(over_prompt) < 15:
                    over_prompt.append({'id': rid, 'prompt_tokens': a, 'history_turns': idx})
                if a+b+1 > total_budget and len(over_total) < 15:
                    over_total.append({'id': rid, 'total_tokens': a+b+1, 'history_turns': idx})
        write_jsonl(output_dir / 'sft' / f'{split}.jsonl', sft_rows)
        if split == 'train':
            write_jsonl(output_dir / 'grpo' / 'train.jsonl', grpo_rows)
        if split == 'dev':
            write_jsonl(output_dir / 'eval' / 'dev.jsonl', eval_rows)
        counts[split] = len(sft_rows)
        stats = {
            'turn_rows': len(sft_rows), 'conversation_count': len(conv_ids),
            'turn_distribution': dict(sorted(turn_counts.items(), key=lambda kv: int(kv[0]))),
            'cur_type_distribution': dict(types), 'gold_answer_python_types': dict(answers_types),
            'user_chars': percentiles(lengths), 'assistant_chars': percentiles(targets),
        }
        if tok is not None:
            stats.update({
                'prompt_tokens': percentiles(prompt_lengths),
                'prompt_plus_target_tokens': percentiles(total_lengths),
                'n_prompt_over_budget': sum(x > prompt_budget for x in prompt_lengths),
                'n_total_over_budget': sum(x > total_budget for x in total_lengths),
                'prompt_over_budget_examples': over_prompt,
                'total_over_budget_examples': over_total,
            })
        manifest['splits'][split] = stats
        if preview:
            prev_path = output_dir / 'examples' / f'{split}_turn2_preview.json'
            prev_path.parent.mkdir(parents=True, exist_ok=True)
            prev_path.write_text(json.dumps(preview, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'{split}: {len(sft_rows)} rows, {len(conv_ids)} conversations; '
              f'user chars p95={stats["user_chars"]["p95"]}')
        if tok is not None:
            print(f'  prompt tokens p95={stats["prompt_tokens"]["p95"]}; '
                  f'>{prompt_budget}: {stats["n_prompt_over_budget"]}; '
                  f'total >{total_budget}: {stats["n_total_over_budget"]}')
    intersection = split_convs['train'] & split_convs['dev']
    if intersection:
        raise ValueError(f'Train/Dev identical conversation IDs: {len(intersection)} examples {sorted(intersection)[:3]}')
    shared_docs = split_filenames['train'] & split_filenames['dev']
    manifest['cross_split'] = {
        'identical_conversation_ids': len(intersection),
        'shared_source_filename_count': len(shared_docs),
        'shared_source_filename_examples': sorted(shared_docs)[:12],
        'note': 'Shared source documents are possible; assess actual overlap before claims of independent document generalization.'
    }
    dest = output_dir / 'manifest.json'
    dest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Train/Dev shared source filename count: {len(shared_docs)}')
    print(f'Wrote dataset manifest: {dest}')
    print('No private Test rows were processed; no source files modified.')
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, default=DEFAULT_INPUT)
    ap.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument('--tokenizer', type=str, default=None, help='Optional local tokenizer path for token-length audit.')
    ap.add_argument('--prompt-budget', type=int, default=1664, help='Reference old FinQA SFT prompt budget; do not truncate.')
    ap.add_argument('--total-budget', type=int, default=2048, help='Reference old FinQA SFT total budget; do not truncate.')
    args = ap.parse_args()
    run(args.data_root, args.output_dir, args.tokenizer, args.prompt_budget, args.total_budget)


if __name__ == '__main__':
    main()
