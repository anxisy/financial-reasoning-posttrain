#!/usr/bin/env python3
"""Paired neutral-header Oracle vs self-generated-answer-history ConvFinQA Dev.

Read-only evaluation. Never trains, and never injects the current or future gold answer
into a model prompt. In self mode previous turns are answered in chronological order.
This is answer-only self-history, NOT a full-transcript free-running conversation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

from frozen_baseline import (
    get_adapter_rank, read_examples, score_prediction,
    serialize_manual, summarize_predictions,
)
from frozen_paired import comparison, read_predictions

BASE = '/root/autodl-tmp/models/Qwen3-1.7B-Base'
DEV = '/root/autodl-tmp/project/convfinqa_posttrain/data/processed_v2/eval/dev.jsonl'
RESULTS = Path('/root/autodl-tmp/project/convfinqa_posttrain/experiments/convfinqa_sft_v2')
ADAPTERS = {
    'transfer': RESULTS / '01_finqa_sft_adapter/full/final_adapter',
    'direct': RESULTS / '02_qwen3_base/full/final_adapter',
}
OLD_HEADER = 'Previous conversation (oracle answers):\n'
NEW_HEADER = 'Previous conversation:\n'
ANS_RE = re.compile(r'(?m)^Answer ([1-9][0-9]*): <answer>(.*?)</answer>$')


def build_prompt(row: dict, prior_answers: dict[tuple[str, int], str] | None):
    """Neutral header in BOTH matched-oracle and self-answer settings."""
    msgs = copy.deepcopy(row['prompt'])
    meta = row['metadata']
    turn = int(meta['turn_ind'])
    cid = str(meta['conversation_id'])
    user = msgs[1]['content']
    if user.count(OLD_HEADER) != 1:
        raise ValueError(f'{row["id"]}: expected exactly one history header')
    if user.count('\n\nCurrent question:\n') != 1:
        raise ValueError(f'{row["id"]}: current question separator missing')
    matches = list(ANS_RE.finditer(user))
    if len(matches) != turn or [int(m.group(1)) for m in matches] != list(range(1, turn + 1)):
        raise ValueError(f'{row["id"]}: malformed history or impossible turn count')
    if prior_answers is not None:
        for past in range(turn):
            if (cid, past) not in prior_answers:
                raise ValueError(f'{row["id"]}: missing model-generated history for turn {past}')
        def sub(m):
            past = int(m.group(1)) - 1
            value = prior_answers[(cid, past)]
            if '\n' in value or '</answer>' in value or '<answer>' in value:
                raise ValueError(f'{row["id"]}: unsafe answer-only history')
            return f'Answer {past + 1}: <answer>{value}</answer>'
        user = ANS_RE.sub(sub, user)
    user = user.replace(OLD_HEADER, NEW_HEADER, 1)
    msgs[1]['content'] = user
    return serialize_manual(msgs)


def choose_rows(rows, max_conversations):
    if max_conversations <= 0:
        return rows
    seen = set()
    chosen = []
    for row in rows:
        cid = row['metadata']['conversation_id']
        if cid not in seen:
            if len(seen) >= max_conversations:
                continue
            seen.add(cid)
        chosen.append(row)
    return chosen


def get_history_answer(scored):
    # No gold-answer fallback. The string comes from the current model output ONLY.
    s = scored.get('prediction_answer')
    s = str(s).strip() if s is not None else ''
    if not s:
        return '[NO_ANSWER]'
    # Normalizer ordinarily returns a scalar, but protect history syntax.
    s = s.replace('\n', ' ').replace('\r', ' ').replace('<answer>', '').replace('</answer>', '')
    return s[:300] or '[NO_ANSWER]'


def hash_prompt_ids(rows, prompt_ids):
    raw = [(r['id'], seq) for r, seq in zip(rows, prompt_ids)]
    return hashlib.sha256(json.dumps(raw, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def summarize_for_mode(preds, mode):
    m = summarize_predictions(preds)
    m['conversation_exact_match']['note'] = (
        'All turns correct with previous GOLD answers (matched neutral history header)'
        if mode == 'matched_oracle' else
        'All turns correct with previous MODEL-PREDICTED final answers (answer-only self-history)'
    )
    m['history_mode'] = mode
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variants', default='transfer,direct', help='comma subset of transfer,direct')
    ap.add_argument('--input', default=DEV)
    ap.add_argument('--base', default=BASE)
    ap.add_argument('--output-dir', default=str(RESULTS / 'self_history_v1'))
    ap.add_argument('--max-conversations', type=int, default=0, help='0 = all 421; 5 for smoke test')
    ap.add_argument('--max-model-len', type=int, default=4096)
    ap.add_argument('--max-new-tokens', type=int, default=384)
    ap.add_argument('--max-num-seqs', type=int, default=32)
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.70)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--bootstrap', type=int, default=2000)
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(',') if v.strip()]
    if not variants or len(variants) != len(set(variants)) or set(variants) - set(ADAPTERS):
        ap.error('--variants must be unique comma subset of transfer,direct')
    if args.max_conversations < 0 or args.max_new_tokens <= 0:
        ap.error('invalid count or generation length')
    for v in variants:
        get_adapter_rank(ADAPTERS[v])

    rows = choose_rows(read_examples(args.input, 0, 1490), args.max_conversations)
    by_turn = defaultdict(list)
    conv_turns = defaultdict(set)
    for row in rows:
        k = int(row['metadata']['turn_ind'])
        c = str(row['metadata']['conversation_id'])
        by_turn[k].append(row)
        conv_turns[c].add(k)
    if not rows or any(turns != set(range(max(turns) + 1)) for turns in conv_turns.values()):
        raise ValueError('Incomplete conversation records; cannot test self-answer history')
    print('Dev rows:', len(rows), 'Conversations:', len(conv_turns), 'turns:', dict(sorted((t, len(x)) for t, x in by_turn.items())), flush=True)

    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=True)
    llm = LLM(model=args.base, tokenizer=args.base, dtype='bfloat16',
              max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
              gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True,
              seed=args.seed, enable_lora=True, max_lora_rank=max(8, *(get_adapter_rank(ADAPTERS[v]) for v in variants)),
              max_loras=1)
    sampler = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens,
                             seed=args.seed, skip_special_tokens=True,
                             stop_token_ids=[tok.eos_token_id])
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    written = {}
    sha_per_mode = {}

    for iv, variant in enumerate(variants, 1):
        print(f'\n========== {variant.upper()} ==========', flush=True)
        req = LoRARequest(f'convfinqa_{variant}_sft', iv, str(ADAPTERS[variant]))
        def evaluate_batch(batch_rows, model_prior, mode):
            prompts = [build_prompt(r, model_prior) for r in batch_rows]
            input_ids = tok(prompts, truncation=False, padding=False, add_special_tokens=True)['input_ids']
            overflow = [(r['id'], len(seq)) for r, seq in zip(batch_rows, input_ids)
                        if len(seq) + args.max_new_tokens > args.max_model_len]
            if overflow:
                raise ValueError(f'{mode}: {len(overflow)} overflow prompts; first examples: {overflow[:5]}')
            results = llm.generate([{'prompt_token_ids': seq} for seq in input_ids],
                                   sampling_params=sampler, lora_request=req, use_tqdm=False)
            if len(results) != len(batch_rows):
                raise RuntimeError('Missing vLLM generations')
            preds = []
            for r, result in zip(batch_rows, results):
                output = result.outputs[0]
                scored = score_prediction(r, output.text.strip(), len(output.token_ids), args.max_new_tokens, variant)
                scored['metadata'] = dict(scored['metadata'], history_mode=mode)
                preds.append(scored)
            return preds, input_ids

        # Warmup LoRA kernels, exclude warmup from scored samples.
        evaluate_batch(rows[:min(3, len(rows))], None, 'warmup')
        t0 = time.perf_counter()
        oracle_preds, oracle_ids = evaluate_batch(rows, None, 'matched_oracle')
        oracle_seconds = time.perf_counter() - t0
        preds_by_mode = {'matched_oracle': oracle_preds}
        fprints = {'matched_oracle': hash_prompt_ids(rows, oracle_ids)}

        own_answers = {}
        own_predictions = {}
        own_ids = {}
        t1 = time.perf_counter()
        for turn in sorted(by_turn):
            batch = by_turn[turn]
            preds, input_ids = evaluate_batch(batch, own_answers, 'model_answer_history')
            for r, pred, seq in zip(batch, preds, input_ids):
                c = str(r['metadata']['conversation_id'])
                own_predictions[r['id']] = pred
                own_ids[r['id']] = seq
                own_answers[(c, turn)] = get_history_answer(pred)
            print(f'  turn {turn}: {len(batch)} generated, model-history correct {sum(p["correct"] for p in preds)}', flush=True)
        self_seconds = time.perf_counter() - t1
        own_preds = [own_predictions[r['id']] for r in rows]
        own_ids_in_order = [own_ids[r['id']] for r in rows]
        preds_by_mode['model_answer_history'] = own_preds
        fprints['model_answer_history'] = hash_prompt_ids(rows, own_ids_in_order)

        target = out_root / variant
        target.mkdir(parents=True, exist_ok=True)
        for mode, preds in preds_by_mode.items():
            sub = target / mode
            sub.mkdir(parents=True, exist_ok=True)
            m = summarize_for_mode(preds, mode)
            m.update(engine='vllm', variant=variant, input_fingerprint=fprints[mode],
                     generated_answers_used_only_from_previous_turns=(mode == 'model_answer_history'),
                     neutral_history_header=True,
                     elapsed_inference_seconds=oracle_seconds if mode == 'matched_oracle' else self_seconds)
            with (sub / 'predictions.jsonl').open('w', encoding='utf-8') as f:
                for p in preds:
                    f.write(json.dumps(p, ensure_ascii=False) + '\n')
            (sub / 'metrics.json').write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'  {mode:<21}: turn acc {m["overall"]["accuracy"]:.2%}; all-turn conv {m["conversation_exact_match"]["accuracy"]:.2%}', flush=True)
        paired = comparison({r['id']: r for r in oracle_preds}, {r['id']: r for r in own_preds}, args.seed, args.bootstrap)
        (target / 'oracle_vs_self_paired.json').write_text(json.dumps(paired, ensure_ascii=False, indent=2), encoding='utf-8')
        written[variant] = {mode: (target / mode / 'predictions.jsonl') for mode in preds_by_mode}
        sha_per_mode[variant] = fprints
        print(f'  matched-oracle -> model-history: {paired["delta_accuracy_pp"]:+.2f}pp; CI {paired["cluster_bootstrap_95pct_ci_pp"]}', flush=True)

    if len(variants) == 2:
        v1, v2 = variants
        if sha_per_mode[v1]['matched_oracle'] != sha_per_mode[v2]['matched_oracle']:
            raise RuntimeError('Matched oracle prompts do not match between variants')
        # Self-answer prompts generally DIFFER between models because their predicted histories differ.
        paired = comparison(read_predictions(written['direct']['model_answer_history']),
                            read_predictions(written['transfer']['model_answer_history']),
                            args.seed, args.bootstrap)
        (out_root / 'self_history_transfer_vs_direct.json').write_text(json.dumps(paired, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'\nSELF-HISTORY transfer - direct: {paired["delta_accuracy_pp"]:+.2f}pp; CI {paired["cluster_bootstrap_95pct_ci_pp"]}', flush=True)

    config = vars(args) | {'adapters': {v: str(ADAPTERS[v]) for v in variants},
                           'prompt_fingerprints': sha_per_mode,
                           'evaluation_note': 'Answer-only previous MODEL final answers, not whole-response dialogue. Matched oracle control uses the SAME neutral history header; no Dev gold answer enters self-history prompts.'}
    (out_root / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nDone. Results:', out_root, flush=True)


if __name__ == '__main__':
    main()
