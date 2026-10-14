#!/usr/bin/env python3
"""Read-only 421-conversation self-history comparisons. Does not rerun inference."""
import argparse
from pathlib import Path
from common import (EXP, OUT, N_CONV, N_CONVERSATIONS, predictions,
                    validate_metrics, assert_aligned, ci_paired, jsonwrite)

PATHS = {
    'transfer': EXP/'self_history_v1/transfer',
    'direct': EXP/'self_history_v1/direct',
    'replay': EXP/'03_finqa_sft_replay/eval_self_history/replay',
}
MODES = ('matched_oracle','model_answer_history')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bootstrap',type=int,default=5000)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    model={};summary={}; oracle_fp={}
    for name, folder in PATHS.items():
        model[name]={}
        summary[name]={}
        for mode in MODES:
            p=folder/mode
            r=predictions(p/'predictions.jsonl',N_CONV)
            sha=validate_metrics(p,r,kind='conv')
            if not sha:raise ValueError(f'{name}/{mode}: missing input fingerprint')
            m = __import__('json').loads((p/'metrics.json').read_text(encoding='utf-8'))
            if m.get('history_mode')!=mode or m.get('neutral_history_header') is not True:
                raise ValueError(f'{name}/{mode}: not neutral-header matched evaluator, or mode mismatch')
            nconvs={x['metadata'].get('conversation_id') for x in r.values()}
            if None in nconvs or len(nconvs)!=N_CONVERSATIONS:
                raise ValueError(f'{name}/{mode}: expected {N_CONVERSATIONS} actual conversations, got {len(nconvs)}')
            model[name][mode]=r
            summary[name][mode]={'accuracy':sum(x['correct'] for x in r.values())/N_CONV,
                                  'correct':sum(x['correct'] for x in r.values()),
                                  'conversations':len(nconvs), 'fingerprint':sha,
                                  'conversation_exact_match':m['conversation_exact_match']['accuracy']}
            if mode=='matched_oracle': oracle_fp[name]=sha
        assert_aligned(model[name]['matched_oracle'],model[name]['model_answer_history'],name+' Oracle vs Self',by_conv=True)
    if len(set(oracle_fp.values()))!=1:raise ValueError('Matched-oracle prompts different between models; STOP')
    for name in ('direct','replay'):
        for mode in MODES:
            assert_aligned(model['transfer'][mode],model[name][mode],f'transfer vs {name} {mode}',by_conv=True)
    print('PASS: 1490 IDs, 421 conversations, same GOLD and matched-oracle neutral-header fingerprint')
    result={'dataset':'ConvFinQA Dev','n':N_CONV,'n_conversations':N_CONVERSATIONS,
            'history':'previous MODEL final answers only, not full generated reasoning or transcript',
            'matched_oracle_prompt_fingerprint':next(iter(oracle_fp.values())),
            'models':summary,'comparisons':{},'bootstrap':args.bootstrap,'seed':args.seed,
            'caveats':['Dev is not unseen Test.','Self generated histories differ by model, so Self prompts do not have matching fingerprints.',
                       'Bootstrap resamples whole conversations; no claim that CIs including 0 establish equivalence.']}
    print('\n=== ConvFinQA Dev neutral-header history ===')
    for name in PATHS:
        print(f'{name:10}: oracle={summary[name]["matched_oracle"]["accuracy"]:.2%} self={summary[name]["model_answer_history"]["accuracy"]:.2%} '
              f'self all-turn={summary[name]["model_answer_history"]["conversation_exact_match"]:.2%}')
    for name in PATHS:
        a=model[name]['matched_oracle'];b=model[name]['model_answer_history']
        r=ci_paired(a,b,lambda row:str(row['metadata']['conversation_id']),args.bootstrap,args.seed)
        result['comparisons'][f'{name}_oracle_vs_self']=r
        print(f'{name:10} Oracle -> Self: {r["delta_pp"]:+.2f}pp; CI {[round(x,2) for x in r["cluster_bootstrap_ci95_pp"]]}')
    for a,b in (('transfer','replay'),('direct','replay'),('direct','transfer')):
        r=ci_paired(model[a]['model_answer_history'],model[b]['model_answer_history'],
                    lambda row:str(row['metadata']['conversation_id']),args.bootstrap,args.seed)
        result['comparisons'][f'{a}_vs_{b}_self']=r
        print(f'{a:10} -> {b:10} (Self): {r["delta_pp"]:+.2f}pp CI {[round(x,2) for x in r["cluster_bootstrap_ci95_pp"]]} wins/losses {r["candidate_wins"]}/{r["candidate_losses"]}')
    target=OUT/'analysis/self_history_paired.json'
    if target.exists():raise FileExistsError(f'Analysis already exists: {target} (do not overwrite)')
    jsonwrite(target,result)
    print('Saved:',target)

if __name__=='__main__':main()
