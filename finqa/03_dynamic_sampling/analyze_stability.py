#!/usr/bin/env python3
"""Post-hoc KL/grad_norm outlier report. This does not change training."""
import argparse
import json
import math
from pathlib import Path

DEFAULT = '/root/autodl-tmp/project/finqa_posttrain/experiments/grpo_dynamic_qwen3_1.7b/full_v1'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', default=DEFAULT)
    p.add_argument('--top-k', type=int, default=10)
    args = p.parse_args()
    out = Path(args.output_dir)
    path = out / 'trainer_log_history.json'
    if not path.exists():
        raise SystemExit(f'Missing {path}; run this after Full completes.')
    rows = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(rows, dict): rows = rows.get('log_history', [])
    report = {'log_path': str(path), 'metrics': {}}
    for metric in ('kl', 'grad_norm'):
        valid = [r for r in rows if isinstance(r.get(metric), (int,float)) and math.isfinite(r[metric])]
        top = sorted(valid, key=lambda r:r[metric], reverse=True)[:args.top_k]
        report['metrics'][metric] = {
            'count': len(valid),
            'max': top[0][metric] if top else None,
            'top': [{k:r.get(k) for k in (
                'step','epoch','kl','grad_norm','loss','reward','reward_std',
                'frac_reward_zero_std','learning_rate')} for r in top]
        }
        print(f'\nTOP {args.top_k} {metric}')
        for r in report['metrics'][metric]['top']:
            print(r)
    (out / 'stability_outliers.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nSaved:', out / 'stability_outliers.json')
    print('NOTE: logged KL can aggregate multiple steps; grad_norm may reflect last optimizer step.')
if __name__ == '__main__': main()
