#!/usr/bin/env python3
"""Compare frozen pretraining baselines and newly trained ConvFinQA SFT, with cluster bootstrap."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from frozen_paired import comparison,read_predictions
PROJECT=Path('/root/autodl-tmp/project/convfinqa_posttrain/experiments')
MAP={
 'base':PROJECT/'oracle_baselines_v1/full_dev/qwen3_base',
 'finqa_sft':PROJECT/'oracle_baselines_v1/full_dev/finqa_sft',
 'finqa_dynamic':PROJECT/'oracle_baselines_v1/full_dev/finqa_dynamic',
 'direct_sft':PROJECT/'convfinqa_sft_v2/02_qwen3_base/eval_dev',
 'transfer_sft':PROJECT/'convfinqa_sft_v2/01_finqa_sft_adapter/eval_dev',
}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',default=str(PROJECT/'convfinqa_sft_v2/summary.json'))
    args=ap.parse_args()
    metrics={}; preds={}; configs={}
    print(f"{'Model':<20}{'Overall':>10}{'Number':>10}{'Program':>10}{'ConvExact':>11}{'Format':>10}")
    for name,path in MAP.items():
        if not (path/'metrics.json').exists():
            print(f'{name:<20} NOT YET AVAILABLE')
            continue
        m=json.loads((path/'metrics.json').read_text());metrics[name]=m
        preds[name]=read_predictions(path/'predictions.jsonl')
        for score in ('overall','by_cur_type','conversation_exact_match'): pass
        print(f"{name:<20}{m['overall']['accuracy']*100:>9.2f}%"
              f"{m['by_cur_type']['number_turn']['accuracy']*100:>9.2f}%"
              f"{m['by_cur_type']['program_turn']['accuracy']*100:>9.2f}%"
              f"{m['conversation_exact_match']['accuracy']*100:>10.2f}%"
              f"{m['overall']['format_accuracy']*100:>9.2f}%")
    if 'direct_sft' in metrics and 'transfer_sft' in metrics:
        fa=metrics['direct_sft'].get('input_fingerprint')
        fb=metrics['transfer_sft'].get('input_fingerprint')
        if fa!=fb: raise ValueError('Direct/transfer input fingerprints differ')
        for old in ('base','finqa_sft','finqa_dynamic'):
            if old in metrics and metrics[old].get('input_fingerprint')!=fa:
                raise ValueError(f'{old}: baseline prompt fingerprint differs. STOP comparison.')
    pairs=[('base','direct_sft'),('finqa_sft','transfer_sft'),('direct_sft','transfer_sft'),
           ('base','transfer_sft'),('finqa_sft','direct_sft')]
    comp={}
    for a,b in pairs:
        if a not in preds or b not in preds: continue
        result=comparison(preds[a],preds[b],seed=42,bootstrap_replicates=2000)
        comp[f'{a}_vs_{b}']=result
        print(f'{a} -> {b}: {result["delta_accuracy_pp"]:+.2f} pp, cluster CI={result["cluster_bootstrap_95pct_ci_pp"]}')
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'metrics':metrics,'paired':comp},ensure_ascii=False,indent=2))
    print('Saved:',out)
if __name__=='__main__':main()
