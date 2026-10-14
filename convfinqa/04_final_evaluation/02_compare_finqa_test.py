#!/usr/bin/env python3
"""Final one-time FinQA Test paired comparison. Reads frozen predictions only."""
import argparse
from common import OUT, N_TEST, MODEL_ORDER, document_cluster, ci_paired, assert_aligned, jsonwrite
from verify_one_test import verify

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bootstrap',type=int,default=5000)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    preds={name:verify(OUT/'finqa_test'/name/'full_test',name) for name in MODEL_ORDER}
    for name in MODEL_ORDER[1:]:assert_aligned(preds['original_sft'],preds[name],'FinQA Test '+name)
    identified=sum(document_cluster(p)[1] for p in preds['original_sft'].values())
    print(f'FinQA source PDF recovery: {identified}/{N_TEST} rows')
    # Always disclose item fallback explicitly rather than falsely claim pure document clustering.
    mode='document_clustered_with_item_fallback' if identified<N_TEST else 'document_clustered'
    pairs=[('original_sft','transfer'),('original_sft','direct'),('original_sft','replay'),
           ('transfer','direct'),('transfer','replay'),('direct','replay')]
    output={'split':'FinQA Test','sample_count':N_TEST,'seed':args.seed,
            'bootstrap_replicates':args.bootstrap,'cluster_method':mode,
            'pdf_identified_rows':identified,
            'models':{m:{'correct':sum(x['correct'] for x in preds[m].values()),
                          'accuracy':sum(x['correct'] for x in preds[m].values())/N_TEST} for m in MODEL_ORDER},'pairs':{},
            'caveats':['One-time final Test use only; do not select models or retune on Test.',
                       'Earlier project phases already inspected FinQA Test results; not a never-observed blind benchmark.',
                       'Same numerical evaluator / vLLM settings; no claim of zero forgetting without equivalence testing.',
                       'Document clustering partially falls back to per-item clusters if PDF IDs cannot be parsed.']}
    print('\n=== FINAL FinQA Test (same 1147 rows and scorer) ===')
    for m in MODEL_ORDER:print(f'{m:15} {output["models"][m]["correct"]:4d}/{N_TEST}: {output["models"][m]["accuracy"]:.2%}')
    print('\n=== PAIRED TEST: candidate - reference ===')
    for a,b in pairs:
        r=ci_paired(preds[a],preds[b],lambda row:document_cluster(row)[0],args.bootstrap,args.seed)
        output['pairs'][f'{a}_vs_{b}']=r
        print(f'{a:12} -> {b:12}: {r["delta_pp"]:+.2f}pp CI {[round(x,2) for x in r["cluster_bootstrap_ci95_pp"]]} wins/losses {r["candidate_wins"]}/{r["candidate_losses"]}')
    target=OUT/'analysis/finqa_test_paired.json'
    if target.exists(): raise FileExistsError(f'Analysis already exists: {target} (do not overwrite final test)')
    jsonwrite(target,output)
    print('Saved:',target)

if __name__=='__main__': main()
