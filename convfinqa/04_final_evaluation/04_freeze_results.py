#!/usr/bin/env python3
"""Create an immutable-by-convention, independently verifiable archive (no original file mutation)."""
from __future__ import annotations
import argparse, datetime, os, shutil, tempfile
from pathlib import Path
from common import (ROOT, PROJECT, FINQA, EXP, OUT, BASE, BENCH, ADAPTERS,
                    FINQA_TEST, FINQA_DEV, CONV_DEV, sha256, jsonread, jsonwrite)

# Copy small derived evidence only, not weights nor raw datasets.
REQUIRED = {
  'preflight.json':OUT/'preflight.json',
  'analysis/finqa_test_paired.json':OUT/'analysis/finqa_test_paired.json',
  'analysis/self_history_paired.json':OUT/'analysis/self_history_paired.json',
  'analysis/finqa_retention_dev.json':EXP/'finqa_retention_v1/analysis/retention_summary.json',
  'analysis/conv_replay_dev.json':EXP/'03_finqa_sft_replay/analysis/replay_comparison.json',
  'analysis/conv_transfer_direct_dev.json':EXP/'paired_transfer_vs_direct_v1/paired_summary.json',
  'data/replay_mix_manifest.json':EXP/'03_finqa_sft_replay/data/mix_manifest.json',
}
for variant in ('original_sft','transfer','direct','replay'):
  base=OUT/'finqa_test'/variant/'full_test'
  for f in ('predictions.jsonl','metrics.json','benchmark.json'):
    REQUIRED[f'finqa_test/{variant}/{f}']=base/f
for variant in ('original_sft','transfer','direct','replay'):
  base=(EXP/'03_finqa_sft_replay/eval_finqa/full_dev' if variant=='replay' else
        EXP/'finqa_retention_v1'/variant/'full_dev')
  for f in ('predictions.jsonl','metrics.json','benchmark.json'):
    REQUIRED[f'finqa_dev/{variant}/{f}']=base/f
for variant in ('transfer','direct','replay'):
  base=EXP/({'transfer':'01_finqa_sft_adapter/eval_dev','direct':'02_qwen3_base/eval_dev',
             'replay':'03_finqa_sft_replay/eval_convfinqa_oracle'}[variant])
  for f in ('predictions.jsonl','metrics.json'):
    REQUIRED[f'conv_oracle_dev/{variant}/{f}']=base/f
  if variant in ('transfer','direct'):
    REQUIRED[f'conv_oracle_dev/{variant}/run_config.json']=base/'run_config.json'
  else:
    REQUIRED[f'conv_oracle_dev/{variant}/run_config.json']=base/'run_config.json'
  hist=(EXP/'self_history_v1'/variant if variant!='replay' else
        EXP/'03_finqa_sft_replay/eval_self_history/replay')
  for mode in ('matched_oracle','model_answer_history'):
    for f in ('predictions.jsonl','metrics.json'):
      REQUIRED[f'self_history_dev/{variant}/{mode}/{f}']=hist/mode/f
# Include original reports when available, tracked as optional evidence.
OPTIONAL = {
    'training/transfer_training_report.json': EXP/'01_finqa_sft_adapter/full/training_report.json',
    'training/direct_training_report.json': EXP/'02_qwen3_base/full/training_report.json',
    'training/replay_training_report.json': EXP/'03_finqa_sft_replay/full/training_report.json',
    'training/replay_mix_data.jsonl': EXP/'03_finqa_sft_replay/data/replay_train.jsonl',
}
REFERENCES = [FINQA_TEST,FINQA_DEV,CONV_DEV,BASE/'config.json',
              BENCH/'bench_vllm.py',BENCH/'common.py',BENCH/'metrics.py',BENCH/'answer_normalizer.py']
REFERENCES.extend(PROJECT.rglob('*.py'))
REFERENCES.extend(PROJECT.rglob('*.sh'))
# Historical FinQA SFT / Vanilla GRPO / Dynamic Sampling run metadata is optional.
# Archive small, already-existing metric/config reports when available, without
# guessing scores or requiring any historical directory to have a particular name.
FINQA_EXPERIMENT_ROOT=FINQA/'experiments'
HISTORICAL_FINQA_REPORTS={}
if FINQA_EXPERIMENT_ROOT.is_dir():
  accepted={'metrics.json','benchmark.json','comparison.json','run_config.json',
            'training_report.json','retention_summary.json','paired_summary.json',
            'diagnostics_summary.json','dynamic_sampling_summary.json',
            'results.json','result.json','summary.json','trainer_state.json'}
  for p in FINQA_EXPERIMENT_ROOT.rglob('*.json'):
    if p.name in accepted and p.is_file() and p.stat().st_size <= 2_000_000:
      HISTORICAL_FINQA_REPORTS['finqa_historical_runs/'+str(p.relative_to(FINQA_EXPERIMENT_ROOT))]=p
  for conf in FINQA_EXPERIMENT_ROOT.rglob('adapter_config.json'):
    REFERENCES.append(conf)
    REFERENCES.extend(conf.parent.glob('*.safetensors'))
    REFERENCES.extend(conf.parent.glob('*.bin'))
# Previously recorded original FinQA adapter performance on ConvFinQA Dev.
for variant in ('qwen3_base','finqa_sft','finqa_dynamic'):
  folder=ROOT/'project/convfinqa_posttrain/experiments/oracle_baselines_v1/full_dev'/variant
  for filename in ('metrics.json','predictions.jsonl'):
    p=folder/filename
    if p.is_file():HISTORICAL_FINQA_REPORTS[f'original_conv_baselines/{variant}/{filename}']=p
for adapter in ADAPTERS.values():
  REFERENCES.append(adapter/'adapter_config.json')
  REFERENCES.extend(sorted(adapter.glob('*.safetensors')))
  REFERENCES.extend(sorted(adapter.glob('*.bin')))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dest',type=Path,default=OUT/'frozen_release_v1')
    args=ap.parse_args()
    dest=args.dest
    if dest.exists():raise FileExistsError(f'Refusing to overwrite frozen archive: {dest}')
    missing=[str(f) for f in REQUIRED.values() if not f.is_file()]
    if missing:raise FileNotFoundError('Required evaluation/evidence missing:\n'+'\n'.join(missing))
    if any(not p.is_file() for p in (FINQA_TEST,FINQA_DEV,CONV_DEV)):
        raise FileNotFoundError('Required source datasets unavailable')
    # Require same successful preflight before freezing.
    pf=jsonread(OUT/'preflight.json')
    if sha256(FINQA_TEST)!=pf['test_input_sha256']:
        raise ValueError('STOP: FinQA Test data changed since first preflight')
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix='.frozen_build_',dir=dest.parent))
    try:
        manifest={'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'project':'Qwen3-1.7B FinQA + ConvFinQA Post-training',
                  'freeze_version':1, 'entry_count':len(REQUIRED),
                  'evidence':{},'source_references':{},'missing_optional_evidence':[],
                  'notes':['Frozen only after the final one-time FinQA Test evaluation.',
                           'Source weights and datasets are NOT copied; their hashes are recorded.',
                           'Self-History uses generated FINAL ANSWERS, not full generated explanations.',
                           'FinQA Test has been inspected earlier in the overall project; do not call it blind.',
                           'CIs on analyzed Dev sets are exploratory.']}
        files={**REQUIRED,**HISTORICAL_FINQA_REPORTS}
        for label, src in OPTIONAL.items():
            if src.is_file():files[label]=src
            else:manifest['missing_optional_evidence'].append(label)
        for label,src in sorted(files.items()):
            target=tmp/'evidence'/label
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,target)
            manifest['evidence'][f'evidence/{label}']={'sha256':sha256(target),'bytes':target.stat().st_size,'origin':str(src)}
        for ref in sorted(set(REFERENCES),key=str):
            if ref.is_file():
                # Exclude release/outputs if user installed this experiment in project folder.
                if str(ref).startswith(str(OUT)): continue
                manifest['source_references'][str(ref)]={'sha256':sha256(ref),'bytes':ref.stat().st_size}
        for f in (OUT/'analysis/finqa_test_paired.json',OUT/'analysis/self_history_paired.json'):
            if not f.is_file():raise FileNotFoundError(f)
        finqa=jsonread(OUT/'analysis/finqa_test_paired.json')
        history=jsonread(OUT/'analysis/self_history_paired.json')
        replay=jsonread(EXP/'03_finqa_sft_replay/analysis/replay_comparison.json')
        md='''# Final results — frozen v1\n\nThis archive is a **snapshot**, not a retraining or a benchmark tuning script.\n\n## FinQA Test (1147 questions)\n\n'''
        md+='| Model | Correct | Accuracy |\n|---|---:|---:|\n'
        for name, row in finqa['models'].items():
            md+=f'| {name} | {row["correct"]}/1147 | {row["accuracy"]:.2%} |\n'
        md+='\n## ConvFinQA Dev Self-History (1490 turns / 421 conversations)\n\n'
        md+='| Model | Matched Oracle | Model Self-History | Self all-turn exact |\n|---|---:|---:|---:|\n'
        for name, d in history['models'].items():
            md+=(f'| {name} | {d["matched_oracle"]["accuracy"]:.2%} | '
                 f'{d["model_answer_history"]["accuracy"]:.2%} | '
                 f'{d["model_answer_history"]["conversation_exact_match"]:.2%} |\n')
        md+='\n## FinQA Dev retention / ConvFinQA Oracle Dev\n\n'
        md+='Experience Replay was evaluated with the historical frozen methods. Detailed paired results are in `evidence/analysis/`.\n\n'
        md+='## Key caveats\n\n- FinQA Test was consulted in earlier project phases: it is not an untouched blind test. No further tuning should be performed on it.\n- ConvFinQA metrics are Dev results. Its underlying source documents overlap the original FinQA corpus.\n- Oracle History uses previous GOLD answers. Self-History uses previous MODEL final-answer strings, not an unconstrained chat transcript.\n- Document/conversation cluster bootstrap estimates sampling uncertainty for a single-seed comparison; it does not prove equivalence.\n- Source model weights and raw datasets are referenced with SHA-256 hashes, NOT copied into this archive. Keep the original files or back them up separately.\n'
        (tmp/'RESULTS.md').write_text(md,encoding='utf-8')
        # Hash all snapshot-derived summary files but exclude the manifest itself.
        manifest['historical_finqa_report_count']=len(HISTORICAL_FINQA_REPORTS)
        manifest['snapshot_summary']={'RESULTS.md':sha256(tmp/'RESULTS.md')}
        jsonwrite(tmp/'MANIFEST.json',manifest)
        (tmp/'MANIFEST.sha256').write_text(sha256(tmp/'MANIFEST.json')+'  MANIFEST.json\n',encoding='ascii')
        if dest.exists():raise FileExistsError('Destination appeared during build '+str(dest))
        tmp.rename(dest)
        print('FROZEN archive:',dest)
        print('Saved evidence files:',len(manifest['evidence']),'source hashes:',len(manifest['source_references']))
        print('Archive manifest:',dest/'MANIFEST.json')
        print('Source weights are not copied; back up model directories separately.')
    except BaseException:
        shutil.rmtree(tmp,ignore_errors=True)
        raise

if __name__=='__main__': main()
