#!/usr/bin/env python3
"""Reuse the exact frozen neutral-header self-history evaluator for Replay only."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT=Path('/root/autodl-tmp')
COMMON=ROOT/'ConvFinQA/00_common'
ADAPTER=ROOT/'project/convfinqa_posttrain/experiments/convfinqa_sft_v2/03_finqa_sft_replay/full/final_adapter'
n=0
if '--max-conversations' in sys.argv:
    index=sys.argv.index('--max-conversations')
    n=int(sys.argv[index+1])
OUTPUT=ROOT/f'project/convfinqa_posttrain/experiments/convfinqa_sft_v2/03_finqa_sft_replay/eval_self_history{f"_smoke{n}" if n else ""}'
if not (COMMON/'eval_self_history_v1.py').exists():
    raise FileNotFoundError('Prior self-history evaluator missing: '+str(COMMON/'eval_self_history_v1.py'))
if not (ADAPTER/'adapter_config.json').exists():raise FileNotFoundError(ADAPTER)
if (OUTPUT/'replay'/'model_answer_history'/'metrics.json').exists():
    raise FileExistsError('Existing self-history evaluation; refusing overwrite: '+str(OUTPUT))
sys.path.insert(0,str(COMMON))
import eval_self_history_v1 as evaluator
evaluator.ADAPTERS['replay']=ADAPTER
# Avoid invoking its fixed direct-vs-transfer pair comparison: one variant only.
sys.argv=[sys.argv[0],'--variants','replay','--output-dir',str(OUTPUT)]+sys.argv[1:]
evaluator.main()
