#!/usr/bin/env python3
"""Config-driven entrypoints for the *unmodified* archived experiment algorithms.

No training or inference happens in init/check/plan. `run` uses the active
Python environment, never invokes a shell, and rejects nonempty output paths.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONFIG_DEFAULT = REPO / 'configs' / 'paths.local.json'

TASKS = (
    'finqa-sft-smoke', 'finqa-sft-full',
    'finqa-grpo-smoke', 'finqa-grpo-full',
    'finqa-dynamic-smoke', 'finqa-dynamic-full',
    'conv-transfer-smoke', 'conv-transfer-full',
    'conv-direct-smoke', 'conv-direct-full',
    'replay-build', 'replay-verify', 'replay-smoke', 'replay-full',
    'conv-transfer-oracle', 'conv-direct-oracle', 'conv-replay-oracle',
    'conv-transfer-self', 'conv-direct-self', 'conv-replay-self',
    'finqa-original-dev', 'finqa-transfer-dev', 'finqa-direct-dev', 'finqa-replay-dev',
    'finqa-original-test', 'finqa-transfer-test', 'finqa-direct-test', 'finqa-replay-test',
)

# These paths are from the actually validated AutoDL project, not repo-relative fixtures.
def default_config(workspace: Path) -> dict[str, str]:
    w = workspace.expanduser().resolve()
    fin = w / 'project' / 'finqa_posttrain'
    con = w / 'project' / 'convfinqa_posttrain'
    cexp = con / 'experiments' / 'convfinqa_sft_v2'
    fdata = fin / 'data' / 'processed_v1_1_1'
    cdata = con / 'data' / 'processed_v2'
    return {k: str(v) for k, v in {
        'base_model': w / 'models' / 'Qwen3-1.7B-Base',
        'finqa_sft_train': fdata / 'sft' / 'finqa_sft_train.jsonl',
        'finqa_sft_dev': fdata / 'sft' / 'finqa_sft_dev.jsonl',
        'finqa_grpo_train': fdata / 'grpo' / 'finqa_grpo_train.jsonl',
        'finqa_eval_dev': fdata / 'eval' / 'finqa_eval_dev.jsonl',
        'finqa_eval_test': fdata / 'eval' / 'finqa_eval_test.jsonl',
        'conv_sft_train': cdata / 'sft' / 'train.jsonl',
        'conv_sft_dev': cdata / 'sft' / 'dev.jsonl',
        'conv_eval_dev': cdata / 'eval' / 'dev.jsonl',
        'original_sft_adapter': fin / 'experiments' / 'sft_qwen3_1.7b' / 'full_v1' / 'final_adapter',
        'transfer_adapter': cexp / '01_finqa_sft_adapter' / 'full' / 'final_adapter',
        'direct_adapter': cexp / '02_qwen3_base' / 'full' / 'final_adapter',
        'replay_adapter': cexp / '03_finqa_sft_replay' / 'full' / 'final_adapter',
        'output_root': w / 'financial_portable_runs_v1',
    }.items()}


@dataclass
class Plan:
    task: str
    cwd: Path
    command: list[str]
    required: list[tuple[str, Path, str]]
    output: Path
    env: str  # 'train' or 'inference' or 'data'


def load_config(path: Path) -> dict[str, Path]:
    if not path.is_file():
        raise FileNotFoundError(f'Config missing: {path}. Run `python tools/portable_runner.py init --workspace /root/autodl-tmp`.')
    raw = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError('Config must be a JSON object')
    expected = default_config(Path('/root/autodl-tmp'))
    missing = sorted(set(expected) - set(raw))
    if missing:
        raise ValueError(f'Missing config keys: {missing}')
    ans: dict[str, Path] = {}
    for key in expected:
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f'Invalid path config for {key}')
        p = Path(raw[key]).expanduser()
        if not p.is_absolute():
            raise ValueError(f'Use absolute paths in config: {key}={p}')
        ans[key] = p
    if ans['output_root'].resolve() == REPO.resolve() or REPO.resolve().is_relative_to(ans['output_root'].resolve()):
        raise ValueError('Output root must NOT contain the GitHub source repository')
    return ans


def build_plan(task: str, c: dict[str, Path], limit: int = 0) -> Plan:
    if task not in TASKS:
        raise ValueError(f'Unknown task {task}; use `list`')
    if limit < 0:
        raise ValueError('--limit must be >= 0')
    output_root = c['output_root']
    required: list[tuple[str, Path, str]] = []
    def need(name: str, kind: str = 'file') -> str:
        p = c[name]
        required.append((name, p, kind))
        return str(p)

    def code(relative: str) -> tuple[Path, str]:
        p = REPO / relative
        required.append(('source', p, 'file'))
        return p.parent, p.name

    env = 'train'
    if task.startswith('finqa-sft-'):
        cwd, src = code('finqa/01_lora_sft/train_sft_1p7b.py')
        out = output_root / 'finqa' / 'sft' / ('preflight_v1' if task.endswith('smoke') else 'full_v1')
        cmd = [sys.executable, src, '--model-path', need('base_model', 'dir'),
               '--train-file', need('finqa_sft_train'), '--eval-file', need('finqa_sft_dev'),
               '--output-root', str(output_root / 'finqa' / 'sft'), '--no-wandb']
        if task.endswith('smoke'): cmd.append('--preflight')
    elif task.startswith('finqa-grpo-') or task.startswith('finqa-dynamic-'):
        dynamic = task.startswith('finqa-dynamic-')
        rel = 'finqa/03_dynamic_sampling/train_dynamic_1p7b.py' if dynamic else 'finqa/02_vanilla_grpo/train_grpo_1p7b.py'
        cwd, src = code(rel)
        mode = 'dynamic' if dynamic else 'grpo'
        out = output_root / 'finqa' / mode / ('smoke' if task.endswith('smoke') else 'full')
        cmd = [sys.executable, src, '--base-model-path', need('base_model', 'dir'),
               '--sft-adapter-path', need('original_sft_adapter', 'adapter'),
               '--train-file', need('finqa_grpo_train'), '--output-dir', str(out), '--report-to', 'none']
        if task.endswith('smoke'): cmd.append('--preflight')
    elif task.startswith('conv-') and (task.endswith('-smoke') or task.endswith('-full')):
        variant = 'transfer' if task.startswith('conv-transfer-') else 'direct'
        cwd, src = code('convfinqa/00_common/train_sft.py')
        out = output_root / 'convfinqa' / variant / ('smoke' if task.endswith('smoke') else 'full')
        cmd = [sys.executable, src, '--variant', variant,
               '--base', need('base_model', 'dir'), '--train', need('conv_sft_train'),
               '--dev', need('conv_sft_dev'), '--output-dir', str(out),
               '--grad-accum', '2' if task.endswith('smoke') else '8']
        if variant == 'transfer': cmd.extend(['--finqa-adapter', need('original_sft_adapter', 'adapter')])
        if task.endswith('smoke'): cmd.extend(['--smoke', '--smoke-steps', '5', '--smoke-train-rows', '32', '--smoke-eval-rows', '8'])
        else: cmd.extend(['--epochs', '1', '--learning-rate', '5e-5', '--save-steps', '250'])
    elif task == 'replay-build':
        cwd, src = code('convfinqa/03_replay/01_data/build_replay.py')
        out = output_root / 'convfinqa' / 'replay' / 'data'
        cmd = [sys.executable, src, '--conv', need('conv_sft_train'), '--finqa', need('finqa_sft_train'),
               '--output-dir', str(out), '--finqa-count', '1110', '--seed', '42']
        env = 'data'
    elif task == 'replay-verify':
        cwd, src = code('convfinqa/03_replay/01_data/verify_replay.py')
        out = output_root / 'convfinqa' / 'replay' / 'data' / 'replay_train.jsonl'
        required.append(('replay-data', out, 'file'))
        cmd = [sys.executable, src, '--input', str(out), '--model', need('base_model', 'dir')]
        env = 'train'
    elif task.startswith('replay-') and (task.endswith('-smoke') or task.endswith('-full')):
        cwd, src = code('convfinqa/03_replay/train_replay.py')
        mixed = output_root / 'convfinqa' / 'replay' / 'data' / 'replay_train.jsonl'
        required.append(('replay-data', mixed, 'file'))
        out = output_root / 'convfinqa' / 'replay' / ('smoke' if task.endswith('smoke') else 'full')
        cmd = [sys.executable, src, '--variant', 'transfer', '--base', need('base_model', 'dir'),
               '--finqa-adapter', need('original_sft_adapter', 'adapter'), '--train', str(mixed),
               '--dev', need('conv_sft_dev'), '--output-dir', str(out), '--grad-accum',
               '2' if task.endswith('smoke') else '8', '--max-seq-length', '4096']
        if task.endswith('smoke'): cmd.extend(['--smoke', '--smoke-steps', '5', '--smoke-train-rows', '32', '--smoke-eval-rows', '8'])
        else: cmd.extend(['--epochs', '1', '--learning-rate', '5e-5', '--save-steps', '250'])
    elif task.startswith('conv-') and task.endswith('-oracle'):
        variant = task.removeprefix('conv-').removesuffix('-oracle')
        cwd, src = code('convfinqa/00_common/eval_sft_vllm.py')
        out = output_root / 'evaluation' / 'convfinqa' / f'{variant}_oracle' / (f'smoke_{limit}' if limit else 'full_dev')
        adapter_key = {'transfer': 'transfer_adapter', 'direct': 'direct_adapter', 'replay': 'replay_adapter'}[variant]
        cmd = [sys.executable, src, '--variant', 'direct' if variant == 'direct' else 'transfer',
               '--base', need('base_model', 'dir'), '--input', need('conv_eval_dev'),
               '--adapter', need(adapter_key, 'adapter'), '--output-dir', str(out),
               '--max-samples', str(limit), '--seed', '42']
        env = 'inference'
    elif task.startswith('conv-') and task.endswith('-self'):
        variant = task.removeprefix('conv-').removesuffix('-self')
        cwd, src = code('tools/portable_self_history.py')
        required.append(('evaluator', REPO / 'convfinqa/00_common/eval_self_history_v1.py', 'file'))
        adapter_key = {'transfer': 'transfer_adapter', 'direct': 'direct_adapter', 'replay': 'replay_adapter'}[variant]
        out = output_root / 'evaluation' / 'convfinqa' / f'{variant}_self' / (f'smoke_{limit}' if limit else 'full_dev')
        cmd = [sys.executable, src, '--variant', variant, '--adapter', need(adapter_key, 'adapter'),
               '--base', need('base_model', 'dir'), '--input', need('conv_eval_dev'),
               '--output-dir', str(out), '--max-conversations', str(limit), '--seed', '42']
        env = 'inference'
    elif task.startswith('finqa-') and (task.endswith('-dev') or task.endswith('-test')):
        _, variant, split = task.split('-')
        adapter_key = {'original': 'original_sft_adapter', 'transfer': 'transfer_adapter',
                       'direct': 'direct_adapter', 'replay': 'replay_adapter'}[variant]
        cwd, src = code('finqa/05_frozen_vllm_evaluator/bench_vllm.py')
        out = output_root / 'evaluation' / 'finqa' / variant / split / (f'smoke_{limit}' if limit else 'full')
        cmd = [sys.executable, src, '--base', need('base_model', 'dir'),
               '--adapter', need(adapter_key, 'adapter'), '--input', need(f'finqa_eval_{split}'),
               '--output-dir', str(out), '--max-samples', str(limit),
               '--request-batch-size', '0', '--warmup-samples', '4', '--max-new-tokens', '384',
               '--max-model-len', '4096', '--max-num-seqs', '32',
               '--gpu-memory-utilization', '0.70', '--seed', '42']
        env = 'inference'
    else:
        raise AssertionError(f'Unknown builder for task {task}')
    if out.is_relative_to(REPO):
        raise ValueError('Output would be placed INSIDE GitHub repository; refusing')
    return Plan(task, cwd, cmd, required, out, env)


def validate(plan: Plan, reject_existing: bool = False) -> list[str]:
    issues = []
    for name, p, kind in plan.required:
        ok = p.is_file() if kind == 'file' else p.is_dir() if kind == 'dir' else (p/'adapter_config.json').is_file()
        if not ok:
            issues.append(f'MISSING {kind}: {name} -> {p}')
    if reject_existing and plan.task != 'replay-verify' and plan.output.exists():
        # New runs must never overwrite frozen/history folders.
        issues.append(f'OUTPUT EXISTS (refusing overwrite): {plan.output}')
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG_DEFAULT, help='Local paths JSON; never commit')
    sub = parser.add_subparsers(dest='action', required=True)
    ini = sub.add_parser('init', help='Create config for existing AutoDL layout; no file overwrite')
    ini.add_argument('--workspace', type=Path, required=True)
    sub.add_parser('list', help='Show supported tasks')
    for name in ('check', 'plan', 'run'):
        p = sub.add_parser(name)
        p.add_argument('task', choices=TASKS)
        p.add_argument('--limit', type=int, default=0, help='Evaluation: maximum examples/conversations; 0=all')
    args = parser.parse_args()
    if args.action == 'init':
        path = args.config.expanduser().resolve()
        if path.exists():
            raise FileExistsError(f'Config already exists; edit it manually or pass --config to a new filename: {path}')
        if not args.workspace.is_dir():
            raise NotADirectoryError(args.workspace)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(default_config(args.workspace), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print('CREATED:', path, '\nEdit paths if your layout differs. Local config is gitignored.')
        return
    if args.action == 'list':
        print('\n'.join(TASKS)); return
    c = load_config(args.config)
    plan = build_plan(args.task, c, args.limit)
    issues = validate(plan, reject_existing=args.action == 'run')
    print(f'TASK: {plan.task}\nENV: {plan.env}\nWORKING DIRECTORY: {plan.cwd}\nOUTPUT: {plan.output}')
    print('COMMAND (JSON argv; NOT a shell script):', json.dumps(plan.command, ensure_ascii=False, indent=2))
    if issues:
        for item in issues: print('[ERROR]', item, file=sys.stderr)
        raise SystemExit(2)
    if args.action == 'run':
        env = dict(os.environ)
        env.setdefault('OMP_NUM_THREADS', '8')
        env.setdefault('MKL_NUM_THREADS', '8')
        env.setdefault('TOKENIZERS_PARALLELISM', 'false')
        if plan.env == 'inference': env['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
        else: env.setdefault('WANDB_DISABLED', 'true')
        plan.output.parent.mkdir(parents=True, exist_ok=True)
        print('LAUNCHING (no source files modified)', flush=True)
        proc = subprocess.run(plan.command, cwd=plan.cwd, env=env, check=False)
        raise SystemExit(proc.returncode)
    print('PASS:', args.action.upper(), '(no GPU workload executed)')


if __name__ == '__main__':
    main()
