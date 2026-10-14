# -*- coding: utf-8 -*-
"""FinQA GRPO + Capped Dynamic Sampling v1.

Fair ablation against frozen Vanilla GRPO v1: only prompt sampling changes.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import re
import shutil
import sys
import time
from pathlib import Path

import torch
import transformers
import peft
import trl
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig

from grpo_data import discover_train_file, normalize_row, read_jsonl, stratified_smoke
from reward_utils import GRPORewardWithDiagnostics, summarize_diagnostics, summarize_trainer_logs
from dynamic_sampling_trainer import CappedDynamicSamplingGRPOTrainer
from dynamic_summary import summarize_dynamic_sampling

DEFAULT_FULL_OUTPUT = (
    '/root/autodl-tmp/project/finqa_posttrain/experiments/'
    'grpo_dynamic_qwen3_1.7b/full_v1'
)
DEFAULT_PREFLIGHT_OUTPUT = (
    '/root/autodl-tmp/project/finqa_posttrain/experiments/'
    'grpo_dynamic_qwen3_1.7b/preflight_v1'
)


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def find_adapter_weight_file(adapter_dir):
    p = Path(adapter_dir)
    for c in [p / 'adapter_model.safetensors', p / 'adapter_model.bin']:
        if c.exists():
            return c
    return None


def latest_checkpoint(output_dir: Path):
    items = []
    for p in output_dir.glob('checkpoint-*'):
        if p.is_dir():
            m = re.fullmatch(r'checkpoint-(\d+)', p.name)
            if m:
                items.append((int(m.group(1)), p))
    return max(items, key=lambda x: x[0])[1] if items else None


def resolve_resume(value, output_dir):
    if value is None:
        return None
    if value == 'latest':
        p = latest_checkpoint(output_dir)
        if p is None:
            raise FileNotFoundError(f'No checkpoint-* found under {output_dir}')
        return str(p)
    p = Path(value)
    if not p.exists():
        raise FileNotFoundError(f'Checkpoint not found: {p}')
    return str(p)


def choose_dtype():
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def model_parameter_stats(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, trainable / total if total else 0.0


def snapshot_trainable_parameters(model):
    return {
        n: p.detach().float().cpu().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }


def compare_trainable_snapshot(model, before):
    changed = 0
    compared = 0
    max_abs = 0.0
    l2_sq = 0.0
    for n, p in model.named_parameters():
        if n not in before:
            continue
        compared += 1
        delta = p.detach().float().cpu() - before[n]
        local = float(delta.abs().max().item()) if delta.numel() else 0.0
        if local > 0:
            changed += 1
        max_abs = max(max_abs, local)
        l2_sq += float((delta * delta).sum().item())
    return {
        'compared_trainable_tensors': compared,
        'changed_trainable_tensors': changed,
        'max_abs_parameter_delta': max_abs,
        'l2_parameter_delta': l2_sq ** 0.5,
    }


def make_config(args, preflight=False):
    params = inspect.signature(GRPOConfig.__init__).parameters
    kw = {
        'output_dir': args.output_dir,
        'overwrite_output_dir': False,
        'per_device_train_batch_size': args.train_batch_size,
        'gradient_accumulation_steps': 1,
        'learning_rate': args.learning_rate,
        'warmup_ratio': args.warmup_ratio,
        'logging_steps': 1 if preflight else args.logging_steps,
        'logging_strategy': 'steps',
        'save_strategy': 'steps',
        'save_steps': args.preflight_steps if preflight else args.save_steps,
        'save_total_limit': args.save_total_limit,
        'max_grad_norm': 1.0,  # frozen Transformers default; do not change the recipe
        'report_to': [] if args.report_to == 'none' else [args.report_to],
        'run_name': args.run_name + ('-preflight' if preflight else ''),
        'seed': args.seed,
        'data_seed': args.seed,
        'bf16': bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        'fp16': bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
        'gradient_checkpointing': True,
        'remove_unused_columns': False,
        'num_generations': args.num_generations,
        'max_prompt_length': args.max_prompt_length,
        'max_completion_length': args.max_completion_length,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'beta': args.beta,
        'epsilon': args.epsilon,
        'dataloader_num_workers': 0,
    }
    if preflight:
        kw['max_steps'] = args.preflight_steps
    else:
        kw['num_train_epochs'] = args.epochs
    if 'loss_type' in params:
        kw['loss_type'] = 'grpo'
    if 'scale_rewards' in params:
        default = params['scale_rewards'].default
        kw['scale_rewards'] = True if isinstance(default, bool) else 'group'
    if 'num_iterations' in params:
        kw['num_iterations'] = 1
    if 'steps_per_generation' in params:
        kw['steps_per_generation'] = 1
    if 'use_vllm' in params:
        kw['use_vllm'] = False
    if 'shuffle_dataset' in params:
        kw['shuffle_dataset'] = True
    if 'disable_dropout' in params:
        kw['disable_dropout'] = True
    if 'gradient_checkpointing_kwargs' in params:
        kw['gradient_checkpointing_kwargs'] = {'use_reentrant': False}
    kw = {k: v for k, v in kw.items() if k in params}
    return GRPOConfig(**kw), kw


def prepare_output_dir(out, resume, overwrite):
    if resume:
        out.mkdir(parents=True, exist_ok=True)
        return
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f'Output dir is not empty: {out}\n'
                'Use another output dir, resume, or intentionally pass '
                '--overwrite-output-dir.'
            )
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def load_policy(base_model_path, adapter_path, dtype):
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    policy = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    if hasattr(policy.config, 'use_cache'):
        policy.config.use_cache = False
    if hasattr(policy, 'enable_input_require_grads'):
        policy.enable_input_require_grads()
    return policy


def load_frozen_sft_reference(base_model_path, adapter_path, dtype):
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    ref = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    for p in ref.parameters():
        p.requires_grad_(False)
    if hasattr(ref.config, 'use_cache'):
        ref.config.use_cache = False
    ref.eval()
    return ref


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-model-path', default='/root/autodl-tmp/models/Qwen3-1.7B-Base')
    p.add_argument(
        '--sft-adapter-path',
        default=(
            '/root/autodl-tmp/project/finqa_posttrain/experiments/'
            'sft_qwen3_1.7b/full_v1/final_adapter'
        ),
    )
    p.add_argument('--train-file', default=None)
    p.add_argument('--output-dir', default=DEFAULT_FULL_OUTPUT)

    p.add_argument('--epochs', type=float, default=1.0)
    p.add_argument('--num-generations', type=int, default=4)
    p.add_argument('--max-prompt-length', type=int, default=1792)
    p.add_argument('--max-completion-length', type=int, default=256)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--train-batch-size', type=int, default=4)
    p.add_argument('--learning-rate', type=float, default=5e-6)
    p.add_argument('--warmup-ratio', type=float, default=0.1)
    p.add_argument('--beta', type=float, default=0.04)
    p.add_argument('--epsilon', type=float, default=0.2)
    p.add_argument('--max-resample-attempts', type=int, default=4)

    p.add_argument('--logging-steps', type=int, default=10)
    p.add_argument('--save-steps', type=int, default=250)
    p.add_argument('--save-total-limit', type=int, default=2)
    p.add_argument('--report-to', choices=['none', 'wandb'], default='wandb')
    p.add_argument('--wandb-project', default='finqa-posttrain')
    p.add_argument('--run-name', default='qwen3-1.7b-grpo-dynamic-sampling-v1')
    p.add_argument('--seed', type=int, default=42)

    p.add_argument('--resume-from-checkpoint', default=None)
    p.add_argument('--overwrite-output-dir', action='store_true')
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--preflight-prompts', type=int, default=64)
    p.add_argument('--preflight-steps', type=int, default=12)
    args = p.parse_args()

    if trl.__version__ != '0.18.0':
        raise RuntimeError(
            f'Dynamic Sampling v1 is version-locked to TRL 0.18.0; found {trl.__version__}.'
        )
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('Dynamic Sampling v1 is single-GPU only.')
    if args.train_batch_size != args.num_generations:
        raise RuntimeError('--train-batch-size must equal --num-generations (default 4).')
    if args.max_resample_attempts < 1:
        raise RuntimeError('--max-resample-attempts must be >= 1')

    low = args.sft_adapter_path.lower()
    if 'grpo' in low or 'smoke' in low or 'preflight' in low:
        raise RuntimeError(
            'Dynamic Sampling must start from the ORIGINAL Full SFT adapter, '
            'not Vanilla GRPO / Smoke / Preflight.'
        )

    if args.preflight and args.output_dir == DEFAULT_FULL_OUTPUT:
        args.output_dir = DEFAULT_PREFLIGHT_OUTPUT

    out = Path(args.output_dir)
    resume = resolve_resume(args.resume_from_checkpoint, out)
    prepare_output_dir(out, resume, args.overwrite_output_dir)

    set_seed(args.seed)
    if args.report_to == 'wandb':
        os.environ.setdefault('WANDB_PROJECT', args.wandb_project)

    train_file = discover_train_file(args.train_file)
    raw = read_jsonl(train_file)
    normalized = [normalize_row(row, i) for i, row in enumerate(raw)]
    if len(normalized) != 6251:
        raise RuntimeError(f'Expected frozen FinQA Train size 6251, got {len(normalized)}')

    train_rows = (
        stratified_smoke(normalized, args.preflight_prompts, args.seed)
        if args.preflight else normalized
    )
    if not train_rows:
        raise RuntimeError('No training rows loaded.')
    assert train_rows[0]['prompt'].endswith('Assistant:')
    assert not train_rows[0]['prompt'].endswith('Assistant:\n')

    print('=' * 100)
    print('Dynamic Sampling Preflight' if args.preflight else 'FULL GRPO + Dynamic Sampling v1')
    print('=' * 100)
    print('Base model             :', args.base_model_path)
    print('SFT init adapter       :', args.sft_adapter_path)
    print('Train file             :', train_file)
    print('Rows total             :', len(normalized))
    print('Rows in dataloader     :', len(train_rows))
    print('G                      :', args.num_generations)
    print('max resample attempts  :', args.max_resample_attempts)
    print('temperature            :', args.temperature)
    print('top_p                  :', args.top_p)
    print('beta                   :', args.beta)
    print('epsilon                :', args.epsilon)
    print('learning rate          :', args.learning_rate)
    print('epochs                 :', args.epochs)
    print('resume                 :', resume)

    adapter_weight = find_adapter_weight_file(args.sft_adapter_path)
    if adapter_weight is None:
        raise FileNotFoundError(f'SFT adapter weights missing: {args.sft_adapter_path}')
    adapter_sha = sha256_file(adapter_weight)
    dtype = choose_dtype()
    print('Max grad norm         : 1.0 (frozen Trainer default)')

    tok = AutoTokenizer.from_pretrained(
        args.base_model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    tok.padding_side = 'left'
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print('\nLoading trainable policy from ORIGINAL SFT adapter...')
    policy = load_policy(args.base_model_path, args.sft_adapter_path, dtype)
    total, trainable, ratio = model_parameter_stats(policy)
    print(f'Policy total params     : {total:,}')
    print(f'Policy trainable params : {trainable:,}')
    print(f'Policy trainable ratio  : {ratio:.6%}')

    reward = GRPORewardWithDiagnostics(
        tokenizer=tok,
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        append_existing=bool(resume),
    )

    def finqa_reward_func(completions, **kwargs):
        return reward(completions=completions, **kwargs)

    config, config_used = make_config(args, preflight=args.preflight)
    ds = Dataset.from_list(train_rows)

    print('\nInitializing Dynamic Sampling GRPOTrainer...')
    trainer = CappedDynamicSamplingGRPOTrainer(
        model=policy,
        reward_funcs=finqa_reward_func,
        args=config,
        train_dataset=ds,
        processing_class=tok,
        dynamic_pool_rows=normalized,
        diagnostic_reward=reward,
        dynamic_output_dir=args.output_dir,
        max_resample_attempts=args.max_resample_attempts,
        dynamic_seed=args.seed,
    )
    if hasattr(trainer.model, 'enable_input_require_grads'):
        trainer.model.enable_input_require_grads()

    print('\nLoading EXPLICIT frozen SFT reference policy for KL...')
    ref = load_frozen_sft_reference(args.base_model_path, args.sft_adapter_path, dtype)
    ref_total, ref_trainable, _ = model_parameter_stats(ref)
    if ref_trainable != 0:
        raise RuntimeError('Reference policy unexpectedly has trainable parameters.')
    ref = trainer.accelerator.prepare_model(ref, evaluation_mode=True)
    trainer.ref_model = ref
    print(f'Reference total params  : {ref_total:,}')
    print('Reference trainable      : 0')
    print('Reference mode           : explicit_frozen_sft_copy')

    run_config = {
        'stage': 'grpo_dynamic_sampling_preflight_v1' if args.preflight else 'grpo_dynamic_sampling_full_v1',
        'only_algorithmic_change_vs_vanilla': 'capped_dynamic_prompt_resampling',
        'base_model_path': args.base_model_path,
        'sft_adapter_path': args.sft_adapter_path,
        'sft_adapter_sha256': adapter_sha,
        'train_file': train_file,
        'rows_total': len(normalized),
        'rows_in_dataloader': len(train_rows),
        'epochs': args.epochs,
        'num_generations': args.num_generations,
        'max_prompt_length': args.max_prompt_length,
        'max_completion_length': args.max_completion_length,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'train_batch_size': args.train_batch_size,
        'learning_rate': args.learning_rate,
        'warmup_ratio': args.warmup_ratio,
        'beta': args.beta,
        'epsilon': args.epsilon,
        'max_grad_norm': 1.0,
        'loss_type': 'grpo',
        'scale_rewards': 'group',
        'reward': {'correctness_weight': 1.0, 'format_weight': 0.1},
        'dynamic_sampling': {
            'criterion': 'keep only correctness-mixed groups when possible',
            'max_resample_attempts': args.max_resample_attempts,
            'fallback_after_cap': True,
            'replacement_pool': 'all 6251 FinQA Train prompts',
        },
        'reference_mode': 'explicit_frozen_sft_copy',
        'resume_from_checkpoint': resume,
        'grpo_config_kwargs_actually_used': config_used,
        'versions': {
            'python': sys.version,
            'torch': torch.__version__,
            'transformers': transformers.__version__,
            'peft': peft.__version__,
            'trl': trl.__version__,
            'platform': platform.platform(),
        },
    }
    (out / 'run_config.json').write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    param_snapshot = snapshot_trainable_parameters(trainer.model) if args.preflight else None

    print('\nStarting Dynamic Sampling training...')
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
    elapsed = time.time() - t0

    final_adapter = out / 'final_adapter'
    trainer.save_model(str(final_adapter))
    tok.save_pretrained(str(final_adapter))

    log_history = trainer.state.log_history
    (out / 'trainer_log_history.json').write_text(
        json.dumps(log_history, ensure_ascii=False, indent=2, default=str), encoding='utf-8'
    )

    candidate_diagnostics = summarize_diagnostics(out)
    trainer_summary = summarize_trainer_logs(log_history)
    dynamic_summary = summarize_dynamic_sampling(
        out, num_generations=args.num_generations, vanilla_group_budget=6251
    )

    parameter_update_check = None
    if param_snapshot is not None:
        parameter_update_check = compare_trainable_snapshot(trainer.model, param_snapshot)

    report = {
        'elapsed_seconds': elapsed,
        'global_step': trainer.state.global_step,
        'train_metrics': result.metrics,
        'trainer_metrics_summary': trainer_summary,
        'candidate_diagnostics': candidate_diagnostics,
        'dynamic_sampling_summary': dynamic_summary,
        'parameter_update_check': parameter_update_check,
        'final_adapter': str(final_adapter),
        'reference_mode': 'explicit_frozen_sft_copy',
        'sft_adapter_sha256': adapter_sha,
    }
    report_name = 'preflight_report.json' if args.preflight else 'full_report.json'
    (out / report_name).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8'
    )

    if args.beta > 0 and not trainer_summary.get('kl_seen', False):
        raise RuntimeError('beta > 0 but no KL metric was logged.')

    if args.preflight:
        if dynamic_summary['total_candidate_groups'] <= dynamic_summary['optimizer_steps']:
            raise RuntimeError(
                'Preflight rejected no zero-variance group. Increase --preflight-steps '
                'before treating Dynamic Sampling as validated.'
            )
        if dynamic_summary['accepted_mixed_groups'] <= 0:
            raise RuntimeError('No informative mixed group was accepted.')
        if parameter_update_check is None or parameter_update_check['changed_trainable_tensors'] <= 0:
            raise RuntimeError('No trainable parameter update detected.')

    print('\n' + '=' * 100)
    print('DYNAMIC SAMPLING PREFLIGHT COMPLETED' if args.preflight else 'FULL GRPO + DYNAMIC SAMPLING COMPLETED')
    print('=' * 100)
    print('Final adapter   :', final_adapter)
    print('Report          :', out / report_name)
    print('Dynamic summary :', out / 'dynamic_sampling_summary.json')
    print('Events          :', out / 'dynamic_sampling_events.jsonl')

    print('\nDynamic Sampling key statistics:')
    for k in [
        'optimizer_steps',
        'total_candidate_groups',
        'total_candidate_rollouts',
        'filtered_zero_variance_groups',
        'accepted_mixed_groups',
        'fallback_zero_variance_groups',
        'effective_update_ratio',
        'fallback_ratio',
        'candidate_mixed_ratio',
        'candidate_zero_variance_ratio',
        'sampling_group_multiplier',
        'candidate_rollout_multiplier_vs_vanilla_6251xG',
    ]:
        print(f'  {k:44s}: {dynamic_summary.get(k)}')

    if parameter_update_check:
        print('\nParameter update check:')
        for k, v in parameter_update_check.items():
            print(f'  {k:32s}: {v}')

    print('\nTrainer stability:')
    for k in ['kl', 'grad_norm', 'clip_ratio/region_mean', 'frac_reward_zero_std']:
        if k in trainer_summary:
            print(f'  {k:32s}: {trainer_summary[k]}')


if __name__ == '__main__':
    main()
