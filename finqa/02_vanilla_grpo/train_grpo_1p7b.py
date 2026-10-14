# -*- coding: utf-8 -*-
"""
FinQA Full Vanilla GRPO v1

Formal 1.7B GRPO baseline:
- Start from Qwen3-1.7B-Base + ORIGINAL Full SFT adapter.
- Use all 6251 FinQA train prompts for 1 epoch.
- G=4, temperature=0.8, top_p=0.95.
- max prompt=1792, max completion=256.
- correctness reward 1.0 + format reward 0.1.
- loss_type='grpo', scale_rewards='group'.
- beta=0.04 with an EXPLICIT frozen copy of the SFT policy as reference.

Why the explicit reference copy?
TRL 0.18.0 otherwise obtains PEFT reference log-probs by disabling the
adapter, which would compare a pretrained SFT adapter against the bare Base
model. For this project the intended reference is the SFT starting policy.

This script is single-GPU by design.
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
from trl import GRPOConfig, GRPOTrainer

from grpo_data import (
    discover_train_file,
    normalize_row,
    read_jsonl,
    stratified_smoke,
)
from reward_utils import (
    GRPORewardWithDiagnostics,
    summarize_diagnostics,
    summarize_trainer_logs,
)

DEFAULT_FULL_OUTPUT = (
    "/root/autodl-tmp/project/finqa_posttrain/experiments/"
    "grpo_qwen3_1.7b/full_v1"
)
DEFAULT_PREFLIGHT_OUTPUT = (
    "/root/autodl-tmp/project/finqa_posttrain/experiments/"
    "grpo_qwen3_1.7b/preflight_kl_v1"
)


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_adapter_weight_file(adapter_dir: str):
    p = Path(adapter_dir)
    candidates = [
        p / "adapter_model.safetensors",
        p / "adapter_model.bin",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def latest_checkpoint(output_dir: Path):
    candidates = []
    for p in output_dir.glob("checkpoint-*"):
        if p.is_dir():
            m = re.fullmatch(r"checkpoint-(\d+)", p.name)
            if m:
                candidates.append((int(m.group(1)), p))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def resolve_resume(value, output_dir: Path):
    if value is None:
        return None
    if value == "latest":
        p = latest_checkpoint(output_dir)
        if p is None:
            raise FileNotFoundError(
                f"No checkpoint-* found under {output_dir}"
            )
        return str(p)
    p = Path(value)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return str(p)


def choose_dtype():
    if torch.cuda.is_available():
        return (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
    return torch.float32


def model_parameter_stats(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, (trainable / total if total else 0.0)


def snapshot_trainable_parameters(model):
    """CPU snapshot used only in cheap preflight to prove parameters update."""
    return {
        name: param.detach().float().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def compare_trainable_snapshot(model, before):
    max_abs_delta = 0.0
    l2_sq = 0.0
    changed_tensors = 0
    compared_tensors = 0

    for name, param in model.named_parameters():
        if name not in before:
            continue

        compared_tensors += 1
        after = param.detach().float().cpu()
        delta = after - before[name]

        local_max = float(delta.abs().max().item()) if delta.numel() else 0.0
        if local_max > 0.0:
            changed_tensors += 1

        max_abs_delta = max(max_abs_delta, local_max)
        l2_sq += float((delta * delta).sum().item())

    return {
        "compared_trainable_tensors": compared_tensors,
        "changed_trainable_tensors": changed_tensors,
        "max_abs_parameter_delta": max_abs_delta,
        "l2_parameter_delta": l2_sq ** 0.5,
    }


def make_config(args, preflight=False):
    params = inspect.signature(GRPOConfig.__init__).parameters

    kw = {
        "output_dir": args.output_dir,
        "overwrite_output_dir": False,
        "per_device_train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": (1 if preflight else args.logging_steps),
        "logging_strategy": "steps",
        "save_strategy": "steps",
        "save_steps": (
            args.preflight_steps if preflight else args.save_steps
        ),
        "save_total_limit": args.save_total_limit,
        "report_to": (
            [] if args.report_to == "none" else [args.report_to]
        ),
        "run_name": (
            args.run_name + "-preflight"
            if preflight
            else args.run_name
        ),
        "seed": args.seed,
        "data_seed": args.seed,
        "bf16": bool(
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ),
        "fp16": bool(
            torch.cuda.is_available()
            and not torch.cuda.is_bf16_supported()
        ),
        "gradient_checkpointing": True,
        "remove_unused_columns": False,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "beta": args.beta,
        "epsilon": args.epsilon,
        "dataloader_num_workers": 0,
    }

    if preflight:
        kw["max_steps"] = args.preflight_steps
    else:
        kw["num_train_epochs"] = args.epochs

    # Explicit Vanilla GRPO settings.
    if "loss_type" in params:
        kw["loss_type"] = "grpo"

    if "scale_rewards" in params:
        # TRL 0.18 accepts bool; later versions accept "group".
        default = params["scale_rewards"].default
        kw["scale_rewards"] = (
            True if isinstance(default, bool) else "group"
        )

    if "num_iterations" in params:
        kw["num_iterations"] = 1

    if "steps_per_generation" in params:
        kw["steps_per_generation"] = 1

    if "use_vllm" in params:
        kw["use_vllm"] = False

    if "shuffle_dataset" in params:
        kw["shuffle_dataset"] = True

    if "disable_dropout" in params:
        kw["disable_dropout"] = True

    # PEFT + gradient checkpointing compatibility:
    # reentrant checkpointing requires at least one input with requires_grad=True
    # and produced "Gradients will be None" in the first KL preflight.
    # Non-reentrant checkpointing avoids that requirement and is also what we
    # used successfully in the formal SFT stage.
    if "gradient_checkpointing_kwargs" in params:
        kw["gradient_checkpointing_kwargs"] = {
            "use_reentrant": False
        }

    kw = {k: v for k, v in kw.items() if k in params}
    return GRPOConfig(**kw), kw


def check_init_adapter_path(path: str, allow_non_sft_init: bool):
    low = str(path).lower()
    suspicious = any(
        token in low
        for token in [
            "grpo_qwen",
            "/grpo_",
            "smoke_v1/final_adapter",
        ]
    )
    if suspicious and not allow_non_sft_init:
        raise RuntimeError(
            "The initialization adapter path looks like a GRPO/Smoke "
            "checkpoint. Formal Full GRPO must start again from the ORIGINAL "
            "Full SFT adapter. If this is intentional, pass "
            "--allow-non-sft-init."
        )


def prepare_output_dir(
    output_dir: Path,
    resume_checkpoint,
    overwrite: bool,
):
    if resume_checkpoint:
        output_dir.mkdir(parents=True, exist_ok=True)
        return

    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"Output directory is not empty: {output_dir}\n"
                "Use a new directory, resume from a checkpoint, or pass "
                "--overwrite-output-dir intentionally."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def load_policy(base_model_path, adapter_path, dtype):
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    policy = PeftModel.from_pretrained(
        base,
        adapter_path,
        is_trainable=True,
    )

    if hasattr(policy.config, "use_cache"):
        policy.config.use_cache = False

    if hasattr(policy, "enable_input_require_grads"):
        policy.enable_input_require_grads()

    return policy


def load_frozen_sft_reference(
    base_model_path,
    adapter_path,
    dtype,
):
    ref_base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    ref = PeftModel.from_pretrained(
        ref_base,
        adapter_path,
        is_trainable=False,
    )

    for p in ref.parameters():
        p.requires_grad_(False)

    if hasattr(ref.config, "use_cache"):
        ref.config.use_cache = False

    ref.eval()
    return ref


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--base-model-path",
        default="/root/autodl-tmp/models/Qwen3-1.7B-Base",
    )
    p.add_argument(
        "--sft-adapter-path",
        default=(
            "/root/autodl-tmp/project/finqa_posttrain/experiments/"
            "sft_qwen3_1.7b/full_v1/final_adapter"
        ),
    )
    p.add_argument("--train-file", default=None)
    p.add_argument("--output-dir", default=DEFAULT_FULL_OUTPUT)

    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--max-prompt-length", type=int, default=1792)
    p.add_argument("--max-completion-length", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)

    p.add_argument("--train-batch-size", type=int, default=4)
    p.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
    )

    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--warmup-ratio", type=float, default=0.1)

    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--epsilon", type=float, default=0.2)

    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=250)
    p.add_argument("--save-total-limit", type=int, default=2)

    p.add_argument(
        "--report-to",
        choices=["none", "wandb"],
        default="wandb",
    )
    p.add_argument("--wandb-project", default="finqa-posttrain")
    p.add_argument(
        "--run-name",
        default="qwen3-1.7b-vanilla-grpo-full-v1",
    )

    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="checkpoint-* path or 'latest'",
    )
    p.add_argument(
        "--overwrite-output-dir",
        action="store_true",
    )
    p.add_argument(
        "--allow-non-sft-init",
        action="store_true",
    )

    # Cheap formal-config preflight. It NEVER feeds the Full run.
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--preflight-prompts", type=int, default=32)
    p.add_argument("--preflight-steps", type=int, default=8)

    args = p.parse_args()

    if args.train_batch_size % args.num_generations != 0:
        raise SystemExit(
            f"--train-batch-size ({args.train_batch_size}) must be "
            f"divisible by --num-generations ({args.num_generations})."
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise SystemExit(
            "Full Vanilla GRPO v1 is intentionally single-GPU. "
            "WORLD_SIZE must be 1."
        )

    check_init_adapter_path(
        args.sft_adapter_path,
        args.allow_non_sft_init,
    )

    # Preflight has a completely separate output path.
    if args.preflight and args.output_dir == DEFAULT_FULL_OUTPUT:
        args.output_dir = DEFAULT_PREFLIGHT_OUTPUT

    out = Path(args.output_dir)

    # Resolve resume only for the formal run.
    if args.preflight and args.resume_from_checkpoint:
        raise RuntimeError(
            "--preflight cannot be combined with --resume-from-checkpoint."
        )

    resume_checkpoint = resolve_resume(
        args.resume_from_checkpoint,
        out,
    )

    prepare_output_dir(
        out,
        resume_checkpoint,
        args.overwrite_output_dir,
    )

    set_seed(args.seed)

    if args.report_to == "wandb":
        os.environ.setdefault(
            "WANDB_PROJECT",
            args.wandb_project,
        )

    train_file = discover_train_file(args.train_file)
    raw = read_jsonl(train_file)
    normalized = [
        normalize_row(row, i)
        for i, row in enumerate(raw)
    ]

    if args.preflight:
        train_rows = stratified_smoke(
            normalized,
            args.preflight_prompts,
            args.seed,
        )
    else:
        train_rows = normalized

    if not train_rows:
        raise RuntimeError("No GRPO training rows loaded.")

    if len(normalized) != 6251:
        print(
            f"WARNING: expected 6251 FinQA train rows, "
            f"found {len(normalized)}"
        )

    difficulty_counts = {}
    for row in train_rows:
        difficulty_counts[row["difficulty"]] = (
            difficulty_counts.get(row["difficulty"], 0) + 1
        )

    print("=" * 100)
    print(
        "Vanilla GRPO KL Preflight"
        if args.preflight
        else "FULL Vanilla GRPO v1"
    )
    print("=" * 100)
    print("Base model       :", args.base_model_path)
    print("SFT init adapter :", args.sft_adapter_path)
    print("Train file       :", train_file)
    print("Rows total       :", len(normalized))
    print("Rows this run    :", len(train_rows))
    print("Difficulty       :", difficulty_counts)
    print("G                :", args.num_generations)
    print("max prompt       :", args.max_prompt_length)
    print("max completion   :", args.max_completion_length)
    print("temperature      :", args.temperature)
    print("top_p            :", args.top_p)
    print("beta             :", args.beta)
    print("epsilon          :", args.epsilon)
    print("epochs           :", args.epochs)
    print("resume           :", resume_checkpoint)

    assert train_rows[0]["prompt"].endswith("Assistant:")
    assert not train_rows[0]["prompt"].endswith("Assistant:\n")

    # Reproducibility / contamination guard.
    adapter_weight = find_adapter_weight_file(
        args.sft_adapter_path
    )
    adapter_sha = (
        sha256_file(adapter_weight)
        if adapter_weight is not None
        else None
    )

    dtype = choose_dtype()

    tok = AutoTokenizer.from_pretrained(
        args.base_model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("\nLoading trainable policy from ORIGINAL SFT adapter...")
    policy = load_policy(
        args.base_model_path,
        args.sft_adapter_path,
        dtype,
    )

    total, trainable, ratio = model_parameter_stats(policy)
    print(f"Policy total params     : {total:,}")
    print(f"Policy trainable params : {trainable:,}")
    print(f"Policy trainable ratio  : {ratio:.6%}")

    if trainable <= 0:
        raise RuntimeError(
            "SFT adapter did not load as trainable."
        )

    reward = GRPORewardWithDiagnostics(
        tokenizer=tok,
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        append_existing=bool(resume_checkpoint),
    )

    # TRL 0.18 expects a normal function exposing __name__.
    def finqa_reward_func(completions, **kwargs):
        return reward(
            completions=completions,
            **kwargs,
        )

    config, config_used = make_config(
        args,
        preflight=args.preflight,
    )

    ds = Dataset.from_list(train_rows)

    trainer_kwargs = {
        "model": policy,
        "reward_funcs": finqa_reward_func,
        "args": config,
        "train_dataset": ds,
    }

    trainer_params = inspect.signature(
        GRPOTrainer.__init__
    ).parameters

    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tok
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tok

    print("\nInitializing GRPOTrainer...")
    trainer = GRPOTrainer(**trainer_kwargs)

    # Second PEFT compatibility guard. Some Trainer setup paths may reconfigure
    # gradient checkpointing after the model was initially prepared.
    if hasattr(trainer.model, "enable_input_require_grads"):
        trainer.model.enable_input_require_grads()

    reference_mode = "none_beta_zero"

    # CRITICAL compatibility fix:
    # TRL 0.18.0 with an already-loaded PeftModel and beta>0 otherwise
    # computes reference log-probs with adapters disabled, i.e. bare Base.
    # We explicitly inject a frozen Base+SFT reference instead.
    if args.beta > 0.0:
        print(
            "\nLoading EXPLICIT frozen SFT reference policy "
            "for KL..."
        )
        ref = load_frozen_sft_reference(
            args.base_model_path,
            args.sft_adapter_path,
            dtype,
        )

        ref_total, ref_trainable, _ = model_parameter_stats(ref)
        if ref_trainable != 0:
            raise RuntimeError(
                f"Reference policy has {ref_trainable} trainable params."
            )

        # Same preparation path TRL uses for a normal reference model.
        ref = trainer.accelerator.prepare_model(
            ref,
            evaluation_mode=True,
        )
        trainer.ref_model = ref

        if trainer.ref_model is None:
            raise RuntimeError(
                "Failed to install explicit SFT reference model."
            )

        reference_mode = "explicit_frozen_sft_copy"

        print(f"Reference total params  : {ref_total:,}")
        print("Reference trainable      : 0")
        print("Reference mode           :", reference_mode)

    run_config = {
        "stage": (
            "vanilla_grpo_preflight_v1"
            if args.preflight
            else "vanilla_grpo_full_v1"
        ),
        "base_model_path": args.base_model_path,
        "sft_adapter_path": args.sft_adapter_path,
        "sft_adapter_weight_file": (
            str(adapter_weight)
            if adapter_weight is not None
            else None
        ),
        "sft_adapter_sha256": adapter_sha,
        "train_file": train_file,
        "rows_total": len(normalized),
        "rows_this_run": len(train_rows),
        "difficulty_counts": difficulty_counts,
        "epochs": args.epochs,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": (
            args.gradient_accumulation_steps
        ),
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "reward": {
            "correctness_weight": 1.0,
            "format_weight": 0.1,
        },
        "beta": args.beta,
        "epsilon": args.epsilon,
        "reference_mode": reference_mode,
        "reference_is_initial_sft_policy": args.beta > 0.0,
        "requested_loss_type": "grpo",
        "requested_scale_rewards": "group",
        "resume_from_checkpoint": resume_checkpoint,
        "grpo_config_kwargs_actually_used": config_used,
        "versions": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "trl": trl.__version__,
            "platform": platform.platform(),
        },
    }

    (out / "run_config.json").write_text(
        json.dumps(
            run_config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nStarting "
        + (
            "KL preflight..."
            if args.preflight
            else "FULL Vanilla GRPO..."
        )
    )

    preflight_param_snapshot = (
        snapshot_trainable_parameters(trainer.model)
        if args.preflight
        else None
    )

    t0 = time.time()
    result = trainer.train(
        resume_from_checkpoint=resume_checkpoint
    )
    elapsed = time.time() - t0

    final_adapter = out / "final_adapter"
    trainer.save_model(str(final_adapter))
    tok.save_pretrained(str(final_adapter))

    log_history = trainer.state.log_history

    (out / "trainer_log_history.json").write_text(
        json.dumps(
            log_history,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    diagnostics = summarize_diagnostics(out)
    trainer_summary = summarize_trainer_logs(log_history)

    parameter_update_check = None
    if args.preflight and preflight_param_snapshot is not None:
        parameter_update_check = compare_trainable_snapshot(
            trainer.model,
            preflight_param_snapshot,
        )

    report = {
        **diagnostics,
        "elapsed_seconds": elapsed,
        "train_metrics": result.metrics,
        "trainer_metrics_summary": trainer_summary,
        "global_step": trainer.state.global_step,
        "final_adapter": str(final_adapter),
        "reference_mode": reference_mode,
        "beta": args.beta,
        "sft_adapter_sha256": adapter_sha,
        "parameter_update_check": parameter_update_check,
    }

    report_name = (
        "preflight_report.json"
        if args.preflight
        else "full_report.json"
    )

    (out / report_name).write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    if args.beta > 0.0 and not trainer_summary.get(
        "kl_seen",
        False,
    ):
        raise RuntimeError(
            "beta > 0 but no `kl` metric appeared in trainer logs. "
            "Do not treat this run as a valid formal KL-regularized GRPO run."
        )

    if args.preflight:
        mixed_ratio = float(diagnostics.get("mixed_group_ratio", 0.0) or 0.0)

        if parameter_update_check is None:
            raise RuntimeError(
                "Preflight parameter-update check was not produced."
            )

        if mixed_ratio > 0.0 and (
            parameter_update_check["max_abs_parameter_delta"] <= 0.0
            or parameter_update_check["changed_trainable_tensors"] <= 0
        ):
            raise RuntimeError(
                "A mixed-reward group occurred, but no trainable parameter "
                "change was detected. Do NOT start Full GRPO."
            )

    print("\n" + "=" * 100)
    print(
        "GRPO PREFLIGHT COMPLETED"
        if args.preflight
        else "FULL VANILLA GRPO COMPLETED"
    )
    print("=" * 100)
    print("Final adapter :", final_adapter)
    print("Report        :", out / report_name)
    print(
        "Diagnostics   :",
        out / "diagnostics_summary.json",
    )
    print(
        "Trainer logs  :",
        out / "trainer_log_history.json",
    )

    print("\nKey diagnostics:")
    for key in [
        "reward_mean",
        "reward_std",
        "correctness_reward_mean",
        "format_reward_mean",
        "zero_variance_group_ratio",
        "all_correct_group_ratio",
        "all_wrong_group_ratio",
        "mixed_group_ratio",
        "unique_completion_ratio_mean",
        "response_length_mean",
        "response_length_p95",
        "max_length_hit_rate",
    ]:
        print(
            f"  {key:32s}: "
            f"{diagnostics.get(key)}"
        )

    if parameter_update_check is not None:
        print("\nParameter update check:")
        for key, value in parameter_update_check.items():
            print(f"  {key:32s}: {value}")

    print("\nTrainer stability:")
    for key in [
        "kl",
        "grad_norm",
        "clip_ratio/region_mean",
        "frac_reward_zero_std",
    ]:
        if key in trainer_summary:
            print(
                f"  {key:32s}: "
                f"{trainer_summary[key]}"
            )


if __name__ == "__main__":
    main()
