#!/usr/bin/env python3
import argparse
import hashlib
import inspect
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch
import transformers
import peft
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

DEFAULT_MODEL = "/root/autodl-tmp/models/Qwen3-1.7B-Base"
DEFAULT_TRAIN = "/root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1/sft/finqa_sft_train.jsonl"
DEFAULT_DEV = "/root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1/sft/finqa_sft_dev.jsonl"
DEFAULT_ROOT = "/root/autodl-tmp/project/finqa_posttrain/experiments/sft_qwen3_1.7b"
MAX_SEQ_LENGTH = 2048
RESPONSE_RESERVE = 384
PROMPT_BUDGET = MAX_SEQ_LENGTH - RESPONSE_RESERVE
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"Invalid JSONL {path}:{line_no}: {e}")
    return rows


def get_message(row, role):
    for m in row.get("messages", []):
        if m.get("role") == role:
            return m.get("content", "")
    raise KeyError(f"Missing role={role} for id={row.get('id')}")


def serialize_prompt_target(row):
    system = get_message(row, "system").strip()
    user = get_message(row, "user").strip()
    assistant = get_message(row, "assistant").strip()
    prompt = f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:"
    target = "\n" + assistant
    return prompt, target


class FinQAAssistantOnlyDataset(Dataset):
    def __init__(self, rows, tokenizer, max_seq_length=2048):
        self.items = []
        self.stats = {
            "n": len(rows), "final_lengths": [], "prompt_lengths": [], "target_lengths": [],
            "prompt_truncated_runtime": 0, "target_truncated_runtime": 0,
        }
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise RuntimeError("Tokenizer has no eos_token_id")
        for row in rows:
            prompt, target = serialize_prompt_target(row)
            prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
            target_ids = tokenizer(target, add_special_tokens=False, truncation=False)["input_ids"]
            self.stats["prompt_lengths"].append(len(prompt_ids))
            self.stats["target_lengths"].append(len(target_ids))
            if len(prompt_ids) > PROMPT_BUDGET:
                prompt_ids = prompt_ids[:PROMPT_BUDGET]
                self.stats["prompt_truncated_runtime"] += 1
            available = max_seq_length - len(prompt_ids) - 1
            if available < 1:
                raise RuntimeError(f"No target budget for id={row.get('id')}")
            if len(target_ids) > available:
                target_ids = target_ids[:available]
                self.stats["target_truncated_runtime"] += 1
            input_ids = prompt_ids + target_ids + [eos_id]
            labels = [-100] * len(prompt_ids) + target_ids + [eos_id]
            attention_mask = [1] * len(input_ids)
            if len(input_ids) > max_seq_length:
                raise AssertionError("Sequence overflow")
            if not any(v != -100 for v in labels):
                raise AssertionError("No supervised assistant tokens")
            self.items.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})
            self.stats["final_lengths"].append(len(input_ids))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class CausalLMCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = int(math.ceil(max_len / m) * m)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for x in features:
            pad = max_len - len(x["input_ids"])
            out["input_ids"].append(x["input_ids"] + [pad_id] * pad)
            out["attention_mask"].append(x["attention_mask"] + [0] * pad)
            out["labels"].append(x["labels"] + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


def summarize(vals):
    vals = sorted(vals)
    n = len(vals)
    def pct(p):
        return vals[min(n - 1, max(0, int(round((n - 1) * p))))]
    return {"min": vals[0], "mean": sum(vals) / n, "p50": pct(0.5), "p95": pct(0.95), "max": vals[-1]}


def dataset_report(ds):
    return {
        "n": len(ds),
        "final_length": summarize(ds.stats["final_lengths"]),
        "original_prompt_length": summarize(ds.stats["prompt_lengths"]),
        "original_target_length": summarize(ds.stats["target_lengths"]),
        "prompt_truncated_runtime": ds.stats["prompt_truncated_runtime"],
        "target_truncated_runtime": ds.stats["target_truncated_runtime"],
    }


def resolve_checkpoint(output_dir, value):
    if value is None:
        return None
    if value != "latest":
        if not Path(value).exists():
            raise FileNotFoundError(value)
        return value
    found = []
    for p in Path(output_dir).glob("checkpoint-*"):
        try:
            found.append((int(p.name.split("-")[-1]), p))
        except Exception:
            pass
    if not found:
        raise FileNotFoundError(f"No checkpoint-* in {output_dir}")
    return str(sorted(found)[-1][1])


def build_training_args(kwargs):
    sig = inspect.signature(TrainingArguments.__init__)
    strategy = kwargs.pop("_eval_strategy")
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = strategy
    else:
        kwargs["evaluation_strategy"] = strategy
    return TrainingArguments(**kwargs)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--train-file", default=DEFAULT_TRAIN)
    p.add_argument("--eval-file", default=DEFAULT_DEV)
    p.add_argument("--output-root", default=DEFAULT_ROOT)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--resume-from-checkpoint", default=None, help="Path or latest")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    mode = "preflight_v1" if args.preflight else "full_v1"
    out = Path(args.output_root) / mode
    out.mkdir(parents=True, exist_ok=True)
    if args.preflight and args.resume_from_checkpoint:
        raise ValueError("Preflight cannot resume")

    print("=" * 100)
    print("Qwen3-1.7B FinQA LoRA-SFT", "PREFLIGHT" if args.preflight else "FULL")
    print("=" * 100)
    print("Model         :", args.model_path)
    print("Train         :", args.train_file)
    print("Dev           :", args.eval_file)
    print("Output        :", out)
    print("Max seq       :", MAX_SEQ_LENGTH)
    print("Prompt budget :", PROMPT_BUDGET)
    print("LoRA          : r=8 alpha=16 dropout=0.05")

    for pth in [args.model_path, args.train_file, args.eval_file]:
        if not Path(pth).exists():
            raise FileNotFoundError(pth)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_rows = read_jsonl(args.train_file)
    eval_rows = read_jsonl(args.eval_file)
    if args.preflight:
        train_rows = train_rows[:128]
        eval_rows = eval_rows[:64]
    print(f"Rows: train={len(train_rows)} eval={len(eval_rows)}")

    train_ds = FinQAAssistantOnlyDataset(train_rows, tokenizer, MAX_SEQ_LENGTH)
    eval_ds = FinQAAssistantOnlyDataset(eval_rows, tokenizer, MAX_SEQ_LENGTH)
    train_stats = dataset_report(train_ds)
    eval_stats = dataset_report(eval_ds)
    print("Train stats:", json.dumps(train_stats, indent=2))
    print("Eval stats :", json.dumps(eval_stats, indent=2))

    prompt0, target0 = serialize_prompt_target(train_rows[0])
    with open(out / "serialization_preview.txt", "w", encoding="utf-8") as f:
        f.write("=== PROMPT ===\n" + prompt0)
        f.write("\n\n=== TARGET repr ===\n" + repr(target0))
        f.write("\n\n=== TARGET ===\n" + target0)

    print("\nLoading Qwen3-1.7B base in BF16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
        task_type="CAUSAL_LM", target_modules=LORA_TARGETS,
    ))

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nLoRA parameter summary:")
    model.print_trainable_parameters()
    print(f"total={total:,} trainable={trainable:,} ratio={100*trainable/total:.6f}%")

    report_to = [] if args.no_wandb else ["wandb"]
    kwargs = dict(
        output_dir=str(out), overwrite_output_dir=False,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=8, learning_rate=2e-4,
        lr_scheduler_type="cosine", warmup_ratio=0.05,
        logging_strategy="steps", logging_steps=10,
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        report_to=report_to,
        run_name="qwen3-1.7b-lora-sft-preflight-v1" if args.preflight else "qwen3-1.7b-lora-sft-full-v1",
        seed=42, data_seed=42, bf16=True, fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False, dataloader_num_workers=0,
        optim="adamw_torch", save_safetensors=True,
    )
    if args.preflight:
        kwargs.update(max_steps=12, _eval_strategy="steps", eval_steps=12, save_strategy="steps", save_steps=12)
    else:
        kwargs.update(num_train_epochs=2.0, _eval_strategy="epoch", save_strategy="epoch")
    training_args = build_training_args(kwargs)

    manifest = {
        "stage": "qwen3_1p7b_lora_sft_preflight_v1" if args.preflight else "qwen3_1p7b_lora_sft_full_v1",
        "boundary_fix": {
            "baseline_prompt_ends_exactly_with": "Assistant:",
            "training_prompt_has_trailing_newline": False,
            "assistant_target_begins_with_newline": True,
            "status": "FROZEN_FROM_0.6B",
        },
        "assistant_only_loss": True,
        "model_path": args.model_path,
        "train_file": args.train_file,
        "eval_file": args.eval_file,
        "train_file_sha256": sha256_file(args.train_file),
        "eval_file_sha256": sha256_file(args.eval_file),
        "train_examples_this_run": len(train_ds),
        "eval_examples_this_run": len(eval_ds),
        "epochs": None if args.preflight else 2.0,
        "max_steps": 12 if args.preflight else None,
        "max_seq_length": MAX_SEQ_LENGTH,
        "response_reserve": RESPONSE_RESERVE,
        "prompt_budget": PROMPT_BUDGET,
        "train_batch_size": 1, "eval_batch_size": 1,
        "gradient_accumulation_steps": 8, "effective_batch": 8,
        "learning_rate": 2e-4, "lr_scheduler_type": "cosine", "warmup_ratio": 0.05,
        "dtype": "torch.bfloat16",
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "target_modules": LORA_TARGETS},
        "parameters": {"total": total, "trainable": trainable, "trainable_ratio": trainable/total},
        "train_stats": train_stats, "eval_stats": eval_stats,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "versions": {
            "python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
            "peft": peft.__version__, "platform": platform.platform(),
        },
    }
    with open(out / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=CausalLMCollator(tokenizer),
    )
    resume_ckpt = resolve_checkpoint(out, args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    if resume_ckpt:
        print("Resuming from:", resume_ckpt)

    print("\nStarting training...")
    start = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
    elapsed = time.time() - start
    print("\nRunning final eval with best-loaded model...")
    eval_metrics = trainer.evaluate()

    final_adapter = out / "final_adapter"
    final_adapter.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(final_adapter, safe_serialization=True)
    tokenizer.save_pretrained(final_adapter)

    report = dict(manifest)
    report.update({
        "elapsed_seconds": elapsed, "train_metrics": train_result.metrics, "eval_metrics": eval_metrics,
        "best_model_checkpoint": trainer.state.best_model_checkpoint, "best_metric": trainer.state.best_metric,
        "global_step": trainer.state.global_step, "final_adapter": str(final_adapter),
    })
    with open(out / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 100)
    print("TRAINING COMPLETED")
    print("=" * 100)
    print("Final adapter :", final_adapter)
    print("Report        :", out / "training_report.json")
    print("Best ckpt     :", trainer.state.best_model_checkpoint)
    print("Best eval loss:", trainer.state.best_metric)
    print("Train metrics :", train_result.metrics)
    print("Eval metrics  :", eval_metrics)


if __name__ == "__main__":
    main()
