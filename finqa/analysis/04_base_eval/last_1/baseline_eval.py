# -*- coding: utf-8 -*-
"""
Qwen / Hugging Face baseline evaluation for frozen FinQA eval JSONL.

Designed for:
- Qwen3-0.6B-Base engineering baseline
- later reuse with the final ~1.5B model by changing --model-path

Important:
- deterministic greedy decoding by default
- no gold answer is inserted into the prompt
- predictions are written incrementally
- --resume can continue an interrupted run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from answer_normalizer import (
    compare_answer_strings,
    extract_answer_from_response,
)
from metrics import compute_metrics


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_existing(path: Path):
    if not path.exists():
        return [], set()

    rows = []
    ids = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            if row.get("id") is not None:
                ids.add(row["id"])

    return rows, ids


def manual_serialize(messages: List[Dict[str, str]]) -> str:
    """
    Stable fallback format for Base models without a chat template.
    Keep this format fixed for fair Base/SFT/GRPO comparisons.
    """
    parts = []

    for msg in messages:
        role = str(msg.get("role", "")).lower()
        content = str(msg.get("content", "")).strip()

        if role == "system":
            parts.append(f"System:\n{content}")
        elif role == "user":
            parts.append(f"User:\n{content}")
        elif role == "assistant":
            parts.append(f"Assistant:\n{content}")
        else:
            parts.append(f"{role.title()}:\n{content}")

    parts.append("Assistant:")
    return "\n\n".join(parts)


def render_prompt(tokenizer, messages, prompt_format: str):
    """
    prompt_format:
      auto   -> use tokenizer chat template if available, else manual
      chat   -> require chat template
      manual -> stable System/User/Assistant text
    """
    if prompt_format == "manual":
        return manual_serialize(messages), "manual"

    if prompt_format in {"auto", "chat"}:
        has_template = bool(getattr(tokenizer, "chat_template", None))

        if has_template:
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return text, "chat_template"
            except Exception:
                if prompt_format == "chat":
                    raise

        if prompt_format == "chat":
            raise RuntimeError(
                "Tokenizer has no usable chat_template. "
                "Use --prompt-format manual or auto."
            )

    return manual_serialize(messages), "manual"


def choose_dtype(name: str):
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16

    # auto
    if torch.cuda.is_available():
        # Ampere/Ada GPUs support bf16 well.
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    return torch.float32


def score_response(response: str, gold_answer: str, args) -> dict:
    pred_answer, has_tag = extract_answer_from_response(
        response,
        allow_fallback=True,
    )

    correct, match_reason = compare_answer_strings(
        pred_answer,
        gold_answer,
        numeric_rel_tol=args.numeric_rel_tol,
        numeric_abs_tol=args.numeric_abs_tol,
        percent_abs_tol=args.percent_abs_tol,
    )

    return {
        "prediction_answer": pred_answer,
        "gold_answer": gold_answer,
        "correct": bool(correct),
        "format_ok": bool(has_tag),
        "strict_correct": bool(correct and has_tag),
        "match_reason": match_reason,
    }


def batch_iter(rows, batch_size):
    for i in range(0, len(rows), batch_size):
        yield rows[i:i + batch_size]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 means all examples.",
    )

    parser.add_argument(
        "--prompt-format",
        choices=["auto", "chat", "manual"],
        default="auto",
    )

    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--print-every", type=int, default=20)

    parser.add_argument("--numeric-rel-tol", type=float, default=1e-3)
    parser.add_argument("--numeric-abs-tol", type=float, default=1e-2)
    parser.add_argument("--percent-abs-tol", type=float, default=0.10)

    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / "predictions.jsonl"
    metrics_path = out_dir / "metrics.json"
    errors_path = out_dir / "errors.jsonl"
    run_config_path = out_dir / "run_config.json"

    rows = read_jsonl(args.input)

    if args.max_samples > 0:
        rows = rows[:args.max_samples]

    existing_rows, completed_ids = (
        load_existing(pred_path)
        if args.resume
        else ([], set())
    )

    todo = [
        row for row in rows
        if row.get("id") not in completed_ids
    ]

    print("=" * 80, flush=True)
    print("FinQA Baseline Evaluation", flush=True)
    print("=" * 80, flush=True)
    print("Model       :", args.model_path, flush=True)
    print("Input       :", args.input, flush=True)
    print("Output      :", out_dir, flush=True)
    print("Total       :", len(rows), flush=True)
    print("Already done:", len(completed_ids), flush=True)
    print("To generate :", len(todo), flush=True)
    print("CUDA        :", torch.cuda.is_available(), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = choose_dtype(args.dtype)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if not torch.cuda.is_available():
        model = model.to("cpu")

    model.eval()

    # Detect actual formatting mode using the first available example.
    sample_for_format = todo[0] if todo else (rows[0] if rows else None)

    if sample_for_format:
        _, actual_prompt_format = render_prompt(
            tokenizer,
            sample_for_format["prompt"],
            args.prompt_format,
        )
    else:
        actual_prompt_format = "unknown"

    run_config = {
        "model_path": args.model_path,
        "input": args.input,
        "num_examples_requested": len(rows),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "requested_prompt_format": args.prompt_format,
        "actual_prompt_format": actual_prompt_format,
        "dtype": str(dtype),
        "seed": args.seed,
        "do_sample": False,
        "numeric_rel_tol": args.numeric_rel_tol,
        "numeric_abs_tol": args.numeric_abs_tol,
        "percent_abs_tol": args.percent_abs_tol,
    }

    with run_config_path.open("w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    mode = "a" if args.resume and pred_path.exists() else "w"

    generated_count = 0
    start_time = time.time()

    with pred_path.open(mode, encoding="utf-8") as pred_f:
        for batch_idx, batch in enumerate(batch_iter(todo, args.batch_size), 1):
            prompts = []
            prompt_modes = []

            for row in batch:
                prompt_text, used_mode = render_prompt(
                    tokenizer,
                    row["prompt"],
                    args.prompt_format,
                )
                prompts.append(prompt_text)
                prompt_modes.append(used_mode)

            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=True,
            )

            if torch.cuda.is_available():
                # device_map="auto" places this small model on one device in practice.
                device = next(model.parameters()).device
                enc = {k: v.to(device) for k, v in enc.items()}

            with torch.inference_mode():
                generated = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            prompt_width = enc["input_ids"].shape[1]
            completion_ids = generated[:, prompt_width:]

            for i, row in enumerate(batch):
                response = tokenizer.decode(
                    completion_ids[i],
                    skip_special_tokens=True,
                ).strip()

                response_tokens = int(
                    (completion_ids[i] != tokenizer.pad_token_id).sum().item()
                )

                scored = score_response(
                    response,
                    row.get("gold_answer", ""),
                    args,
                )

                out_row = {
                    "id": row.get("id"),
                    "prediction": response,
                    **scored,
                    "response_tokens": response_tokens,
                    "prompt_format_used": prompt_modes[i],
                    "metadata": row.get("metadata", {}),
                }

                pred_f.write(
                    json.dumps(out_row, ensure_ascii=False) + "\n"
                )
                pred_f.flush()
                generated_count += 1

            if batch_idx % args.print_every == 0 or generated_count == len(todo):
                elapsed = max(time.time() - start_time, 1e-6)
                rate = generated_count / elapsed
                print(
                    f"Generated {generated_count}/{len(todo)} "
                    f"({rate:.2f} examples/s)",
                    flush=True,
                )

    # Re-read all predictions (including resumed rows) and compute final metrics.
    all_predictions, _ = load_existing(pred_path)
    metrics = compute_metrics(all_predictions)

    metrics["run"] = run_config
    metrics["num_predictions"] = len(all_predictions)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with errors_path.open("w", encoding="utf-8") as f:
        for row in all_predictions:
            if not row.get("correct", False):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n" + "=" * 80, flush=True)
    print("RESULT", flush=True)
    print("=" * 80, flush=True)

    overall = metrics["overall"]

    print(f"N                 : {overall['n']}", flush=True)
    print(f"Answer accuracy   : {overall['accuracy']:.4f}", flush=True)
    print(f"Strict accuracy   : {overall['strict_accuracy']:.4f}", flush=True)
    print(f"Format accuracy   : {overall['format_accuracy']:.4f}", flush=True)
    print(f"Avg response toks : {overall['avg_response_tokens']:.2f}", flush=True)

    print("\nBy difficulty:", flush=True)
    for name, stat in metrics["by_difficulty"].items():
        print(
            f"  {name:10s} n={stat['n']:4d} "
            f"acc={stat['accuracy']:.4f} "
            f"fmt={stat['format_accuracy']:.4f}",
            flush=True,
        )

    print("\nBy operator:", flush=True)
    for name, stat in metrics["by_operator"].items():
        print(
            f"  {name:10s} n={stat['n']:4d} "
            f"acc={stat['accuracy']:.4f}",
            flush=True,
        )

    print("\nSaved:", flush=True)
    print(" ", pred_path, flush=True)
    print(" ", metrics_path, flush=True)
    print(" ", errors_path, flush=True)
    print(" ", run_config_path, flush=True)


if __name__ == "__main__":
    main()
