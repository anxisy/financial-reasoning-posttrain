#!/usr/bin/env python3
"""ConvFinQA oracle-history baseline with frozen FinQA manual prompt and numeric matcher.

Evaluates Qwen3-1.7B Base and optional FinQA SFT/Dynamic LoRA sequentially
using a single vLLM engine. No training, no private Test, no dataset mutation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from answer_normalizer import compare_answer_strings, extract_answer_with_method

BASE_PATH = "/root/autodl-tmp/models/Qwen3-1.7B-Base"
SFT_ADAPTER = "/root/autodl-tmp/project/finqa_posttrain/experiments/sft_qwen3_1.7b/full_v1/final_adapter"
DYNAMIC_ADAPTER = "/root/autodl-tmp/project/finqa_posttrain/experiments/grpo_dynamic_qwen3_1.7b/full_v1/final_adapter"
DEV_PATH = "/root/autodl-tmp/project/convfinqa_posttrain/data/processed_v1/eval/dev.jsonl"
OUTPUT_ROOT = "/root/autodl-tmp/project/convfinqa_posttrain/experiments/oracle_baselines_v1"
LABELS = {"base": "qwen3_base", "sft": "finqa_sft", "dynamic": "finqa_dynamic"}


def serialize_manual(messages):
    """Use same serialization as frozen FinQA evaluation. No newline after Assistant:."""
    parts = []
    for message in messages:
        role = str(message.get("role", "")).lower()
        content = str(message.get("content", "")).strip()
        if role == "system":
            parts.append(f"System:\n{content}")
        elif role == "user":
            parts.append(f"User:\n{content}")
        elif role == "assistant":
            parts.append(f"Assistant:\n{content}")
        else:
            raise ValueError(f"Unsupported role {role}")
    if not parts:
        raise ValueError("Empty messages")
    prompt = "\n\n".join(parts) + "\n\nAssistant:"
    assert prompt.endswith("Assistant:") and not prompt.endswith("Assistant:\n")
    return prompt


def read_examples(path, limit, expected_size):
    examples = []
    seen_ids = set()
    with Path(path).open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            rid = str(row.get("id", ""))
            if not rid or rid in seen_ids:
                raise ValueError(f"Missing/duplicate id at line {lineno}: {rid!r}")
            seen_ids.add(rid)
            messages = row.get("prompt")
            if not isinstance(messages, list) or len(messages) != 2:
                raise ValueError(f"Invalid prompt schema: {rid}")
            if [v.get("role") for v in messages] != ["system", "user"]:
                raise ValueError(f"Unexpected roles: {rid}")
            if "gold_answer" not in row:
                raise ValueError(f"Missing gold: {rid}")
            meta = row.get("metadata", {})
            if meta.get("history_mode") != "oracle_gold_answers":
                raise ValueError(f"Non-oracle dataset detected: {rid}")
            if str(meta.get("source_split")) != "dev":
                raise ValueError(f"Only Dev split supported, got {rid}")
            if int(meta.get("turn_ind", -1)) < 0 or not meta.get("conversation_id"):
                raise ValueError(f"Missing turn metadata: {rid}")
            examples.append(row)
            if limit and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"No data in {path}")
    if not limit and len(examples) != expected_size:
        raise ValueError(f"Expected {expected_size} rows, found {len(examples)}")
    return examples


def get_adapter_rank(path):
    cfg_path = Path(path) / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    rank = int(config["r"])
    return max([rank] + [int(x) for x in (config.get("rank_pattern") or {}).values()])


def score_prediction(row, text, response_tokens, max_new_tokens, variant):
    answer, has_tag, extraction = extract_answer_with_method(text, allow_fallback=True)
    good, reason = compare_answer_strings(
        answer, row["gold_answer"], numeric_rel_tol=1e-3,
        numeric_abs_tol=1e-2, percent_abs_tol=0.10,
    )
    return {
        "id": row["id"],
        "variant": variant,
        "prediction": text.strip(),
        "prediction_answer": answer,
        "gold_answer": row["gold_answer"],
        "correct": bool(good),
        "format_ok": bool(has_tag),
        "match_reason": reason,
        "extraction_method": extraction,
        "response_tokens": int(response_tokens),
        "hit_max_new_tokens": response_tokens >= max_new_tokens,
        "metadata": row["metadata"],
    }


def summarize_predictions(preds):
    n = len(preds)
    if not n:
        raise ValueError("No predictions")

    def stats(seq):
        count = len(seq)
        return {"n": count, "correct": sum(r["correct"] for r in seq),
                "accuracy": sum(r["correct"] for r in seq) / count} if count else {"n": 0, "correct": 0, "accuracy": None}

    by_turn, by_type, by_op, by_conv = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for r in preds:
        meta = r["metadata"]
        by_turn[str(meta["turn_ind"])].append(r)
        by_type[str(meta.get("cur_type", "unknown"))].append(r)
        by_conv[str(meta["conversation_id"])].append(r)
        for op in set(meta.get("operators") or []):
            by_op[op].append(r)
    conv_count = len(by_conv)
    fully_correct = sum(all(q["correct"] for q in rows) for rows in by_conv.values())
    return {
        "overall": {
            **stats(preds),
            "strict_accuracy": sum(r["correct"] and r["format_ok"] for r in preds) / n,
            "format_accuracy": sum(r["format_ok"] for r in preds) / n,
            "answer_found_rate": sum(bool(r["prediction_answer"]) for r in preds) / n,
            "max_token_hit_rate": sum(r["hit_max_new_tokens"] for r in preds) / n,
            "avg_response_tokens": sum(r["response_tokens"] for r in preds) / n,
        },
        "by_turn": {k: stats(v) for k, v in sorted(by_turn.items(), key=lambda kv: int(kv[0]))},
        "by_cur_type": {k: stats(v) for k, v in sorted(by_type.items())},
        "by_operator": {k: stats(v) for k, v in sorted(by_op.items())},
        "conversation_exact_match": {
            "n_conversations": conv_count,
            "all_turns_correct": fully_correct,
            "accuracy": fully_correct / conv_count,
            "note": "All-turn correctness under ORACLE history, not free-running dialogue success",
        },
        "scoring_note": "Uses frozen FinQA numeric matching. Numeric 0.14136 and 14.136% are not equivalent in this STRICT baseline; inspect ratio edge cases separately.",
    }


def parse_variants(s):
    values = [x.strip().lower() for x in s.split(",") if x.strip()]
    if not values or len(values) != len(set(values)) or any(v not in LABELS for v in values):
        raise ValueError("--variants must be a comma-separated unique subset of base,sft,dynamic")
    return values


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=BASE_PATH)
    ap.add_argument("--sft-adapter", default=SFT_ADAPTER)
    ap.add_argument("--dynamic-adapter", default=DYNAMIC_ADAPTER)
    ap.add_argument("--input", default=DEV_PATH)
    ap.add_argument("--output-dir", default=OUTPUT_ROOT)
    ap.add_argument("--variants", default="base,sft,dynamic")
    ap.add_argument("--max-samples", type=int, default=20, help="0 means full 1490 Dev")
    ap.add_argument("--expected-full-n", type=int, default=1490)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--request-batch-size", type=int, default=0, help="0 submits all requests in one vLLM call")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    ap.add_argument("--warmup-samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--enforce-eager", action="store_true")
    args = ap.parse_args()
    variants = parse_variants(args.variants)
    if args.max_samples < 0 or args.max_new_tokens <= 0 or args.request_batch_size < 0:
        ap.error("Invalid max-samples/max-new-tokens/request-batch-size")

    for name in ["base", "input"]:
        if not Path(getattr(args, name)).exists():
            raise FileNotFoundError(getattr(args, name))
    rank = 8
    if "sft" in variants:
        rank = max(rank, get_adapter_rank(args.sft_adapter))
    if "dynamic" in variants:
        rank = max(rank, get_adapter_rank(args.dynamic_adapter))

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    rows = read_examples(args.input, args.max_samples, args.expected_full_n)
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=True)
    prompts = [serialize_manual(r["prompt"]) for r in rows]
    enc = tok(prompts, truncation=False, padding=False, add_special_tokens=True)
    ids = enc["input_ids"]
    length_max = max(map(len, ids))
    too_long = [(r["id"], len(s)) for r, s in zip(rows, ids)
                if len(s) + args.max_new_tokens > args.max_model_len]
    if too_long:
        raise ValueError(f"{len(too_long)} inputs exceed max-model-len {args.max_model_len} with requested new tokens. Samples: {too_long[:5]}")
    sha = hashlib.sha256(json.dumps([(r["id"], seq) for r, seq in zip(rows, ids)],
                                     ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    print(f"Dev rows={len(rows)}; max prompt tokens={length_max}; input fingerprint={sha}", flush=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    t0 = time.perf_counter()
    llm = LLM(
        model=args.base, tokenizer=args.base, dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
        trust_remote_code=True, seed=args.seed, enforce_eager=args.enforce_eager,
        enable_lora=any(v != "base" for v in variants), max_lora_rank=rank,
        max_loras=2,
    )
    engine_load_seconds = time.perf_counter() - t0
    sampler = SamplingParams(temperature=0.0, top_p=1.0,
                             max_tokens=args.max_new_tokens, seed=args.seed,
                             skip_special_tokens=True,
                             stop_token_ids=[tok.eos_token_id])
    req = {
        "base": None,
        "sft": LoRARequest("finqa_sft", 1, args.sft_adapter) if "sft" in variants else None,
        "dynamic": LoRARequest("finqa_dynamic", 2, args.dynamic_adapter) if "dynamic" in variants else None,
    }
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    param_log = {
        "input": args.input, "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "prompt_token_ids_sha256": sha, "max_input_tokens": length_max,
        "history_mode": "oracle_gold_answers", "num_rows": len(rows),
        "variants": variants, "base": args.base, "sft_adapter": args.sft_adapter,
        "dynamic_adapter": args.dynamic_adapter,
        "max_new_tokens": args.max_new_tokens, "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs, "request_batch_size": args.request_batch_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "warmup_samples": args.warmup_samples, "seed": args.seed,
        "enforce_eager": args.enforce_eager,
        "vllm_version": importlib.metadata.version("vllm"),
        "engine_load_seconds": engine_load_seconds,
        "flashinfer_sampler_disabled": os.getenv("VLLM_USE_FLASHINFER_SAMPLER") == "0",
    }
    (root / "run_config.json").write_text(json.dumps(param_log, ensure_ascii=False, indent=2))
    bs = args.request_batch_size or len(rows)
    for variant in variants:
        print(f"\n===== Evaluating {variant} =====", flush=True)
        lora_request = req[variant]

        def generate(seq_list):
            packed = [{"prompt_token_ids": s} for s in seq_list]
            outputs = llm.generate(packed, sampling_params=sampler,
                                   lora_request=lora_request, use_tqdm=False)
            if len(outputs) != len(seq_list):
                raise RuntimeError("Incomplete vLLM generation")
            return [(q.outputs[0].text.strip(), len(q.outputs[0].token_ids)) for q in outputs]

        w = min(args.warmup_samples, len(ids))
        if w:
            print(f"Warming up {variant} with {w} examples...", flush=True)
            generate(ids[:w])
        predictions = []
        total_toks = 0
        t1 = time.perf_counter()
        for start in range(0, len(ids), bs):
            chunk = generate(ids[start:start + bs])
            for source, (completion, tokens) in zip(rows[start:start + bs], chunk):
                predictions.append(score_prediction(source, completion, tokens,
                                                    args.max_new_tokens, variant))
                total_toks += tokens
            print(f"  {variant}: {len(predictions)}/{len(ids)}", flush=True)
        elapsed = time.perf_counter() - t1
        report = summarize_predictions(predictions)
        report["engine"] = "vllm"
        report["variant"] = variant
        report["input_fingerprint"] = sha
        report["num_rows"] = len(rows)
        report["elapsed_inference_seconds"] = elapsed
        report["samples_per_second"] = len(rows) / elapsed
        report["generated_tokens_per_second"] = total_toks / elapsed
        output = root / LABELS[variant]
        output.mkdir(exist_ok=True, parents=True)
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as h:
            for p in predictions:
                h.write(json.dumps(p, ensure_ascii=False) + "\n")
        (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{variant}: accuracy={report['overall']['accuracy']:.2%}; "
              f"format={report['overall']['format_accuracy']:.2%}; "
              f"conversation_exact_match={report['conversation_exact_match']['accuracy']:.2%}; "
              f"inference={elapsed:.1f}s", flush=True)
    print(f"\nAll requested variants completed. Results: {root}")
    print("Run: python analyze_baselines.py --input-root " + str(root))


if __name__ == "__main__":
    main()
