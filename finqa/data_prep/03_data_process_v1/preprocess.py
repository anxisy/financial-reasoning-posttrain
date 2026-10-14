# -*- coding: utf-8 -*-
"""
FinQA data pipeline v1.1 - LOW MEMORY VERSION

Key change:
- process ONE split at a time
- write JSONL incrementally
- do NOT keep canonical/SFT/GRPO/eval full lists in RAM
- only keep small counters/sets for reporting

This is designed for a ~2GB cgroup memory limit.
"""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from transformers import AutoTokenizer

from config import PipelineConfig
from answer_normalizer import choose_canonical_gold
from quality_audit import audit_quality
from reasoning_builder import build_sft_response
from prompt_builder import build_messages, build_sections, truncate_report_to_budget


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_jsonl_line(handle, row: Dict[str, Any]):
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def difficulty_from_steps(num_steps: int) -> str:
    if num_steps <= 1:
        return "easy"
    if num_steps == 2:
        return "medium"
    return "hard"


def canonicalize(example: dict) -> dict:
    qa = example.get("qa", {})
    chosen = choose_canonical_gold(qa)
    audit = audit_quality(qa)
    steps = qa.get("steps") or []

    return {
        "id": example.get("id"),
        "filename": example.get("filename"),
        "question": qa.get("question", ""),
        "pre_text": example.get("pre_text", []),
        "post_text": example.get("post_text", []),
        "table": example.get("table", []),

        "gold_answer": chosen["gold_answer"],
        "gold_source": chosen["gold_source"],
        "answer": chosen["answer"],
        "step_result": chosen["step_result"],
        "exe_ans": chosen["exe_ans"],

        "steps": steps,
        "program": qa.get("program"),
        "program_re": qa.get("program_re"),
        "explanation": qa.get("explanation", ""),

        "num_steps": len(steps),
        "difficulty": difficulty_from_steps(len(steps)),

        "annotation_quality": audit["annotation_quality"],
        "annotation_flags": audit["annotation_flags"],
        "semantic_status": audit["semantic_status"],
        "semantic_flags": audit["semantic_flags"],
        "semantic_verified": audit["semantic_verified"],

        "program_ops_normalized": audit["program_ops_normalized"],
        "step_ops_normalized": audit["step_ops_normalized"],

        # Audit-only. Never inserted into prompts.
        "gold_inds": qa.get("gold_inds"),
    }


def make_prompt_record(tokenizer, cfg: PipelineConfig, canonical: dict, max_prompt_tokens: int):
    ex = {
        "pre_text": canonical["pre_text"],
        "post_text": canonical["post_text"],
        "table": canonical["table"],
        "qa": {"question": canonical["question"]},
    }

    sections = build_sections(ex, cfg.system_prompt)
    sections, trunc_meta = truncate_report_to_budget(
        tokenizer,
        sections,
        max_prompt_tokens,
    )
    return build_messages(sections), trunc_meta


def build_sft_row(tokenizer, cfg: PipelineConfig, c: dict):
    messages, trunc_meta = make_prompt_record(
        tokenizer,
        cfg,
        c,
        cfg.sft_prompt_max_tokens,
    )

    response = build_sft_response(
        {
            "steps": c["steps"],
            "explanation": c["explanation"],
        },
        c["gold_answer"],
    )

    return {
        "id": c["id"],
        "messages": messages + [{"role": "assistant", "content": response}],
        "metadata": {
            "filename": c["filename"],
            "gold_answer": c["gold_answer"],
            "gold_source": c["gold_source"],
            "annotation_quality": c["annotation_quality"],
            "semantic_status": c["semantic_status"],
            "semantic_flags": c["semantic_flags"],
            "num_steps": c["num_steps"],
            "difficulty": c["difficulty"],
            "operators": c["step_ops_normalized"] or c["program_ops_normalized"],
            **trunc_meta,
        },
    }


def build_grpo_row(tokenizer, cfg: PipelineConfig, c: dict, split_role="train"):
    messages, trunc_meta = make_prompt_record(
        tokenizer,
        cfg,
        c,
        cfg.grpo_prompt_max_tokens,
    )

    row = {
        "id": c["id"],
        "prompt": messages,
        "gold_answer": c["gold_answer"],
        "metadata": {
            "filename": c["filename"],
            "gold_source": c["gold_source"],
            "annotation_quality": c["annotation_quality"],
            "semantic_status": c["semantic_status"],
            "semantic_flags": c["semantic_flags"],
            "num_steps": c["num_steps"],
            "difficulty": c["difficulty"],
            "operators": c["step_ops_normalized"] or c["program_ops_normalized"],
            **trunc_meta,
        },
    }

    if split_role != "train":
        row["split_role"] = split_role

    return row


class SplitStats:
    def __init__(self):
        self.num_examples = 0
        self.annotation_quality = Counter()
        self.annotation_flags = Counter()
        self.semantic_status = Counter()
        self.semantic_flags = Counter()
        self.difficulty = Counter()
        self.gold_source = Counter()
        self.operator_counts = Counter()

    def add(self, c: dict):
        self.num_examples += 1
        self.annotation_quality[c["annotation_quality"]] += 1
        self.semantic_status[c["semantic_status"]] += 1
        self.difficulty[c["difficulty"]] += 1
        self.gold_source[c["gold_source"]] += 1

        for x in c["annotation_flags"]:
            self.annotation_flags[x] += 1

        for x in c["semantic_flags"]:
            self.semantic_flags[x] += 1

        ops = c["step_ops_normalized"] or c["program_ops_normalized"]
        for op in ops:
            self.operator_counts[op] += 1

    def to_dict(self):
        return {
            "num_examples": self.num_examples,
            "annotation_quality": dict(self.annotation_quality),
            "annotation_flags": dict(self.annotation_flags),
            "semantic_status": dict(self.semantic_status),
            "semantic_flags": dict(self.semantic_flags),
            "difficulty": dict(self.difficulty),
            "gold_source": dict(self.gold_source),
            "normalized_operator_counts": dict(self.operator_counts),
        }


class TruncStats:
    def __init__(self):
        self.n = 0
        self.truncated = 0
        self.original_sum = 0
        self.final_sum = 0
        self.max_original = 0
        self.max_final = 0

    def add(self, meta: dict):
        self.n += 1
        self.truncated += int(meta.get("was_truncated", 0))
        orig = int(meta.get("original_prompt_tokens", 0))
        final = int(meta.get("final_prompt_tokens", 0))
        self.original_sum += orig
        self.final_sum += final
        self.max_original = max(self.max_original, orig)
        self.max_final = max(self.max_final, final)

    def to_dict(self):
        return {
            "num_examples": self.n,
            "num_truncated": self.truncated,
            "truncation_rate": (self.truncated / self.n if self.n else 0.0),
            "mean_original_prompt_tokens": (self.original_sum / self.n if self.n else 0.0),
            "mean_final_prompt_tokens": (self.final_sum / self.n if self.n else 0.0),
            "max_original_prompt_tokens": self.max_original,
            "max_final_prompt_tokens": self.max_final,
        }


def process_split(split, raw_path: Path, out: Path, tokenizer, cfg: PipelineConfig):
    print(f"\n[1/3] Loading split: {split} -> {raw_path}", flush=True)
    raw = read_json(raw_path)
    print(f"      examples: {len(raw)}", flush=True)

    canonical_path = out / "canonical" / f"{split}.jsonl"
    canonical_path.parent.mkdir(parents=True, exist_ok=True)

    sft_handle = None
    grpo_or_eval_handle = None

    if split == "train":
        sft_path = out / "sft" / "finqa_sft_train.jsonl"
        grpo_path = out / "grpo" / "finqa_grpo_train.jsonl"
        sft_path.parent.mkdir(parents=True, exist_ok=True)
        grpo_path.parent.mkdir(parents=True, exist_ok=True)
        sft_handle = sft_path.open("w", encoding="utf-8")
        grpo_or_eval_handle = grpo_path.open("w", encoding="utf-8")

    elif split == "dev":
        sft_path = out / "sft" / "finqa_sft_dev.jsonl"
        eval_path = out / "eval" / "finqa_eval_dev.jsonl"
        sft_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        sft_handle = sft_path.open("w", encoding="utf-8")
        grpo_or_eval_handle = eval_path.open("w", encoding="utf-8")

    else:
        eval_path = out / "eval" / "finqa_eval_test.jsonl"
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        grpo_or_eval_handle = eval_path.open("w", encoding="utf-8")

    ids = set()
    filenames = set()
    split_stats = SplitStats()
    sft_trunc = TruncStats()
    grpo_eval_trunc = TruncStats()

    try:
        with canonical_path.open("w", encoding="utf-8") as canonical_handle:
            for idx, example in enumerate(raw, 1):
                c = canonicalize(example)

                if (not cfg.keep_suspicious) and c["semantic_status"] == "suspicious":
                    continue

                dump_jsonl_line(canonical_handle, c)

                ids.add(c["id"])
                if c["filename"]:
                    filenames.add(c["filename"])

                split_stats.add(c)

                if split in {"train", "dev"}:
                    sft_row = build_sft_row(tokenizer, cfg, c)
                    dump_jsonl_line(sft_handle, sft_row)
                    sft_trunc.add(sft_row["metadata"])

                if split == "train":
                    row = build_grpo_row(tokenizer, cfg, c, split_role="train")
                elif split == "dev":
                    row = build_grpo_row(tokenizer, cfg, c, split_role="evaluation")
                else:
                    row = build_grpo_row(tokenizer, cfg, c, split_role="evaluation")

                dump_jsonl_line(grpo_or_eval_handle, row)
                grpo_eval_trunc.add(row["metadata"])

                # Drop references to large per-example strings ASAP.
                del c, row
                if split in {"train", "dev"}:
                    del sft_row

                if idx % 500 == 0:
                    print(f"      processed {idx}/{len(raw)}", flush=True)

    finally:
        if sft_handle is not None:
            sft_handle.close()
        if grpo_or_eval_handle is not None:
            grpo_or_eval_handle.close()

    count_after_filter = split_stats.num_examples

    # Critical for 2GB environments.
    del raw
    gc.collect()

    print(f"      finished {split}: {count_after_filter} examples", flush=True)

    return {
        "ids": ids,
        "filenames": filenames,
        "summary": split_stats.to_dict(),
        "sft_truncation": sft_trunc.to_dict() if split in {"train", "dev"} else None,
        "grpo_eval_truncation": grpo_eval_trunc.to_dict(),
    }


def overlap_report(split_results):
    def overlap(a, b, field):
        return len(split_results[a][field] & split_results[b][field])

    return {
        "id_overlap_train_dev": overlap("train", "dev", "ids"),
        "filename_overlap_train_dev": overlap("train", "dev", "filenames"),
        "id_overlap_train_test": overlap("train", "test", "ids"),
        "filename_overlap_train_test": overlap("train", "test", "filenames"),
        "id_overlap_dev_test": overlap("dev", "test", "ids"),
        "filename_overlap_dev_test": overlap("dev", "test", "filenames"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument(
        "--drop-suspicious",
        action="store_true",
        help="For ablation only. Baseline should NOT use this flag.",
    )
    args = parser.parse_args()

    cfg = PipelineConfig()

    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.model_path:
        cfg.model_path = args.model_path
    if args.max_seq_length:
        cfg.max_seq_length = args.max_seq_length
    if args.drop_suspicious:
        cfg.keep_suspicious = False

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    print("Tokenizer loaded.", flush=True)

    split_results = {}

    for split, raw_path in cfg.split_paths().items():
        split_results[split] = process_split(
            split=split,
            raw_path=raw_path,
            out=out,
            tokenizer=tokenizer,
            cfg=cfg,
        )

    report = {
        "pipeline_version": "v1.1-lowmem",
        "config": cfg.to_dict(),
        "overlap_check": overlap_report(split_results),
        "canonical_summary": {
            split: split_results[split]["summary"]
            for split in ("train", "dev", "test")
        },
        "truncation_summary": {
            "sft_train": split_results["train"]["sft_truncation"],
            "sft_dev": split_results["dev"]["sft_truncation"],
            "grpo_train": split_results["train"]["grpo_eval_truncation"],
            "eval_dev": split_results["dev"]["grpo_eval_truncation"],
            "eval_test": split_results["test"]["grpo_eval_truncation"],
        },
        "notes": [
            "Low-memory version: one split at a time, incremental JSONL writes.",
            "SFT train and GRPO train intentionally share FinQA train.",
            "Dev/test are never used for optimization.",
            "gold_inds/retrieved evidence are never inserted into prompts.",
            "semantic_status=not_flagged does NOT mean human-verified correctness.",
            "Baseline keeps suspicious samples; filtering is reserved for ablation.",
        ],
    }

    report_path = out / "reports" / "preprocess_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("FinQA v1.1-lowmem preprocessing complete")
    print("=" * 80)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nOutput:", out)


if __name__ == "__main__":
    main()
