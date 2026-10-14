# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from transformers import AutoTokenizer

from config import PipelineConfig
from answer_normalizer import build_quality_metadata
from reasoning_builder import build_sft_response
from prompt_builder import build_messages, build_sections, truncate_report_to_budget

def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def op_names(qa: dict) -> List[str]:
    names = []
    for step in qa.get("steps") or []:
        if isinstance(step, dict):
            op = str(step.get("op", "")).strip()
            if op:
                names.append(op)
    return names

def difficulty_from_steps(num_steps: int) -> str:
    if num_steps <= 1:
        return "easy"
    if num_steps == 2:
        return "medium"
    return "hard"

def canonicalize(example: dict) -> dict:
    qa = example.get("qa", {})
    quality = build_quality_metadata(qa)
    steps = qa.get("steps") or []

    return {
        "id": example.get("id"),
        "filename": example.get("filename"),
        "question": qa.get("question", ""),
        "pre_text": example.get("pre_text", []),
        "post_text": example.get("post_text", []),
        "table": example.get("table", []),

        "gold_answer": quality["gold_answer"],
        "gold_source": quality["gold_source"],
        "quality_flag": quality["quality_flag"],
        "answer": quality["answer"],
        "step_result": quality["step_result"],
        "exe_ans": quality["exe_ans"],
        "answer_step_match": quality["answer_step_match"],
        "answer_step_reason": quality["answer_step_reason"],

        "steps": steps,
        "program": qa.get("program"),
        "program_re": qa.get("program_re"),
        "explanation": qa.get("explanation", ""),

        "num_steps": len(steps),
        "difficulty": difficulty_from_steps(len(steps)),
        "operators": op_names(qa),

        # Audit-only; NEVER inserted into the model prompt.
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
        tokenizer=tokenizer,
        sections=sections,
        max_prompt_tokens=max_prompt_tokens,
    )
    return build_messages(sections), trunc_meta

def build_sft_row(tokenizer, cfg: PipelineConfig, c: dict) -> dict:
    messages, trunc_meta = make_prompt_record(
        tokenizer, cfg, c, cfg.sft_prompt_max_tokens
    )
    response = build_sft_response(
        {"steps": c["steps"], "explanation": c["explanation"]},
        c["gold_answer"],
    )
    return {
        "id": c["id"],
        "messages": messages + [{"role": "assistant", "content": response}],
        "metadata": {
            "filename": c["filename"],
            "gold_answer": c["gold_answer"],
            "gold_source": c["gold_source"],
            "quality_flag": c["quality_flag"],
            "num_steps": c["num_steps"],
            "difficulty": c["difficulty"],
            "operators": c["operators"],
            **trunc_meta,
        },
    }

def build_grpo_row(tokenizer, cfg: PipelineConfig, c: dict) -> dict:
    messages, trunc_meta = make_prompt_record(
        tokenizer, cfg, c, cfg.grpo_prompt_max_tokens
    )
    return {
        "id": c["id"],
        "prompt": messages,
        "gold_answer": c["gold_answer"],
        "metadata": {
            "filename": c["filename"],
            "gold_source": c["gold_source"],
            "quality_flag": c["quality_flag"],
            "num_steps": c["num_steps"],
            "difficulty": c["difficulty"],
            "operators": c["operators"],
            **trunc_meta,
        },
    }

def build_eval_row(tokenizer, cfg: PipelineConfig, c: dict) -> dict:
    row = build_grpo_row(tokenizer, cfg, c)
    row["split_role"] = "evaluation"
    return row

def check_overlap(raw_by_split: Dict[str, List[dict]]) -> dict:
    result = {}
    ids = {
        split: {x.get("id") for x in rows if x.get("id") is not None}
        for split, rows in raw_by_split.items()
    }
    filenames = {
        split: {x.get("filename") for x in rows if x.get("filename")}
        for split, rows in raw_by_split.items()
    }

    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        result[f"id_overlap_{a}_{b}"] = len(ids[a] & ids[b])
        result[f"filename_overlap_{a}_{b}"] = len(filenames[a] & filenames[b])
    return result

def summarize(rows: List[dict]) -> dict:
    return {
        "num_examples": len(rows),
        "quality_flag": dict(Counter(x["quality_flag"] for x in rows)),
        "difficulty": dict(Counter(x["difficulty"] for x in rows)),
        "gold_source": dict(Counter(x["gold_source"] for x in rows)),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
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

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    raw_by_split = {}
    canonical_by_split = {}

    for split, path in cfg.split_paths().items():
        raw = read_json(path)
        raw_by_split[split] = raw

        canonical = [canonicalize(ex) for ex in raw]
        if not cfg.keep_low_quality:
            canonical = [x for x in canonical if x["quality_flag"] != "low"]

        canonical_by_split[split] = canonical
        write_jsonl(out / "canonical" / f"{split}.jsonl", canonical)

    sft_train = [build_sft_row(tokenizer, cfg, x) for x in canonical_by_split["train"]]
    sft_dev = [build_sft_row(tokenizer, cfg, x) for x in canonical_by_split["dev"]]

    grpo_train = [build_grpo_row(tokenizer, cfg, x) for x in canonical_by_split["train"]]

    eval_dev = [build_eval_row(tokenizer, cfg, x) for x in canonical_by_split["dev"]]
    eval_test = [build_eval_row(tokenizer, cfg, x) for x in canonical_by_split["test"]]

    write_jsonl(out / "sft" / "finqa_sft_train.jsonl", sft_train)
    write_jsonl(out / "sft" / "finqa_sft_dev.jsonl", sft_dev)
    write_jsonl(out / "grpo" / "finqa_grpo_train.jsonl", grpo_train)
    write_jsonl(out / "eval" / "finqa_eval_dev.jsonl", eval_dev)
    write_jsonl(out / "eval" / "finqa_eval_test.jsonl", eval_test)

    report = {
        "config": cfg.to_dict(),
        "overlap_check": check_overlap(raw_by_split),
        "canonical_summary": {
            split: summarize(rows)
            for split, rows in canonical_by_split.items()
        },
        "output_counts": {
            "sft_train": len(sft_train),
            "sft_dev": len(sft_dev),
            "grpo_train": len(grpo_train),
            "eval_dev": len(eval_dev),
            "eval_test": len(eval_test),
        },
        "notes": [
            "SFT train and GRPO train intentionally share the original FinQA train split.",
            "Dev/test are never used for optimization.",
            "gold_inds and retrieved evidence are never inserted into prompts.",
            "Low-quality samples are kept by default and marked for later filtering ablations.",
        ],
    }

    report_path = out / "reports" / "preprocess_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("FinQA preprocessing complete")
    print("=" * 80)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nOutput directory:", out)

if __name__ == "__main__":
    main()
