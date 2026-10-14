# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Iterable, List

from answer_normalizer import compare_answer_strings, extract_answer_from_response

def score_response(
    response: str,
    gold_answer: str,
    numeric_rel_tol: float = 1e-3,
    numeric_abs_tol: float = 1e-2,
    percent_abs_tol: float = 0.10,
) -> dict:
    pred_answer, has_tag = extract_answer_from_response(response, allow_fallback=True)
    correct, reason = compare_answer_strings(
        pred_answer,
        gold_answer,
        numeric_rel_tol=numeric_rel_tol,
        numeric_abs_tol=numeric_abs_tol,
        percent_abs_tol=percent_abs_tol,
    )
    return {
        "prediction_answer": pred_answer,
        "gold_answer": gold_answer,
        "correct": bool(correct),
        "match_reason": reason,
        "format_ok": bool(has_tag),
    }

def correctness_reward(
    responses: Iterable[str],
    gold_answers: Iterable[str],
    **kwargs,
) -> List[float]:
    rewards = []
    for response, gold in zip(responses, gold_answers):
        result = score_response(response, gold, **kwargs)
        rewards.append(1.0 if result["correct"] else 0.0)
    return rewards

def format_reward(responses: Iterable[str]) -> List[float]:
    rewards = []
    for response in responses:
        _, has_tag = extract_answer_from_response(response, allow_fallback=False)
        rewards.append(1.0 if has_tag else 0.0)
    return rewards

def combined_reward(
    responses: Iterable[str],
    gold_answers: Iterable[str],
    correctness_weight: float = 1.0,
    format_weight: float = 0.1,
    **kwargs,
) -> List[float]:
    c = correctness_reward(responses, gold_answers, **kwargs)
    f = format_reward(responses)
    return [
        correctness_weight * cr + format_weight * fr
        for cr, fr in zip(c, f)
    ]

def evaluate_jsonl(
    input_path: str,
    response_field: str = "prediction",
    gold_field: str = "gold_answer",
) -> dict:
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    scored = []
    for row in rows:
        result = score_response(row.get(response_field, ""), row.get(gold_field, ""))
        scored.append({**row, **result})

    n = len(scored)
    accuracy = sum(int(x["correct"]) for x in scored) / n if n else 0.0
    format_accuracy = sum(int(x["format_ok"]) for x in scored) / n if n else 0.0

    return {
        "num_examples": n,
        "accuracy": accuracy,
        "format_accuracy": format_accuracy,
        "rows": scored,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--response-field", default="prediction")
    parser.add_argument("--gold-field", default="gold_answer")
    args = parser.parse_args()

    result = evaluate_jsonl(
        args.input,
        response_field=args.response_field,
        gold_field=args.gold_field,
    )

    print("Examples        :", result["num_examples"])
    print("Accuracy        :", f"{result['accuracy']:.4f}")
    print("Format accuracy :", f"{result['format_accuracy']:.4f}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for row in result["rows"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("Saved:", out)

if __name__ == "__main__":
    main()
