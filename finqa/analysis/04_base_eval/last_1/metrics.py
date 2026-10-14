# -*- coding: utf-8 -*-
"""
FinQA evaluation metrics.

Reports:
- overall answer accuracy
- strict accuracy (correct AND <answer> tag present)
- format accuracy
- average response tokens
- difficulty slices
- operator slices
- context-length slices
- annotation-quality slices
- semantic-status slices
- truncated/non-truncated slices
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List


def context_bucket(n_tokens: int) -> str:
    n = int(n_tokens or 0)
    if n <= 1024:
        return "<=1024"
    if n <= 1536:
        return "1025-1536"
    if n <= 2048:
        return "1537-2048"
    return ">2048"


def _summarize(rows: List[dict]) -> dict:
    n = len(rows)

    if n == 0:
        return {
            "n": 0,
            "accuracy": 0.0,
            "strict_accuracy": 0.0,
            "format_accuracy": 0.0,
            "avg_response_tokens": 0.0,
        }

    correct = sum(int(bool(x.get("correct"))) for x in rows)
    strict = sum(
        int(bool(x.get("correct")) and bool(x.get("format_ok")))
        for x in rows
    )
    fmt = sum(int(bool(x.get("format_ok"))) for x in rows)
    resp_tokens = sum(int(x.get("response_tokens", 0) or 0) for x in rows)

    return {
        "n": n,
        "accuracy": correct / n,
        "strict_accuracy": strict / n,
        "format_accuracy": fmt / n,
        "avg_response_tokens": resp_tokens / n,
    }


def _group_by(rows: List[dict], key_fn) -> Dict[str, dict]:
    groups = defaultdict(list)
    for row in rows:
        key = str(key_fn(row))
        groups[key].append(row)
    return {
        key: _summarize(items)
        for key, items in sorted(groups.items())
    }


def compute_metrics(rows: List[dict]) -> dict:
    metrics = {
        "overall": _summarize(rows),
    }

    metrics["by_difficulty"] = _group_by(
        rows,
        lambda x: x.get("metadata", {}).get("difficulty", "unknown"),
    )

    metrics["by_context_length"] = _group_by(
        rows,
        lambda x: context_bucket(
            x.get("metadata", {}).get("original_prompt_tokens", 0)
        ),
    )

    metrics["by_annotation_quality"] = _group_by(
        rows,
        lambda x: x.get("metadata", {}).get("annotation_quality", "unknown"),
    )

    metrics["by_semantic_status"] = _group_by(
        rows,
        lambda x: x.get("metadata", {}).get("semantic_status", "unknown"),
    )

    metrics["by_truncation"] = _group_by(
        rows,
        lambda x: (
            "truncated"
            if int(x.get("metadata", {}).get("was_truncated", 0))
            else "not_truncated"
        ),
    )

    # Multi-label operator slices: an example can contribute to multiple operators,
    # but only once per operator even if the operator appears multiple times.
    op_groups = defaultdict(list)

    for row in rows:
        ops = row.get("metadata", {}).get("operators", []) or []
        for op in sorted(set(map(str, ops))):
            op_groups[op].append(row)

    metrics["by_operator"] = {
        op: _summarize(items)
        for op, items in sorted(op_groups.items())
    }

    return metrics
