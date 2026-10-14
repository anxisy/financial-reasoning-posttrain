# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Any, Dict, List

from answer_normalizer import choose_canonical_gold, compare_answer_strings, parse_answer
from operator_utils import extract_program_ops, extract_step_ops

MAX_PATTERNS = (
    r"\bhighest\b",
    r"\bmaximum\b",
    r"\bgreatest\b",
    r"\bmax\b",
)

MIN_PATTERNS = (
    r"\blowest\b",
    r"\bminimum\b",
    r"\bsmallest\b",
    r"\bmin\b",
)

# Conservative exception words for questions where a "greater" loss/decline
# can correspond to a more negative numeric value.
NEGATIVE_MAGNITUDE_PATTERNS = (
    r"\bloss\b",
    r"\blosses\b",
    r"\bdecline\b",
    r"\bdecrease\b",
    r"\breduction\b",
    r"\bdrop\b",
    r"\bdeficit\b",
    r"\bnegative\b",
)

PERCENT_PATTERNS = (
    r"\bpercentage\b",
    r"\bpercent\b",
    r"\bgrowth rate\b",
    r"\bmargin\b",
    r"\bratio\b",
    r"\bwhat percent\b",
    r"\bwhat percentage\b",
)

COMPARISON_PATTERNS = (
    r"\bexceed\b",
    r"\boutperform\b",
    r"\bhigher than\b",
    r"\bgreater than\b",
    r"\bless than\b",
    r"\bmeet\b",
    r"\bmet\b",
)


def _matches_any(text: str, patterns) -> bool:
    text = (text or "").lower()
    return any(re.search(pattern, text) is not None for pattern in patterns)


def audit_annotation(qa: Dict[str, Any]) -> Dict[str, Any]:
    flags: List[str] = []
    chosen = choose_canonical_gold(qa)

    answer = chosen["answer"]
    step_result = chosen["step_result"]

    if chosen["gold_source"] == "missing":
        flags.append("missing_gold")

    if answer and step_result:
        ok, _ = compare_answer_strings(answer, step_result)
        if not ok:
            flags.append("answer_step_mismatch")

    program_ops = extract_program_ops(qa.get("program"))
    step_ops = extract_step_ops(qa.get("steps"))

    if program_ops and step_ops and program_ops != step_ops:
        flags.append("program_step_operator_mismatch")

    if "missing_gold" in flags:
        quality = "low"
    elif flags:
        quality = "medium"
    else:
        quality = "high"

    return {
        "annotation_quality": quality,
        "annotation_flags": flags,
        "program_ops_normalized": program_ops,
        "step_ops_normalized": step_ops,
    }


def audit_semantics(
    qa: Dict[str, Any],
    annotation: Dict[str, Any],
) -> Dict[str, Any]:
    question = str(qa.get("question", "")).lower()
    gold_answer = choose_canonical_gold(qa)["gold_answer"]
    parsed_gold = parse_answer(gold_answer)

    ops = annotation["program_ops_normalized"] or annotation["step_ops_normalized"]
    op_set = set(ops)

    flags: List[str] = []

    asks_max = _matches_any(question, MAX_PATTERNS)
    asks_min = _matches_any(question, MIN_PATTERNS)
    negative_magnitude_context = _matches_any(
        question,
        NEGATIVE_MAGNITUDE_PATTERNS,
    )

    # Fix 1:
    # "highest/greatest ... loss" can legitimately map to min when values are signed.
    if (
        asks_max
        and "min" in op_set
        and "max" not in op_set
        and not negative_magnitude_context
    ):
        flags.append("question_asks_max_but_ops_use_min")

    if (
        asks_min
        and "max" in op_set
        and "min" not in op_set
        and not negative_magnitude_context
    ):
        flags.append("question_asks_min_but_ops_use_max")

    # Fix 2:
    # If the expected answer is yes/no, percentages in the question are context,
    # not necessarily the expected output type.
    asks_percent = _matches_any(question, PERCENT_PATTERNS)
    if asks_percent and parsed_gold.kind != "yesno":
        if parsed_gold.kind in {"text", "empty"}:
            flags.append("percentage_question_but_gold_type_is_non_numeric")

    comparison_question = _matches_any(question, COMPARISON_PATTERNS)
    if (
        parsed_gold.kind == "yesno"
        and comparison_question
        and "greater" not in op_set
    ):
        flags.append("comparison_yesno_question_without_greater_op")

    status = "suspicious" if flags else "not_flagged"

    return {
        "semantic_status": status,
        "semantic_flags": flags,
        "semantic_verified": False,
    }


def audit_quality(qa: Dict[str, Any]) -> Dict[str, Any]:
    annotation = audit_annotation(qa)
    semantic = audit_semantics(qa, annotation)
    return {**annotation, **semantic}
