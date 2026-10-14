# -*- coding: utf-8 -*-
"""
Answer parsing/comparison used by FinQA baseline evaluation.

Keep this logic aligned with the frozen data pipeline so Base/SFT/GRPO use
the same correctness definition.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Optional, Tuple


ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)
NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)")
YES_VALUES = {"yes", "y", "true"}
NO_VALUES = {"no", "n", "false"}


@dataclass
class ParsedAnswer:
    raw: str
    kind: str
    value: Any
    has_percent_sign: bool = False


def _string(value) -> str:
    return "" if value is None else str(value).strip()


def extract_tagged_answer(text: str) -> Optional[str]:
    if text is None:
        return None
    matches = ANSWER_TAG_RE.findall(str(text))
    return matches[-1].strip() if matches else None


def extract_answer_from_response(
    text: str,
    allow_fallback: bool = True,
) -> Tuple[str, bool]:
    """
    Returns:
        (extracted_answer, has_strict_answer_tag)

    Fallback is intentionally permissive for Base evaluation:
    if <answer> is missing, use the last non-empty line.
    Format accuracy is reported separately.
    """
    tagged = extract_tagged_answer(text)
    if tagged is not None:
        return tagged, True

    if not allow_fallback:
        return "", False

    lines = [x.strip() for x in str(text).splitlines() if x.strip()]
    if not lines:
        return "", False

    candidate = lines[-1]
    candidate = re.sub(
        r"^(final\s+answer|answer)\s*[:：]\s*",
        "",
        candidate,
        flags=re.I,
    )
    return candidate.strip(), False


def parse_answer(value: Any) -> ParsedAnswer:
    raw = _string(value)

    if not raw:
        return ParsedAnswer(raw, "empty", None)

    lowered = raw.lower().strip().strip(" .,:;")

    if lowered in YES_VALUES:
        return ParsedAnswer(raw, "yesno", "yes")

    if lowered in NO_VALUES:
        return ParsedAnswer(raw, "yesno", "no")

    match = NUMBER_RE.search(raw)
    if match:
        try:
            number = float(match.group(0).replace(",", ""))
            is_percent = "%" in raw
            return ParsedAnswer(
                raw,
                "percent" if is_percent else "number",
                number,
                is_percent,
            )
        except ValueError:
            pass

    return ParsedAnswer(raw, "text", lowered)


def compare_parsed_answers(
    pred: ParsedAnswer,
    gold: ParsedAnswer,
    numeric_rel_tol: float = 1e-3,
    numeric_abs_tol: float = 1e-2,
    percent_abs_tol: float = 0.10,
):
    if gold.kind == "empty":
        return False, "missing_gold"

    if pred.kind == "empty":
        return False, "empty_prediction"

    if gold.kind == "yesno":
        ok = pred.kind == "yesno" and pred.value == gold.value
        return ok, "yesno_match" if ok else "yesno_mismatch"

    if gold.kind == "text":
        ok = pred.kind == "text" and pred.value == gold.value
        return ok, "text_match" if ok else "text_mismatch"

    if pred.kind not in {"number", "percent"} or gold.kind not in {"number", "percent"}:
        return False, "type_mismatch"

    p = float(pred.value)
    g = float(gold.value)

    if gold.kind == "percent":
        # 53% vs 53%
        if math.isclose(p, g, rel_tol=0.0, abs_tol=percent_abs_tol):
            return True, "percent_point_match"

        # 0.53 vs 53%
        if pred.kind == "number" and math.isclose(
            p * 100.0,
            g,
            rel_tol=numeric_rel_tol,
            abs_tol=percent_abs_tol,
        ):
            return True, "percent_fraction_match"

        return False, "percent_mismatch"

    if pred.kind == "percent":
        return False, "unexpected_percent"

    ok = math.isclose(
        p,
        g,
        rel_tol=numeric_rel_tol,
        abs_tol=numeric_abs_tol,
    )

    return ok, "numeric_match" if ok else "numeric_mismatch"


def compare_answer_strings(prediction: Any, gold: Any, **kwargs):
    return compare_parsed_answers(
        parse_answer(prediction),
        parse_answer(gold),
        **kwargs,
    )
