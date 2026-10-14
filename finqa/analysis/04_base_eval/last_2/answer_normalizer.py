# -*- coding: utf-8 -*-
"""
FinQA baseline answer parser v1.1.

Main fix:
- Ignore placeholder tags such as <answer>...</answer>.
- Prefer a real <answer>...</answer>.
- Otherwise try \\boxed{...}, explicit final-answer phrases, then a conservative
  last-line fallback.

`format_ok=True` ONLY when a valid non-placeholder <answer>...</answer> exists.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Optional, Tuple


ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)
BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]+)\}", re.I)
NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)\s*%?")
YESNO_TOKEN_RE = re.compile(r"\b(yes|no|true|false)\b", re.I)

FINAL_CUE_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|=|:|：)\s*(.+)",
    re.I,
)

PLACEHOLDER_VALUES = {
    "",
    "...",
    "…",
    "xxx",
    "x",
    "answer",
    "your answer",
    "final answer",
    "result",
    "value",
    "number",
    "n/a",
    "na",
}

ROLE_ONLY_VALUES = {
    "assistant",
    "assistant:",
    "user",
    "user:",
    "system",
    "system:",
}


@dataclass
class ParsedAnswer:
    raw: str
    kind: str
    value: Any
    has_percent_sign: bool = False


def _string(value) -> str:
    return "" if value is None else str(value).strip()


def _clean_candidate(value: str) -> str:
    value = _string(value)
    value = value.strip("` \t\r\n")
    value = value.strip()
    return value


def _is_placeholder(value: str) -> bool:
    value = _clean_candidate(value)
    normalized = re.sub(r"\s+", " ", value.lower()).strip(" .,:;!?")
    if normalized in PLACEHOLDER_VALUES:
        return True
    if normalized in ROLE_ONLY_VALUES:
        return True
    if normalized and set(normalized) <= {".", "…", "-", "_"}:
        return True
    return False


def _atomic_answer_from_text(text: str) -> str:
    """
    Extract a compact answer from a sentence/line.

    Preference:
    1) final yes/no token if present
    2) last numeric/percentage token
    3) cleaned text
    """
    text = _clean_candidate(text)
    if not text or _is_placeholder(text):
        return ""

    yesno_matches = list(YESNO_TOKEN_RE.finditer(text))
    number_matches = list(NUMBER_RE.finditer(text))

    # If a yes/no token is very near the end, prefer it.
    if yesno_matches:
        last = yesno_matches[-1]
        tail = text[last.end():].strip(" .,:;!?")
        if not tail:
            token = last.group(1).lower()
            return "yes" if token in {"yes", "true"} else "no"

    if number_matches:
        return number_matches[-1].group(0).strip()

    return text


def extract_valid_tagged_answer(text: str) -> Optional[str]:
    if text is None:
        return None

    matches = ANSWER_TAG_RE.findall(str(text))
    if not matches:
        return None

    # Search from the last tag backwards and ignore placeholders like "...".
    for candidate in reversed(matches):
        candidate = _clean_candidate(candidate)
        if not _is_placeholder(candidate):
            return _atomic_answer_from_text(candidate)

    return None


def extract_answer_with_method(
    text: str,
    allow_fallback: bool = True,
) -> Tuple[str, bool, str]:
    """
    Returns:
        answer, format_ok, extraction_method

    extraction_method is one of:
        answer_tag
        boxed
        final_cue
        last_line
        none
    """
    text = "" if text is None else str(text)

    # 1) Real <answer>...</answer>
    tagged = extract_valid_tagged_answer(text)
    if tagged:
        return tagged, True, "answer_tag"

    if not allow_fallback:
        return "", False, "none"

    # 2) LaTeX boxed result.
    boxed = BOXED_RE.findall(text)
    for candidate in reversed(boxed):
        candidate = _atomic_answer_from_text(candidate)
        if candidate and not _is_placeholder(candidate):
            return candidate, False, "boxed"

    # 3) Explicit final-answer / answer cue.
    cue_matches = list(FINAL_CUE_RE.finditer(text))
    for match in reversed(cue_matches):
        candidate = _atomic_answer_from_text(match.group(1))
        if candidate and not _is_placeholder(candidate):
            return candidate, False, "final_cue"

    # 4) Conservative last non-empty line fallback.
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    for line in reversed(lines):
        if _is_placeholder(line):
            continue

        # Avoid role labels / repeated role-token garbage.
        normalized = line.lower().strip()
        if normalized in ROLE_ONLY_VALUES:
            continue

        candidate = _atomic_answer_from_text(line)
        if candidate and not _is_placeholder(candidate):
            return candidate, False, "last_line"

    return "", False, "none"


def extract_answer_from_response(
    text: str,
    allow_fallback: bool = True,
) -> Tuple[str, bool]:
    answer, format_ok, _ = extract_answer_with_method(
        text,
        allow_fallback=allow_fallback,
    )
    return answer, format_ok


def parse_answer(value: Any) -> ParsedAnswer:
    raw = _string(value)

    if not raw:
        return ParsedAnswer(raw, "empty", None)

    lowered = raw.lower().strip().strip(" .,:;")

    if lowered in {"yes", "y", "true"}:
        return ParsedAnswer(raw, "yesno", "yes")

    if lowered in {"no", "n", "false"}:
        return ParsedAnswer(raw, "yesno", "no")

    # The extraction stage usually returns an atomic answer already,
    # but keep this permissive for evaluator compatibility.
    match = re.search(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)", raw)
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
