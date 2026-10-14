# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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

def extract_answer_from_response(text: str, allow_fallback: bool = True) -> Tuple[str, bool]:
    tagged = extract_tagged_answer(text)
    if tagged is not None:
        return tagged, True
    if not allow_fallback:
        return "", False
    lines = [x.strip() for x in str(text).splitlines() if x.strip()]
    if not lines:
        return "", False
    candidate = re.sub(
        r"^(final\s+answer|answer)\s*[:：]\s*",
        "",
        lines[-1],
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
            value = float(match.group(0).replace(",", ""))
            is_percent = "%" in raw
            return ParsedAnswer(raw, "percent" if is_percent else "number", value, is_percent)
        except ValueError:
            pass

    return ParsedAnswer(raw, "text", lowered)

def final_step_result(qa: Dict[str, Any]) -> str:
    steps = qa.get("steps") or []
    if not steps or not isinstance(steps[-1], dict):
        return ""
    return _string(steps[-1].get("res"))

def choose_canonical_gold(qa: Dict[str, Any]) -> Dict[str, Any]:
    answer = _string(qa.get("answer"))
    step_result = final_step_result(qa)
    exe_ans = _string(qa.get("exe_ans"))

    if answer:
        gold, source = answer, "answer"
    elif step_result:
        gold, source = step_result, "steps[-1].res"
    elif exe_ans:
        gold, source = exe_ans, "exe_ans"
    else:
        gold, source = "", "missing"

    return {
        "gold_answer": gold,
        "gold_source": source,
        "answer": answer,
        "step_result": step_result,
        "exe_ans": exe_ans,
    }

def compare_parsed_answers(
    pred: ParsedAnswer,
    gold: ParsedAnswer,
    numeric_rel_tol: float = 1e-3,
    numeric_abs_tol: float = 1e-2,
    percent_abs_tol: float = 0.10,
) -> Tuple[bool, str]:
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

    p, g = float(pred.value), float(gold.value)

    if gold.kind == "percent":
        if math.isclose(p, g, rel_tol=0.0, abs_tol=percent_abs_tol):
            return True, "percent_point_match"
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

    ok = math.isclose(p, g, rel_tol=numeric_rel_tol, abs_tol=numeric_abs_tol)
    return ok, "numeric_match" if ok else "numeric_mismatch"

def compare_answer_strings(prediction: Any, gold: Any, **kwargs) -> Tuple[bool, str]:
    return compare_parsed_answers(parse_answer(prediction), parse_answer(gold), **kwargs)
