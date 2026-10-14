# -*- coding: utf-8 -*-
from typing import Any, Dict, List
from operator_utils import normalize_operator

def _str(x: Any) -> str:
    return "" if x is None else str(x).strip()

def build_reasoning_lines(qa: Dict[str, Any]) -> List[str]:
    lines = []
    for idx, step in enumerate(qa.get("steps") or [], 1):
        if not isinstance(step, dict):
            continue
        op = normalize_operator(step.get("op", "operation"))
        arg1 = _str(step.get("arg1"))
        arg2 = _str(step.get("arg2"))
        res = _str(step.get("res"))

        if arg2 and arg2.lower() not in {"none", "null"}:
            expr = f"{op}({arg1}, {arg2})"
        else:
            expr = f"{op}({arg1})"

        lines.append(f"Step {idx}: {expr}" + (f" = {res}" if res else ""))

    return lines

def build_sft_response(qa: Dict[str, Any], gold_answer: str) -> str:
    parts = []
    lines = build_reasoning_lines(qa)
    if lines:
        parts.append("\n".join(lines))

    explanation = _str(qa.get("explanation"))
    if explanation:
        parts.append(f"Explanation: {explanation}")

    parts.append(f"<answer>{gold_answer}</answer>")
    return "\n\n".join(parts)
