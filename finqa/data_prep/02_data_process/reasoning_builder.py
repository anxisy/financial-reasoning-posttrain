# -*- coding: utf-8 -*-
from typing import Any, Dict, List

def _str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()

def normalize_op_name(op: str) -> str:
    op = _str(op).lower()
    mapping = [
        ("compare_larger", "greater"),
        ("greater", "greater"),
        ("minus", "subtract"),
        ("subtract", "subtract"),
        ("add", "add"),
        ("divide", "divide"),
        ("multiply", "multiply"),
        ("average", "average"),
        ("max", "max"),
        ("min", "min"),
        ("sum", "sum"),
        ("exp", "exp"),
    ]
    for prefix, normalized in mapping:
        if op.startswith(prefix):
            return normalized
    return op

def build_reasoning_lines(qa: Dict[str, Any]) -> List[str]:
    steps = qa.get("steps") or []
    lines = []
    for idx, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            continue
        op = normalize_op_name(step.get("op", "operation"))
        arg1 = _str(step.get("arg1"))
        arg2 = _str(step.get("arg2"))
        res = _str(step.get("res"))
        if arg2 and arg2.lower() not in {"none", "null"}:
            expr = f"{op}({arg1}, {arg2})"
        else:
            expr = f"{op}({arg1})"
        if res:
            lines.append(f"Step {idx}: {expr} = {res}")
        else:
            lines.append(f"Step {idx}: {expr}")
    return lines

def build_sft_response(qa: Dict[str, Any], gold_answer: str) -> str:
    lines = build_reasoning_lines(qa)
    parts = []
    if lines:
        parts.append("\n".join(lines))
    explanation = _str(qa.get("explanation"))
    if explanation:
        parts.append(f"Explanation: {explanation}")
    parts.append(f"<answer>{gold_answer}</answer>")
    return "\n\n".join(parts)
