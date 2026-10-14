# -*- coding: utf-8 -*-
import re
from typing import Any, List

PROGRAM_OP_RE = re.compile(r"([A-Za-z_]+)\s*\(")

def _s(x: Any) -> str:
    return "" if x is None else str(x).strip().lower()

def normalize_operator(op: str) -> str:
    op = _s(op)
    mapping = [
        ("table_average", "average"),
        ("table_max", "max"),
        ("table_min", "min"),
        ("table_sum", "sum"),
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

def extract_program_ops(program: Any) -> List[str]:
    if program is None:
        return []
    text = " ".join(map(str, program)) if isinstance(program, list) else str(program)
    return [normalize_operator(x) for x in PROGRAM_OP_RE.findall(text)]

def extract_step_ops(steps: Any) -> List[str]:
    result = []
    for step in steps or []:
        if isinstance(step, dict) and step.get("op"):
            result.append(normalize_operator(step["op"]))
    return result
