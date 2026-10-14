# -*- coding: utf-8 -*-
from __future__ import annotations
import math, re
from dataclasses import dataclass
from typing import Any, Optional, Tuple

ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)
BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]+)\}", re.I)
NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)\s*%?")
YESNO_TOKEN_RE = re.compile(r"\b(yes|no|true|false)\b", re.I)

PLACEHOLDERS = {"", "...", "…", "xxx", "x", "answer", "your answer", "final answer",
                "result", "value", "number", "n/a", "na"}
ROLE_ONLY = {"assistant", "assistant:", "user", "user:", "system", "system:"}

@dataclass
class ParsedAnswer:
    raw: str
    kind: str
    value: Any
    has_percent_sign: bool = False

def _s(v): return "" if v is None else str(v).strip()

def _clean(v):
    return _s(v).strip("` \t\r\n").strip()

def _is_placeholder(v):
    n = re.sub(r"\s+", " ", _clean(v).lower()).strip(" .,:;!?")
    if n in PLACEHOLDERS or n in ROLE_ONLY:
        return True
    return bool(n) and set(n) <= {".", "…", "-", "_"}

def _atomic(text):
    text = _clean(text)
    if not text or _is_placeholder(text):
        return ""
    yn = list(YESNO_TOKEN_RE.finditer(text))
    nums = list(NUMBER_RE.finditer(text))
    if yn:
        m = yn[-1]
        if not text[m.end():].strip(" .,:;!?"):
            return "yes" if m.group(1).lower() in {"yes", "true"} else "no"
    if nums:
        return nums[-1].group(0).strip()
    return text

def extract_answer_with_method(text: str, allow_fallback: bool = True) -> Tuple[str, bool, str]:
    text = "" if text is None else str(text)

    tags = ANSWER_TAG_RE.findall(text)
    for c in reversed(tags):
        c = _clean(c)
        if not _is_placeholder(c):
            a = _atomic(c)
            if a:
                return a, True, "answer_tag"

    if not allow_fallback:
        return "", False, "none"

    for c in reversed(BOXED_RE.findall(text)):
        a = _atomic(c)
        if a and not _is_placeholder(a):
            return a, False, "boxed"

    lines = [x.strip() for x in text.splitlines() if x.strip()]
    cue = re.compile(r"(?:final\s+answer|answer)\s*(?:is|=|:|：)\s*(.+)$", re.I)
    for line in reversed(lines):
        m = cue.search(line)
        if not m:
            continue
        raw = _clean(m.group(1))
        if not raw or _is_placeholder(raw):
            continue
        nums = list(NUMBER_RE.finditer(raw))
        yn = list(YESNO_TOKEN_RE.finditer(raw))
        if (len(raw) <= 120 and len(nums) <= 1) or (yn and not nums and len(raw) <= 120):
            a = _atomic(raw)
            if a:
                return a, False, "final_cue"

    for line in reversed(lines):
        if _is_placeholder(line) or line.lower().strip() in ROLE_ONLY:
            continue
        a = _atomic(line)
        if a:
            return a, False, "last_line"

    return "", False, "none"

def parse_answer(v):
    raw = _s(v)
    if not raw:
        return ParsedAnswer(raw, "empty", None)
    low = raw.lower().strip().strip(" .,:;")
    if low in {"yes", "y", "true"}:
        return ParsedAnswer(raw, "yesno", "yes")
    if low in {"no", "n", "false"}:
        return ParsedAnswer(raw, "yesno", "no")
    m = re.search(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)", raw)
    if m:
        try:
            n = float(m.group(0).replace(",", ""))
            isp = "%" in raw
            return ParsedAnswer(raw, "percent" if isp else "number", n, isp)
        except ValueError:
            pass
    return ParsedAnswer(raw, "text", low)

def compare_answer_strings(prediction, gold, numeric_rel_tol=1e-3,
                           numeric_abs_tol=1e-2, percent_abs_tol=0.10):
    p, g = parse_answer(prediction), parse_answer(gold)
    if g.kind == "empty": return False, "missing_gold"
    if p.kind == "empty": return False, "empty_prediction"
    if g.kind == "yesno":
        ok = p.kind == "yesno" and p.value == g.value
        return ok, "yesno_match" if ok else "yesno_mismatch"
    if g.kind == "text":
        ok = p.kind == "text" and p.value == g.value
        return ok, "text_match" if ok else "text_mismatch"
    if p.kind not in {"number", "percent"} or g.kind not in {"number", "percent"}:
        return False, "type_mismatch"
    pv, gv = float(p.value), float(g.value)
    if g.kind == "percent":
        if math.isclose(pv, gv, rel_tol=0.0, abs_tol=percent_abs_tol):
            return True, "percent_point_match"
        if p.kind == "number" and math.isclose(pv * 100.0, gv,
                                               rel_tol=numeric_rel_tol,
                                               abs_tol=percent_abs_tol):
            return True, "percent_fraction_match"
        return False, "percent_mismatch"
    if p.kind == "percent":
        return False, "unexpected_percent"
    ok = math.isclose(pv, gv, rel_tol=numeric_rel_tol, abs_tol=numeric_abs_tol)
    return ok, "numeric_match" if ok else "numeric_mismatch"
