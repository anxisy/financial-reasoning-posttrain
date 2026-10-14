# -*- coding: utf-8 -*-
import re
from typing import Iterable, List, Sequence

_WS_RE = re.compile(r"\s+")
_ONLY_PUNCT_RE = re.compile(r"^[\s\.\,\;\:\-\_]+$")

def clean_text_line(text) -> str:
    if text is None:
        return ""
    text = str(text).replace("\u00a0", " ").strip()
    text = _WS_RE.sub(" ", text)
    if not text or _ONLY_PUNCT_RE.fullmatch(text):
        return ""
    return text

def clean_text_list(items: Iterable) -> List[str]:
    cleaned = []
    previous = None
    for item in items or []:
        line = clean_text_line(item)
        if not line:
            continue
        if line == previous:
            continue
        cleaned.append(line)
        previous = line
    return cleaned

def clean_table(table: Sequence[Sequence]) -> List[List[str]]:
    if not table:
        return []
    cleaned = []
    for row in table:
        new_row = [clean_text_line(cell) for cell in row]
        if any(cell for cell in new_row):
            cleaned.append(new_row)
    return cleaned

def table_to_text(table: Sequence[Sequence[str]]) -> str:
    table = clean_table(table)
    if not table:
        return "(empty table)"
    return "\n".join(" | ".join(row) for row in table)

def clean_example_context(example: dict) -> dict:
    return {
        "pre_text": clean_text_list(example.get("pre_text", [])),
        "post_text": clean_text_list(example.get("post_text", [])),
        "table": clean_table(example.get("table", [])),
    }
