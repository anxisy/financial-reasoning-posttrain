# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple
from clean_context import clean_example_context, table_to_text

def build_sections(example: dict, system_prompt: str) -> Dict[str, str]:
    cleaned = clean_example_context(example)
    qa = example.get("qa", {})
    return {
        "system": system_prompt.strip(),
        "question": str(qa.get("question", "")).strip(),
        "table": table_to_text(cleaned["table"]),
        "pre_text": "\n".join(cleaned["pre_text"]),
        "post_text": "\n".join(cleaned["post_text"]),
    }

def compose_user_prompt(sections: Dict[str, str]) -> str:
    return (
        f"Question:\n{sections['question']}\n\n"
        f"Financial Table:\n{sections['table']}\n\n"
        f"Financial Report - Pre Text:\n{sections['pre_text'] or '(none)'}\n\n"
        f"Financial Report - Post Text:\n{sections['post_text'] or '(none)'}"
    )

def build_messages(sections: Dict[str, str]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": sections["system"]},
        {"role": "user", "content": compose_user_prompt(sections)},
    ]

def _token_len(tokenizer, text: str) -> int:
    return len(
        tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
    )

def _decode_prefix(tokenizer, text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return ""
    ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True).strip()

def _decode_suffix(tokenizer, text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return ""
    ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]
    return tokenizer.decode(ids[-max_tokens:], skip_special_tokens=True).strip()

def truncate_report_to_budget(
    tokenizer,
    sections: Dict[str, str],
    max_prompt_tokens: int,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    original_user = compose_user_prompt(sections)
    original_len = _token_len(
        tokenizer,
        sections["system"] + "\n\n" + original_user,
    )

    if original_len <= max_prompt_tokens:
        return sections, {
            "original_prompt_tokens": original_len,
            "final_prompt_tokens": original_len,
            "was_truncated": 0,
        }

    base = dict(sections)
    base["pre_text"] = ""
    base["post_text"] = ""

    protected_len = _token_len(
        tokenizer,
        base["system"] + "\n\n" + compose_user_prompt(base),
    )
    remaining = max(0, max_prompt_tokens - protected_len)

    pre_budget = remaining // 2
    post_budget = remaining - pre_budget

    base["pre_text"] = _decode_prefix(tokenizer, sections["pre_text"], pre_budget)
    base["post_text"] = _decode_suffix(tokenizer, sections["post_text"], post_budget)

    final_len = _token_len(
        tokenizer,
        base["system"] + "\n\n" + compose_user_prompt(base),
    )

    overflow = max(0, final_len - max_prompt_tokens)
    if overflow > 0 and base["post_text"]:
        post_ids = tokenizer(
            base["post_text"],
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        keep = max(0, len(post_ids) - overflow - 4)
        base["post_text"] = tokenizer.decode(
            post_ids[-keep:] if keep else [],
            skip_special_tokens=True,
        ).strip()

    final_len = _token_len(
        tokenizer,
        base["system"] + "\n\n" + compose_user_prompt(base),
    )

    overflow = max(0, final_len - max_prompt_tokens)
    if overflow > 0 and base["pre_text"]:
        pre_ids = tokenizer(
            base["pre_text"],
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        keep = max(0, len(pre_ids) - overflow - 4)
        base["pre_text"] = tokenizer.decode(
            pre_ids[:keep],
            skip_special_tokens=True,
        ).strip()

    final_len = _token_len(
        tokenizer,
        base["system"] + "\n\n" + compose_user_prompt(base),
    )

    return base, {
        "original_prompt_tokens": original_len,
        "final_prompt_tokens": final_len,
        "was_truncated": 1,
    }
