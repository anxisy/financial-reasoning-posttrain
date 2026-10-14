# -*- coding: utf-8 -*-
from typing import Iterable, List
from answer_normalizer import compare_answer_strings, extract_answer_from_response

def score_response(response: str, gold_answer: str, **kwargs) -> dict:
    pred_answer, has_tag = extract_answer_from_response(response, allow_fallback=True)
    correct, reason = compare_answer_strings(pred_answer, gold_answer, **kwargs)
    return {
        "prediction_answer": pred_answer,
        "gold_answer": gold_answer,
        "correct": bool(correct),
        "match_reason": reason,
        "format_ok": bool(has_tag),
    }

def correctness_reward(responses: Iterable[str], gold_answers: Iterable[str], **kwargs) -> List[float]:
    return [
        1.0 if score_response(r, g, **kwargs)["correct"] else 0.0
        for r, g in zip(responses, gold_answers)
    ]

def format_reward(responses: Iterable[str]) -> List[float]:
    rewards = []
    for response in responses:
        _, ok = extract_answer_from_response(response, allow_fallback=False)
        rewards.append(1.0 if ok else 0.0)
    return rewards

def combined_reward(
    responses: Iterable[str],
    gold_answers: Iterable[str],
    correctness_weight: float = 1.0,
    format_weight: float = 0.1,
    **kwargs,
) -> List[float]:
    c = correctness_reward(responses, gold_answers, **kwargs)
    f = format_reward(responses)
    return [correctness_weight * x + format_weight * y for x, y in zip(c, f)]
