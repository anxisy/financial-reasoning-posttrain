# -*- coding: utf-8 -*-
from answer_normalizer import compare_answer_strings, extract_answer_from_response
from metrics import compute_metrics, context_bucket


def main():
    cases = [
        ("53%", "53%", True),
        ("0.53", "53%", True),
        ("13.0", "$ 13 million", True),
        ("yes", "yes", True),
        ("42%", "53%", False),
    ]

    for pred, gold, expected in cases:
        ok, reason = compare_answer_strings(pred, gold)
        print(pred, "|", gold, "=>", ok, reason)
        assert ok == expected

    ans, fmt = extract_answer_from_response(
        "Reasoning...\n<answer>53%</answer>"
    )
    assert ans == "53%"
    assert fmt is True

    rows = [
        {
            "correct": True,
            "format_ok": True,
            "response_tokens": 10,
            "metadata": {
                "difficulty": "easy",
                "operators": ["divide"],
                "original_prompt_tokens": 900,
                "annotation_quality": "high",
                "semantic_status": "not_flagged",
                "was_truncated": 0,
            },
        },
        {
            "correct": False,
            "format_ok": False,
            "response_tokens": 20,
            "metadata": {
                "difficulty": "hard",
                "operators": ["divide", "subtract"],
                "original_prompt_tokens": 1700,
                "annotation_quality": "medium",
                "semantic_status": "not_flagged",
                "was_truncated": 1,
            },
        },
    ]

    m = compute_metrics(rows)
    assert m["overall"]["n"] == 2
    assert abs(m["overall"]["accuracy"] - 0.5) < 1e-9
    assert m["by_operator"]["divide"]["n"] == 2
    assert m["by_operator"]["subtract"]["n"] == 1
    assert context_bucket(900) == "<=1024"
    assert context_bucket(1700) == "1537-2048"

    print("\nSELF TEST PASSED")


if __name__ == "__main__":
    main()
