# -*- coding: utf-8 -*-
from answer_normalizer import (
    compare_answer_strings,
    extract_answer_with_method,
)
from metrics import compute_metrics


def main():
    # Regression: placeholder must NOT count as a real answer tag.
    text1 = r"""Assistant:
To solve it:
Place the final answer inside <answer>...</answer>.
637 / 5 = 127.4
\[
\boxed{127.4}
\]
"""
    ans, fmt, method = extract_answer_with_method(text1)
    print("case1:", ans, fmt, method)
    assert ans == "127.4"
    assert fmt is False
    assert method == "boxed"
    ok, _ = compare_answer_strings(ans, "127.40")
    assert ok

    # Regression: another exact smoke-test failure.
    text2 = r"""Assistant:
555.3 - 536.7 = 18.6
Place the final answer inside <answer>...</answer>.
\[
\boxed{18.6}
\]
"""
    ans, fmt, method = extract_answer_with_method(text2)
    print("case2:", ans, fmt, method)
    assert ans == "18.6"
    assert fmt is False
    assert method == "boxed"
    ok, _ = compare_answer_strings(ans, "18.6")
    assert ok

    # Real tagged answer.
    ans, fmt, method = extract_answer_with_method(
        "Reasoning...\n<answer>53%</answer>"
    )
    print("case3:", ans, fmt, method)
    assert ans == "53%"
    assert fmt is True
    assert method == "answer_tag"

    # Repeated Assistant garbage should not become the answer.
    ans, fmt, method = extract_answer_with_method(
        "\n".join(["Assistant"] * 100)
    )
    print("case4:", repr(ans), fmt, method)
    assert ans == ""
    assert fmt is False
    assert method == "none"

    # Answer sentence: extract compact percentage.
    ans, fmt, method = extract_answer_with_method(
        "Answer: The deferred fuel cost revisions percentage is 6.2%."
    )
    print("case5:", ans, fmt, method)
    assert ans == "6.2%"
    assert fmt is False
    assert method == "final_cue"

    rows = [
        {
            "correct": True,
            "format_ok": False,
            "prediction_answer": "127.4",
            "hit_max_new_tokens": False,
            "role_repetition_flag": False,
            "response_tokens": 50,
            "extraction_method": "boxed",
            "metadata": {
                "difficulty": "easy",
                "operators": ["divide"],
                "original_prompt_tokens": 800,
                "annotation_quality": "high",
                "semantic_status": "not_flagged",
                "was_truncated": 0,
            },
        },
        {
            "correct": False,
            "format_ok": False,
            "prediction_answer": "",
            "hit_max_new_tokens": True,
            "role_repetition_flag": True,
            "response_tokens": 256,
            "extraction_method": "none",
            "metadata": {
                "difficulty": "medium",
                "operators": ["multiply"],
                "original_prompt_tokens": 1100,
                "annotation_quality": "medium",
                "semantic_status": "not_flagged",
                "was_truncated": 0,
            },
        },
    ]

    m = compute_metrics(rows)
    assert m["overall"]["n"] == 2
    assert abs(m["overall"]["accuracy"] - 0.5) < 1e-9
    assert abs(m["overall"]["answer_found_rate"] - 0.5) < 1e-9
    assert abs(m["overall"]["max_token_hit_rate"] - 0.5) < 1e-9
    assert abs(m["overall"]["role_repetition_rate"] - 0.5) < 1e-9

    print("\nSELF TEST PASSED")


if __name__ == "__main__":
    main()
