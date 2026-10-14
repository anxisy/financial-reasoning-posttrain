# -*- coding: utf-8 -*-
from answer_normalizer import extract_answer_with_method

def check(text, expected, method=None):
    ans, fmt, got_method = extract_answer_with_method(text)
    print(repr(ans), fmt, got_method)
    assert ans == expected, (ans, expected)
    if method is not None:
        assert got_method == method, (got_method, method)

def main():
    # Original placeholder regression.
    check(
        r"""Place the final answer inside <answer>...</answer>.
\[
\boxed{127.4}
\]""",
        "127.4",
        "boxed",
    )

    # Edge case discovered in chat smoke:
    # "Answer:" is only a heading; do not swallow the next prose line.
    check(
        """Answer:
To calculate the increase between 2016 and 2017, we subtract the values.
Therefore, the increase is 1,095,000 (in thousands).""",
        "1,095,000",
        "last_line",
    )

    # Normal same-line answer cue should still work.
    check(
        "Answer: The deferred fuel cost revisions percentage is 6.2%.",
        "6.2%",
        "final_cue",
    )

    # Empty Answer heading followed by direct answer.
    check(
        """Answer:
$ 2255 million""",
        "2255",
        "last_line",
    )

    # Valid answer tag remains highest priority.
    check(
        "Reasoning...\n<answer>24.72%</answer>",
        "24.72%",
        "answer_tag",
    )

    print("\nPARSER PATCH SELF TEST PASSED")

if __name__ == "__main__":
    main()
