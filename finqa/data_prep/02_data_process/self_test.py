# -*- coding: utf-8 -*-
from answer_normalizer import compare_answer_strings, choose_canonical_gold
from reasoning_builder import build_sft_response

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

    qa = {
        "answer": "",
        "exe_ans": "yes",
        "steps": [
            {"op": "compare_larger1-1", "arg1": "10", "arg2": "3", "res": "yes"}
        ],
        "explanation": "",
    }
    chosen = choose_canonical_gold(qa)
    assert chosen["gold_answer"] == "yes"
    assert chosen["gold_source"] == "steps[-1].res"

    response = build_sft_response(qa, chosen["gold_answer"])
    print("\nSFT response:\n", response)
    print("\nSELF TEST PASSED")

if __name__ == "__main__":
    main()
