# -*- coding: utf-8 -*-
from answer_normalizer import compare_answer_strings
from quality_audit import audit_quality

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
        "question": "what was the highest revenue observed in this period?",
        "answer": "4501",
        "exe_ans": "1267.0",
        "program": "table_min(revenue, none)",
        "steps": [{"op": "min2-1", "arg1": "revenue", "arg2": "none", "res": "4501"}],
        "explanation": "it is the maximum value .",
    }

    audit = audit_quality(qa)
    print("\nAudit:", audit)

    assert audit["annotation_quality"] == "high"
    assert audit["semantic_status"] == "suspicious"
    assert "question_asks_max_but_ops_use_min" in audit["semantic_flags"]

    print("\nSELF TEST PASSED")

if __name__ == "__main__":
    main()
