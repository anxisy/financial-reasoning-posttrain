# -*- coding: utf-8 -*-
from quality_audit import audit_quality


def show(name, qa):
    result = audit_quality(qa)
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print("annotation_quality:", result["annotation_quality"])
    print("annotation_flags  :", result["annotation_flags"])
    print("semantic_status   :", result["semantic_status"])
    print("semantic_flags    :", result["semantic_flags"])
    return result


def main():
    qa1 = {
        "question": "what is the highest revenue observed in this period?",
        "answer": "4501",
        "exe_ans": "1267.0",
        "program": "table_min(revenue, none)",
        "steps": [{"op": "min2-1", "arg1": "revenue", "arg2": "none", "res": "4501"}],
        "explanation": "it is the maximum value .",
    }
    r1 = show("CASE 1: highest revenue + min", qa1)
    assert r1["semantic_status"] == "suspicious"
    assert "question_asks_max_but_ops_use_min" in r1["semantic_flags"]

    qa2 = {
        "question": "what was the greatest foreign currency translation loss, in millions?",
        "answer": "-139.9",
        "exe_ans": "-139.6",
        "program": "table_min(foreign currency translation, none)",
        "steps": [{"op": "min2-1", "arg1": "foreign currency translation", "arg2": "none", "res": "-139.9"}],
        "explanation": "",
    }
    r2 = show("CASE 2: greatest loss + min", qa2)
    assert "question_asks_max_but_ops_use_min" not in r2["semantic_flags"]

    qa3 = {
        "question": "as of the current year, did the firm meet the eventual objective of an estimated basel iii tier i common ratio of 9.5%?",
        "answer": "no",
        "exe_ans": "no",
        "program": "greater(8.7, 9.5)",
        "steps": [{"op": "compare_larger2-1", "arg1": "8.7", "arg2": "9.5", "res": "no"}],
        "explanation": "",
    }
    r3 = show("CASE 3: percentage context + yes/no answer", qa3)
    assert "percentage_question_but_gold_type_is_non_numeric" not in r3["semantic_flags"]
    assert r3["semantic_status"] == "not_flagged"

    qa4 = {
        "question": "does the current estimated basel iii tier 1 ratio exceed the requirement under basel iii rules?",
        "answer": "no",
        "exe_ans": "no",
        "program": "greater(8.7%, 9.5%)",
        "steps": [{"op": "compare_larger1-1", "arg1": "8.7%", "arg2": "9.5%", "res": "no"}],
        "explanation": "",
    }
    r4 = show("CASE 4: ratio comparison + no", qa4)
    assert "percentage_question_but_gold_type_is_non_numeric" not in r4["semantic_flags"]
    assert r4["semantic_status"] == "not_flagged"

    print("\n" + "=" * 80)
    print("SEMANTIC AUDIT v1.1.1 SELF TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
