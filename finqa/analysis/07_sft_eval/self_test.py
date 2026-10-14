# -*- coding: utf-8 -*-
from answer_normalizer import extract_answer_with_method, compare_answer_strings
from evaluate_finqa_peft import manual_serialize

def main():
    p=manual_serialize([
        {"role":"system","content":"SYS"},
        {"role":"user","content":"QUESTION"},
    ])
    assert p=="System:\nSYS\n\nUser:\nQUESTION\n\nAssistant:"
    assert not p.endswith("Assistant:\n")

    a,fmt,m=extract_answer_with_method("Put it in <answer>...</answer>.\n\\boxed{127.4}")
    assert a=="127.4" and fmt is False and m=="boxed"
    ok,_=compare_answer_strings(a,"127.40")
    assert ok

    a,fmt,m=extract_answer_with_method("<answer>24.72%</answer>")
    assert a=="24.72%" and fmt and m=="answer_tag"

    print("Frozen prompt:",repr(p))
    print("Parser:",a,fmt,m)
    print("\nSELF TEST PASSED")

if __name__=="__main__":
    main()
