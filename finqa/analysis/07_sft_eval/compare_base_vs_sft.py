# -*- coding: utf-8 -*-
import argparse, json

def load(p):
    with open(p,"r",encoding="utf-8") as f:return json.load(f)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",required=True); p.add_argument("--sft",required=True)
    a=p.parse_args(); b=load(a.base); s=load(a.sft)

    keys=["accuracy","strict_accuracy","format_accuracy","answer_found_rate",
          "max_token_hit_rate","role_repetition_rate","avg_response_tokens"]

    print("="*88)
    print(f"{'metric':26s}{'Base':>18s}{'SFT':>18s}{'delta':>18s}")
    print("="*88)
    for k in keys:
        bv=b["overall"][k]; sv=s["overall"][k]
        if k=="avg_response_tokens":
            print(f"{k:26s}{bv:18.2f}{sv:18.2f}{sv-bv:+18.2f}")
        else:
            print(f"{k:26s}{100*bv:17.2f}%{100*sv:17.2f}%{100*(sv-bv):+17.2f}pp")

    print("\nDifficulty accuracy:")
    for name in sorted(set(b.get("by_difficulty",{}))|set(s.get("by_difficulty",{}))):
        bv=b.get("by_difficulty",{}).get(name,{}).get("accuracy",0)
        sv=s.get("by_difficulty",{}).get(name,{}).get("accuracy",0)
        print(f"  {name:10s} Base={100*bv:6.2f}% SFT={100*sv:6.2f}% delta={100*(sv-bv):+.2f}pp")

    print("\nOperator accuracy (n>=50):")
    for op in sorted(set(b.get("by_operator",{}))|set(s.get("by_operator",{}))):
        n=s.get("by_operator",{}).get(op,{}).get("n",0)
        if n<50: continue
        bv=b.get("by_operator",{}).get(op,{}).get("accuracy",0)
        sv=s.get("by_operator",{}).get(op,{}).get("accuracy",0)
        print(f"  {op:10s} n={n:4d} Base={100*bv:6.2f}% SFT={100*sv:6.2f}% delta={100*(sv-bv):+.2f}pp")

if __name__=="__main__":
    main()
