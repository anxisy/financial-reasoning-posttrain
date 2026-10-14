# -*- coding: utf-8 -*-
import argparse
import json


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--before-name", default="SFT")
    p.add_argument("--after-name", default="GRPO")
    args = p.parse_args()

    before = load(args.before)
    after = load(args.after)

    keys = [
        "accuracy",
        "strict_accuracy",
        "format_accuracy",
        "answer_found_rate",
        "max_token_hit_rate",
        "role_repetition_rate",
        "avg_response_tokens",
    ]

    print("=" * 92)
    print(
        f"{'metric':26s}"
        f"{args.before_name:>18s}"
        f"{args.after_name:>18s}"
        f"{'delta':>18s}"
    )
    print("=" * 92)

    for key in keys:
        b = before["overall"][key]
        a = after["overall"][key]

        if key == "avg_response_tokens":
            print(
                f"{key:26s}"
                f"{b:18.2f}"
                f"{a:18.2f}"
                f"{a-b:+18.2f}"
            )
        else:
            print(
                f"{key:26s}"
                f"{100*b:17.2f}%"
                f"{100*a:17.2f}%"
                f"{100*(a-b):+17.2f}pp"
            )

    print("\nDifficulty accuracy:")
    names = sorted(
        set(before.get("by_difficulty", {}))
        | set(after.get("by_difficulty", {}))
    )

    for name in names:
        b = (
            before.get("by_difficulty", {})
            .get(name, {})
            .get("accuracy", 0.0)
        )
        a = (
            after.get("by_difficulty", {})
            .get(name, {})
            .get("accuracy", 0.0)
        )

        print(
            f"  {name:10s} "
            f"{args.before_name}={100*b:6.2f}% "
            f"{args.after_name}={100*a:6.2f}% "
            f"delta={100*(a-b):+.2f}pp"
        )

    print("\nOperator accuracy (n>=50):")
    ops = sorted(
        set(before.get("by_operator", {}))
        | set(after.get("by_operator", {}))
    )

    for op in ops:
        n = (
            after.get("by_operator", {})
            .get(op, {})
            .get("n", 0)
        )
        if n < 50:
            continue

        b = (
            before.get("by_operator", {})
            .get(op, {})
            .get("accuracy", 0.0)
        )
        a = (
            after.get("by_operator", {})
            .get(op, {})
            .get("accuracy", 0.0)
        )

        print(
            f"  {op:10s} n={n:4d} "
            f"{args.before_name}={100*b:6.2f}% "
            f"{args.after_name}={100*a:6.2f}% "
            f"delta={100*(a-b):+.2f}pp"
        )


if __name__ == "__main__":
    main()
