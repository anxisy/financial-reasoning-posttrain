# -*- coding: utf-8 -*-
import tempfile

from grpo_data import manual_prompt
from reward_utils import GRPORewardWithDiagnostics, summarize_diagnostics


class Tok:
    def __call__(self, text, add_special_tokens=False, truncation=False):
        return {"input_ids": list(range(len(text.split())))}


def main():
    prompt = manual_prompt(
        [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "Q"},
        ]
    )

    assert prompt == "System:\nSYS\n\nUser:\nQ\n\nAssistant:"
    assert prompt.endswith("Assistant:")
    assert not prompt.endswith("Assistant:\n")

    with tempfile.TemporaryDirectory() as d:
        reward = GRPORewardWithDiagnostics(
            Tok(),
            d,
            num_generations=4,
            max_completion_length=256,
        )

        rewards = reward(
            completions=[
                "<answer>10</answer>",
                "<answer>10</answer>",
                "<answer>9</answer>",
                "10",
            ],
            gold_answer=["10"],
            example_id=["x"],
            difficulty=["easy"],
            operators_json=['["divide"]'],
        )

        assert rewards == [1.1, 1.1, 0.1, 1.0]

        summary = summarize_diagnostics(d)
        assert summary["num_groups"] == 1
        assert summary["mixed_group_ratio"] == 1.0
        assert summary["all_correct_group_ratio"] == 0.0
        assert summary["all_wrong_group_ratio"] == 0.0
        assert summary["by_operator"]["divide"]["groups"] == 1

    print("Manual prompt:", repr(prompt))
    print("Rewards:", rewards)
    print("SELF TEST PASSED")


if __name__ == "__main__":
    main()
