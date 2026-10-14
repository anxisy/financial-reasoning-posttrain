import json
from pathlib import Path

path = Path(
    "/root/autodl-tmp/project/finqa_posttrain/"
    "experiments/grpo_qwen3_1.7b/full_v1/"
    "trainer_log_history.json"
)

data = json.loads(path.read_text())

if isinstance(data, dict):
    data = data.get("log_history", [])

for metric in ["kl", "grad_norm"]:
    rows = [
        r for r in data
        if isinstance(r.get(metric), (int, float))
    ]
    rows.sort(key=lambda r: r[metric], reverse=True)

    print(f"\nTOP 10 {metric}")
    for r in rows[:10]:
        print({
            "step": r.get("step"),
            "kl": r.get("kl"),
            "grad_norm": r.get("grad_norm"),
            "loss": r.get("loss"),
            "reward": r.get("reward"),
            "reward_std": r.get("reward_std"),
            "zero_std": r.get("frac_reward_zero_std"),
            "learning_rate": r.get("learning_rate"),
        })