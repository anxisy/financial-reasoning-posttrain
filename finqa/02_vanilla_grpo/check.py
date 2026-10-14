import json
from pathlib import Path

root = Path(
    "/root/autodl-tmp/project/finqa_posttrain/"
    "experiments/grpo_qwen3_1.7b/full_v1"
)

data = json.loads(
    (root / "trainer_log_history.json").read_text()
)

if isinstance(data, dict):
    data = data.get("log_history", [])

for r in data:
    step = r.get("step")
    if isinstance(step, (int, float)) and 3500 <= step <= 3600:
        print({
            "step": step,
            "kl": r.get("kl"),
            "grad_norm": r.get("grad_norm"),
            "loss": r.get("loss"),
            "reward": r.get("reward"),
            "reward_std": r.get("reward_std"),
            "zero_std": r.get("frac_reward_zero_std"),
        })
