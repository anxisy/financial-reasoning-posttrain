# -*- coding: utf-8 -*-
"""
Clean custom Dynamic Sampling diagnostics back to a saved Trainer checkpoint
before resuming training.

Why:
If training is interrupted after checkpoint-N, custom JSONL diagnostics may
already contain rows from steps N+1 ... current_step. Resuming from checkpoint-N
would regenerate those steps and append duplicate diagnostics.

Usage:
python prepare_dynamic_resume.py \
  --output-dir /root/autodl-tmp/project/finqa_posttrain/experiments/grpo_dynamic_qwen3_1.7b/full_v1 \
  --checkpoint latest
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


def checkpoint_step(value: str, out: Path) -> tuple[int, Path]:
    if value == "latest":
        candidates = []
        for p in out.glob("checkpoint-*"):
            m = re.fullmatch(r"checkpoint-(\d+)", p.name)
            if p.is_dir() and m:
                candidates.append((int(m.group(1)), p))
        if not candidates:
            raise SystemExit(f"No checkpoint-* found under {out}")
        return max(candidates, key=lambda x: x[0])

    p = Path(value)
    m = re.fullmatch(r"checkpoint-(\d+)", p.name)
    if not p.is_dir() or not m:
        raise SystemExit(f"Invalid checkpoint path: {p}")
    return int(m.group(1)), p


def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def backup(path: Path):
    if not path.exists():
        return None
    bak = path.with_suffix(path.suffix + ".before_resume.bak")
    shutil.copy2(path, bak)
    return bak


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--checkpoint", default="latest")
    args = ap.parse_args()

    out = Path(args.output_dir)
    step, ckpt = checkpoint_step(args.checkpoint, out)

    events_path = out / "dynamic_sampling_events.jsonl"
    groups_path = out / "group_diagnostics.jsonl"
    rollouts_path = out / "rollout_diagnostics.jsonl"

    events = read_jsonl(events_path)
    if not events:
        raise SystemExit(f"No events found: {events_path}")

    kept_events = [
        x for x in events
        if int(x.get("optimizer_step", 0)) <= step
    ]
    removed_events = len(events) - len(kept_events)

    retained_reward_calls = {
        int(x["reward_call"])
        for x in kept_events
        if x.get("reward_call") is not None
    }

    groups = read_jsonl(groups_path)
    kept_groups = [
        x for x in groups
        if int(x.get("reward_call", -1)) in retained_reward_calls
    ]

    rollouts = read_jsonl(rollouts_path)
    kept_rollouts = [
        x for x in rollouts
        if int(x.get("reward_call", -1)) in retained_reward_calls
    ]

    for p in [events_path, groups_path, rollouts_path]:
        b = backup(p)
        if b:
            print("Backup:", b)

    write_jsonl(events_path, kept_events)
    write_jsonl(groups_path, kept_groups)
    write_jsonl(rollouts_path, kept_rollouts)

    print("=" * 88)
    print("Dynamic Sampling resume diagnostics prepared")
    print("Checkpoint:", ckpt)
    print("Checkpoint step:", step)
    print("Events:  ", len(events), "->", len(kept_events),
          f"(removed {removed_events})")
    print("Groups:  ", len(groups), "->", len(kept_groups))
    print("Rollouts:", len(rollouts), "->", len(kept_rollouts))
    print("Retained reward calls:", len(retained_reward_calls))
    print("=" * 88)
    print("Now resume with:")
    print("python train_dynamic_1p7b.py --resume-from-checkpoint latest")


if __name__ == "__main__":
    main()
