# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import json
import random
from collections import defaultdict
from pathlib import Path

from trl import GRPOTrainer


class CappedDynamicSamplingGRPOTrainer(GRPOTrainer):
    """Single-GPU TRL 0.18.0 Dynamic Sampling.

    Keep a prompt group only when correctness contains both correct and wrong
    completions. For all-correct/all-wrong groups, resample another prompt.
    Retry up to ``max_resample_attempts``; if the cap is exhausted, transparently
    keep the final zero-variance group as a fallback.
    """

    def __init__(
        self,
        *args,
        dynamic_pool_rows,
        diagnostic_reward,
        dynamic_output_dir,
        max_resample_attempts=4,
        dynamic_seed=42,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if self.accelerator.num_processes != 1:
            raise RuntimeError("Dynamic Sampling v1 is single-GPU only.")
        if self.args.per_device_train_batch_size != self.num_generations:
            raise RuntimeError(
                "per_device_train_batch_size must equal num_generations so each "
                "optimizer step contains one unique prompt group."
            )
        if self.args.gradient_accumulation_steps != 1:
            raise RuntimeError("gradient_accumulation_steps must be 1.")
        if self.args.steps_per_generation != 1 or self.num_iterations != 1:
            raise RuntimeError("steps_per_generation and num_iterations must both be 1.")

        self.dynamic_pool_rows = [dict(x) for x in dynamic_pool_rows]
        if not self.dynamic_pool_rows:
            raise RuntimeError("dynamic_pool_rows is empty")

        self.diagnostic_reward = diagnostic_reward
        self.max_resample_attempts = int(max_resample_attempts)
        if self.max_resample_attempts < 1:
            raise ValueError("max_resample_attempts must be >= 1")
        self.dynamic_seed = int(dynamic_seed)

        self.dynamic_output_dir = Path(dynamic_output_dir)
        self.dynamic_output_dir.mkdir(parents=True, exist_ok=True)
        self.dynamic_events_path = self.dynamic_output_dir / "dynamic_sampling_events.jsonl"

    def _snapshot_trl_logging_state(self):
        return {
            "metrics": {
                mode: defaultdict(list, {k: list(v) for k, v in values.items()})
                for mode, values in self._metrics.items()
            },
            "textual_logs": copy.deepcopy(self._textual_logs),
            "num_input_tokens_seen": self.state.num_input_tokens_seen,
        }

    def _restore_trl_logging_state(self, snap):
        self._metrics = snap["metrics"]
        self._textual_logs = snap["textual_logs"]
        self.state.num_input_tokens_seen = snap["num_input_tokens_seen"]

    def _deterministic_replacement_row(self, optimizer_step, attempt, avoid_example_id=None):
        seed = self.dynamic_seed + optimizer_step * 1_000_003 + attempt * 97_409
        rng = random.Random(seed)
        n = len(self.dynamic_pool_rows)
        idx = rng.randrange(n)
        if avoid_example_id is not None and n > 1:
            for _ in range(16):
                row = self.dynamic_pool_rows[idx]
                if row.get("example_id") != avoid_example_id:
                    break
                idx = rng.randrange(n)
        return dict(self.dynamic_pool_rows[idx])

    def _repeat_row_as_group(self, row):
        return [dict(row) for _ in range(self.num_generations)]

    def _write_dynamic_event(self, event):
        with self.dynamic_events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _generate_and_score_completions(self, inputs):
        if not self.model.training:
            return super()._generate_and_score_completions(inputs)

        if len(inputs) != self.num_generations:
            raise RuntimeError(
                f"Expected {self.num_generations} repeated rows, got {len(inputs)}"
            )

        optimizer_step = int(self.state.global_step) + 1
        candidate_inputs = inputs
        original_id = inputs[0].get("example_id")

        for attempt in range(1, self.max_resample_attempts + 1):
            logging_snap = self._snapshot_trl_logging_state()
            result = super()._generate_and_score_completions(candidate_inputs)

            group_rows = self.diagnostic_reward.last_group_rows
            if len(group_rows) != 1:
                raise RuntimeError(
                    "Expected exactly one reward group per generation call; "
                    f"got {len(group_rows)}"
                )

            group = dict(group_rows[0])
            mixed = bool(group["mixed_correctness"])
            is_last = attempt >= self.max_resample_attempts
            accepted = mixed or is_last
            fallback = bool(is_last and not mixed)

            event = {
                "optimizer_step": optimizer_step,
                "attempt": attempt,
                "max_resample_attempts": self.max_resample_attempts,
                "source": "dataloader" if attempt == 1 else "replacement",
                "original_example_id": original_id,
                "candidate_example_id": group.get("example_id"),
                "reward_call": group.get("reward_call"),
                "difficulty": group.get("difficulty"),
                "operators_json": group.get("operators_json"),
                "correctness": group.get("correctness"),
                "reward_mean": group.get("reward_mean"),
                "reward_variance": group.get("reward_variance"),
                "correctness_mean": group.get("correctness_mean"),
                "correctness_variance": group.get("correctness_variance"),
                "all_correct": group.get("all_correct"),
                "all_wrong": group.get("all_wrong"),
                "mixed_correctness": mixed,
                "accepted_for_update": accepted,
                "fallback_zero_variance": fallback,
            }
            self._write_dynamic_event(event)

            if mixed or fallback:
                return result

            # Candidate was zero-variance and rejected. Remove it from ordinary
            # Trainer/W&B aggregates while preserving our own JSONL diagnostics.
            self._restore_trl_logging_state(logging_snap)

            replacement = self._deterministic_replacement_row(
                optimizer_step=optimizer_step,
                attempt=attempt + 1,
                avoid_example_id=group.get("example_id"),
            )
            candidate_inputs = self._repeat_row_as_group(replacement)

        raise AssertionError("unreachable")
