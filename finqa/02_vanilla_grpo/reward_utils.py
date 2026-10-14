# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)
BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]+)\}", re.I)
NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)\s*%?")
YESNO_TOKEN_RE = re.compile(r"\b(yes|no|true|false)\b", re.I)

PLACEHOLDERS = {
    "", "...", "…", "xxx", "x", "answer", "your answer",
    "final answer", "result", "value", "number", "n/a", "na",
}
ROLE_ONLY = {"assistant", "assistant:", "user", "user:", "system", "system:"}


def _s(v):
    return "" if v is None else str(v).strip()


def _clean(v):
    return _s(v).strip("` \t\r\n").strip()


def _is_placeholder(v):
    n = re.sub(r"\s+", " ", _clean(v).lower()).strip(" .,:;!?")
    if n in PLACEHOLDERS or n in ROLE_ONLY:
        return True
    return bool(n) and set(n) <= {".", "…", "-", "_"}


def _atomic(text):
    text = _clean(text)
    if not text or _is_placeholder(text):
        return ""

    yn = list(YESNO_TOKEN_RE.finditer(text))
    nums = list(NUMBER_RE.finditer(text))

    if yn:
        m = yn[-1]
        if not text[m.end():].strip(" .,:;!?"):
            return "yes" if m.group(1).lower() in {"yes", "true"} else "no"

    if nums:
        return nums[-1].group(0).strip()

    return text


def extract_answer_with_method(text: str):
    text = "" if text is None else str(text)

    for c in reversed(ANSWER_TAG_RE.findall(text)):
        c = _clean(c)
        if not _is_placeholder(c):
            a = _atomic(c)
            if a:
                return a, True, "answer_tag"

    for c in reversed(BOXED_RE.findall(text)):
        a = _atomic(c)
        if a and not _is_placeholder(a):
            return a, False, "boxed"

    lines = [x.strip() for x in text.splitlines() if x.strip()]
    cue = re.compile(
        r"(?:final\s+answer|answer)\s*(?:is|=|:|：)\s*(.+)$",
        re.I,
    )

    for line in reversed(lines):
        m = cue.search(line)
        if not m:
            continue
        raw = _clean(m.group(1))
        if not raw or _is_placeholder(raw):
            continue
        nums = list(NUMBER_RE.finditer(raw))
        yn = list(YESNO_TOKEN_RE.finditer(raw))
        if (
            (len(raw) <= 120 and len(nums) <= 1)
            or (yn and not nums and len(raw) <= 120)
        ):
            a = _atomic(raw)
            if a:
                return a, False, "final_cue"

    for line in reversed(lines):
        if _is_placeholder(line) or line.lower().strip() in ROLE_ONLY:
            continue
        a = _atomic(line)
        if a:
            return a, False, "last_line"

    return "", False, "none"


def _parse(v):
    raw = _s(v)
    if not raw:
        return "empty", None

    low = raw.lower().strip().strip(" .,:;")
    if low in {"yes", "y", "true"}:
        return "yesno", "yes"
    if low in {"no", "n", "false"}:
        return "yesno", "no"

    m = re.search(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)", raw)
    if m:
        try:
            n = float(m.group(0).replace(",", ""))
            return ("percent" if "%" in raw else "number"), n
        except ValueError:
            pass

    return "text", low


def compare_answer_strings(
    prediction,
    gold,
    numeric_rel_tol=1e-3,
    numeric_abs_tol=1e-2,
    percent_abs_tol=0.10,
):
    pk, pv = _parse(prediction)
    gk, gv = _parse(gold)

    if gk == "empty":
        return False, "missing_gold"
    if pk == "empty":
        return False, "empty_prediction"

    if gk == "yesno":
        ok = pk == "yesno" and pv == gv
        return ok, "yesno_match" if ok else "yesno_mismatch"

    if gk == "text":
        ok = pk == "text" and pv == gv
        return ok, "text_match" if ok else "text_mismatch"

    if pk not in {"number", "percent"} or gk not in {"number", "percent"}:
        return False, "type_mismatch"

    pv, gv = float(pv), float(gv)

    if gk == "percent":
        if math.isclose(pv, gv, rel_tol=0.0, abs_tol=percent_abs_tol):
            return True, "percent_point_match"
        if pk == "number" and math.isclose(
            pv * 100.0,
            gv,
            rel_tol=numeric_rel_tol,
            abs_tol=percent_abs_tol,
        ):
            return True, "percent_fraction_match"
        return False, "percent_mismatch"

    if pk == "percent":
        return False, "unexpected_percent"

    ok = math.isclose(
        pv,
        gv,
        rel_tol=numeric_rel_tol,
        abs_tol=numeric_abs_tol,
    )
    return ok, "numeric_match" if ok else "numeric_mismatch"


def normalize_completion(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for item in c:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(c, dict):
        return str(c.get("content", c))
    return str(c)


def expand_to_n(value, n, group_size):
    if value is None:
        return [None] * n

    if isinstance(value, (str, int, float, bool)):
        return [value] * n

    value = list(value)
    if len(value) == n:
        return value

    if len(value) * group_size == n:
        out = []
        for x in value:
            out.extend([x] * group_size)
        return out

    if len(value) == 1:
        return value * n

    raise ValueError(
        f"Cannot align metadata length {len(value)} "
        f"to completions length {n} with group_size={group_size}"
    )


def p95(xs):
    if not xs:
        return 0.0
    ys = sorted(xs)
    idx = max(0, min(len(ys) - 1, math.ceil(0.95 * len(ys)) - 1))
    return float(ys[idx])


def _existing_max_reward_call(path: Path):
    if not path.exists():
        return 0
    max_call = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                max_call = max(max_call, int(row.get("reward_call", 0) or 0))
            except Exception:
                continue
    return max_call


class GRPORewardWithDiagnostics:
    """
    Combined reward:
        correctness (1.0) + format (0.1)

    Saves both rollout-level and prompt-group-level diagnostics.
    """

    def __init__(
        self,
        tokenizer,
        output_dir,
        num_generations=4,
        max_completion_length=256,
        numeric_rel_tol=1e-3,
        numeric_abs_tol=1e-2,
        percent_abs_tol=0.10,
        correctness_weight=1.0,
        format_weight=0.1,
        append_existing=False,
    ):
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_generations = int(num_generations)
        self.max_completion_length = int(max_completion_length)

        self.numeric_rel_tol = numeric_rel_tol
        self.numeric_abs_tol = numeric_abs_tol
        self.percent_abs_tol = percent_abs_tol

        self.correctness_weight = correctness_weight
        self.format_weight = format_weight

        self.rollout_path = self.output_dir / "rollout_diagnostics.jsonl"
        self.group_path = self.output_dir / "group_diagnostics.jsonl"

        if append_existing:
            self.call_index = max(
                _existing_max_reward_call(self.rollout_path),
                _existing_max_reward_call(self.group_path),
            )
        else:
            self.call_index = 0
            for p in [self.rollout_path, self.group_path]:
                if p.exists():
                    p.unlink()

    def __call__(self, completions, gold_answer=None, **kwargs):
        texts = [normalize_completion(x) for x in completions]
        n = len(texts)
        g = self.num_generations

        golds = expand_to_n(gold_answer, n, g)
        example_ids = expand_to_n(kwargs.get("example_id"), n, g)
        difficulties = expand_to_n(kwargs.get("difficulty"), n, g)
        operators_json = expand_to_n(kwargs.get("operators_json"), n, g)

        correctness = []
        format_scores = []
        total_rewards = []
        lengths = []
        parsed_answers = []
        extraction_methods = []

        for text, gold in zip(texts, golds):
            pred, format_ok, method = extract_answer_with_method(text)
            ok, reason = compare_answer_strings(
                pred,
                gold,
                numeric_rel_tol=self.numeric_rel_tol,
                numeric_abs_tol=self.numeric_abs_tol,
                percent_abs_tol=self.percent_abs_tol,
            )

            c = 1.0 if ok else 0.0
            f = 1.0 if format_ok else 0.0
            r = self.correctness_weight * c + self.format_weight * f

            ids = self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]

            correctness.append(c)
            format_scores.append(f)
            total_rewards.append(float(r))
            lengths.append(len(ids))
            parsed_answers.append(pred)
            extraction_methods.append((method, reason))

        self.call_index += 1
        call_idx = self.call_index

        with self.rollout_path.open("a", encoding="utf-8") as f:
            for i in range(n):
                f.write(
                    json.dumps(
                        {
                            "reward_call": call_idx,
                            "completion_index": i,
                            "example_id": example_ids[i],
                            "difficulty": difficulties[i],
                            "operators_json": operators_json[i],
                            "gold_answer": golds[i],
                            "prediction_answer": parsed_answers[i],
                            "correctness_reward": correctness[i],
                            "format_reward": format_scores[i],
                            "total_reward": total_rewards[i],
                            "response_tokens": lengths[i],
                            "hit_max_completion_length": (
                                lengths[i] >= self.max_completion_length
                            ),
                            "extraction_method": extraction_methods[i][0],
                            "match_reason": extraction_methods[i][1],
                            "completion": texts[i],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        if n % g != 0:
            raise RuntimeError(
                f"Reward call got {n} completions, "
                f"not divisible by num_generations={g}"
            )

        group_rows = []

        for start in range(0, n, g):
            end = start + g
            rr = total_rewards[start:end]
            cc = correctness[start:end]
            ff = format_scores[start:end]
            tt = texts[start:end]
            ll = lengths[start:end]

            mean_r = sum(rr) / g
            var_r = sum((x - mean_r) ** 2 for x in rr) / g

            mean_c = sum(cc) / g
            var_c = sum((x - mean_c) ** 2 for x in cc) / g

            row = {
                "reward_call": call_idx,
                "group_index_in_call": start // g,
                "example_id": example_ids[start],
                "difficulty": difficulties[start],
                "operators_json": operators_json[start],
                "rewards": rr,
                "correctness": cc,
                "format": ff,
                "reward_mean": mean_r,
                "reward_variance": var_r,
                "correctness_mean": mean_c,
                "correctness_variance": var_c,
                "total_zero_variance": var_r <= 1e-12,
                "correctness_zero_variance": var_c <= 1e-12,
                "all_correct": all(x == 1.0 for x in cc),
                "all_wrong": all(x == 0.0 for x in cc),
                "mixed_correctness": (
                    any(x == 1.0 for x in cc)
                    and any(x == 0.0 for x in cc)
                ),
                "unique_completion_ratio": len(set(tt)) / g,
                "response_length_mean": sum(ll) / g,
                "response_length_max": max(ll),
                "max_length_hit_count": sum(
                    x >= self.max_completion_length for x in ll
                ),
            }
            group_rows.append(row)

        with self.group_path.open("a", encoding="utf-8") as f:
            for row in group_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # W&B is supplemental; JSONL is the source of truth.
        try:
            import wandb

            if wandb.run is not None and group_rows:
                wandb.log(
                    {
                        "grpo_diag/reward_mean": (
                            sum(total_rewards) / len(total_rewards)
                        ),
                        "grpo_diag/correctness_mean": (
                            sum(correctness) / len(correctness)
                        ),
                        "grpo_diag/format_mean": (
                            sum(format_scores) / len(format_scores)
                        ),
                        "grpo_diag/zero_variance_group_ratio": (
                            sum(
                                r["correctness_zero_variance"]
                                for r in group_rows
                            )
                            / len(group_rows)
                        ),
                        "grpo_diag/all_correct_group_ratio": (
                            sum(r["all_correct"] for r in group_rows)
                            / len(group_rows)
                        ),
                        "grpo_diag/all_wrong_group_ratio": (
                            sum(r["all_wrong"] for r in group_rows)
                            / len(group_rows)
                        ),
                        "grpo_diag/mixed_group_ratio": (
                            sum(r["mixed_correctness"] for r in group_rows)
                            / len(group_rows)
                        ),
                        "grpo_diag/unique_completion_ratio": (
                            sum(
                                r["unique_completion_ratio"]
                                for r in group_rows
                            )
                            / len(group_rows)
                        ),
                        "grpo_diag/response_length_mean": (
                            sum(lengths) / len(lengths)
                        ),
                        "grpo_diag/max_length_hit_rate": (
                            sum(
                                x >= self.max_completion_length
                                for x in lengths
                            )
                            / len(lengths)
                        ),
                    },
                    commit=False,
                )
        except Exception:
            pass

        return total_rewards


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if not xs:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _slice_group_stats(rows):
    return {
        "groups": len(rows),
        "zero_variance_group_ratio": _mean(
            [1.0 if x["correctness_zero_variance"] else 0.0 for x in rows]
        ),
        "all_correct_group_ratio": _mean(
            [1.0 if x["all_correct"] else 0.0 for x in rows]
        ),
        "all_wrong_group_ratio": _mean(
            [1.0 if x["all_wrong"] else 0.0 for x in rows]
        ),
        "mixed_group_ratio": _mean(
            [1.0 if x["mixed_correctness"] else 0.0 for x in rows]
        ),
        "unique_completion_ratio_mean": _mean(
            [float(x["unique_completion_ratio"]) for x in rows]
        ),
        "correctness_mean": _mean(
            [float(x["correctness_mean"]) for x in rows]
        ),
    }


def summarize_diagnostics(output_dir):
    out = Path(output_dir)
    rollout_path = out / "rollout_diagnostics.jsonl"
    group_path = out / "group_diagnostics.jsonl"

    rollouts = []
    groups = []

    if rollout_path.exists():
        with rollout_path.open("r", encoding="utf-8") as f:
            rollouts = [json.loads(x) for x in f if x.strip()]

    if group_path.exists():
        with group_path.open("r", encoding="utf-8") as f:
            groups = [json.loads(x) for x in f if x.strip()]

    rewards = [float(x["total_reward"]) for x in rollouts]
    correctness = [float(x["correctness_reward"]) for x in rollouts]
    formats = [float(x["format_reward"]) for x in rollouts]
    lengths = [int(x["response_tokens"]) for x in rollouts]

    summary = {
        "num_rollouts": len(rollouts),
        "num_groups": len(groups),
        "reward_mean": _mean(rewards),
        "reward_std": _std(rewards),
        "correctness_reward_mean": _mean(correctness),
        "format_reward_mean": _mean(formats),
        "group_reward_variance_mean": _mean(
            [float(x["reward_variance"]) for x in groups]
        ),
        "zero_variance_group_ratio": _mean(
            [
                1.0 if x["correctness_zero_variance"] else 0.0
                for x in groups
            ]
        ),
        "all_correct_group_ratio": _mean(
            [1.0 if x["all_correct"] else 0.0 for x in groups]
        ),
        "all_wrong_group_ratio": _mean(
            [1.0 if x["all_wrong"] else 0.0 for x in groups]
        ),
        "mixed_group_ratio": _mean(
            [1.0 if x["mixed_correctness"] else 0.0 for x in groups]
        ),
        "unique_completion_ratio_mean": _mean(
            [float(x["unique_completion_ratio"]) for x in groups]
        ),
        "response_length_mean": _mean(lengths),
        "response_length_p95": p95(lengths),
        "response_length_max": max(lengths) if lengths else 0,
        "max_length_hit_rate": _mean(
            [
                1.0 if x["hit_max_completion_length"] else 0.0
                for x in rollouts
            ]
        ),
    }

    # Difficulty slices.
    by_diff = defaultdict(list)
    for row in groups:
        by_diff[str(row.get("difficulty", "unknown"))].append(row)

    summary["by_difficulty"] = {
        key: _slice_group_stats(rows)
        for key, rows in sorted(by_diff.items())
    }

    # Operator slices. A group can belong to multiple operators.
    by_operator = defaultdict(list)
    for row in groups:
        raw = row.get("operators_json", "[]")
        try:
            ops = json.loads(raw) if isinstance(raw, str) else list(raw or [])
        except Exception:
            ops = []
        for op in sorted(set(map(str, ops))):
            by_operator[op].append(row)

    summary["by_operator"] = {
        key: _slice_group_stats(rows)
        for key, rows in sorted(by_operator.items())
    }

    (out / "diagnostics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def summarize_trainer_logs(log_history):
    numeric_keys = [
        "loss",
        "grad_norm",
        "learning_rate",
        "reward",
        "reward_std",
        "frac_reward_zero_std",
        "kl",
        "clip_ratio/low_mean",
        "clip_ratio/high_mean",
        "clip_ratio/region_mean",
        "completions/mean_length",
        "completions/clipped_ratio",
    ]

    out = {}

    for key in numeric_keys:
        vals = []
        for row in log_history:
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                vals.append(float(value))

        if vals:
            out[key] = {
                "count": len(vals),
                "mean": _mean(vals),
                "min": min(vals),
                "max": max(vals),
                "last": vals[-1],
            }

    out["kl_seen"] = "kl" in out
    return out
