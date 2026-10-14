# -*- coding: utf-8 -*-
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict

@dataclass
class PipelineConfig:
    data_dir: str = "/root/autodl-tmp/project/finqa_posttrain/data/raw/FinQA/dataset"
    output_dir: str = "/root/autodl-tmp/project/finqa_posttrain/data/processed"
    model_path: str = "/root/autodl-tmp/models/Qwen3-0.6B-Base"

    max_seq_length: int = 2048
    sft_response_reserve: int = 384
    grpo_prompt_max_tokens: int = 1792

    system_prompt: str = (
        "You are a financial numerical reasoning assistant. "
        "Use the financial report and table to solve the question. "
        "Show concise calculation steps and place the final answer inside "
        "<answer>...</answer>."
    )

    numeric_rel_tol: float = 1e-3
    numeric_abs_tol: float = 1e-2
    percent_abs_tol: float = 0.10

    keep_low_quality: bool = True

    @property
    def sft_prompt_max_tokens(self) -> int:
        return max(128, self.max_seq_length - self.sft_response_reserve)

    def split_paths(self) -> Dict[str, Path]:
        base = Path(self.data_dir)
        return {
            "train": base / "train.json",
            "dev": base / "dev.json",
            "test": base / "test.json",
        }

    def to_dict(self):
        return asdict(self)
