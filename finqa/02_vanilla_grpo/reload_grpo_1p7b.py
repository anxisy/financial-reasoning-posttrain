# -*- coding: utf-8 -*-
import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def manual(messages):
    parts = []
    for m in messages:
        role = str(m.get("role", "")).lower()
        content = str(m.get("content", "")).strip()
        if role == "system":
            parts.append(f"System:\n{content}")
        elif role == "user":
            parts.append(f"User:\n{content}")

    parts.append("Assistant:")
    prompt = "\n\n".join(parts)

    assert prompt.endswith("Assistant:")
    assert not prompt.endswith("Assistant:\n")
    return prompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base-model-path",
        default="/root/autodl-tmp/models/Qwen3-1.7B-Base",
    )
    p.add_argument(
        "--adapter-path",
        default=(
            "/root/autodl-tmp/project/finqa_posttrain/experiments/"
            "grpo_qwen3_1.7b/full_v1/final_adapter"
        ),
    )
    p.add_argument(
        "--eval-file",
        default=(
            "/root/autodl-tmp/project/finqa_posttrain/data/"
            "processed_v1_1_1/eval/finqa_eval_dev.jsonl"
        ),
    )
    p.add_argument("--num-examples", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=384)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(
        args.base_model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        else (
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        )
    )

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    model = PeftModel.from_pretrained(
        base,
        args.adapter_path,
        is_trainable=False,
    )

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device).eval()

    rows = read_jsonl(args.eval_file)[: args.num_examples]

    for i, row in enumerate(rows, 1):
        prompt = manual(row["prompt"])
        enc = tok(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

        prompt_len = enc["input_ids"].shape[1]
        text = tok.decode(
            out[0, prompt_len:],
            skip_special_tokens=True,
        ).strip()

        print("\n" + "=" * 100)
        print(f"EXAMPLE {i} | id={row.get('id')}")
        print("=" * 100)
        print(text)


if __name__ == "__main__":
    main()
