#!/usr/bin/env python3
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
DEFAULT_BASE = "/root/autodl-tmp/models/Qwen3-1.7B-Base"
DEFAULT_ADAPTER = "/root/autodl-tmp/project/finqa_posttrain/experiments/sft_qwen3_1.7b/full_v1/final_adapter"
p = argparse.ArgumentParser()
p.add_argument("--base-model-path", default=DEFAULT_BASE)
p.add_argument("--adapter-path", default=DEFAULT_ADAPTER)
args = p.parse_args()
print("Loading tokenizer...")
AutoTokenizer.from_pretrained(args.base_model_path, local_files_only=True)
print("Loading base...")
base = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True)
print("Loading adapter...")
model = PeftModel.from_pretrained(base, args.adapter_path, is_trainable=False)
model.eval()
print("Model class      :", type(base).__name__)
print("Adapter loaded   :", args.adapter_path)
print("Total params     :", f"{sum(p.numel() for p in model.parameters()):,}")
print("Trainable params :", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print("Device           :", next(model.parameters()).device)
print("RELOAD TEST PASSED")
