import json
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

BASE = "/root/autodl-tmp/models/Qwen3-1.7B-Base"

ADAPTER = (
    "/root/autodl-tmp/project/finqa_posttrain/"
    "experiments/grpo_dynamic_qwen3_1.7b/"
    "full_v1/final_adapter"
)

assert Path(BASE).is_dir(), "Base model not found"
assert Path(ADAPTER).is_dir(), "LoRA adapter not found"

config_path = Path(ADAPTER) / "adapter_config.json"
config = json.loads(config_path.read_text())
rank = int(config["r"])

print("Base model:", BASE)
print("LoRA adapter:", ADAPTER)
print("LoRA rank:", rank)

print("\n[1/3] Loading vLLM model...")

llm = LLM(
    model=BASE,
    dtype="bfloat16",
    max_model_len=3072,
    gpu_memory_utilization=0.70,
    enable_lora=True,
    max_lora_rank=rank,
    enforce_eager=True,
)

print("\n[2/3] Running inference...")

prompt = (
    "System:\n"
    "You are a financial numerical reasoning assistant. "
    "Show calculations and put the final answer in <answer> tags.\n\n"
    "User:\n"
    "A company's revenue increased from 200 million to "
    "250 million. What is the percentage increase?\n\n"
    "Assistant:"
)

params = SamplingParams(
    temperature=0.0,
    max_tokens=128,
)

outputs = llm.generate(
    [prompt],
    sampling_params=params,
    lora_request=LoRARequest(
        "finqa_dynamic",
        1,
        ADAPTER,
    ),
)

print("\n[3/3] Result:")
print(outputs[0].outputs[0].text)
print("\nSMOKE TEST COMPLETED")
