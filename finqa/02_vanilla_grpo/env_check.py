# -*- coding: utf-8 -*-
import inspect
import torch
import transformers
import peft

try:
    import trl
    from trl import GRPOConfig, GRPOTrainer
except Exception as e:
    raise SystemExit(f"TRL import failed: {e}")

try:
    import datasets
    datasets_version = datasets.__version__
except Exception as e:
    datasets_version = f"IMPORT FAILED: {e}"

print("torch       :", torch.__version__)
print("transformers:", transformers.__version__)
print("peft        :", peft.__version__)
print("trl         :", trl.__version__)
print("datasets    :", datasets_version)
print("CUDA        :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU         :", torch.cuda.get_device_name(0))
    print("BF16        :", torch.cuda.is_bf16_supported())
    props = torch.cuda.get_device_properties(0)
    print("VRAM GiB    :", round(props.total_memory / 1024**3, 2))

cfg = inspect.signature(GRPOConfig.__init__).parameters
trn = inspect.signature(GRPOTrainer.__init__).parameters

print("\nImportant GRPOConfig args:")
for key in [
    "num_generations",
    "max_prompt_length",
    "max_completion_length",
    "temperature",
    "top_p",
    "beta",
    "epsilon",
    "loss_type",
    "scale_rewards",
    "num_iterations",
    "steps_per_generation",
    "use_vllm",
]:
    if key in cfg:
        print(f"  {key:24s} supported; default={cfg[key].default!r}")
    else:
        print(f"  {key:24s} NOT EXPOSED")

print(
    "\nTrainer tokenizer arg:",
    "processing_class"
    if "processing_class" in trn
    else ("tokenizer" if "tokenizer" in trn else "none"),
)

print(
    "\nNOTE: Full v1 uses an explicit frozen SFT reference copy when beta > 0, "
    "so KL is measured against the SFT starting policy rather than the bare base model."
)
print("\nENV CHECK COMPLETE")
