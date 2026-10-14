#!/usr/bin/env python3
import os, sys
import torch, transformers, peft
print("=" * 80)
print("Qwen3-1.7B SFT environment check")
print("=" * 80)
print("Python      :", sys.version.split()[0])
print("torch       :", torch.__version__)
print("transformers:", transformers.__version__)
print("peft        :", peft.__version__)
print("CUDA        :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU         :", torch.cuda.get_device_name(0))
    print("VRAM GB     :", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
    print("BF16        :", torch.cuda.is_bf16_supported())
print("OMP_NUM_THREADS:", repr(os.environ.get("OMP_NUM_THREADS")))
print("MKL_NUM_THREADS:", repr(os.environ.get("MKL_NUM_THREADS")))
