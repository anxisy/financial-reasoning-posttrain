import importlib.metadata
from pathlib import Path
from common import DEFAULT_BASE,DEFAULT_ADAPTER,DEFAULT_DEV

def ver(name):
    try:return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:return 'NOT INSTALLED'

for key,path in [('BASE',DEFAULT_BASE),('ADAPTER',DEFAULT_ADAPTER),('DEV',DEFAULT_DEV)]:
    print(f'{key:8s} {path} exists={Path(path).exists()}')
for name in ('torch','transformers','peft','vllm','nvidia-ml-py'):
    print(f'{name:15s}: {ver(name)}')
try:
    import torch
    print('CUDA usable:',torch.cuda.is_available())
    if torch.cuda.is_available(): print('GPU:',torch.cuda.get_device_name(0),'BF16:',torch.cuda.is_bf16_supported())
except Exception as e:print('torch info unavailable:',e)
