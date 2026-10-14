#!/usr/bin/env python3
"""Conservative static checks before publishing a staged GitHub repository."""
from pathlib import Path
import argparse
import os
import re
import sys

FORBIDDEN_EXTS = {'.safetensors','.pt','.pth','.ckpt','.gguf','.bin','.whl',
                  '.parquet','.arrow','.jsonl','.sqlite','.db','.csv','.pem','.key',
                  '.zip','.tar','.gz'}
FORBIDDEN_DIRS = {'.venv','venv','__pycache__','data','datasets','raw','processed',
                  'logs','outputs','wandb','checkpoint','checkpoints','weights','models',
                  '.git','.cache','.idea'}
SECRET_PATTERNS = {
    'private-key': re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----'),
    'openai-key': re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{25,}\b'),
    'github-token': re.compile(r'\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b'),
    'hf-token': re.compile(r'\bhf_[A-Za-z0-9]{24,}\b'),
    'aws-key': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'wandb-key': re.compile(r'(?i)\bWANDB_API_KEY\s*[=:]\s*[\"\']?[a-f0-9]{40}\b'),
}

def run(root: Path):
    errors, warnings, count = [], [], 0
    root = root.resolve()
    for base, dirs, files in os.walk(root,followlinks=False):
        for d in list(dirs):
            p = Path(base)/d
            if p.is_symlink(): errors.append(f'symlink folder: {p.relative_to(root)}'); dirs.remove(d)
            elif d.lower() in {'.git','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache'}: dirs.remove(d)
            elif d.lower() in FORBIDDEN_DIRS: errors.append(f'blocked directory: {p.relative_to(root)}');dirs.remove(d)
        for name in files:
            p=Path(base)/name
            if p.is_symlink(): errors.append(f'symlink file: {p.relative_to(root)}');continue
            relative=p.relative_to(root)
            count+=1
            if p.suffix.lower() in FORBIDDEN_EXTS: errors.append(f'blocked file extension: {relative}')
            if name.startswith('.env') and name!='.env.example': errors.append(f'private .env file: {relative}')
            if p.stat().st_size>5_000_000: errors.append(f'file >5MB: {relative}')
            if p.stat().st_size <= 4_000_000 and p.suffix.lower() in {'.py','.sh','.md','.txt','.json','.yml','.yaml','.toml','.ini','.cfg'}:
                text=p.read_text(encoding='utf8',errors='replace')
                for label,pattern in SECRET_PATTERNS.items():
                    if pattern.search(text): errors.append(f'potential secret {label}: {relative}')
    print(f'RELEASE AUDIT: {count} files checked, {len(errors)} errors.')
    for item in errors[:40]: print('[ERROR]',item)
    if warnings:
        for item in warnings[:20]: print('[WARN]',item)
    return 1 if errors else 0

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('.'))
    a=p.parse_args()
    sys.exit(run(a.root))

if __name__=='__main__': main()
