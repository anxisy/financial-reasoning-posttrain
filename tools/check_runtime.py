#!/usr/bin/env python3
"""Lightweight runtime checks, without installing or modifying dependencies."""
import argparse
import importlib.metadata as metadata
import sys

TRAIN = {'torch': '2.6.0', 'transformers': '4.52.3', 'peft': '0.21.0', 'trl': '0.18.0'}
INFERENCE = {'vllm': '0.31.0', 'torch': None}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('train', 'inference'))
    args = parser.parse_args()
    print('Python:', sys.version.split()[0], 'executable:', sys.executable)
    required = TRAIN if args.mode == 'train' else INFERENCE
    issues = []
    for package, expected in required.items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            issues.append(f'MISSING package: {package}')
            continue
        status = 'OK' if expected is None or actual.split('+')[0] == expected else 'MISMATCH'
        print(f'{status} {package}: {actual}' + (f' (expected {expected})' if expected else ''))
        if status != 'OK':
            issues.append(f'{package}: expected {expected}, found {actual}')
    try:
        import torch
        print('CUDA available:', torch.cuda.is_available(), 'torch CUDA build:', torch.version.cuda)
        if torch.cuda.is_available():
            print('GPU:', torch.cuda.get_device_name(0))
        else:
            print('NOTE: this machine has no visible CUDA GPU; GPU training/inference cannot be validated.')
    except Exception as e:
        issues.append(f'Could not import torch: {e}')
    if issues:
        for issue in issues: print('CHECK:', issue)
        return 1
    print('PASS: core packages match the recorded environment. GPU and full dependency compatibility still require a smoke test.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
