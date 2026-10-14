#!/usr/bin/env python3
"""Recover a successfully generated FinQA Test run left in .running_*.

Checks the *exact* original predictions, metrics, inference config, frozen
prompt fingerprint and adapter sha256 before promoting the directory. Refuses
to delete, overwrite, or guess among multiple temporary runs.
"""
import argparse
from pathlib import Path
from common import OUT, MODEL_ORDER
from verify_one_test import verify


def recover_variant(variant):
    root = OUT / 'finqa_test' / variant
    dest = root / 'full_test'
    if dest.exists():
        print(f'Already published, verifying: {dest}')
        verify(dest, variant)
        return
    pending = sorted(p for p in root.glob('.running_full_test_*') if p.is_dir()) if root.exists() else []
    if not pending:
        print(f'No pending run for {variant} (nothing to recover).')
        return
    if len(pending) != 1:
        raise RuntimeError(f'{variant}: multiple unfinished/finished temporary directories: {pending}; '
                           'do not guess, inspect each folder first.')
    source = pending[0]
    print(f'Verifying existing inference output before publishing: {source}')
    verify(source, variant)
    if dest.exists():
        raise FileExistsError(f'Unexpected destination appeared: {dest}')
    source.rename(dest)
    print(f'RECOVERED {variant} -> {dest}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('variant', choices=(*MODEL_ORDER, 'all'))
    args = p.parse_args()
    for variant in (MODEL_ORDER if args.variant == 'all' else (args.variant,)):
        recover_variant(variant)


if __name__ == '__main__':
    main()
