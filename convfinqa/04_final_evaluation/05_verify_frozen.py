#!/usr/bin/env python3
"""Independent local verification of the frozen evidence and optionally source files."""
import argparse, sys
from pathlib import Path
from common import OUT, sha256, jsonread

def verify(path, check_sources=False):
    manifest=jsonread(path/'MANIFEST.json')
    expected=(path/'MANIFEST.sha256').read_text(encoding='ascii').strip().split()[0]
    actual=sha256(path/'MANIFEST.json')
    if actual!=expected:raise ValueError('MANIFEST.json changed after freeze')
    for name, info in manifest['evidence'].items():
        f=path/name
        if not f.is_file() or sha256(f)!=info['sha256'] or f.stat().st_size!=info['bytes']:
            raise ValueError(f'Copied evidence failed SHA check: {f}')
    for name, sha in manifest['snapshot_summary'].items():
        if sha256(path/name)!=sha:raise ValueError(f'Summary changed: {name}')
    if check_sources:
        for name, info in manifest['source_references'].items():
            f=Path(name)
            if not f.is_file() or sha256(f)!=info['sha256']:
                raise ValueError(f'Source changed, moved, or deleted: {name}')
    print(f'PASS frozen archive: {len(manifest["evidence"])} copied files all SHA-256 match')
    print('Source references:',len(manifest['source_references']),
          '(checked)' if check_sources else '(not checked; use --check-sources)')
    print('Manifest:',path/'MANIFEST.json')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--archive',type=Path,default=OUT/'frozen_release_v1')
    p.add_argument('--check-sources',action='store_true')
    a=p.parse_args()
    verify(a.archive,a.check_sources)

if __name__=='__main__':main()
