#!/usr/bin/env python3
"""Report original AutoDL absolute-path references; do NOT mutate experiment source."""
from pathlib import Path
import argparse
import sys

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--limit',type=int,default=45)
    a=parser.parse_args()
    root=a.root.resolve()
    hits=[]
    for subtree in ('finqa','convfinqa'):
        if not (root/subtree).exists():continue
        for p in sorted((root/subtree).rglob('*')):
            if not p.is_file() or p.suffix not in {'.py','.sh','.md','.yml','.yaml','.json','.toml','.cfg'}:
                continue
            if p.stat().st_size>1_000_000:continue
            for line_no,line in enumerate(p.read_text(encoding='utf-8',errors='replace').splitlines(),1):
                if '/root/autodl-tmp' in line or '/root/.cache' in line:
                    hits.append((p.relative_to(root),line_no,line.strip()[:140]))
    print(f'Absolute AutoDL path references found: {len(hits)}')
    for path,num,line in hits[:a.limit]:
        print(f' {path}:{num}: {line}')
    if len(hits)>a.limit:
        print(f'... {len(hits)-a.limit} further references; use --limit N')
    print('Source was not modified. This is a portability audit, not an error.')
    return 0
if __name__=='__main__':sys.exit(main())
