#!/usr/bin/env python3
"""Check unchanged bytes of copied experiment source files against export manifest."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent)
    a=ap.parse_args()
    root=a.root.resolve()
    provenance=root/'provenance/source_files.json'
    obj=json.loads(provenance.read_text(encoding='utf8'))
    problems=[]
    for row in obj['files']:
        p=root/row['repo_path']
        if not p.is_file() or p.is_symlink():problems.append(f'MISSING/SYMLINK {row["repo_path"]}');continue
        if digest(p)!=row['sha256']:problems.append(f'CHANGED {row["repo_path"]}')
    print('SOURCE SNAPSHOT:',len(obj['files']),'files,',len(problems),'issues')
    for item in problems[:50]:print('[FAIL]',item)
    print('Note: this checks byte-for-byte preservation of source snapshots, not GPU result reproduction.')
    sys.exit(bool(problems))
if __name__=='__main__':main()
