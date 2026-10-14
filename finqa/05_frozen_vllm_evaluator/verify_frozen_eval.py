"""Compare a newly generated Transformers benchmark against the frozen FinQA Dev evaluator.

This is a read-only verification, and does NOT assume exact equality of generation.
"""
import argparse
import json
from pathlib import Path


def read_jsonl(path):
    out={}
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r=json.loads(line)
            k=str(r.get('id'))
            if k in out:
                raise ValueError(f'Duplicate example id: {k}')
            out[k]=r
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--benchmark-dir', required=True)
    p.add_argument('--reference-dir', default='/root/autodl-tmp/project/finqa_posttrain/experiments/grpo_dynamic_qwen3_1.7b/eval_dev_v1')
    a=p.parse_args()
    b=Path(a.benchmark_dir); ref=Path(a.reference_dir)
    ours=read_jsonl(b/'predictions.jsonl'); theirs=read_jsonl(ref/'predictions.jsonl')
    if set(ours)!=set(theirs):
        missing=sorted(set(theirs)-set(ours)); extra=sorted(set(ours)-set(theirs))
        raise ValueError(f'Different sample ID sets: missing={missing[:5]}, extra={extra[:5]}, size={len(ours)}/{len(theirs)}')
    text_equal=sum(ours[k].get('prediction')==theirs[k].get('prediction') for k in ours)
    answer_equal=sum(str(ours[k].get('prediction_answer'))==str(theirs[k].get('prediction_answer')) for k in ours)
    correctness_equal=sum(bool(ours[k].get('correct'))==bool(theirs[k].get('correct')) for k in ours)
    orig_accuracy=sum(bool(theirs[k].get('correct')) for k in theirs)/len(theirs)
    curr_accuracy=sum(bool(ours[k].get('correct')) for k in ours)/len(ours)
    print('Frozen evaluator rows:',len(theirs))
    print('Benchmark rows:',len(ours))
    print('Identical generated text:',text_equal)
    print('Identical parsed answers:',answer_equal)
    print('Identical correctness flags:',correctness_equal)
    print(f'Original accuracy: {orig_accuracy:.8%}')
    print(f'Benchmark accuracy: {curr_accuracy:.8%}')
    print(f'Delta: {(curr_accuracy-orig_accuracy)*100:+.3f} pp')
    if text_equal<len(ours):
        print('INFO: Not all generations identical. Investigate engine/tool version and decoding before claiming exact parity.')


if __name__=='__main__':
    main()
