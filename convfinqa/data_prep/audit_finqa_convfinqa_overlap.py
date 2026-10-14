#!/usr/bin/env python3
"""Audit possible FinQA -> ConvFinQA cross-dataset overlap (READ ONLY).

Checks train/dev/test FinQA against train/dev ConvFinQA for:
(1) source PDF paths, (2) original QA record IDs,
(3) normalized question matches tied to source PDFs,
(4) canonical financial table content tied to source PDFs.

Outputs a JSON report and CSV details. Does not claim that unmatched sources
are independent: IDs and content can be transformed between datasets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

FINQA_DEFAULT = Path('/root/autodl-tmp/project/finqa_posttrain/data/raw/FinQA/dataset')
CONV_DEFAULT = Path('/root/autodl-tmp/project/convfinqa_posttrain/data/raw/ConvFinQA/data_extracted/data')
OUT_DEFAULT = Path('/root/autodl-tmp/project/convfinqa_posttrain/analysis/cross_dataset_overlap_v1')

# Both datasets usually use paths similar to Single_KIM/2010/page_94.pdf-4.
DOC_RE = re.compile(r'(?P<company>[\w.\-]+)/(?P<year>(?:19|20)\d{2})/(?P<page>page[^/]*?\.pdf)', re.I)
TURN_RE = re.compile(r'^(.*)_(\d+)$')
SPACES = re.compile(r'\s+')


def get_doc(value):
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.replace('\\', '/').strip().lower()
    # Can be "Single_XXX/..." or ".../XXX/..."; keep only document suffix.
    m = DOC_RE.search(text)
    if not m:
        return None
    company = re.sub(r'^(single|double)_', '', m.group('company'), flags=re.I)
    return f"{company}/{m.group('year')}/{m.group('page')}"


def get_record_id(value, *, conv=False):
    if value is None:
        return None
    s = str(value).strip().lower()
    if conv:
        m = TURN_RE.fullmatch(s)
        if m:
            s = m.group(1)
    return s or None


def normal_question(value):
    if not isinstance(value, str):
        return None
    s = SPACES.sub(' ', value.strip().lower())
    s = re.sub(r'[?!.。？！\s]+$', '', s)
    return s or None


def canonical(value):
    if value is None:
        return None
    if isinstance(value, str):
        return SPACES.sub(' ', value.strip().lower())
    if isinstance(value, dict):
        return {str(k).lower(): canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    return value


def table_fingerprints(row):
    out = set()
    for field in ('table', 'table_ori'):
        if row.get(field) is None:
            continue
        value = canonical(row[field])
        if not value:
            continue
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        out.add(hashlib.sha256(raw.encode('utf8')).hexdigest())
    return out


def finqa_questions(row):
    qa = row.get('qa')
    candidates = []
    if isinstance(qa, dict):
        for field in ('question', 'original_question', 'query'):
            if field in qa and isinstance(qa[field], str):
                candidates.append(qa[field])
    elif isinstance(qa, list):
        for item in qa:
            if isinstance(item, dict) and isinstance(item.get('question'), str):
                candidates.append(item['question'])
    return {s for q in candidates if (s := normal_question(q))}


def conv_questions(row):
    ann = row.get('annotation') or {}
    qs = ann.get('dialogue_break', []) if isinstance(ann, dict) else []
    if not isinstance(qs, list):
        qs = []
    return {s for q in qs if (s := normal_question(q))}


def load_json_array(path):
    with path.open(encoding='utf-8') as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f'Expected JSON list: {path}')
    return rows


def extract_info(rows, *, is_conv=False):
    result = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Every dataset row must be a JSON object')
        rid = get_record_id(row.get('id'), conv=is_conv)
        doc_from_filename = get_doc(row.get('filename'))
        doc_from_id = get_doc(str(row.get('id', '')))
        # Prefer filename when present, but retain ID origin for diagnostic mismatch.
        doc = doc_from_filename or doc_from_id
        result.append({
            'id': rid, 'doc': doc,
            'filename_doc': doc_from_filename, 'id_doc': doc_from_id,
            'questions': conv_questions(row) if is_conv else finqa_questions(row),
            'tables': table_fingerprints(row),
        })
    return result


def build_index(items):
    idx = {'doc': defaultdict(set), 'id': defaultdict(set), 'question': defaultdict(set), 'table': defaultdict(set)}
    for row in items:
        rid, doc = row['id'], row['doc']
        if doc and rid:
            idx['doc'][doc].add(rid)
        if rid:
            idx['id'][rid].add(rid)
        if doc:
            for q in row['questions']:
                idx['question'][(doc, q)].add(rid or doc)
            for t in row['tables']:
                idx['table'][(doc, t)].add(rid or doc)
    return idx


def compare(finqa_idx, conv_items, num_turns_by_id):
    details = []
    matched = Counter()
    examples = {k: [] for k in ('source_pdf', 'original_record_id', 'same_source_question', 'same_source_table')}
    unknown = 0
    for row in conv_items:
        doc, rid = row['doc'], row['id']
        if not doc:
            unknown += 1
        pdf = bool(doc and doc in finqa_idx['doc'])
        same_id = bool(rid and rid in finqa_idx['id'])
        question_hits = sorted(q for q in row['questions'] if (doc, q) in finqa_idx['question']) if doc else []
        table_hits = sorted(t for t in row['tables'] if (doc, t) in finqa_idx['table']) if doc else []
        flags = {
            'source_pdf': pdf,
            'original_record_id': same_id,
            'same_source_question': bool(question_hits),
            'same_source_table': bool(table_hits),
        }
        turns = num_turns_by_id.get(rid, 0)
        for flag, yes in flags.items():
            if yes:
                matched[f'{flag}_conversations'] += 1
                matched[f'{flag}_turns'] += turns
                if len(examples[flag]) < 10:
                    examples[flag].append({'conv_id': rid, 'doc': doc, 'turns': turns,
                                           'questions': question_hits[:2] if flag == 'same_source_question' else []})
        if any(flags.values()):
            details.append({
                'conv_id': rid or '', 'source_pdf': doc or '', 'turns': turns,
                'pdf_match': int(pdf), 'original_record_id_match': int(same_id),
                'matched_questions_on_same_pdf': len(question_hits),
                'matched_tables_on_same_pdf': len(table_hits),
                'matched_question_examples': ' | '.join(question_hits[:3]),
            })
    matched['unparsed_conv_document_ids'] = unknown
    return dict(matched), examples, details


def run(finqa_dir, conv_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = {}
    for split in ('train', 'dev', 'test'):
        p = finqa_dir / f'{split}.json'
        if not p.is_file():
            raise FileNotFoundError(p)
        raw[split] = extract_info(load_json_array(p))
    conv = {}
    turn_counts = {}
    for split in ('train', 'dev'):
        p = conv_dir / f'{split}.json'
        t = conv_dir / f'{split}_turn.json'
        if not p.is_file() or not t.is_file():
            raise FileNotFoundError(f'ConvFinQA files missing: {p}, {t}')
        conv[split] = extract_info(load_json_array(p), is_conv=True)
        counts = Counter()
        for row in load_json_array(t):
            rid = get_record_id(row.get('id'), conv=True)
            counts[rid] += 1
        turn_counts[split] = counts
    indices = {s: build_index(rows) for s, rows in raw.items()}
    results = {'scope': 'FinQA raw train/dev/test vs ConvFinQA raw train/dev; private tests not used',
               'warnings': [
                   'A matched PDF does not prove identical questions or target leakage.',
                   'Exact ID/question/table matches are stronger indicators; unobserved semantic overlap is possible.',
                   'FinQA DEV and TEST were not SFT/GRPO training data; interpret them separately.',
                   'ConvFinQA turn-level rows from the same conversation are not independent samples.',
                   'Text/formatted-table changes can hide real overlaps; zero matches is not proof of independence.'
               ], 'dataset_counts': {}, 'comparisons': {}}
    for split, rows in raw.items():
        results['dataset_counts'][f'finqa_{split}'] = {
            'rows': len(rows), 'ids_parsed': sum(bool(r['id']) for r in rows),
            'documents_parsed': sum(bool(r['doc']) for r in rows),
            'questions_extracted': sum(bool(r['questions']) for r in rows),
        }
    for split, rows in conv.items():
        nturns = sum(turn_counts[split].values())
        conv_ids = {r['id'] for r in rows}
        missing = [rid for rid in conv_ids if rid not in turn_counts[split]]
        results['dataset_counts'][f'convfinqa_{split}'] = {
            'conversations': len(rows), 'turns': nturns,
            'docs_parsed': sum(bool(r['doc']) for r in rows),
            'questions_extracted': sum(bool(r['questions']) for r in rows),
            'missing_turns_for_conversations': len(missing),
            'filename_id_doc_disagreements': sum(bool(r['filename_doc'] and r['id_doc'] and r['filename_doc'] != r['id_doc']) for r in rows),
        }
        for s in ('train', 'dev', 'test'):
            stats, examples, details = compare(indices[s], rows, turn_counts[split])
            key = f'finqa_{s}__convfinqa_{split}'
            results['comparisons'][key] = {'conv_conversations': len(rows), 'conv_turns': nturns,
                                            'stats': stats, 'examples': examples}
            outcsv = output_dir / f'{key}_matched.csv'
            with outcsv.open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'conv_id', 'source_pdf', 'turns', 'pdf_match',
                    'original_record_id_match', 'matched_questions_on_same_pdf',
                    'matched_tables_on_same_pdf', 'matched_question_examples'])
                writer.writeheader()
                writer.writerows(details)
    report_file = output_dir / 'overlap_report.json'
    report_file.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    print('\n=== DATASET COVERAGE ===')
    for k, val in results['dataset_counts'].items():
        print(k, val)
    print('\n=== CROSS-DATASET OVERLAP (conversation / turn counts) ===')
    print(f"{'Pair':<36} {'PDF':>10} {'ID':>10} {'Question':>10} {'Table':>10}")
    for name, d in results['comparisons'].items():
        s = d['stats']
        pairs = [f"{s.get(x+'_conversations',0)}/{s.get(x+'_turns',0)}" for x in
                 ('source_pdf', 'original_record_id', 'same_source_question', 'same_source_table')]
        print(f"{name:<36} " + ' '.join(f'{v:>10}' for v in pairs))
    print('\nCOUNTS: conversation / turn (not independent samples)')
    print('Interpret finqa_train__convfinqa_dev first: potential overlap with previously trained model.')
    print(f'\nReport: {report_file}')
    print(f'Details: {output_dir}/*_matched.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finqa-dir', type=Path, default=FINQA_DEFAULT)
    parser.add_argument('--conv-dir', type=Path, default=CONV_DEFAULT)
    parser.add_argument('--output-dir', type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    run(args.finqa_dir, args.conv_dir, args.output_dir)


if __name__ == '__main__':
    main()
