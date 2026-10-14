"""Shared contracts for the frozen FinQA/ConvFinQA final evaluation."""
from __future__ import annotations
import hashlib, json, os, random, re
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get('AUTODL_ROOT', '/root/autodl-tmp'))
PROJECT = ROOT / 'ConvFinQA'
EXP = ROOT / 'project/convfinqa_posttrain/experiments/convfinqa_sft_v2'
OUT = EXP / '04_final_evaluation_v1'
FINQA = ROOT / 'project/finqa_posttrain'
BENCH = ROOT / 'open-r1/20_finqa_vllm_benchmark_v2'
BASE = ROOT / 'models/Qwen3-1.7B-Base'
FINQA_TEST = Path(os.environ.get('FINQA_TEST_INPUT', str(FINQA / 'data/processed_v1_1_1/eval/finqa_eval_test.jsonl')))
FINQA_DEV = FINQA / 'data/processed_v1_1_1/eval/finqa_eval_dev.jsonl'
CONV_DEV = ROOT / 'project/convfinqa_posttrain/data/processed_v2/eval/dev.jsonl'
ADAPTERS = {
    'original_sft': FINQA / 'experiments/sft_qwen3_1.7b/full_v1/final_adapter',
    'transfer': EXP / '01_finqa_sft_adapter/full/final_adapter',
    'direct': EXP / '02_qwen3_base/full/final_adapter',
    'replay': EXP / '03_finqa_sft_replay/full/final_adapter',
}
FINQA_DEV_SHA = '752ba9302ba1153bbdcae95d0d9e743b1f00576ab75761939e821dbd678a58af'
CONV_DEV_SHA = '4fb737022d435bb472c316bb1b5de7776b8238a623bbc1efe900767eab09952f'
N_TEST, N_DEV, N_CONV, N_CONVERSATIONS = 1147, 883, 1490, 421
MODEL_ORDER = ('original_sft', 'transfer', 'direct', 'replay')
PDF_PATTERN = re.compile(r'([a-z0-9_.-]+/\d{4}/page_\d+\.pdf)', re.I)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def jsonread(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonwrite(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def predictions(path, n=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = {}
    with path.open(encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            rid = str(r['id'])
            if not rid or rid in rows:
                raise ValueError(f'{path}:{lineno}: blank/duplicate id {rid}')
            if type(r['correct']) is not bool:
                raise ValueError(f'{path}:{lineno}: correct is not a boolean')
            if 'gold_answer' not in r or not isinstance(r.get('metadata'), dict):
                raise ValueError(f'{path}:{lineno}: missing gold_answer or metadata')
            rows[rid] = r
    if n is not None and len(rows) != n:
        raise ValueError(f'{path}: expected {n} predictions, got {len(rows)}')
    return rows


def assert_aligned(a, b, name='pair', by_conv=False):
    if set(a) != set(b):
        raise ValueError(f'{name}: IDs mismatch; first only A={list(set(a)-set(b))[:3]}, B={list(set(b)-set(a))[:3]}')
    for rid, x in a.items():
        y = b[rid]
        if json.dumps(x['gold_answer'], sort_keys=True) != json.dumps(y['gold_answer'], sort_keys=True):
            raise ValueError(f'{name}: gold mismatch for {rid}')
        if by_conv:
            for key in ('conversation_id', 'turn_ind', 'cur_type'):
                if x['metadata'].get(key) != y['metadata'].get(key):
                    raise ValueError(f'{name}: {key} mismatch at {rid}')


def document_cluster(r):
    md = r.get('metadata') or {}
    for key in ('source_filename', 'filename', 'source_document', 'document_id'):
        v = str(md.get(key) or '')
        match = PDF_PATTERN.search(v)
        if match:
            return match.group(1).lower(), True
    match = PDF_PATTERN.search(str(r['id']))
    if match:
        return match.group(1).lower(), True
    return 'item:' + str(r['id']), False


def ci_paired(a, b, cluster_fn, boots=5000, seed=42):
    if boots < 200:
        raise ValueError('Use at least 200 bootstrap draws')
    assert_aligned(a, b)
    groups = defaultdict(list)
    wins = losses = 0
    for rid in sorted(a):
        x, y = int(a[rid]['correct']), int(b[rid]['correct'])
        groups[cluster_fn(a[rid])].append((x, y))
        wins += (x == 0 and y == 1)
        losses += (x == 1 and y == 0)
    packed = [(len(rows), sum(x for x, _ in rows), sum(y for _, y in rows))
              for _, rows in sorted(groups.items())]
    rng = random.Random(seed)
    samples = []
    for _ in range(boots):
        selected = rng.choices(packed, k=len(packed))
        denom = sum(z[0] for z in selected)
        samples.append(100 * sum(z[2] - z[1] for z in selected) / denom)
    samples.sort()
    def quantile(q):
        x = (len(samples)-1)*q
        lo = int(x)
        return samples[lo] + (samples[min(lo+1,len(samples)-1)]-samples[lo])*(x-lo)
    n = len(a)
    return {
        'n':n, 'cluster_count':len(groups),
        'reference_correct':sum(r['correct'] for r in a.values()),
        'candidate_correct':sum(r['correct'] for r in b.values()),
        'reference_acc':sum(r['correct'] for r in a.values())/n,
        'candidate_acc':sum(r['correct'] for r in b.values())/n,
        'delta_pp':100*(wins-losses)/n,
        'candidate_wins':wins, 'candidate_losses':losses,
        'cluster_bootstrap_ci95_pp':[quantile(.025),quantile(.975)],
        'bootstrap_replicates':boots,'seed':seed,
    }


def validate_metrics(folder, rows, *, kind):
    """Validate prediction-derived numbers against both frozen evaluator schemas.

    Historical FinQA/vLLM metrics keep overall.n and overall.accuracy,
    *not* overall.correct. ConvFinQA evaluator includes overall.correct.
    Recompute the count and accuracy from the original predictions, and
    check all fields that are actually present without changing the scorer.
    """
    folder = Path(folder)
    metrics = jsonread(folder/'metrics.json')
    all_stats = metrics.get('overall') or {}
    n = len(rows)
    if not n:
        raise ValueError(f'{folder}: empty predictions')
    correct = sum(int(x['correct']) for x in rows.values())
    if all_stats.get('n') != n:
        raise ValueError(f'{folder}: metrics n={all_stats.get("n")} but predictions n={n}')
    if 'correct' in all_stats and all_stats['correct'] != correct:
        raise ValueError(f'{folder}: metrics correct={all_stats["correct"]} but predictions correct={correct}')
    if 'num_predictions' in metrics and metrics['num_predictions'] != n:
        raise ValueError(f'{folder}: metrics num_predictions inconsistent with predictions')
    try:
        observed = float(all_stats['accuracy'])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f'{folder}: missing/invalid accuracy') from exc
    if abs(observed - correct/n) > 1e-10:
        raise ValueError(f'{folder}: metric accuracy={observed} but predictions {correct}/{n}={correct/n}')

    if kind == 'finqa':
        # The established FinQA evaluator stores only rates in overall.
        # Verify its additional fields using the original prediction flags.
        expected_fields = {
            'strict_accuracy': sum(bool(x.get('correct')) and bool(x.get('format_ok')) for x in rows.values()) / n,
            'format_accuracy': sum(bool(x.get('format_ok')) for x in rows.values()) / n,
            'answer_found_rate': sum(bool(x.get('prediction_answer')) for x in rows.values()) / n,
            'max_token_hit_rate': sum(bool(x.get('hit_max_new_tokens')) for x in rows.values()) / n,
        }
        for key, expected in expected_fields.items():
            if key not in all_stats or abs(float(all_stats[key]) - expected) > 1e-10:
                raise ValueError(f'{folder}: metric {key} disagrees with predictions')
        b = jsonread(folder/'benchmark.json')
        if b.get('engine') != 'vllm' or b.get('examples') != n:
            raise ValueError(f'{folder}: not complete frozen vLLM benchmark')
        settings = b.get('run') or {}
        if (settings.get('max_new_tokens') != 384 or settings.get('seed') != 42
                or settings.get('max_model_len') != 4096 or settings.get('dtype') != 'bfloat16'):
            raise ValueError(f'{folder}: unexpected generation config {settings}')
        fingerprint = b.get('prompt_token_ids_sha256')
        if not fingerprint:
            raise ValueError(f'{folder}: missing prompt fingerprint')
        return fingerprint
    return metrics.get('input_fingerprint')
