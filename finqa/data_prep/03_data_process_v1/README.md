# FinQA Semantic Audit Patch v1.1.1

This patch only changes `quality_audit.py`.

## Fixes

1. `greatest/highest ... loss` is no longer automatically flagged when the operator is `min`.
2. Percentage/ratio questions with a yes/no gold answer are no longer flagged as non-numeric percentage answers.

## Apply

Back up the old file:

```bash
cp quality_audit.py quality_audit_v1_1_backup.py
```

Replace `quality_audit.py` with the new one from this package.

Copy `self_test_semantic_v1_1_1.py` into the same folder and run:

```bash
python self_test_semantic_v1_1_1.py
```

Expected ending:

```text
SEMANTIC AUDIT v1.1.1 SELF TEST PASSED
```

Then rerun preprocessing to a NEW output directory:

```text
/root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1
```

Do not overwrite or delete your successful `processed_v1_1` snapshot.
