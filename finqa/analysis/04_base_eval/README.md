# Baseline parser patch v1.1.1

This is a small patch on top of `finqa_baseline_eval_v1_1`.

It fixes an edge case seen in the chat smoke run:

```text
Answer:
To calculate ... 2016 and 2017 ...
...
Therefore ... 1,095,000
```

v1.1 could treat a number from the explanatory line as the answer because
the `Answer:` cue was allowed to span the newline.

v1.1.1 only accepts answer cues on the same physical line and lets the
last-line fallback handle `Answer:` headings.

## Apply

Back up the current file:

```bash
cp answer_normalizer.py answer_normalizer_v1_1_backup.py
```

Replace `answer_normalizer.py` with the file in this patch.

Run:

```bash
python self_test_parser_patch.py
```

Expected:

```text
PARSER PATCH SELF TEST PASSED
```
