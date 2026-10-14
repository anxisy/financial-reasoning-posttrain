# Final results — frozen v1

This archive is a **snapshot**, not a retraining or a benchmark tuning script.

## FinQA Test (1147 questions)

| Model | Correct | Accuracy |
|---|---:|---:|
| direct | 228/1147 | 19.88% |
| original_sft | 621/1147 | 54.14% |
| replay | 606/1147 | 52.83% |
| transfer | 433/1147 | 37.75% |

## ConvFinQA Dev Self-History (1490 turns / 421 conversations)

| Model | Matched Oracle | Model Self-History | Self all-turn exact |
|---|---:|---:|---:|
| direct | 81.01% | 73.62% | 57.48% |
| replay | 80.60% | 71.95% | 56.06% |
| transfer | 82.08% | 73.42% | 57.96% |

## FinQA Dev retention / ConvFinQA Oracle Dev

Experience Replay was evaluated with the historical frozen methods. Detailed paired results are in `evidence/analysis/`.

## Key caveats

- FinQA Test was consulted in earlier project phases: it is not an untouched blind test. No further tuning should be performed on it.
- ConvFinQA metrics are Dev results. Its underlying source documents overlap the original FinQA corpus.
- Oracle History uses previous GOLD answers. Self-History uses previous MODEL final-answer strings, not an unconstrained chat transcript.
- Document/conversation cluster bootstrap estimates sampling uncertainty for a single-seed comparison; it does not prove equivalence.
- Source model weights and raw datasets are referenced with SHA-256 hashes, NOT copied into this archive. Keep the original files or back them up separately.
