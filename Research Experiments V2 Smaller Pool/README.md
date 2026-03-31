# V2 — Smaller Pool

Reduced the Pokemon pool from 20 to 13 and increased statistical power with 30 runs per model.

## Changes from V1

- **Pool size:** 20 to 13 (top 13 OU-viable species)
- **Runs per model:** 5 to 30 for tighter confidence intervals
- **Battles per run:** 10M to 1M (shorter per run, more runs)
- **Gamma:** 0.995 to 0.99 (better for shorter episodes)
- **Lambda:** 0.6967 to 0.9 (from RL literature, Singh & Sutton 1996)
- **Epsilon decay:** 500K battles (50% of training)

## Configuration

| Parameter | Value |
|-----------|-------|
| Pool Size | 13 |
| Battles/Run | 1,000,000 |
| Runs/Model | 30 |
| Alpha | 0.1 |
| Gamma | 0.99 |
| Lambda | 0.9 |
| Epsilon Decay | 500K battles |

## Notes

- The increase to Lambda=0.9 was later found to cause trace interference in hierarchical models (see V3 README). V1's Lambda=0.6967 performs better for hierarchical architectures.

## Running

```bash
python run_all.py                        # Run all 4 models x 30 runs
python run_all.py --tests                # Run test suite
python shared/plot.py                    # Generate per-model plots
python shared/compare.py                 # Cross-model comparison
```
