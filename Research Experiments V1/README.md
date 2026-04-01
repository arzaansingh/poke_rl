# V1 — Base Experiment

The foundational 2x2 factorial experiment testing Flat vs Hierarchical action decomposition crossed with Zero vs Smart Q-table initialization.

## Configuration

| Parameter | Value |
|-----------|-------|
| Pool Size | 20 (full OU pool) |
| Battles/Run | 10,000,000 |
| Runs/Model | 5 |
| Alpha (learning rate) | 0.1 |
| Gamma (discount) | 0.995 |
| Lambda (trace decay) | 0.6967 |
| Epsilon Decay | 5M battles (linear 1.0 to 0.05) |
| Logging Interval | Every 1,000 battles |

## Key Findings

- All 4 models learn to beat the heuristic opponent given enough training.
- **Hierarchical + Smart Init (Model 4)** converges fastest and achieves the highest win rate (~55%+).
- **Smart initialization** provides a significant head start, especially for the hierarchical model.
- **Hierarchical decomposition** outperforms flat when combined with smart init.
- Lambda=0.6967 provides good trace decay for this setting without cross-agent interference.

## Running

```bash
python run_all.py                        # Run all 4 models x 5 runs (parallel)
python run_all.py --tests                # Run test suite
python shared/plot.py                    # Generate per-model plots
python shared/compare.py                 # Generate cross-model comparison plots
```
