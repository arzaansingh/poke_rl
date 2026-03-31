# V4 — Hyperparameter Grid Search

Systematic grid search over the three core Q-learning hyperparameters to find the optimal configuration for each model variant. Each combo is evaluated across multiple runs with different randomly-sampled Pokemon pools.

## Changes from V3

- **Grid search:** 3x3x3 = 27 hyperparameter combinations over alpha, gamma, lambda.
- **Per-run pools:** Each run uses a different randomly-sampled subset of Pokemon (deterministic by run_id). All 4 models in the same run share the same pool.
- **Parallel orchestration:** `run_grid.py` manages concurrent experiments with slot-based port allocation and a live cursor-based dashboard.
- **Per-battle epsilon decay:** Epsilon updates after every battle (smooth decay) rather than per-batch.

## Grid

| Parameter | Values |
|-----------|--------|
| Alpha (learning rate) | 0.05, 0.1, 0.2 |
| Gamma (discount factor) | 0.9, 0.99, 0.999 |
| Lambda (trace decay) | 0.5, 0.7, 0.9 |

## Configuration

| Parameter | Value |
|-----------|-------|
| Pool Size | 9 (configurable) |
| Battles/Run | 100,000 |
| Runs/Combo | 10 |
| Epsilon Decay | 100K battles |
| Batch Size | 2,000 (subprocess resume granularity) |
| Save Frequency | 2,000 battles |

## Directory Structure

```
grid_results/
  hp_001/                          # One HP combo (alpha=0.05, gamma=0.9, lam=0.5)
    params.json                    # {alpha, gamma, lam, combo_id}
    run_1/
      pool.json                    # The 9 pokemon for this run
      model_1_flat_zero/logs/run_1.csv
      model_2_flat_smart/logs/run_1.csv
      model_3_hier_zero/logs/run_1.csv
      model_4_hier_smart/logs/run_1.csv
    run_2/
      ...
  hp_002/
    ...
  summary.csv                      # Ranked results across all combos
```

## Commands

### Training

```bash
python run_grid.py                            # Full grid search (8 parallel)
python run_grid.py --max-parallel 16          # More parallelism
python run_grid.py --combo hp_001             # Single combo (all runs x 4 models)
python run_grid.py --combo hp_001 --run 3     # Single combo, single run
python run_grid.py --list-grid                # Show all 27 HP combos
python run_grid.py --summary                  # Ranked results table
python run_grid.py --reset                    # Delete all results
```

### Plotting

```bash
python shared/plot.py --combo hp_001                           # 4 models on one chart (mean + std)
python shared/plot.py --combo hp_001 --panel                   # 4-panel: WR, TableSize, Eps, Reward
python shared/plot.py --combo hp_001 --metric OverallWin       # Overall win rate
python shared/plot.py --combo hp_001 --metric AvgReward        # Average reward
python shared/plot.py --combo hp_001 --metric TableSize        # Q-table growth
python shared/plot.py --combo hp_001 --model model_4_hier_smart  # Single model
python shared/plot.py --combo hp_001 hp_010 hp_019             # Compare combos
python shared/plot.py --combo hp_001 hp_010 --model model_2_flat_smart  # Compare for one model
python shared/plot.py --heatmap                                # Best WR heatmap across grid
python shared/plot.py                                          # Plot all combos + heatmap
python shared/plot.py --combo hp_001 --no-save                 # Show interactively
```

### Testing

```bash
python run_grid.py --tests                    # Full test suite (140 tests)
python run_grid.py --tests --skip-integration # Skip Showdown-dependent tests
```
