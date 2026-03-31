# V3 — Controlled Teams

Introduced deterministic per-battle team generation so all 4 models face the exact same opponent teams, isolating the model architecture as the only variable.

## Changes from V2

- **IndexedTeambuilder:** Battle N always produces the same learner and opponent teams across all 4 models. Uses `Random(f"{base_seed}_{battle_index}")` for deterministic generation.
- **Separate seeds:** Opponent and learner use different base seeds (seed+1000 vs seed+2000) so they get different teams from each other, but the same teams across models.

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

## Known Issue: Lambda=0.9 Trace Interference

V3 results showed hierarchical models (Models 3 & 4) underperforming compared to V1, converging to ~37% rolling win rate while flat smart (Model 2) reached 46%.

**Root cause:** Lambda=0.9 creates eligibility traces that persist ~170x longer than V1's Lambda=0.6967. Hierarchical models have two interacting trace systems (master + sub-agent) that interfere with each other at high lambda values. Flat models only have one trace chain and are unaffected.

| Version | Gamma x Lambda | Trace at 20 steps |
|---------|---------------|-------------------|
| V1 | 0.995 x 0.6967 = 0.693 | 0.06% |
| V3 | 0.99 x 0.9 = 0.891 | 10.1% |

**Fix:** Restoring Lambda=0.6967 resolves the interference. This motivated the V4 grid search.

## Running

```bash
python run_all.py                        # Run all 4 models x 30 runs
python run_all.py --tests                # Run test suite
python shared/plot.py                    # Generate per-model plots
python shared/compare.py                 # Cross-model comparison
```
