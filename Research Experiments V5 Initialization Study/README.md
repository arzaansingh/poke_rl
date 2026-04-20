# V5 — Initialization Study

V5 is intended to stay as close to V4 as possible, with only two substantive changes:

- Smart initialization now standardizes heuristic scores directly into `[-0.01, 0.01]`.
- Each original model has a matched fixed-epsilon variant, giving 8 total models.

## Model Set

Decay schedule:

- `model_1_flat_zero`
- `model_2_flat_smart`
- `model_3_hier_zero`
- `model_4_hier_smart`

Fixed epsilon schedule:

- `model_5_flat_zero_fixed_eps`
- `model_6_flat_smart_fixed_eps`
- `model_7_hier_zero_fixed_eps`
- `model_8_hier_smart_fixed_eps`

The fixed-epsilon models use the terminal epsilon (`0.05`) from battle 0 onward.

## Grid

| Parameter | Values |
|-----------|--------|
| Alpha | 0.10, 0.20 |
| Gamma | 0.99, 0.999 |
| Lambda | 0.7, 0.9 |

Reduced to 2×2×2 = 8 combos (V4 showed λ and low α/γ had minimal impact).

## Smart Init Change

V4 seeded unseen actions with softmax probabilities, which effectively turned initialization into a policy prior.  
V5 still ranks actions heuristically, but now maps those scores linearly into `[-0.01, 0.01]` so the agent gets a bounded head start without locking itself into the heuristic.

## Commands

```bash
python run_grid.py
python run_grid.py --max-parallel 16
python run_grid.py --combo hp_001
python run_grid.py --combo hp_001 --run 3
python run_grid.py --list-grid
python run_grid.py --summary
python shared/plot.py --combo hp_001
python shared/plot.py --combo hp_001 --panel
python run_grid.py --tests --skip-integration
```
