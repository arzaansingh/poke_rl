# PokeAgent

Reinforcement learning agents for competitive Pokemon battles using a **2x2 factorial experiment design**: **Flat vs Hierarchical** action decomposition crossed with **Zero vs Smart** Q-table initialization.

Agents are trained via **Q-learning with eligibility traces (TD(lambda))** against a heuristic opponent in the Gen 4 OU metagame, using the [poke-env](https://github.com/hsahovic/poke-env) framework and a local [Pokemon Showdown](https://github.com/smogon/pokemon-showdown) server.

## Experiment Design

### 2x2 Factorial

|                  | Zero Init (Q=0)       | Smart Init (softmax priors) |
|------------------|-----------------------|-----------------------------|
| **Flat**         | Model 1: Flat + Zero  | Model 2: Flat + Smart       |
| **Hierarchical** | Model 3: Hier + Zero  | Model 4: Hier + Smart       |

- **Flat models** select from all available actions (moves + switches) using a single Q-table with a 35-dimensional state.
- **Hierarchical models** use a master policy (move vs switch, 20-dim state) and a sub-agent policy (which Pokemon to switch to, 17-dim state), with separate Q-tables.
- **Zero init** starts all Q-values at 0.
- **Smart init** seeds Q-values with softmax-normalized heuristic scores (type effectiveness, base stats) to bootstrap learning.

### State Representation

- **Battle state (20-tuple):** Active Pokemon species, HP, moves (4), type matchups, opponent species/HP/status, weather, hazards.
- **Bench detail (15-tuple):** 5 bench Pokemon, each with species, HP bucket, and status (sorted alphabetically for order invariance).
- **Flat state = battle_state + bench_detail** (35 features).
- **Hierarchical master = battle_state** (20 features), **sub-agent = opponent info + bench_detail** (17 features).

### Training

- **Algorithm:** Q-learning with replacing eligibility traces, linear epsilon decay.
- **Opponent:** `SimpleHeuristicsPlayer` from poke-env (deterministic heuristic baseline).
- **Dense rewards:** Step-level rewards for HP differential, faints, and status changes, plus +1/-1 for win/loss.
- **Teams:** Built from a pool of OU-legal Gen 4 Pokemon with role-based movesets, EVs, and items.

## Experiment Versions

| Version | Description | Battles/Run | Runs | Pool | Key Params |
|---------|-------------|-------------|------|------|------------|
| [V1](Research%20Experiments/) | Base experiment | 10M | 5 | 20 | alpha=0.1, gamma=0.995, lambda=0.6967 |
| [V2](Research%20Experiments%20V2%20Smaller%20Pool/) | Smaller pool, more runs | 1M | 30 | 13 | alpha=0.1, gamma=0.99, lambda=0.9 |
| [V3](Research%20Experiments%20V3%20Controlled%20Teams/) | Deterministic teams | 1M | 30 | 13 | alpha=0.1, gamma=0.99, lambda=0.9 |
| [V4](Research%20Experiments%20V4%20Grid%20Search/) | HP grid search | 100K | 10/combo | 9 | 3x3x3 grid over alpha, gamma, lambda |

Each version builds on the previous one. See individual folder READMEs for details.

## Project Structure

```
PokeAgent/
  Research Experiments/              # V1 — Base experiment
  Research Experiments V2 Smaller Pool/   # V2 — Reduced pool, 30 runs
  Research Experiments V3 Controlled Teams/  # V3 — Deterministic teams
  Research Experiments V4 Grid Search/      # V4 — Hyperparameter optimization
  pokemon-showdown/                  # Local Showdown server (not tracked)
```

Each experiment folder contains:
```
shared/
  config.py          # Hyperparameters, pool, grid definitions
  features.py        # State extraction (battle_state, bench_detail, etc.)
  rewards.py         # Dense reward calculation
  heuristics.py      # Smart init heuristic scoring
  team_builder.py    # Team generation from pokemon pool
  train_common.py    # Shared training loop
  tests.py           # Test suite
  plot.py            # Visualization
model_1_flat_zero/
  player.py          # Flat + Zero Init agent
  train.py           # Training entry point
model_2_flat_smart/  # Flat + Smart Init agent
model_3_hier_zero/   # Hierarchical + Zero Init agent
model_4_hier_smart/  # Hierarchical + Smart Init agent
```

## Requirements

- Python 3.8+
- [poke-env](https://github.com/hsahovic/poke-env)
- Node.js (for Pokemon Showdown server)
- numpy, pandas, matplotlib, seaborn

## Quick Start

```bash
# 1. Start a local Showdown server
cd pokemon-showdown
node pokemon-showdown start --no-security 9000

# 2. Run V4 grid search
cd "Research Experiments V4 Grid Search"
python run_grid.py --max-parallel 8

# 3. View results
python run_grid.py --summary
python shared/plot.py --heatmap

# 4. Run tests
python run_grid.py --tests --skip-integration
```
