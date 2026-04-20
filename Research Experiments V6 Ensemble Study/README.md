# Research Experiments V6 — Ensemble Study

Ensemble Q-learning follow-up to V5. Trains **K = 30** independent
`HierSmartPlayer` agents at V5's best config (hp_001), each on the full
20-Pokémon pool, and combines them at inference via configurable voting
(soft / hard / confidence).

**Hypothesis:** 30 stochastically-diverse "weakly better than baseline"
Q-learners beat the single best V5 agent (59.8% WR).

---

## Directory layout

```
Research Experiments V6 Ensemble Study/
├── README.md
├── shared/                              # Self-contained (copied from V5)
│   ├── __init__.py
│   ├── config.py                        # V6 constants + pool + paths
│   ├── features.py                      # AdvancedFeatureExtractor (20/17/35-tuple)
│   ├── heuristics.py                    # HeuristicEngine (smart-init priors)
│   ├── rewards.py                       # Dense shaping rewards
│   ├── team_builder.py                  # IndexedTeambuilder (deterministic gen4ou)
│   └── train_common.py                  # run_training async loop
├── model_ensemble/
│   ├── __init__.py
│   ├── hier_smart_player.py             # V5 M8 player (Watkins Q(λ), hierarchical)
│   ├── player.py                        # EnsemblePlayer + MemberQTable
│   └── train_member.py                  # One-member entry point
├── baseline_single/
│   └── train_single.py                  # Compute-matched 5M single-agent control
├── run_ensemble.py                      # Parallel training orchestrator
├── play_ensemble.py                     # Evaluation CLI (soft/hard/confidence)
├── analyze_ensemble.py                  # Plot generator
└── ensemble_results/
    ├── run_1/                           # Main K=30 run
    │   ├── params.json
    │   ├── pool.json
    │   ├── manifest.json
    │   ├── member_1/ … member_30/
    │   └── plots/
    └── baseline_5m/                     # Compute-matched single control
```

**V6 is fully self-contained** — no imports from V5. All required shared
modules are copied in place.

---

## Configuration (`shared/config.py`)

| Constant | Value | Notes |
|---|---|---|
| `K_MEMBERS` | 30 | ensemble size |
| `BATTLES_PER_MEMBER` | 1,000,000 | 20× V5's per-run budget |
| `BASELINE_SINGLE_BATTLES` | 5,000,000 | compute-matched control |
| `FULL_POKEMON_POOL` | 20 species | same as V5; V6 uses all 20 |
| `HP_ALPHA`, `HP_GAMMA`, `HP_LAM` | 0.1, 0.99, 0.7 | V5 hp_001 winner |
| `FIXED_EPS` | 0.05 | V5 M8 winner |
| `BATTLE_FORMAT` | `gen4ou` | |
| `BASE_SEED` | 10,000 | member_k uses seed `BASE_SEED + k` |
| `DEFAULT_MAX_PARALLEL` | 10 | concurrent training subprocesses |
| `DEFAULT_BASE_PORT` | 9000 | Showdown port slot i uses `base_port + i` |

---

## Prerequisites

1. **Pokemon Showdown** installed at `<project_root>/pokemon-showdown/`.
   Orchestrator spawns it automatically per port; make sure `node` + the
   checked-out pokemon-showdown repo are in place. (Same setup as V5.)
2. **Python deps:** `poke_env`, `numpy`, `matplotlib`. (Already installed for V5.)

---

## Usage

### Smoke test (before real run)

```bash
cd "Research Experiments V6 Ensemble Study"

# Tiny K=2, 500-battle end-to-end test (~2 min)
python run_ensemble.py --k 2 --battles 500 --run-id 999 --max-parallel 2

# Evaluate the two smoke members vs heuristic
python play_ensemble.py --members ensemble_results/run_999 --strategy soft --n-battles 20 --port 9000
python play_ensemble.py --members ensemble_results/run_999 --strategy hard --n-battles 20 --port 9000
```

### Main K=30 run (~9 days wall-clock)

```bash
# Launch the ensemble. Resumable — Ctrl+C and relaunch with --resume to continue.
python run_ensemble.py --k 30 --battles 1000000 --run-id 1 --max-parallel 10 --resume

# Status check at any time:
python run_ensemble.py --status --run-id 1
```

### Compute-matched baseline (5M single agent, ~3 days)

Can run in parallel to the ensemble on a different port.

```bash
python run_ensemble.py --run-baseline --run-id 1 --base-port 9020
```

### Evaluation

```bash
# Headline number: soft-vote ensemble vs heuristic, 500 battles
python play_ensemble.py --members ensemble_results/run_1 --strategy soft --n-battles 500 --log-diagnostics

# Strategy ablation
for s in soft hard confidence; do
  python play_ensemble.py --members ensemble_results/run_1 --strategy $s --n-battles 500
done

# K-saturation curve (free post-hoc)
for n in 1 5 10 20 30; do
  python play_ensemble.py --members ensemble_results/run_1 --strategy soft --subset $n --n-battles 500
done

# Compute-matched comparison
python play_ensemble.py --members ensemble_results/run_1 --strategy soft \
    --opponent baseline_5m --n-battles 500
```

### Human play mode (for expo demo)

```bash
# Ensemble accepts browser challenges
python play_ensemble.py --members ensemble_results/run_1 --strategy soft --human --port 8000
# Then at http://127.0.0.1:8000 search for the username it prints
```

### Generate all paper/poster plots

```bash
python analyze_ensemble.py --run-id 1 --dpi 300
# Outputs to ensemble_results/run_1/plots/
```

---

## Controls & ablations (literature-driven)

The K=30 headline must be reported alongside these:

| Control | How to obtain |
|---|---|
| **V5 M8 single (50K, 59.8%)** | Reference from V5 poster |
| **Compute-matched V6 single (5M)** | `run_ensemble.py --run-baseline` |
| **K-saturation curve** | `play_ensemble.py --subset {1,5,10,20,30}` |
| **Per-member solo WR distribution** | Each member's final `RollingWin` from `member_k/logs/run_1.csv` |
| **Pairwise disagreement rate** | `play_ensemble.py --log-diagnostics` |
| **Unseen-state fallback rate** | `play_ensemble.py --log-diagnostics` |

All are rendered by `analyze_ensemble.py`.

---

## Implementation details

### Diversity source
Random seed only (`seed = BASE_SEED + k`). All HPs, init, algorithm, and
pool are identical across members. Exploration stochasticity drives
decorrelation.

### Voting strategies (in `EnsemblePlayer`)
- **`soft`**: argmax of mean Q across members (bagging-style averaging)
- **`hard`**: plurality vote over per-member argmax (tie-break on mean Q)
- **`confidence`**: weight member k by `(max_k - mean_k)` spread, then mean

### Unseen-state handling
For any `(state, action)` a member hasn't seen, prior is seeded on-the-fly
via the same `HeuristicEngine.build_q_priors` used during training init
(generalized optimistic initialization). Every member always contributes a
vote. Optional `--log-diagnostics` tracks how often this fallback fires.

### Resumability
Subprocess-level: each member trains in 5,000-battle batches.
`train_common.run_training` pickles its Q-tables every 5,000 battles
(`SAVE_FREQ`). On restart, `--historic_battles` passes the resume point.
Interrupting with Ctrl+C and relaunching with `--resume` is safe.

### Memory at inference
30 Q-tables, each potentially 1-3 GB at 1M battles. `play_ensemble.py`
loads them all at startup via `MemberQTable.load`. If your machine RAM
is tight, reduce via `--subset` or load fewer members.

---

## References (for the paper)

1. Wiering & van Hasselt (2008), *IEEE TSMC-B* — seminal tabular ensemble RL.
2. Osband et al. (2016), *NeurIPS* — Bootstrapped DQN; reports K~10 saturation.
3. Anschel, Baram & Shimkin (2017), *ICML* — Averaged-DQN.
4. Lan et al. (2020), *ICLR* — Maxmin Q-learning.
5. Lin et al. (2024), *ICLR* — "Curse of Diversity"; motivates the 5M control.
6. Song et al. (2023), *Applied Soft Computing* — Ensemble RL survey.
7. Sutton & Barto (2018) — Q(λ), optimistic initialization.
8. Sahovic (2020) — poke-env infrastructure.
