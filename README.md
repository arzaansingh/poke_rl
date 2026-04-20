# PokeAgent — Tabular Reinforcement Learning for Competitive Pokémon

A research codebase exploring whether **tabular Q-learning with domain-informed initialization** can produce a competitive Gen 4 OU Pokémon agent without deep networks. Each experiment version tests a specific hypothesis about *what actually makes RL work* in a large, stochastic, two-player game: state representation, action decomposition, Q-value initialization, exploration schedule, and — in V6 — ensembling.

Agents are trained via **Watkins Q(λ) with eligibility traces** against a heuristic opponent (`SimpleHeuristicsPlayer` from [poke-env](https://github.com/hsahovic/poke-env)) on a local [Pokémon Showdown](https://github.com/smogon/pokemon-showdown) server.

---

## Research thesis

> In a tabular setting, the choice of **Q-value initialization** often matters more than the choice of **exploration schedule**. When priors already encode domain knowledge, aggressive exploration becomes counter-productive. Ensembling multiple "weakly-better-than-baseline" Q-learners should then beat any single best-tuned agent.

The V1 → V6 progression is designed to establish this claim incrementally:

- **V1-V3** — build the baseline and isolate variance drivers.
- **V4** — hyperparameter grid search to find (α, γ, λ) sweet spot.
- **V5** — full 2×2×2 factorial (architecture × init × ε-schedule), 640 runs. **Headline result: Hierarchical + Smart Init + Fixed ε = 59.8% WR** at the best HP combo.
- **V6** — ensemble of 30 V5-winner members, targeting Schapire-style weak-learner aggregation.

---

## Experiment versions

| Version | Question | Design | Battles / run | Pool | Key finding |
|---|---|---|---|---|---|
| **[V1](Research%20Experiments%20V1/)** | Can a flat Q-table beat a heuristic? | 2×2 factorial (architecture × init) | 10M | 20 | First working agent; flat+zero was ~8% WR. |
| **[V2](Research%20Experiments%20V2%20Smaller%20Pool/)** | Is the variance from team diversity or from learning? | Same 2×2 but pool=13, 30 runs | 1M | 13 | Variance mostly from teams; smaller pool → clearer signal. |
| **[V3](Research%20Experiments%20V3%20Controlled%20Teams/)** | Do deterministic teams isolate the learning signal? | Indexed teambuilder (same matchup every N battles) | 1M | 13 | Yes — confirms the variance attribution. |
| **[V4](Research%20Experiments%20V4%20Grid%20Search/)** | Which (α, γ, λ) performs best? | 3×3×3 grid × 2×2 factorial × 10 runs | 100K | 9 | hp_001 = (α=0.1, γ=0.99, λ=0.7) wins. |
| **[V5](Research%20Experiments%20V5%20Initialization%20Study/)** | Does hierarchy + smart init + fixed ε dominate? | 2×2×2 × 8 HP × 10 runs = **640 runs** | 50K | 20→8 sampled | **M8 at hp_001 = 59.8% WR.** Smart init = +38.5pp for flat, +9.6pp for hier. Fixed ε beats decay when init is already smart. |
| **[V6](Research%20Experiments%20V6%20Ensemble%20Study/)** | Does an ensemble of 30 V5-winner seeds beat the single best V5 run? | K=30 × 1M battles/member, soft/hard/confidence voting at eval | 1M | 20 (full) | **In progress** (~24h on an 8-core box). |

See each folder's `README.md` for per-version detail; V5 also has a comprehensive [`REPORT.md`](Research%20Experiments%20V5%20Initialization%20Study/REPORT.md) and a poster — see below.

---

## Poster (V5 results)

Presented at Tulane CMPS 4020 Spring 2026:

- Source: [`Research Experiments V5 Initialization Study/poster/main.tex`](Research%20Experiments%20V5%20Initialization%20Study/poster/main.tex)
- Figures: [`Research Experiments V5 Initialization Study/poster/figures/`](Research%20Experiments%20V5%20Initialization%20Study/poster/figures/)
- Interactive diagram builder (React+Vite): [`Research Experiments V5 Initialization Study/poster/poster-visuals/`](Research%20Experiments%20V5%20Initialization%20Study/poster/poster-visuals/)

Headline figures:
- **Heatmap** of win rate across 8 models × 8 HP combos
- **Learning curves** for hp_001 (the winner)
- **Violin + strip plot** of 10-run variance
- **Interaction plot** showing Smart Init × Architecture trade-off
- **Effect sizes** bar chart

Poster PDF can be rebuilt via `pdflatex main.tex` inside the `poster/` directory (requires a TeX distribution). The React diagram builder is run via `cd poster/poster-visuals && npm install && npm run dev`.

---

## Analysis summary

See [**`ANALYSIS.md`**](ANALYSIS.md) for the condensed research narrative (V1 → V6 findings, why each experiment matters, and what V6 is testing).

For the full V5 write-up with statistical tests and per-factor effect sizes:
[`Research Experiments V5 Initialization Study/REPORT.md`](Research%20Experiments%20V5%20Initialization%20Study/REPORT.md).

---

## Shared architecture (V5 onward)

**State representation** (unchanged across V5 and V6, used by the winning M8 config):

- **Battle state (20-tuple):** active Pokémon species/HP/status/ability/boosts, opponent context, speed check, field conditions, hazards.
- **Bench detail (15-tuple):** 5 bench slots, alphabetically sorted, each (species, hp-bucket, status).
- **Flat state (35-tuple):** battle_state + bench_detail.
- **Hierarchical:** master = 20-tuple, sub-agent = 17-tuple (opponent context + bench_detail).

**Algorithm:** Watkins Q(λ) with replacing eligibility traces, cleared on exploratory actions.

**Rewards:** Dense per-step shaping (faint delta × 0.1, HP delta × 0.05, status delta × 0.01, boost × 0.01) + terminal ±1.

**Teams:** Role-based movesets from `pokemon-showdown/data/random-battles/gen4/sets.json`, built via the deterministic `IndexedTeambuilder` so that battle *N* always produces the same matchup.

---

## Project layout

```
PokeAgent/
├── README.md                                      # ← you are here
├── ANALYSIS.md                                    # Research narrative V1-V6
├── .gitignore
│
├── Research Experiments V1/                       # Base 2×2 factorial
├── Research Experiments V2 Smaller Pool/          # Pool=13, 30 runs
├── Research Experiments V3 Controlled Teams/      # Deterministic teams
├── Research Experiments V4 Grid Search/           # HP grid 3×3×3
├── Research Experiments V5 Initialization Study/  # 2×2×2 factorial × 8 HP × 10 runs
│   ├── REPORT.md                                  # Full paper-style writeup
│   ├── poster/                                    # LaTeX poster + figures
│   ├── plots/                                     # 64 per-model learning curves + effect plots
│   └── grid_results/                              # Per-run CSVs (large .pkl models excluded)
└── Research Experiments V6 Ensemble Study/        # K=30 ensemble on V5 winner
    ├── README.md
    ├── run_ensemble.py                            # Orchestrator w/ live dashboard
    ├── play_ensemble.py                           # Eval: soft/hard/confidence voting
    ├── analyze_ensemble.py                        # Plots
    ├── model_ensemble/                            # HierSmartPlayer + EnsemblePlayer
    ├── baseline_single/                           # 5M compute-matched control
    └── shared/                                    # Self-contained V5 shared/ copy

Each `Research Experiments V*/` folder contains a `shared/` package with:
  config.py       — HPs, pool, paths
  features.py     — state extraction
  heuristics.py   — smart-init scoring
  rewards.py      — dense shaping
  team_builder.py — IndexedTeambuilder
  train_common.py — async training loop (graceful SIGINT/SIGTERM save in V6)
```

---

## Quick start

```bash
# 1. Local Pokémon Showdown server (one-time install + launch)
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown && npm install && node pokemon-showdown start --no-security --skip-build 9000 &

# 2. V5 — reproduce the headline result
cd "Research Experiments V5 Initialization Study"
python run_grid.py --max-parallel 8            # ≈ 640 runs
python run_grid.py --summary                    # Ranked WR table

# 3. V6 — train the K=30 ensemble (auto-resumes, live dashboard)
cd "Research Experiments V6 Ensemble Study"
python run_ensemble.py --max-parallel 16        # Terminal 1
python run_ensemble.py --run-baseline           # Terminal 2 (5M single control)

# 4. V6 eval (after training)
python play_ensemble.py --members ensemble_results/run_1 --strategy soft --n-battles 500
python analyze_ensemble.py --run-id 1
```

---

## Requirements

- Python 3.10+
- [poke-env](https://github.com/hsahovic/poke-env) ≥ 0.10
- Node.js + Pokémon Showdown (at `<project_root>/pokemon-showdown/`)
- numpy, pandas, matplotlib, seaborn, scipy

---

## Reproducibility

- Every training run is seeded deterministically (`seed = BASE_SEED + run/member`).
- Teams are deterministic given `(seed, battle_index)` via `IndexedTeambuilder`.
- Q-tables checkpoint to disk every 5,000 battles; `SIGINT`/`SIGTERM` save gracefully (V6).
- All training CSVs (small, versioned) are committed; Q-table pickles (large, regenerable) are `.gitignore`d.

---

## Citation

If you use this work:
```
Arzaan Singh. "PokeAgent: Tabular Reinforcement Learning for Competitive Pokémon Battling."
CMPS 4020, Tulane University, Spring 2026.
```

Supervised by Dr. Zheng.
