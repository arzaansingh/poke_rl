# PokeAgent — Research Analysis (V1 → V6)

One-page condensed narrative of what each experiment version tested, what we learned, and where V6 is headed. For the full V5 paper-style writeup with statistical tests, see [`Research Experiments V5 Initialization Study/REPORT.md`](Research%20Experiments%20V5%20Initialization%20Study/REPORT.md).

---

## V1 — Base 2×2 factorial *(can tabular Q-learning even learn Pokémon?)*

**Design:** Flat vs Hierarchical × Zero vs Smart init, 10M battles/run, pool of 20.

**Finding:** The flat zero-init baseline stalls around **8% WR**. Hierarchical helps a little. Smart init moves the needle dramatically — jumping flat to the mid-40s. Established that the hierarchical decomposition is viable and that *initialization matters more than expected*.

**Open question:** is the smart-init effect real, or is it noise from massively variable team matchups?

---

## V2 — Smaller pool, more runs *(is the variance from learning or from teams?)*

**Design:** Reduced pool from 20 → 13 species, ran 30 independent seeds instead of 5. Same 2×2 design, 1M battles/run.

**Finding:** Per-run standard deviation dropped sharply when the pool shrank. The apparent noise in V1 was primarily **team-matchup variance**, not learning instability. Smart init's win-rate advantage over zero init was confirmed at higher statistical confidence.

**Open question:** even with a smaller pool, teams within a pool still vary battle-to-battle. Can we *fix* the matchup to isolate the pure learning signal?

---

## V3 — Controlled teams *(deterministic matchups isolate the learning signal)*

**Design:** Introduced the `IndexedTeambuilder` — a teambuilder seeded by `(base_seed, battle_index)` so that battle *N* always produces the same matchup across all 4 models. 1M battles × 30 runs.

**Finding:** Variance dropped again. Now the only stochastic factor in comparisons is **the agent's exploration noise**, not the environment. Smart init confirmed to be a real effect (not a lucky team draw). Hierarchical starts to pull ahead more consistently.

**Open question:** we've been using one hyperparameter triple (α, γ, λ) picked by intuition. Is it the right one?

---

## V4 — Hyperparameter grid search *(which (α, γ, λ) is optimal?)*

**Design:** 3×3×3 grid over α ∈ {0.05, 0.1, 0.2}, γ ∈ {0.95, 0.99, 0.999}, λ ∈ {0.5, 0.7, 0.9}, crossed with the 2×2 architecture-init factorial. 100K battles × 10 runs per cell.

**Finding:** α=0.1, γ=0.99, λ=0.7 (`hp_001`) wins. α=0.2 is unstable (trace divergence). γ=0.999 is over-discounting — our episodes are short (tens of turns), so γ=0.99 works fine. λ=0.9 over-credits early actions for terminal outcomes; λ=0.7 is sweet.

**Open question:** we've been running fixed ε-decay. But if Smart Init already encodes a good prior, does aggressive exploration *hurt*?

---

## V5 — Initialization Study *(the headline result)*

**Design:** Full **2×2×2 factorial** — Architecture (Flat / Hier) × Init (Zero / Smart) × Epsilon (Decay / Fixed) — crossed with the 8-combo HP grid, 10 runs each = **640 independent training runs**. 50K battles/run. `IndexedTeambuilder`, gen4ou, same pool.

### Headline numbers

| Model | Config | Best HP | Final WR |
|---|---|---|---|
| M1 | Flat + Zero + Decay | hp_001 | 8.3% |
| M2 | Flat + Smart + Decay | hp_001 | 46.8% |
| M3 | Hier + Zero + Decay | hp_001 | 49.3% |
| M4 | Hier + Smart + Decay | hp_001 | 58.5% |
| M5 | Flat + Zero + Fixed | hp_001 | 10.0% |
| M6 | Flat + Smart + Fixed | hp_001 | 48.1% |
| M7 | Hier + Zero + Fixed | hp_001 | 50.0% |
| **M8** | **Hier + Smart + Fixed** | **hp_001** | **59.8%** |

### Effect sizes (marginal, averaged across HP combos)

- **Smart Init:** +38.5 pp for Flat, +9.6 pp for Hier (biggest single factor)
- **Hierarchy:** +2.0 pp on average (consistent but small)
- **Fixed ε:** +1.4 pp when init is already smart (counter-productive with zero init)
- **λ = 0.7 vs 0.9:** +8.5 pp
- **α = 0.1 vs 0.2:** +4.9 pp

### Conclusions

1. **Initialization > exploration** in large discrete state spaces. When the prior already encodes domain knowledge, wide early exploration becomes noise.
2. **Hierarchy + Smart Init are complementary, not redundant.** Hierarchy reduces effective state-space; Smart Init accelerates learning per state.
3. **Fixed ε=0.05 beats decay** when starting from a smart prior. This is the paper's novelty claim.

See [V5 REPORT.md](Research%20Experiments%20V5%20Initialization%20Study/REPORT.md) for per-factor effect-size plots, interaction plots, and statistical tests. See [the poster](Research%20Experiments%20V5%20Initialization%20Study/poster/main.tex) for the presented summary.

---

## V6 — Ensemble Study *(do 30 weak-learners beat 1 strong-learner?)*

**Design:** Take the exact V5 winner (M8 = Hier + Smart + Fixed ε at hp_001). Train **K = 30 independent members** with different random seeds, each for **1M battles** (20× V5's per-run budget), using the **full 20-Pokémon pool**. Combine at inference via configurable **soft / hard / confidence voting**.

**Literature context:** Wiering & van Hasselt (2008), Osband et al. (2016 — Bootstrapped DQN), Lin et al. (2024 — "Curse of Diversity") are the key references. K=30 is unusually large for tabular work (published norm is 5-10) — the K-saturation ablation will show whether 30 actually helps over 10.

**Hypothesis:** Under the Schapire weak-learner framing, 30 stochastically-diverse Q-tables (individually 55-60% WR) should compose into a 62-70% WR ensemble. The diversity source is **seed-only** (no bagging, no HP perturbation) — this is the risky choice the paper needs to defend via pairwise-disagreement analysis.

**Controls (literature-required):**
- **Compute-matched single**: one 5M-battle `HierSmartPlayer` (= 1/6 ensemble compute) — does concentrating the budget in one agent do as well?
- **K-ablation**: evaluate at K ∈ {1, 5, 10, 20, 30} (free post-hoc) — does K=30 actually help over K=10?
- **Pairwise disagreement**: fraction of eval states where member-argmaxes differ — proves members didn't collapse to identical policies.
- **Unseen-state fallback rate**: how often does a member hit heuristic priors (because it never saw this state during training)?

**Status:** Code complete, smoke-tested. Training starts on workstation (AnyDesk remote), ~14-24h wall-clock at max-parallel=16.

See [V6 README](Research%20Experiments%20V6%20Ensemble%20Study/README.md) for reproduction commands.

---

## Research trajectory at a glance

```
V1 → "tabular Q-learning can learn Pokémon at all"
V2 → "the variance is teams, not learning"
V3 → "control the teams, isolate the learning signal"
V4 → "which (α, γ, λ) is best?"   →  hp_001
V5 → "initialization dominates exploration"  →  M8 at 59.8% WR
V6 → "an ensemble of 30 M8-seeds beats the single best"  →  in progress
```

Each version preserves the same core infrastructure (`shared/` package) so findings compound. V6 is fully self-contained (V5's `shared/` is copied in) — no cross-experiment imports.
