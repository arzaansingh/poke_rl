# V6 Paper Outline — section-by-section plan

> **Target format:** NeurIPS long (9 content pages + unlimited refs + unlimited appendix). Compress to short (4 pages) if Dr. Zheng prefers a workshop submission. Section budgets below are for the long format.
>
> **Working title (placeholder):**
> *Ensembling Tabular Q-Learning Agents for Competitive Pokemon: A Read-Only Inference Method with Heuristic-Prior Fallback*
>
> **Alternate titles (less verbose):**
> - *Q-Function Ensembles for Pokemon Battling*
> - *Memory-Bounded Inference for Tabular Q-Learning Ensembles*

---

## Abstract (~200 words, ¼ page)

**Structure (5 sentences):**

1. **Setup** — Tabular reinforcement learning in large stochastic two-player games is limited by single-agent ceiling effects: throwing more compute at one agent does not break through.
2. **Method** — We train K = 30 independent tabular Q(λ) agents on the same 20-Pokemon pool in Showdown's Gen 4 OU format, then combine them at inference via three voting rules (soft, hard, confidence-weighted) with a read-only heuristic-prior fallback for unseen states.
3. **Key result** — An ensemble of K = 10 agents achieves 66.8% ± 0.3% win rate over 100,000 evaluation battles against poke-env's SimpleHeuristicsPlayer, a 12.6 percentage-point lift over a compute-matched single agent trained for 5× longer (z = 62σ, p < 0.001).
4. **K-saturation curve** — Performance saturates near K = 10 in the predicted Osband-2016 shape; voting rule choice matters less than K.
5. **Significance** — The result establishes that ensembling Q-function tabular learners delivers strictly more value than equivalent single-agent compute in this regime; the read-only inference rule we propose is, to our knowledge, the first such treatment in tabular Q-ensemble literature.

**Don't include in the abstract:** specific algorithms (Q(λ), eligibility traces), implementation details, related-work references, dataset construction.

---

## 1. Introduction (~1 page)

### Goals
- Hook the reader.
- State the problem.
- Preview the contribution.
- List the contributions explicitly.

### Paragraph plan

1. **Motivation paragraph (5 sentences).** Reinforcement learning faces a recurring bottleneck: when a single agent's policy plateaus, throwing more compute at the same agent yields diminishing or even negative returns. Pokemon Showdown is a canonical stochastic, partially observable two-player game with ~10¹² distinct states. Tabular Q-learning solutions to it have plateaued around 55% win rate against poke-env's heuristic baseline; deep-RL agents with much greater compute push this to 85-95% (Wang 2024, Grigsby et al. 2025). The natural question this paper asks: **can we recover most of the deep-RL gain using ensembles of small tabular learners?**

2. **Why ensembling specifically (3 sentences).** Ensembling has a long history in supervised learning (bagging, boosting, random forests). In RL it is empirically successful for value-function variance reduction (Anschel et al. 2017; Lan et al. 2020) and exploration (Osband et al. 2016). But the action-time voting variant — train K agents independently, combine their decisions at evaluation — has been understudied in tabular RL on stochastic adversarial games.

3. **What's new (2 sentences).** We close this gap. Our paper makes the following contributions:

4. **Contributions list (bulleted, 3-4 items):**
   - **C1**: A read-only, memory-bounded inference rule for tabular Q-ensembles with heuristic-prior fallback for unseen states; the rule is asymptotically equivalent to V5-style "smart init" while reducing memory growth from O(unique-states-encountered) to O(1) per battle.
   - **C2**: An empirical comparison of three voting rules (soft, hard, confidence-weighted) at K = 1, 3, 5, 10 on the Pokemon Gen 4 OU format — to our knowledge the first such comparison in tabular Q on a stochastic two-player game.
   - **C3**: A compute-matched evaluation framework: K = 10 agents at 1M training battles each compared against a single agent at 5M battles. This is principled but rare in the prior ensemble RL literature.
   - **C4**: An open-source pipeline (poke-env compatible) released at github.com/arzaansingh/poke_rl, reproducing all results.

5. **Key result preview (1 sentence + 1 number).** Our K=10 ensemble achieves 66.8% win rate vs SimpleHeuristicsPlayer, a +12.6 pp lift over the compute-matched 5M baseline. Paper layout follows.

### Word budget: ~700 words

---

## 2. Related Work (~1 page)

### Structure: three paragraphs, each ~150 words.

#### 2.1 Ensemble RL (tabular)
- Wiering & van Hasselt (2008) — algorithmic ensemble, majority + Boltzmann combination rules
- Faußer & Schwenker (2011) — same-algorithm seed ensemble, averaging + majority
- Lan et al. (2020) Maxmin Q — K-independent estimators for bias control (tabular section)
- Peer et al. (2021) EBQL — K-way generalization of double-Q

#### 2.2 Ensemble RL (deep)
- Osband et al. (2016) Bootstrapped DQN — K = 10 heads, Thompson sampling at episode start
- Anschel et al. (2017) Averaged-DQN — temporal ensemble of target networks
- Chen et al. (2021) REDQ — K = 10 ensemble, min-over-subset target
- Lee et al. (2021) SUNRISE — UCB action selection (closest deep analog to our confidence vote)
- Lin et al. (2024) Curse of Diversity — buffer-sharing failure mode, **which V6 avoids by construction**

#### 2.3 RL for Pokemon
- Lee & Togelius (2017) — Showdown AI Competition, tabular Q baselines vs Random
- Kalose et al. (2018) — Stanford project, tabular Q vs Random, ~70% WR
- Sahovic (2020) — poke-env, the canonical research library
- Simões et al. (2020) — competitive deep RL on a custom Pokemon environment
- Wang (2024) — MIT M.Eng. thesis, PPO + MCTS in gen4randombattle, 85-91% vs SimpleHeuristicsPlayer (random battle, not OU)
- Hu et al. (2024) PokéLLMon — GPT-4 agent, 26% vs heuristic
- Karten et al. (2025) PokeChamp — LLM minimax, no SimpleHeuristics number
- Grigsby et al. (2025) Metamon — 200M-param transformer + offline RL, 64-80% in early-gen OU composites
- Anonymous (2025) VGC-Bench — MARL self-play populations for VGC doubles

**Headline gap (to assert in the paper):** No prior paper trains independent tabular Q agents and combines them by action-time voting on Pokemon.

### Word budget: ~750 words

---

## 3. Background and Method (~1.5 pages)

### 3.1 Preliminaries (~½ page, math-heavy)

Define:
- **MDP / POMDP notation** — states *s*, actions *a*, transition *P*, reward *r*, discount *γ*, partial-observability function *O*
- **Watkins Q(λ)** — eligibility-trace update equation with replacing traces
- **Pokemon Showdown as a POMDP** — observation = own team complete + opponent active visible only, action set varies turn-by-turn
- **Hierarchical action decomposition** — master state s_m ∈ ℝ²⁰, sub-state s_s ∈ ℝ¹⁷, action sets A_master = {move₁, …, move₄, switch}, A_sub = {pokémon₁, …, pokémon₅}
- **Heuristic prior** — *h(s, a)*, defined as the V5 smart-init function from Singh & Zheng (2026) earlier work

### 3.2 The ensemble inference rule (~½ page, this is C1)

State the problem formally:
- Given K trained Q-tables {Qᵏ}, choose action *a* at state *s*
- Each agent k may or may not have observed (s, a) during training
- For unseen (s, a), define fallback *Qᵏ(s, a) := h(s, a)* — pure function of state

**Define three combination rules:**

**Soft (mean argmax):**
$$ a^* = \arg\max_a \frac{1}{K} \sum_{k=1}^{K} Q^k(s, a) $$

**Hard (plurality with mean-Q tiebreak):**
$$ a^* = \arg\max_a |\{k : a = \arg\max_{a'} Q^k(s, a')\}| $$
(Ties broken on mean Q.)

**Confidence-weighted:**
$$ w_k = \max\left(\max_a Q^k(s, a) - \frac{1}{|A|}\sum_a Q^k(s, a),\ 0\right) $$
$$ a^* = \arg\max_a \frac{\sum_k w_k \cdot Q^k(s, a)}{\sum_k w_k} $$
(Fall back to soft when ∑w_k = 0.)

**Key property:** All three rules are pure functions of {Qᵏ}, never mutate any Qᵏ. State this and note the memory-bounded-ness corollary.

### 3.3 Training the ensemble members (~½ page)

- K = 30 independent agents, distinct random seeds
- Each agent: Watkins Q(λ) with α = 0.1, γ = 0.99, λ = 0.7, fixed ε = 0.05
- 20-Pokemon pool, gen4ou format, IndexedTeambuilder (deterministic teams per battle)
- 1,000,000 battles per agent, total 30M training battles
- 8 cloud AWS EC2 c5.4xlarge in parallel, 15 days
- Reproducibility — seed = base_seed + k; fixed pool; saved checkpoints

**Reference earlier work (V5):** Singh & Zheng (2026) for the smart-init function h and hierarchical decomposition; cite the V5 poster if it's been written up, or mark "in preparation."

### Word budget: ~1500 words

---

## 4. Experimental Setup (~½ page)

### Subsections

- **4.1 Environment** — poke-env wrapper over Pokemon Showdown server, gen4ou, IndexedTeambuilder (deterministic teams per battle index)
- **4.2 Evaluation conditions** — 8 conditions tabulated (the 3 strategies × K=10, the K-saturation curve {K=1,3,5,10} at SOFT, the head-to-head, and the compute-matched baseline)
- **4.3 Evaluation protocol** — 100,000 battles per condition, Wilson 95% CIs
- **4.4 Baseline opponents** — SimpleHeuristicsPlayer (poke-env's built-in), our own 5M-battle compute-matched single agent

### Word budget: ~400 words

---

## 5. Results (~2 pages)

### 5.1 Headline: ensemble beats compute-matched single agent (~½ page)

- Main number: K=10 SOFT = **66.8% ± 0.3%**, baseline_5m = **54.2% ± 0.3%**, Δ = **+12.6 pp, z ≈ 62**
- Head-to-head ensemble vs baseline = 59.7% (≈ 61σ above coin flip)
- **Figure 1**: headline_comparison.png

### 5.2 K-saturation curve (~½ page)

- K = 1: 55.9%; K = 3: 62.7%; K = 5: 65.1%; K = 10: 66.8%
- Matches Osband 2016 saturation shape
- **Figure 2**: k_saturation_curve.png

### 5.3 Voting strategy comparison (~½ page)

- SOFT = 66.8%, HARD = 65.6%, CONFIDENCE = 67.6% — all within 2pp
- Confidence is slight winner; hard underperforms (loses Q-magnitude info)
- **Figure 3**: strategy_comparison.png

### 5.4 Per-member diagnostics (~½ page)

- Per-member solo WRs: distribution, mean, std (the "weak learner" framing)
- Pairwise disagreement at K=10 SOFT — proves diversity preserved, not collapsed
- Unseen-state fallback rate — quantifies how often heuristic fires
- **Figure 4**: per_member_final_bar.png + diagnostics.png

### 5.5 Comparison to prior work table (~¼ page)

- The comparison table from our lit review (Wang, Metamon, PokeLLMon, Lee & Togelius, Kalose) with V6 inserted
- Frame: "V6 lands inside Metamon's gen4OU band at orders-of-magnitude less compute, well above all prior tabular and LLM work"
- **Table 1**

### Word budget: ~2000 words

---

## 6. Discussion (~1 page)

### Subsections

- **6.1 Why the ensemble works** — variance reduction across stochastic exploration paths produces uncorrelated errors; voting averages them out
- **6.2 Why ensembling beats more single-agent compute** — diminishing returns + state-space coverage; 30M total battles across 10 agents covers more of the state distribution than 5M battles in one agent
- **6.3 Comparison to deep RL** — concede that Wang 2024 and Metamon outperform V6 but at 10²–10⁴× more compute; argue V6 is a Pareto point on the compute/performance curve, not an absolute peak
- **6.4 Avoiding the Curse of Diversity (Lin et al. 2024)** — independent buffers, no shared off-policy data; explicit
- **6.5 Limitations** — single opponent, single pool, no test-time generalization, no deep-RL ablation

### Word budget: ~900 words

---

## 7. Conclusion (~⅓ page)

- Restate contributions
- Restate headline number
- Future work: (a) deep-RL variants of members, (b) different diversity sources (bagging, HP perturbation), (c) multi-opponent / multi-pool generalization, (d) hierarchical voting (master + sub agents independently)

### Word budget: ~250 words

---

## 8. References (unlimited)

- Estimated ~30–35 references
- All cataloged in `paper/references.bib`

---

## 9. Appendix (unlimited)

### A. Reproducibility
- Hyperparameters table, seed list
- Hardware specs, wall-clock time per training run
- Software versions (poke-env, Showdown server, Python)

### B. Additional experiments / negative results
- Why K = 30 evaluation OOM'd and we capped at K = 10
- Failed strategies considered (e.g., median voting, top-1 vs top-K weighting)

### C. Mathematical detail
- Wilson confidence interval derivation
- Variance bounds on the K-member soft average

### D. All 8 raw eval JSONs (or table of WRs per condition)

---

## Layout for the front-loaded message

The paper's first page should answer four questions in the reader's first 30 seconds:

1. **What problem?** Tabular RL ceilings.
2. **What's new?** Read-only ensemble inference rule with heuristic-prior fallback.
3. **Does it work?** +12.6pp over compute-matched baseline (z = 62).
4. **Why should the reader care?** First Pokemon Q-ensemble paper; methodology generalizes to any partially-observable tabular RL setting.

Every other section serves one of those four answers.

---

## Page budget summary

| Section | Pages | Words (approx) |
|---|---:|---:|
| Abstract | 0.25 | 200 |
| 1. Introduction | 1.0 | 700 |
| 2. Related Work | 1.0 | 750 |
| 3. Background and Method | 1.5 | 1500 |
| 4. Experimental Setup | 0.5 | 400 |
| 5. Results | 2.0 | 2000 |
| 6. Discussion | 1.0 | 900 |
| 7. Conclusion | 0.33 | 250 |
| **Total content** | **~7.5** | **~6700** |
| References (unlimited) | — | — |
| Appendix (unlimited) | — | — |

Leaves 1–1.5 pages of margin under the 9-page NeurIPS limit.

---

## Order to write things in

1. **Method (Section 3)** first — defines the technical contribution; nothing else makes sense until this is solid.
2. **Results (Section 5)** second — the numbers are already in `eval_*.json`; this is mechanical writing.
3. **Introduction (Section 1)** third — once Method and Results exist, the contribution claims write themselves.
4. **Related Work (Section 2)** fourth — use the lit review from `conversation history`.
5. **Discussion (Section 6)** fifth.
6. **Experimental Setup (Section 4)** sixth — short, mostly numbers.
7. **Conclusion (Section 7)** seventh.
8. **Abstract** last (you cannot write a good abstract until everything else exists).
9. **Appendix** in parallel as you go.
