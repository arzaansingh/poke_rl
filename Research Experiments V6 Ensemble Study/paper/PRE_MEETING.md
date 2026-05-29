# Pre-Meeting Prep — Friday 8 AM CDT with Dr. Zheng

> **Goal of the meeting:** lock down (a) format (short 4-pg vs long 9-pg NeurIPS), (b) the precise technical contribution beyond empirical results, and (c) a writing/submission timeline.

---

## 1. The single most important thing Dr. Zheng has said

From his email on **May 18, 2026**:

> "You already have some strong results, and the next step is to identify the **main technical contributions beyond the empirical findings**."

And earlier on **March 27, 2026**:

> "As neither hierarchical RL nor reward shaping is new, the evaluation alone cannot be considered a technical contribution."

He has flagged this twice. Be ready to give him a crisp answer on **what is methodologically new** in this paper, not just what the win-rate numbers are.

---

## 2. Three candidate technical-contribution framings (rank-ordered, pick one before Friday)

### A. Read-only, memory-bounded ensemble inference with heuristic-prior fallback *(strongest framing)*

**Claim:** We propose an inference-time ensemble combination rule for tabular Q-learning that:
- Reads K frozen Q-tables in parallel without ever mutating them
- Falls back to a deterministic domain prior when a state is unseen by member k
- Maintains a constant memory footprint independent of evaluation length
- Permits 100,000-battle evaluations that prior tabular ensemble work could not reach

**Why this is a contribution:** Lin et al. (ICLR 2024, "Curse of Diversity") explicitly identified the failure mode of shared-buffer ensembles. We sidestep it by construction, AND we prove the inference is correct under the heuristic-prior fallback. The naïve implementation (used in V5 training-time inference) caches priors back into the Q-table and OOMs at ~30k battles in our setting. Fixing this is a real technical move.

### B. Three-rule voting comparison + the K-saturation curve in a stochastic two-player game

**Claim:** First published comparison of soft / hard / confidence-weighted voting in tabular Q-ensembles at K ∈ {1, 3, 5, 10} on a non-toy domain.

**Why this is a contribution:** Wiering & van Hasselt (2008) compared majority vs. Boltzmann across *algorithms*, not seeds. SUNRISE (ICML 2021) uses UCB only in deep RL. No prior work fills the (tabular × seed-only × stochastic adversarial game) cell.

**Honest assessment:** This is more empirical than methodological. Dr. Zheng may push back: "the comparison itself is not new — comparing voting rules has been done." Frame it as "the first such comparison **in this regime**" if you go this route.

### C. Hierarchical-decomposition + ensembling interaction

**Claim:** The hierarchical action decomposition from V5 (master Q for move-vs-switch, sub-agent Q for which Pokémon to switch to) composes non-trivially with the ensemble layer — each agent's master *and* sub-agent vote, doubling the effective ensemble width.

**Why this is a contribution:** No prior work studies the interaction of hierarchical action decomposition with ensembling. This is a real architectural composition.

**Honest assessment:** Slightly thin — the composition is mechanically simple. May not survive Dr. Zheng's "is hierarchical RL new?" lens from his March 27 email.

### Recommendation: **Lead with (A), back it up with empirical results from (B), mention (C) in the architecture section.**

---

## 3. Specific questions to ask Dr. Zheng on Friday

1. **Format: 4-page (workshop short) vs 9-page (NeurIPS long).** Your draft email from May 21 already asked this. Given his "submit to arXiv first" framing, my read: **the 9-page long format gives you room to be thorough**, and short workshop versions can be derived later. Get his explicit recommendation.

2. **Which of the three contribution framings does he prefer?** Lay out A/B/C above, ask which one he wants you to lead with.

3. **Should the paper include a Deep RL comparison (DQN/DDQN/PPO baseline)?** Critically important. He said in May 14 reply you mentioned "studying whether agents with different architecture (DQN, DDQN, PPO, etc.) can improve results further." He may want this *before* publishing, or he may say "publish what you have, leave deep-RL extension as future work." Pin him down.

4. **Target venue (rank order):**
   - arXiv preprint (his explicit recommendation — first stop)
   - NeurIPS 2026 RLVG workshop (he linked the 2025 version himself in March 5 email; deadline likely August–September 2026)
   - RLC 2026 (deadline January 2026; missed for this year, target 2027)
   - AAAI 2027 (deadline August 2026)
   - IEEE Conference on Games 2027 (Jan/Feb 2027 deadline)

   Ask him which workshops/conferences he has reviewed for or has connections at.

5. **Co-authorship + acknowledgments.** Will he be a co-author? Standard for senior research projects — confirm so the title page is right.

6. **Self-imposed timeline.** Propose: arXiv preprint by **end of July 2026**, then re-target a workshop for fall.

7. **Statistical-significance angle.** He flagged on March 27 that 10–20 runs are needed for significance. We have 30 trained members + 100,000 eval battles per condition, giving CI of ±0.3%. Lead the conversation with this — it directly answers his earlier concern.

---

## 4. Things to *not* say in the meeting

- "I have something publishable" — this is for him to decide, not you. Frame as "I would like your judgment on whether these results are publishable, and what the right venue is."
- Anything about competing publishability or claiming firsts before confirming with him.
- Don't downplay the work either — the V6 results genuinely are unusual (no prior Pokemon Q-ensemble paper exists per our lit review).

---

## 5. What to bring to the meeting

1. **This document** (skim before the call).
2. **The 6 eval plots** from `ensemble_results/run_1/eval_plots/` — the headline_comparison.png and k_saturation_curve.png especially.
3. **A clean copy of the key numbers**:
   - K=10 SOFT vs heuristic: **66.8% ± 0.3% over 100,000 battles**
   - baseline_5m solo: **54.2% ± 0.3%**
   - Δ = **+12.6 pp, z = 62σ, p < 0.001**
   - Head-to-head (K=10 vs baseline): 59.7%
4. **The Wang 2024 MIT thesis comparison number** (85% in gen4 *random*battle — note the format difference).
5. **The Metamon comparison band** (64–80% in gen1-4 OU composites, Grigsby et al. 2025) — V6 sits inside this band at 1/100 the compute.

---

## 6. Status of the paper materials (what's done, what's left)

| Asset | Status | Location |
|---|---|---|
| Lit review (general ensemble RL) | Done | conversation history |
| Lit review (Pokemon RL) | Done | conversation history |
| Win-rate comparison table | Done | conversation history |
| 30 trained Q-tables, 1M battles each | Done | workstation, synced |
| baseline_5m model, 5M battles | Done | workstation, synced |
| 8 eval JSONs, 100k battles each | Done | `ensemble_results/run_1/` |
| 6 paper-grade plots | Done | `ensemble_results/run_1/eval_plots/` |
| Codebase (run_eval_suite.py + analyze_eval.py) | Done | GitHub |
| Paper outline (this folder) | Done | `paper/OUTLINE.md` |
| Paper guidelines | Done | `paper/GUIDELINES.md` |
| Submission venue plan | Done | `paper/SUBMISSION_PLAN.md` |
| `main.tex` NeurIPS skeleton | Done | `paper/main.tex` |
| Section .tex stubs | Done | `paper/sections/` |
| `references.bib` starter | Done | `paper/references.bib` |
| Architecture diagram | TODO | post-meeting |
| Algorithm pseudocode (LaTeX algorithm block) | TODO | post-meeting |
| Actual prose for each section | TODO | post-meeting |

---

## 7. One-paragraph "elevator pitch" to open the meeting

> "Professor, the V6 ensemble results came out very strong — K=10 ensemble at 66.8% vs SimpleHeuristicsPlayer over 100k battles, +12.6 percentage points over a compute-matched single-agent baseline at z = 62σ. From the literature search I did, no prior paper studies Q-function ensembles on Pokemon, and there's no published comparison of soft/hard/confidence voting in tabular Q at this scale. I've drafted a paper outline that frames the contribution as a read-only memory-bounded ensemble inference rule with heuristic-prior fallback, evaluated empirically. I'd like your guidance on whether that framing is right, whether to write the 4-page or 9-page version, and whether to add a deep-RL baseline before submitting."

That's 30 seconds. Then let him drive.
