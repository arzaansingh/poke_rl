# Figures Plan — V6 Paper

> All figures live in `paper/figures/`. Every figure uses the unified color palette and font conventions from `GUIDELINES.md` (Section 5).
>
> Re-generate via `python analyze_eval.py --run-id 1 --dpi 300` from the repository root.

---

## Figures already produced (300 DPI PNG)

| # | File | What it shows | Section | Status |
|---|---|---|---|---|
| 1 | `headline_comparison.png` | V5 M8 ref / V6 baseline_5m / V6 ensemble bars with +12.6pp lift annotation | §5.1 (Headline) | ✅ Ready |
| 2 | `k_saturation_curve.png` | WR vs K with Wilson 95% CI bands; baseline_5m + V5 M8 reference lines | §5.2 (K-saturation) | ✅ Ready |
| 3 | `strategy_comparison.png` | Soft / Hard / Confidence bars at K=10 with CIs | §5.3 (Voting) | ✅ Ready |
| 4 | `head_to_head.png` | Stacked horizontal bar of ensemble vs baseline_5m wins (100k battles) | §5.1 (Headline) | ✅ Ready |
| 5 | `diagnostics.png` | Pairwise disagreement + unseen-state rate panels | §5.4 (Diagnostics) | ✅ Ready |
| 6 | `effect_lift_table.png` | Publication-quality table of all 8 conditions with Wilson CIs | §5 + Appendix | ✅ Ready (also rendered as LaTeX table in paper body) |
| 7 | `per_member_curves.png` | 30 thin training trajectories + mean ± std band | Appendix | ✅ Ready |
| 8 | `per_member_final_bar.png` | Bar chart of each member's final solo WR | §5.4 (Diagnostics) | ✅ Ready |

## Figures to produce (TODO before submission)

| # | Working name | What to show | Section | Notes |
|---|---|---|---|---|
| 9 | `architecture_diagram.pdf` | System block diagram: 30 trained members → load K → state → Q-matrix → voting rule → action | §3.2 (Method) | Vector graphic. TikZ recommended; SVG fallback. |
| 10 | `algorithm_pseudocode.tex` | Pseudocode of the read-only inference rule (LaTeX `algorithmic` block, not an image) | §3.2 (Method) | Already templated as a `\begin{algorithm}` block in `method.tex` |
| 11 | `comparison_table.tex` | Comparison table — V6 alongside Wang 2024, Metamon, PokeLLMon, Lee & Togelius, Kalose | §5.5 (Comparison) | Render as LaTeX `booktabs` table from data in conversation history's lit review |
| 12 | `q_matrix_example.pdf` | Worked example: 3×3 Q-matrix showing soft / hard / confidence picks | §3.2 (Method) | Optional — only if Method section needs pedagogical clarity |

---

## Figure caption template

Use this template for every figure caption:

> **Figure N.** [One-sentence headline summary in bold.] [Detail sentence with the specific numbers and Ns.] [Method sentence if relevant: "Win rates computed over 100,000 evaluation battles; error bars show Wilson 95% CIs."]

**Example for Figure 1:**

> **Figure 1.** *V6 ensemble outperforms compute-matched single-agent baseline by 12.6 percentage points.* The K=10 ensemble (soft voting) achieves 66.8% win rate over 100,000 evaluation battles against `SimpleHeuristicsPlayer`, compared to 54.2% for `baseline_5m`, a HierSmartPlayer trained for 5× longer on identical hardware. Error bars show Wilson 95% confidence intervals (±0.3 pp); the lift annotation gives the difference and its 95% CI.

---

## Color palette (also documented in GUIDELINES.md §5)

```
Tulane green   #006747   primary ensemble color
Tulane blue    #418FDE   baseline / control color
Gray           #95a5a6   prior-work / reference lines
Orange         #e67e22   hard voting
Kelly green    #43B02A   confidence voting
Light gray     #bdc3c7   coin-flip / null reference
```

Set in `analyze_eval.py`. Keep consistent if you add new figures.

---

## Vectorize before submission

PNG is fine for drafts. For the final submission to arXiv / a conference, regenerate all figures as PDF:

```bash
# In analyze_eval.py, change save call from .png to .pdf:
fig.savefig(out_path, dpi=300, format='pdf', bbox_inches='tight')
```

PDF figures scale cleanly, print at any zoom, and reviewers expect them.
