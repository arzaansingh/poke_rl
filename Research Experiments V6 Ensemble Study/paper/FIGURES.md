# Figures Plan — V6 Paper

> Paper figures live in `paper/figures/`. **Two scripts produce figures:**
>
> - `analyze_paper_figures.py` (preferred for paper) — vector PDFs with Times serif typography matching LaTeX body. Output: `fig1_*.pdf` through `fig7_*.pdf`. Run from repo root: `python analyze_paper_figures.py`.
> - `analyze_eval.py` (informal / poster) — 300 DPI PNGs with sans-serif. Older filenames (`headline_comparison.png`, etc.). Useful for slides and quick checks, not for the paper.

---

## Paper figures (all 7 produced, in `paper/figures/`, vector PDFs)

| # | Figure file | What it shows | Section |
|---|---|---|---|
| 1 | `fig1_headline_comparison.pdf` | V5 M8 ref / V6 baseline_5m / V6 ensemble bars with +12.6pp lift annotation | §5.1 |
| 2 | `fig2_k_saturation.pdf` | WR vs K ∈ {1, 3, 5, 10} with Wilson 95% CI band | §5.2 |
| 3 | `fig3_strategy_comparison.pdf` | Hard / Soft / Confidence bars at K=10 with CIs | §5.3 |
| 4 | `fig4_head_to_head.pdf` | Stacked horizontal bar: ensemble wins vs baseline_5m wins | §5.1 |
| 5 | `fig5_per_member_distribution.pdf` | Violin + strip plot of K=30 final solo WRs | §5.4 |
| 6 | `fig6_diagnostics.pdf` | Pairwise disagreement + unseen-state rate panels | §5.4 |
| 7 | `fig7_prior_work_comparison.pdf` | V6 in context of Wang 2024, Metamon, PokeLLMon, prior tabular work | §5.5 |

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
