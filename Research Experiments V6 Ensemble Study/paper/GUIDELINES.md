# Writing Guidelines — V6 Paper

> **Tone target:** mathematically rigorous, plain English, concise, scientific but engaging. Match the register of recent NeurIPS/ICML papers (e.g., REDQ, SUNRISE, EBQL). No marketing language.

These rules apply to every sentence in the paper. Re-read them before each writing session.

---

## 1. Voice

- **Use first-person plural** ("we train K = 30 agents", "we evaluate over 100k battles") unless writing in the abstract, where you can use either passive or first-person.
- **Past tense for what was done** ("we trained"), **present tense for what holds** ("the ensemble outperforms"), **future tense never** unless discussing future work.
- **No em dashes.** Use commas, colons, parentheses, or semicolons. (User preference, carried over from grant writing.)
- **No hedging adverbs** like "very," "quite," "really," "extremely." If you mean "substantial," write "substantial" or give the number.
- **No marketing language.** Never use "novel," "groundbreaking," "state-of-the-art" without specifying *what* prior bar is being cleared and *how much* it is exceeded. Reviewers actively look for and downrate this.

---

## 2. Structure

- **One claim per sentence.** Long sentences with multiple subordinate clauses tend to lose reviewers.
- **One idea per paragraph.** Topic sentence → evidence → consequence.
- **Use the inverted pyramid in each section:** state the conclusion first, then justify it. The reader skims; do not make them work.
- **Forward references over backward references.** "We will show in Section 5 that …" is fine. "As shown above" is fine but use sparingly.

---

## 3. Mathematical rigor

- **Define every symbol the first time it appears.** No exceptions.
- **State assumptions explicitly.** "We assume the opponent's policy is fixed during evaluation" is better than "we evaluate against a fixed opponent."
- **State the regime / scale.** "K = 10" is not enough — also "trained for 1M episodes each, in a state space of ~10¹² distinct states." Scale tells the reader whether the result is impressive.
- **Use equations for the things that are best read as equations.** Reserve prose for things best read as prose. Do not write equations in prose form or vice versa.
- **Equations should be parsable on first read.** Aliasing intermediate quantities is fine: `w_k := max(Q^k - mean(Q^k), 0)` then use `w_k` afterward.
- **Wilson CIs for proportions.** Never use normal-approximation CIs in this paper — they are wrong near 0.5 and definitely wrong for win rates. Use Wilson everywhere; cite the definition once in the Appendix.
- **Cite every borrowed concept.** "Watkins Q(λ) (Watkins 1989)", "eligibility traces (Sutton & Barto 2018)", etc.

---

## 4. What to claim and what NOT to claim

| Claim safely | Avoid |
|---|---|
| "To our knowledge, no prior paper studies Q-function ensembles on Pokemon" | "We are the first to do X" (without qualification — reviewers always find counterexamples) |
| "The +12.6 pp lift is statistically significant at p < 0.001 (Wilson 95% CI for the difference excludes zero by 62σ)" | "Our method significantly improves performance" (vague) |
| "Performance saturates near K = 10, consistent with Osband et al. 2016's findings" | "K = 10 is optimal" |
| "Voting strategy choice has limited effect (all within 2 pp of each other in our setting)" | "Voting strategy does not matter" |
| "Our method's WR of 66.8% in gen4OU is inside the band reported by Grigsby et al. 2025 for early-generation OU formats (64–80%)" | "Our method is competitive with the state-of-the-art" |

---

## 5. Figures and tables

### Universal rules
- **Every figure must be readable as a thumbnail.** If you can't tell what story it's telling at 30% size, it's too busy.
- **Every figure caption begins with a 1-sentence summary of what the figure shows.** Then provide details. Reviewers read captions first; they may not read the figure body.
- **All figures use the same color palette and font** (see Section 6 below). Inconsistent figure colors are a code smell that reviewers notice.
- **Use confidence intervals on every bar chart with stochastic measurements.** Wilson CIs computed from the underlying battle counts.
- **Always include the N (number of battles or samples) on the figure or in the caption.**
- **Vectorize where possible** (.pdf or .svg, not .png). Matplotlib saves vector PDFs via `savefig(..., format='pdf')`.

### Color palette (consistent with V5 poster)

| Use | Hex | Notes |
|---|---|---|
| Primary (ensemble / V6 main) | `#006747` (Tulane green) | Use for the headline result, the K=10 SOFT line, etc. |
| Secondary (baseline / control) | `#418FDE` (Tulane blue) | Use for baseline_5m, the compute-matched control |
| Reference / historical | `#95a5a6` (gray) | Use for V5 reference lines, prior work in comparison tables |
| Accent — hard voting | `#e67e22` (orange) | Use only in the strategy_comparison figure |
| Accent — confidence voting | `#43B02A` (Tulane kelly) | Use only in the strategy_comparison figure |
| Coin flip / null | `#bdc3c7` (light gray) | Use for 50% reference lines |

### Font
- All figure text in `sans-serif` (matplotlib default DejaVu Sans is fine).
- Title 14pt bold, axis labels 12pt, tick labels 11pt, legend 11pt.
- For LaTeX-rendered math in figure axes, use `\usepackage{amsmath}` in matplotlib's rcParams.

### Tables
- Use `booktabs` (`\toprule`, `\midrule`, `\bottomrule`) for NeurIPS-style horizontal-only rules. **Never** use vertical rules.
- Right-align numbers, left-align text. Use the `S` column type from `siunitx` for decimal-aligned numbers.
- Always include CIs in tables of win rates: `66.8 [66.5, 67.1]`.

---

## 6. LaTeX-specific conventions

- **Use `\citet{key}` for prose citations** (e.g., "Lin et al. (2024) showed …") and `\citep{key}` for parenthetical citations.
- **Bibliography is BibTeX in `references.bib`.** All keys lowercase, format `lastnameYEAR` (e.g., `osband2016`, `lin2024curse`).
- **One sentence per line in source code.** Easier to diff, easier to comment on, easier for `git blame`. Wrap long sentences after natural break points.
- **No hard wrapping in the middle of equations.**
- **All section labels prefixed by `sec:`** (`\label{sec:method}`), all figures by `fig:` (`\label{fig:headline}`), tables by `tab:`, equations by `eq:`.
- **Cross-reference labels never include the section number** — let `\ref{fig:headline}` produce "Figure 3" or whatever it renders to.

---

## 7. Things to actively flag in early drafts

These are the kinds of issues that need fixing before submission:

- "Significant" without a p-value or CI.
- "Outperforms" without a number for the gap.
- "Recent work has shown" without specific citations.
- Any sentence longer than 35 words.
- Any paragraph longer than 8 sentences.
- Any equation with undefined symbols.
- Any number reported without units (battles, percentage points, etc.) or without CI.
- Any color in a figure that isn't on the palette above.
- Any em dash anywhere in the body, including captions.

---

## 8. The 7-point reviewer's quick-rejection checklist (avoid all of these)

1. Method section is hand-wavy — no equations defining the proposed algorithm.
2. Results lack confidence intervals or only report a single seed.
3. No comparison to a strong baseline (in our case: the compute-matched baseline_5m is the answer).
4. Related work is missing 2–3 obviously-relevant recent papers.
5. Figures are PNG screenshots from a notebook.
6. Abstract claims more than the paper delivers.
7. Code/data not released or not reproducible.

We hit none of these if we follow the OUTLINE and reuse the existing eval JSONs + plots.

---

## 9. Word budget enforcement

Pick a paragraph. Aim for 100 words for prose, 50 for math-heavy paragraphs. If you cannot make the point in 100 words, the argument is unclear; rewrite, do not pad.

---

## 10. The single rule that matters most

> **A reviewer cannot accept a paper they cannot understand. The clarity of the writing is the single biggest determinant of whether the result is publishable. Optimize for clarity at the cost of cleverness, every time.**
