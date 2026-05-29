# V6 Paper Workspace

Working directory for the V6 ensemble Q-learning paper. Open in VS Code with the LaTeX Workshop extension to build.

## Read these first (in order)

1. [`PRE_MEETING.md`](PRE_MEETING.md) — talking points + question list for the Friday meeting with Dr. Zheng. **Skim this before 8 AM CDT.**
2. [`OUTLINE.md`](OUTLINE.md) — section-by-section plan with word/page budgets.
3. [`GUIDELINES.md`](GUIDELINES.md) — writing rules + figure thematic guide. Re-read before each writing session.
4. [`FIGURES.md`](FIGURES.md) — figure inventory: 8 already produced, 4 to make.
5. [`SUBMISSION_PLAN.md`](SUBMISSION_PLAN.md) — venue tier list + deadlines.

## Build instructions

### Quickest path: the build script

```bash
cd paper
./build.sh            # auto-downloads neurips_2024.sty if missing, builds main.pdf
./build.sh clean      # remove build artifacts
./build.sh figures    # regenerate figures + build
```

The script auto-downloads `neurips_2024.sty` from a GitHub mirror if it is not already present.

### In VS Code

With the **LaTeX Workshop** extension installed, open `main.tex` and hit `Ctrl+Alt+B` (or `Cmd+Alt+B`). Default recipe runs pdflatex → bibtex → pdflatex → pdflatex. The first build also needs `neurips_2024.sty`; run `./build.sh` once to auto-download it, then VS Code builds will pick it up.

### From the command line directly

```bash
cd paper
latexmk -pdf main
```

Output: `main.pdf` (currently 8 pages, all 7 figures embedded, 0 warnings).

## File map

```
paper/
├── README.md                # ← you are here
├── PRE_MEETING.md           # Friday meeting prep (urgent)
├── OUTLINE.md               # Section-by-section plan
├── GUIDELINES.md            # Writing rules
├── FIGURES.md               # Figure inventory
├── SUBMISSION_PLAN.md       # Venue + deadline plan
├── main.tex                 # Top-level LaTeX with \input{...} for sections
├── references.bib           # All cited papers (29 entries from the lit review)
├── sections/
│   ├── abstract.tex         # ← write last
│   ├── introduction.tex     # ← write third
│   ├── related_work.tex     # ← write fourth
│   ├── method.tex           # ← write first (most important)
│   ├── experimental_setup.tex
│   ├── results.tex          # ← write second
│   ├── discussion.tex
│   ├── conclusion.tex
│   └── appendix.tex
└── figures/                 # 8 figures already at 300 DPI; 4 TODO
    ├── headline_comparison.png
    ├── k_saturation_curve.png
    ├── strategy_comparison.png
    ├── head_to_head.png
    ├── diagnostics.png
    ├── effect_lift_table.png
    ├── per_member_curves.png
    └── per_member_final_bar.png
```

## Order of operations (recommended)

1. **Friday meeting** with Dr. Zheng — lock contribution framing, format (4-pg vs 9-pg), target venue.
2. **Method section** first — defines the contribution; nothing else makes sense until this is solid.
3. **Results section** second — numbers already exist; mostly mechanical writing from `eval_*.json`.
4. **Introduction** third — easier once Method and Results exist.
5. **Related Work** fourth — pull straight from the lit review in the conversation history.
6. **Discussion** + **Conclusion** + **Abstract** + **Appendix** last.

Estimated drafting time: ~3 weeks of focused work to v1 arXiv submission.
