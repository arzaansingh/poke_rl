"""
V6 Paper Figures — publication-quality vector PDFs.

Distinct from analyze_eval.py (which produces PNG poster-style plots): this
script generates paper-grade figures intended for the NeurIPS-format paper:

  * Vector PDF output (scales cleanly at any zoom)
  * Times New Roman serif typography (matches LaTeX body font)
  * Column-width and full-width aspect ratios calibrated to NeurIPS layout
  * Minimal grid, thin axis lines, tight margins
  * Colorblind-safe palette (Wong 2011 / IBM colorblind set)
  * Wilson 95% CIs everywhere

Output: paper/figures/*.pdf (and matching .png at 600 DPI for previews)

Usage:
    python analyze_paper_figures.py --run-id 1

Figures produced:
  fig1_headline_comparison.pdf      — V6 ensemble vs baseline_5m vs V5 ref
  fig2_k_saturation.pdf             — WR vs K with CI band
  fig3_strategy_comparison.pdf      — soft / hard / confidence at K=10
  fig4_head_to_head.pdf             — ensemble wins / baseline wins split
  fig5_per_member_distribution.pdf  — violin plot of K=30 final solo WRs
  fig6_diagnostics.pdf              — disagreement + unseen-rate panels
  fig7_prior_work_comparison.pdf    — V6 in context of Wang, Metamon, etc.
"""

import argparse
import json
import math
import os
import sys
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from shared.config import ENSEMBLE_RESULTS_DIR


# ═══════════════════════════════════════════════════════════════════════
# Palette — dark Tulane + brushed gold (matches V5 poster aesthetic;
# avoids the saturated Pokémon yellow / bright blue look of generic
# matplotlib defaults). All colors are dark, muted, colorblind-friendly.
# ═══════════════════════════════════════════════════════════════════════

C_ENSEMBLE   = "#003D2B"   # Tulane DARK green   — V6 ensemble primary
C_BASELINE   = "#9C7C38"   # Antique brushed gold — compute-matched single agent
C_REFERENCE  = "#4A5568"   # Slate gray          — prior work, V5 reference
C_HARD       = "#7C2D2D"   # Deep burgundy red    — hard voting (Pokemon Red darkened)
C_SOFT       = "#003D2B"   # Tulane dark green    — soft voting
C_CONFIDENCE = "#1F3A5F"   # Pokemon Crystal navy — confidence voting
C_FILL       = "#D4DDD7"   # Very subtle green tint for CI bands
C_COIN_FLIP  = "#A0AEC0"   # Light cool gray — null reference
C_GOLD_FILL  = "#E8DBB8"   # Pale gold tint for highlight regions

V5_M8_REF_WR = 0.598


# ═══════════════════════════════════════════════════════════════════════
# Typography setup — match LaTeX body font
# ═══════════════════════════════════════════════════════════════════════

def setup_typography():
    """Times-family serif for all figure text. Matches NeurIPS body font."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,                       # base size — captioned at 9pt
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.5,
        "grid.alpha": 1.0,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,                   # editable text in PDF (TrueType)
        "ps.fonttype": 42,
    })


# NeurIPS column widths in inches: ~5.5" full text width, ~2.7" half width
COL_FULL = 5.5
COL_HALF = 2.7


# ═══════════════════════════════════════════════════════════════════════
# Stats helpers
# ═══════════════════════════════════════════════════════════════════════

def wilson_ci(wins: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (centre - half, centre + half)


def diff_z(wins_a, n_a, wins_b, n_b):
    p_a = wins_a / n_a
    p_b = wins_b / n_b
    se = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    return (p_a - p_b) / se if se else 0.0


# ═══════════════════════════════════════════════════════════════════════
# Loaders
# ═══════════════════════════════════════════════════════════════════════

def load_evals(run_dir):
    evals = {}
    for path in sorted(glob(os.path.join(run_dir, "eval_*.json"))):
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            continue
        if "id" not in d:
            continue
        evals[d["id"]] = d
    return evals


def save_figure(fig, basename, plots_dir):
    pdf_path = os.path.join(plots_dir, basename + ".pdf")
    png_path = os.path.join(plots_dir, basename + ".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    print(f"  → {basename}.{{pdf,png}}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Figure 1 — headline comparison (V5 ref / baseline / ensemble)
# ═══════════════════════════════════════════════════════════════════════

def fig1_headline(evals, plots_dir):
    print("Figure 1: headline comparison")
    bsl = evals.get("baseline5m_solo_heuristic")
    ens_soft = evals.get("K10_soft_heuristic")
    ens_conf = evals.get("K10_confidence_heuristic")
    ens_best = max([e for e in (ens_soft, ens_conf) if e], key=lambda e: e["win_rate"])

    fig, ax = plt.subplots(figsize=(COL_FULL, 2.6))

    labels = [
        f"V5 M8\n(50K, ref)",
        f"V6 baseline\n(1 agent, 5M)",
        f"V6 ensemble\n(K=10, {ens_best['strategy']})",
    ]
    wrs = [V5_M8_REF_WR, bsl["win_rate"], ens_best["win_rate"]]
    cis = [None,
           wilson_ci(bsl["wins"], bsl["n_battles"]),
           wilson_ci(ens_best["wins"], ens_best["n_battles"])]
    colors = [C_REFERENCE, C_BASELINE, C_ENSEMBLE]

    yerr_lo = [0 if c is None else wr - c[0] for wr, c in zip(wrs, cis)]
    yerr_hi = [0 if c is None else c[1] - wr for wr, c in zip(wrs, cis)]

    x = np.arange(len(labels))
    bars = ax.bar(x, wrs, color=colors, edgecolor="#222", linewidth=0.5,
                  yerr=[yerr_lo, yerr_hi], capsize=4,
                  error_kw={"ecolor": "#222", "elinewidth": 0.8})

    # Numbers on bars
    for rect, wr in zip(bars, wrs):
        ax.text(rect.get_x() + rect.get_width() / 2, wr + 0.013,
                f"{wr:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#222")

    # Lift arrow + annotation
    delta = ens_best["win_rate"] - bsl["win_rate"]
    z = diff_z(ens_best["wins"], ens_best["n_battles"],
               bsl["wins"], bsl["n_battles"])
    x_b, x_e = 1, 2
    y_top = max(wrs) + 0.060
    ax.annotate("", xy=(x_e, y_top), xytext=(x_b, y_top),
                arrowprops=dict(arrowstyle="-|>", color=C_ENSEMBLE,
                                lw=1.2, mutation_scale=12))
    ax.text((x_b + x_e) / 2, y_top + 0.012,
            rf"$\Delta = +{delta * 100:.1f}$ pp  ($z = {z:.0f}$)",
            ha="center", va="bottom", fontsize=9, color=C_ENSEMBLE,
            fontweight="bold")

    # Coin flip reference
    ax.axhline(0.5, color=C_COIN_FLIP, linewidth=0.7, linestyle=":", zorder=0)
    ax.text(len(labels) - 0.45, 0.504, "0.5", fontsize=7, color="#888",
            va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Win rate vs SimpleHeuristicsPlayer")
    ax.set_ylim(0.40, max(wrs) + 0.13)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.6)

    save_figure(fig, "fig1_headline_comparison", plots_dir)


# ═══════════════════════════════════════════════════════════════════════
# Figure 2 — K-saturation curve
# ═══════════════════════════════════════════════════════════════════════

def fig2_k_saturation(evals, plots_dir):
    print("Figure 2: K-saturation curve")
    pairs = [(k, evals.get(f"K{k}_soft_heuristic")) for k in (1, 3, 5, 10)]
    pairs = [(k, e) for k, e in pairs if e]
    if not pairs:
        return

    ks = np.array([k for k, _ in pairs])
    wrs = np.array([e["win_rate"] for _, e in pairs])
    cis = [wilson_ci(e["wins"], e["n_battles"]) for _, e in pairs]
    los = np.array([c[0] for c in cis])
    his = np.array([c[1] for c in cis])

    fig, ax = plt.subplots(figsize=(COL_FULL, 2.6))

    ax.fill_between(ks, los, his, color=C_FILL, alpha=1.0, zorder=1,
                    label="95% Wilson CI")
    ax.plot(ks, wrs, color=C_ENSEMBLE, linewidth=1.6, marker="o",
            markersize=7, markerfacecolor="white",
            markeredgecolor=C_ENSEMBLE, markeredgewidth=1.4,
            label="Soft-voting ensemble", zorder=3)

    # Annotate each point
    for k, wr in zip(ks, wrs):
        ax.annotate(f"{wr:.3f}", xy=(k, wr), xytext=(0, 9),
                    textcoords="offset points",
                    ha="center", fontsize=8, fontweight="bold",
                    color="#222")

    bsl = evals.get("baseline5m_solo_heuristic")
    if bsl:
        ax.axhline(bsl["win_rate"], color=C_BASELINE, linewidth=1.0,
                   linestyle="--", alpha=0.85, zorder=2,
                   label=f"V6 baseline_5m  ({bsl['win_rate']:.3f})")
    ax.axhline(0.5, color=C_COIN_FLIP, linewidth=0.7, linestyle=":", zorder=0)
    ax.axhline(V5_M8_REF_WR, color=C_REFERENCE, linewidth=0.9,
               linestyle="-.", alpha=0.8, zorder=2,
               label=f"V5 M8 reference  ({V5_M8_REF_WR})")

    ax.set_xticks(ks)
    ax.set_xlabel("Ensemble size $K$")
    ax.set_ylabel("Win rate vs SimpleHeuristicsPlayer")
    ax.set_ylim(0.45, max(his.max() + 0.04, 0.72))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(linestyle="-", linewidth=0.4, alpha=0.6)
    ax.legend(loc="lower right", frameon=True, fancybox=False,
              edgecolor="#999", framealpha=0.95)

    save_figure(fig, "fig2_k_saturation", plots_dir)


# ═══════════════════════════════════════════════════════════════════════
# Figure 3 — voting strategy comparison
# ═══════════════════════════════════════════════════════════════════════

def fig3_strategy(evals, plots_dir):
    print("Figure 3: strategy comparison")
    items = [
        ("Hard",       evals.get("K10_hard_heuristic"),       C_HARD),
        ("Soft",       evals.get("K10_soft_heuristic"),       C_SOFT),
        ("Confidence", evals.get("K10_confidence_heuristic"), C_CONFIDENCE),
    ]
    items = [(n, e, c) for n, e, c in items if e]
    if not items:
        return

    names = [n for n, _, _ in items]
    wrs   = np.array([e["win_rate"] for _, e, _ in items])
    cis   = [wilson_ci(e["wins"], e["n_battles"]) for _, e, _ in items]
    err_lo = wrs - np.array([c[0] for c in cis])
    err_hi = np.array([c[1] for c in cis]) - wrs
    colors = [c for _, _, c in items]

    fig, ax = plt.subplots(figsize=(COL_HALF, 2.6))

    x = np.arange(len(items))
    bars = ax.bar(x, wrs, color=colors, edgecolor="#222", linewidth=0.5,
                  yerr=[err_lo, err_hi], capsize=4,
                  error_kw={"ecolor": "#222", "elinewidth": 0.8},
                  width=0.6)

    for rect, wr in zip(bars, wrs):
        ax.text(rect.get_x() + rect.get_width() / 2, wr + 0.010,
                f"{wr:.3f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#222")

    bsl = evals.get("baseline5m_solo_heuristic")
    if bsl:
        ax.axhline(bsl["win_rate"], color=C_BASELINE, linewidth=0.9,
                   linestyle="--", alpha=0.85,
                   label=f"baseline ({bsl['win_rate']:.3f})")
    ax.axhline(0.5, color=C_COIN_FLIP, linewidth=0.6, linestyle=":")

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Win rate (vs heuristic)")
    ax.set_ylim(0.45, max(c[1] for c in cis) + 0.04)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.6)
    ax.legend(loc="lower right", frameon=True, fancybox=False,
              edgecolor="#999", framealpha=0.95, fontsize=7)

    save_figure(fig, "fig3_strategy_comparison", plots_dir)


# ═══════════════════════════════════════════════════════════════════════
# Figure 4 — head-to-head split
# ═══════════════════════════════════════════════════════════════════════

def fig4_head_to_head(evals, plots_dir):
    print("Figure 4: head-to-head")
    h2h = evals.get("K10_soft_baseline5m")
    if not h2h:
        return

    wins = h2h["wins"]
    losses = h2h["n_battles"] - wins
    wr = wins / h2h["n_battles"]
    lo, hi = wilson_ci(wins, h2h["n_battles"])
    z = (wr - 0.5) / math.sqrt(0.5 * 0.5 / h2h["n_battles"])

    fig, ax = plt.subplots(figsize=(COL_FULL, 1.6))

    ax.barh([0], [wins], color=C_ENSEMBLE, edgecolor="#222", linewidth=0.5,
            label=f"Ensemble  ({wr:.3f})")
    ax.barh([0], [losses], left=[wins], color=C_BASELINE,
            edgecolor="#222", linewidth=0.5,
            label=f"baseline_5m  ({1 - wr:.3f})")

    half = h2h["n_battles"] / 2
    ax.axvline(half, color="#444", linewidth=0.7, linestyle=":")
    ax.text(half, 0.55, "  0.5", fontsize=7, color="#555", va="bottom")

    ax.text(wins / 2, 0, f"{wr:.1%}", ha="center", va="center",
            fontsize=14, fontweight="bold", color="white")
    ax.text(wins + losses / 2, 0, f"{1 - wr:.1%}",
            ha="center", va="center",
            fontsize=14, fontweight="bold", color="white")

    ax.set_yticks([])
    ax.set_xlim(0, h2h["n_battles"])
    ax.set_xlabel(f"Battles ($N$ = {h2h['n_battles']:,})")
    ax.set_title(f"K=10 soft-voting ensemble vs baseline_5m   "
                 rf"($z = {z:.0f}\sigma$, 95% CI [{lo:.4f}, {hi:.4f}])",
                 fontsize=9, pad=8)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -1.5),
              ncol=2, frameon=False)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x / 1000)}k" if x else "0"))

    save_figure(fig, "fig4_head_to_head", plots_dir)


# ═══════════════════════════════════════════════════════════════════════
# Figure 5 — per-member distribution (violin + strip overlay)
# ═══════════════════════════════════════════════════════════════════════

def fig5_per_member(plots_dir, run_dir):
    print("Figure 5: per-member distribution")
    import csv
    finals = []
    for k in range(1, 31):
        path = os.path.join(run_dir, f"member_{k}", "logs", "run_1.csv")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            continue
        last = rows[-1]
        try:
            finals.append(float(last[1]))
        except (ValueError, IndexError):
            continue
    if not finals:
        return

    finals = np.array(finals)
    fig, ax = plt.subplots(figsize=(COL_HALF, 2.6))

    # Violin
    parts = ax.violinplot([finals], showmeans=False, showmedians=False,
                          showextrema=False, widths=0.7)
    for body in parts["bodies"]:
        body.set_facecolor(C_FILL)
        body.set_edgecolor(C_ENSEMBLE)
        body.set_linewidth(1.0)
        body.set_alpha(1.0)

    # Strip overlay (jittered points)
    rng = np.random.default_rng(7)
    jitter = rng.normal(0, 0.04, size=len(finals))
    ax.scatter(1 + jitter, finals, s=14, color=C_ENSEMBLE,
               edgecolor="white", linewidth=0.6, zorder=4, alpha=0.85)

    # Mean horizontal line
    ax.hlines(finals.mean(), 0.7, 1.3, color=C_BASELINE,
              linewidth=1.2, linestyle="--",
              label=f"mean = {finals.mean():.3f}")

    # Coin flip
    ax.axhline(0.5, color=C_COIN_FLIP, linewidth=0.6, linestyle=":")

    ax.set_xticks([1])
    ax.set_xticklabels([f"K=30 members"])
    ax.set_ylabel("Final rolling WR (100-battle window)")
    ax.set_ylim(0.40, 0.75)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fancybox=False,
              edgecolor="#999", framealpha=0.95, fontsize=7)

    save_figure(fig, "fig5_per_member_distribution", plots_dir)


# ═══════════════════════════════════════════════════════════════════════
# Figure 6 — diagnostics (disagreement + unseen-rate)
# ═══════════════════════════════════════════════════════════════════════

def fig6_diagnostics(evals, plots_dir):
    print("Figure 6: diagnostics")
    has_diag = [(eid, e) for eid, e in evals.items() if "diagnostics" in e]
    if not has_diag:
        print("  (no diagnostics in evals — skip)")
        return

    # Sort by name for clean ordering
    has_diag.sort(key=lambda x: x[0])
    names = [e[1]["name"][:25] for e in has_diag]
    dis_mean = [e[1]["diagnostics"].get("disagreement", {}).get("mean", None)
                for e in has_diag]
    uns_mean = [e[1]["diagnostics"].get("unseen_rate", {}).get("mean", None)
                for e in has_diag]
    dis_min  = [e[1]["diagnostics"].get("disagreement", {}).get("min", None)
                for e in has_diag]
    dis_max  = [e[1]["diagnostics"].get("disagreement", {}).get("max", None)
                for e in has_diag]
    uns_min  = [e[1]["diagnostics"].get("unseen_rate", {}).get("min", None)
                for e in has_diag]
    uns_max  = [e[1]["diagnostics"].get("unseen_rate", {}).get("max", None)
                for e in has_diag]

    fig, axes = plt.subplots(1, 2, figsize=(COL_FULL, 2.8))
    x = np.arange(len(names))

    # ── Disagreement ──
    ax = axes[0]
    valid = [(i, m, lo, hi) for i, (m, lo, hi) in
             enumerate(zip(dis_mean, dis_min, dis_max)) if m is not None]
    if valid:
        xs = [v[0] for v in valid]
        ms = [v[1] for v in valid]
        los = [v[2] for v in valid]
        his = [v[3] for v in valid]
        ax.errorbar(xs, ms,
                    yerr=[np.array(ms) - np.array(los),
                          np.array(his) - np.array(ms)],
                    fmt="o", color=C_ENSEMBLE, markersize=6,
                    markerfacecolor="white", markeredgewidth=1.2,
                    elinewidth=0.8, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Pairwise argmax disagreement")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.6)
    ax.set_title("Disagreement fraction", fontsize=9, pad=6)

    # ── Unseen rate ──
    ax = axes[1]
    valid = [(i, m, lo, hi) for i, (m, lo, hi) in
             enumerate(zip(uns_mean, uns_min, uns_max)) if m is not None]
    if valid:
        xs = [v[0] for v in valid]
        ms = [v[1] for v in valid]
        los = [v[2] for v in valid]
        his = [v[3] for v in valid]
        ax.errorbar(xs, ms,
                    yerr=[np.array(ms) - np.array(los),
                          np.array(his) - np.array(ms)],
                    fmt="o", color=C_BASELINE, markersize=6,
                    markerfacecolor="white", markeredgewidth=1.2,
                    elinewidth=0.8, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Unseen-state fallback fraction")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.6)
    ax.set_title("Heuristic-fallback rate", fontsize=9, pad=6)

    fig.tight_layout()
    save_figure(fig, "fig6_diagnostics", plots_dir)


# ═══════════════════════════════════════════════════════════════════════
# Figure 7 — V6 in context of prior work
# ═══════════════════════════════════════════════════════════════════════

def fig7_prior_work(evals, plots_dir):
    """V6 ensemble placed in context of published Pokemon-RL win rates
    against SimpleHeuristicsPlayer or equivalent baseline. Numbers from
    our literature review.
    """
    print("Figure 7: prior work comparison")
    ens = evals.get("K10_soft_heuristic")
    if not ens:
        return

    # (method, format, WR, N, color, marker)
    # Numbers from our literature review (see paper/OUTLINE.md §2)
    rows = [
        ("PokeLLMon\n(GPT-4)",        "gen8 random",  0.260, 100,    C_REFERENCE, "o"),
        ("Lee & Togelius\n(MLP-Q vs Random)", "custom random", 0.633, 90, C_REFERENCE, "o"),
        ("Kalose et al.\n(softmax-Q vs Random)", "custom Gen 1",  0.700, 20000, C_REFERENCE, "o"),
        ("V6 baseline\n(1 agent, 5M)", "gen4 OU",      evals["baseline5m_solo_heuristic"]["win_rate"],
            evals["baseline5m_solo_heuristic"]["n_battles"], C_BASELINE, "s"),
        ("V6 ensemble\n(K=10)",        "gen4 OU",      ens["win_rate"],
            ens["n_battles"], C_ENSEMBLE, "*"),
        ("Metamon\n(transformer, OL)", "gen1-4 OU",    0.72, 2000, "#666666", "D"),
        ("Wang (MIT)\n(PPO+MCTS)",     "gen4 random",  0.880, None, "#444444", "D"),
    ]

    fig, ax = plt.subplots(figsize=(COL_FULL, 2.8))

    for i, (name, fmt, wr, n, color, marker) in enumerate(rows):
        ci_str = ""
        if n:
            lo, hi = wilson_ci(int(round(wr * n)), n)
            ax.errorbar([i], [wr],
                        yerr=[[wr - lo], [hi - wr]],
                        fmt=marker, color=color, markersize=11,
                        markerfacecolor=color, markeredgecolor="white",
                        markeredgewidth=1.0, elinewidth=0.8, capsize=4)
        else:
            ax.scatter([i], [wr], color=color, marker=marker, s=120,
                       edgecolor="white", linewidth=1.0)
        ax.text(i, wr + 0.03, f"{wr:.2f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#222")
        ax.text(i, 0.20, fmt, ha="center", va="bottom",
                fontsize=7, color="#666", style="italic")

    ax.axhline(0.5, color=C_COIN_FLIP, linewidth=0.7, linestyle=":")
    ax.text(len(rows) - 0.4, 0.51, "coin flip", fontsize=7, color="#888",
            va="bottom")

    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], fontsize=7)
    ax.set_ylabel("Win rate vs heuristic baseline")
    ax.set_ylim(0.15, 1.00)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.6)
    ax.set_title("Pokemon-RL results vs heuristic baseline  (format in italics)",
                 fontsize=9, pad=6)

    save_figure(fig, "fig7_prior_work_comparison", plots_dir)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="V6 paper-grade figures (vector PDF + PNG preview)")
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--out", type=str, default=None,
                        help="Output dir (default: paper/figures/)")
    args = parser.parse_args()

    run_dir = os.path.join(ENSEMBLE_RESULTS_DIR, f"run_{args.run_id}")
    plots_dir = args.out or os.path.join(_THIS_DIR, "paper", "figures")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Generating V6 paper figures → {plots_dir}\n")
    setup_typography()
    evals = load_evals(run_dir)
    print(f"Loaded {len(evals)} eval JSONs\n")

    fig1_headline(evals, plots_dir)
    fig2_k_saturation(evals, plots_dir)
    fig3_strategy(evals, plots_dir)
    fig4_head_to_head(evals, plots_dir)
    fig5_per_member(plots_dir, run_dir)
    fig6_diagnostics(evals, plots_dir)
    fig7_prior_work(evals, plots_dir)

    print(f"\nAll figures written to: {plots_dir}\n")


if __name__ == "__main__":
    main()
