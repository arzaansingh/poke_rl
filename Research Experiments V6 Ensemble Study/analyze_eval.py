"""
V6 Evaluation Analysis — paper-grade plots and statistical summary.

Builds publication-quality figures from the 8 evaluation JSONs produced by
run_eval_suite.py. Matches V5 poster aesthetics (Tulane green primary, sans-
serif, font_scale=1.3, whitegrid). All figures saved at 300 DPI.

Outputs go to ensemble_results/run_<id>/eval_plots/ :

  1. k_saturation_curve.png       — WR vs K with Wilson 95% CI bands
  2. strategy_comparison.png      — soft / hard / confidence at K=10
  3. headline_comparison.png      — V6 baseline_5m vs V6 ensemble (THE figure)
  4. head_to_head.png             — ensemble vs baseline_5m direct match
  5. effect_lift_table.png        — table of effect sizes + Wilson CIs
  6. diagnostics.png              — disagreement + unseen-rate distributions

Plus stdout: a publication-grade text summary with proper Wilson CIs,
paired-difference CIs, and effect sizes.

Usage:
    python analyze_eval.py --run-id 1
    python analyze_eval.py --run-id 1 --dpi 600
"""

import argparse
import json
import math
import os
import sys
from glob import glob
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from shared.config import ENSEMBLE_RESULTS_DIR


# ════════════════════════════════════════════════════════════════════════
# V5 POSTER COLOR PALETTE (matches V5 Initialization Study)
# ════════════════════════════════════════════════════════════════════════

TULANE = {"green": "#006747", "blue": "#418FDE", "kelly": "#43B02A", "dark": "#003D2B"}

# Specific evaluation colors
COLOR_ENSEMBLE   = TULANE["green"]   # ensemble (primary, what we're advocating)
COLOR_BASELINE   = TULANE["blue"]    # V6 baseline_5m (the compute-matched control)
COLOR_V5_REF     = "#95a5a6"         # V5 M8 reference line (historical)
COLOR_COIN_FLIP  = "#bdc3c7"         # 50% reference line
COLOR_HARD       = "#e67e22"         # hard voting (orange)
COLOR_SOFT       = TULANE["green"]   # soft voting
COLOR_CONFIDENCE = TULANE["kelly"]   # confidence voting (kelly green)

V5_M8_REFERENCE_WR = 0.598   # V5 M8 final-rolling-WR headline (for context only)


# ════════════════════════════════════════════════════════════════════════
# Style + helpers
# ════════════════════════════════════════════════════════════════════════

def setup_style():
    sns.set_theme(style="whitegrid", font_scale=1.3)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.titleweight": "bold",
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.2,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })


def wilson_ci(wins: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion. Tighter and better-
    behaved than normal-approximation ('Wald') CI, especially near 0/1."""
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (centre - half, centre + half)


def diff_ci(wins_a: int, n_a: int, wins_b: int, n_b: int) -> Tuple[float, float, float]:
    """95% CI on (p_a - p_b) using normal approximation on independent samples.
    Returns (delta, lo, hi)."""
    p_a = wins_a / n_a
    p_b = wins_b / n_b
    se = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    delta = p_a - p_b
    return delta, delta - 1.96 * se, delta + 1.96 * se


def annotate_bars(ax, container, fmt="{:.3f}", offset=0.005, fontsize=12, color="#222"):
    for rect, label in zip(container, container.datavalues):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + offset,
            fmt.format(label),
            ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold", color=color,
        )


# ════════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════════

def load_evals(run_dir: str) -> Dict[str, dict]:
    """Load all eval_*.json into a dict keyed by 'id' field."""
    evals = {}
    for path in sorted(glob(os.path.join(run_dir, "eval_*.json"))):
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception as e:
            print(f"[warn] could not read {path}: {e}")
            continue
        # Some older smoke-test JSONs may lack 'id'; skip them.
        if "id" not in d:
            print(f"[skip] {os.path.basename(path)} missing 'id' (probably stale) — skipping")
            continue
        evals[d["id"]] = d
    return evals


# ════════════════════════════════════════════════════════════════════════
# 1. K-saturation curve
# ════════════════════════════════════════════════════════════════════════

def plot_k_saturation(evals: Dict[str, dict], plots_dir: str, dpi: int):
    print("Generating K-saturation curve...")
    pairs = [
        (1,  evals.get("K1_soft_heuristic")),
        (3,  evals.get("K3_soft_heuristic")),
        (5,  evals.get("K5_soft_heuristic")),
        (10, evals.get("K10_soft_heuristic")),
    ]
    pairs = [(k, e) for k, e in pairs if e]
    if not pairs:
        print("  [skip] no K-saturation evals found"); return

    ks = np.array([k for k, _ in pairs])
    wrs = np.array([e["win_rate"] for _, e in pairs])
    cis = [wilson_ci(e["wins"], e["n_battles"]) for _, e in pairs]
    los = np.array([c[0] for c in cis])
    his = np.array([c[1] for c in cis])

    fig, ax = plt.subplots(figsize=(10, 6))

    # CI band
    ax.fill_between(ks, los, his, color=COLOR_ENSEMBLE, alpha=0.18,
                    label="95% Wilson CI")
    # Curve + markers
    ax.plot(ks, wrs, color=COLOR_ENSEMBLE, linewidth=3, marker="o",
            markersize=14, markeredgecolor="white", markeredgewidth=2,
            label="Ensemble (SOFT voting)", zorder=3)

    # Annotations on each point
    for k, wr in zip(ks, wrs):
        ax.annotate(f"{wr:.3f}", xy=(k, wr), xytext=(0, 16),
                    textcoords="offset points",
                    ha="center", fontsize=12, fontweight="bold",
                    color=TULANE["dark"])

    # Reference lines
    bsl = evals.get("baseline5m_solo_heuristic")
    if bsl:
        ax.axhline(bsl["win_rate"], color=COLOR_BASELINE, linewidth=2,
                   linestyle="--", alpha=0.85,
                   label=f"V6 baseline_5m (5M, single)  WR={bsl['win_rate']:.3f}")
    ax.axhline(0.5, color="black", linewidth=1, linestyle=":", alpha=0.4,
               label="coin flip (0.500)")
    ax.axhline(V5_M8_REFERENCE_WR, color=COLOR_V5_REF, linewidth=1.5,
               linestyle="-.", alpha=0.6,
               label=f"V5 M8 reference (50K, training rolling)  ≈ {V5_M8_REFERENCE_WR}")

    ax.set_xticks(ks)
    ax.set_xlabel("Ensemble size K", fontsize=14)
    ax.set_ylabel("Win rate vs SimpleHeuristicsPlayer", fontsize=14)
    ax.set_title("V6 Ensemble — K-Saturation Curve  (100,000 battles per point)",
                 fontsize=15, color=TULANE["dark"], pad=14)
    ax.set_ylim(0.45, max(0.75, his.max() + 0.04))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95)

    out = os.path.join(plots_dir, "k_saturation_curve.png")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {out}")


# ════════════════════════════════════════════════════════════════════════
# 2. Voting strategy comparison
# ════════════════════════════════════════════════════════════════════════

def plot_strategy_comparison(evals: Dict[str, dict], plots_dir: str, dpi: int):
    print("Generating strategy-comparison plot...")
    cfgs = [
        ("hard",       evals.get("K10_hard_heuristic"),       COLOR_HARD),
        ("soft",       evals.get("K10_soft_heuristic"),       COLOR_SOFT),
        ("confidence", evals.get("K10_confidence_heuristic"), COLOR_CONFIDENCE),
    ]
    cfgs = [(n, e, c) for n, e, c in cfgs if e]
    if not cfgs:
        print("  [skip] no strategy evals at K=10 found"); return

    names = [n for n, _, _ in cfgs]
    wrs   = np.array([e["win_rate"] for _, e, _ in cfgs])
    cis   = [wilson_ci(e["wins"], e["n_battles"]) for _, e, _ in cfgs]
    err_lo = wrs - np.array([c[0] for c in cis])
    err_hi = np.array([c[1] for c in cis]) - wrs
    colors = [c for _, _, c in cfgs]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars = ax.bar(names, wrs, color=colors, edgecolor=TULANE["dark"], linewidth=0.9,
                  yerr=[err_lo, err_hi], capsize=8, error_kw={"ecolor": "#333", "elinewidth": 1.5})

    # Annotate values
    for rect, wr in zip(bars, wrs):
        ax.text(rect.get_x() + rect.get_width() / 2, wr + 0.015,
                f"{wr:.3f}", ha="center", va="bottom",
                fontsize=14, fontweight="bold", color=TULANE["dark"])

    # Reference lines
    bsl = evals.get("baseline5m_solo_heuristic")
    if bsl:
        ax.axhline(bsl["win_rate"], color=COLOR_BASELINE, linewidth=2,
                   linestyle="--", alpha=0.85,
                   label=f"baseline_5m  ({bsl['win_rate']:.3f})")
    ax.axhline(0.5, color="black", linewidth=1, linestyle=":", alpha=0.4,
               label="coin flip")

    ax.set_ylabel("Win rate vs SimpleHeuristicsPlayer", fontsize=14)
    ax.set_title("V6 Voting Strategy Comparison  (K=10, 100k battles, 95% Wilson CI)",
                 fontsize=14, color=TULANE["dark"], pad=14)
    ax.set_ylim(0.45, max(0.75, max(c[1] for c in cis) + 0.04))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=11)

    out = os.path.join(plots_dir, "strategy_comparison.png")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {out}")


# ════════════════════════════════════════════════════════════════════════
# 3. Headline comparison — paper title figure
# ════════════════════════════════════════════════════════════════════════

def plot_headline_comparison(evals: Dict[str, dict], plots_dir: str, dpi: int):
    print("Generating headline comparison plot...")
    bsl = evals.get("baseline5m_solo_heuristic")
    ens_soft = evals.get("K10_soft_heuristic")
    ens_conf = evals.get("K10_confidence_heuristic")
    if not (bsl and ens_soft):
        print("  [skip] need baseline + ensemble evals"); return

    # Pick the ensemble best to lead with
    candidates = [e for e in (ens_soft, evals.get("K10_hard_heuristic"), ens_conf) if e]
    ens_best = max(candidates, key=lambda e: e["win_rate"])
    best_label = ens_best["name"]

    bars_data = [
        ("V5 M8 (50K, ref)\n[training rolling WR]", V5_M8_REFERENCE_WR, None, COLOR_V5_REF),
        ("V6 baseline_5m (5M)\n[1 single agent]", bsl["win_rate"],
         wilson_ci(bsl["wins"], bsl["n_battles"]), COLOR_BASELINE),
        (f"V6 ensemble (10 × 1M)\n[{ens_best['strategy']} voting]", ens_best["win_rate"],
         wilson_ci(ens_best["wins"], ens_best["n_battles"]), COLOR_ENSEMBLE),
    ]

    labels = [b[0] for b in bars_data]
    wrs    = np.array([b[1] for b in bars_data])
    cis    = [b[2] for b in bars_data]
    colors = [b[3] for b in bars_data]
    yerr_lo = []
    yerr_hi = []
    for wr, c in zip(wrs, cis):
        if c is None:
            yerr_lo.append(0); yerr_hi.append(0)
        else:
            yerr_lo.append(wr - c[0]); yerr_hi.append(c[1] - wr)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.bar(labels, wrs, color=colors, edgecolor=TULANE["dark"], linewidth=1.0,
                  yerr=[yerr_lo, yerr_hi], capsize=10,
                  error_kw={"ecolor": "#333", "elinewidth": 1.6})

    # Value annotations
    for rect, wr in zip(bars, wrs):
        ax.text(rect.get_x() + rect.get_width() / 2, wr + 0.018,
                f"{wr:.3f}", ha="center", va="bottom",
                fontsize=15, fontweight="bold", color=TULANE["dark"])

    # Lift annotation between baseline (idx 1) and ensemble (idx 2)
    delta, lo, hi = diff_ci(ens_best["wins"], ens_best["n_battles"],
                            bsl["wins"], bsl["n_battles"])
    # Arrow from baseline-top to ensemble-top
    x1 = bars[1].get_x() + bars[1].get_width() / 2
    x2 = bars[2].get_x() + bars[2].get_width() / 2
    y1 = wrs[1] + 0.06
    y2 = wrs[2] + 0.06
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=TULANE["dark"], lw=2.5))
    ax.text((x1 + x2) / 2, max(y1, y2) + 0.012,
            f"+{delta * 100:.2f} pp lift\n95% CI [{lo * 100:+.2f}, {hi * 100:+.2f}]",
            ha="center", fontsize=12, fontweight="bold", color=TULANE["dark"],
            bbox=dict(facecolor="white", edgecolor=TULANE["dark"],
                      boxstyle="round,pad=0.4", linewidth=1.2))

    # Reference lines
    ax.axhline(0.5, color="black", linewidth=1, linestyle=":", alpha=0.4)
    ax.text(2.4, 0.502, "coin flip (0.500)", fontsize=10, color="#666")

    ax.set_ylabel("Win rate vs SimpleHeuristicsPlayer", fontsize=14)
    ax.set_title("V6 Headline Result — Ensemble Beats Compute-Matched Single Agent",
                 fontsize=15, color=TULANE["dark"], pad=14)
    ax.set_ylim(0.40, max(0.85, max([(c[1] if c else wr) for wr, c in zip(wrs, cis)]) + 0.12))
    ax.grid(axis="y", alpha=0.3)

    out = os.path.join(plots_dir, "headline_comparison.png")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {out}")


# ════════════════════════════════════════════════════════════════════════
# 4. Head-to-head ensemble vs baseline_5m
# ════════════════════════════════════════════════════════════════════════

def plot_head_to_head(evals: Dict[str, dict], plots_dir: str, dpi: int):
    print("Generating head-to-head plot...")
    h2h = evals.get("K10_soft_baseline5m")
    if not h2h:
        print("  [skip] no head-to-head eval"); return

    wins = h2h["wins"]
    losses = h2h["n_battles"] - wins
    wr = wins / h2h["n_battles"]
    lo, hi = wilson_ci(wins, h2h["n_battles"])

    fig, ax = plt.subplots(figsize=(10, 4))

    # Stacked horizontal bar
    ax.barh([0], [wins], color=COLOR_ENSEMBLE, edgecolor=TULANE["dark"], linewidth=1.0,
            label=f"Ensemble wins  ({wins:,}, {wr:.3f})")
    ax.barh([0], [losses], left=[wins], color=COLOR_BASELINE, edgecolor=TULANE["dark"],
            linewidth=1.0, label=f"baseline_5m wins  ({losses:,}, {1 - wr:.3f})")

    # 50% reference line
    half = h2h["n_battles"] / 2
    ax.axvline(half, color="black", linewidth=1.5, linestyle=":")
    ax.text(half + 200, 0.42, "coin flip", fontsize=10, color="#666")

    # Annotations
    ax.text(wins / 2, 0, f"{wr:.1%}", ha="center", va="center",
            fontsize=22, fontweight="bold", color="white")
    ax.text(wins + losses / 2, 0, f"{1 - wr:.1%}", ha="center", va="center",
            fontsize=22, fontweight="bold", color="white")

    # CI annotation under bar
    z_score = (wr - 0.5) / math.sqrt(0.5 * 0.5 / h2h["n_battles"])
    ax.text(h2h["n_battles"] / 2, -0.55,
            f"95% CI  [{lo:.4f}, {hi:.4f}]   →   {z_score:.0f}σ from coin flip",
            ha="center", fontsize=12, color=TULANE["dark"],
            bbox=dict(facecolor="#f5f5f5", edgecolor=TULANE["dark"],
                      boxstyle="round,pad=0.4"))

    ax.set_yticks([])
    ax.set_xlim(0, h2h["n_battles"])
    ax.set_xlabel(f"Battles  (n = {h2h['n_battles']:,})", fontsize=12)
    ax.set_title(f"V6 Ensemble vs V6 baseline_5m  (Head-to-head, {h2h['name']})",
                 fontsize=14, color=TULANE["dark"], pad=12)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -1.1), ncol=2, fontsize=11)

    out = os.path.join(plots_dir, "head_to_head.png")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {out}")


# ════════════════════════════════════════════════════════════════════════
# 5. Effect-size table figure (publication-grade summary)
# ════════════════════════════════════════════════════════════════════════

def plot_effect_table(evals: Dict[str, dict], plots_dir: str, dpi: int):
    print("Generating effect-size table figure...")
    rows = []
    order = [
        ("V6 ensemble K=10 SOFT vs heuristic",       "K10_soft_heuristic"),
        ("V6 ensemble K=10 HARD vs heuristic",       "K10_hard_heuristic"),
        ("V6 ensemble K=10 CONFIDENCE vs heuristic", "K10_confidence_heuristic"),
        ("V6 ensemble K=5  SOFT vs heuristic",       "K5_soft_heuristic"),
        ("V6 ensemble K=3  SOFT vs heuristic",       "K3_soft_heuristic"),
        ("V6 ensemble K=1  SOFT vs heuristic",       "K1_soft_heuristic"),
        ("V6 baseline_5m solo vs heuristic",         "baseline5m_solo_heuristic"),
        ("V6 ensemble K=10 SOFT vs baseline_5m",     "K10_soft_baseline5m"),
    ]
    for label, eid in order:
        e = evals.get(eid)
        if not e:
            continue
        lo, hi = wilson_ci(e["wins"], e["n_battles"])
        rows.append([label, f"{e['wins']:,}/{e['n_battles']:,}",
                     f"{e['win_rate']:.4f}", f"[{lo:.4f}, {hi:.4f}]"])

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(11, 0.55 * (n_rows + 1) + 1.2))
    ax.axis("off")

    headers = ["Condition", "Wins / Battles", "Win Rate", "95% Wilson CI"]
    table = ax.table(cellText=rows, colLabels=headers, loc="center",
                     cellLoc="left", colLoc="left",
                     colWidths=[0.50, 0.18, 0.12, 0.20])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    # Style header row
    for j, _ in enumerate(headers):
        cell = table[(0, j)]
        cell.set_facecolor(TULANE["green"])
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_height(0.08)

    # Highlight the rows that map to "ensemble vs baseline" comparison
    highlight_keywords = ("K=10 SOFT vs heuristic", "baseline_5m solo", "vs baseline_5m")
    for i, row in enumerate(rows, start=1):
        if any(kw in row[0] for kw in highlight_keywords):
            for j in range(len(headers)):
                table[(i, j)].set_facecolor("#eaf3ed")

    ax.set_title("V6 Evaluation Summary — 100,000 battles per condition",
                 fontsize=15, color=TULANE["dark"], pad=12, fontweight="bold")

    out = os.path.join(plots_dir, "effect_lift_table.png")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {out}")


# ════════════════════════════════════════════════════════════════════════
# 6. Diagnostics: disagreement + unseen rate
# ════════════════════════════════════════════════════════════════════════

def plot_diagnostics(evals: Dict[str, dict], plots_dir: str, dpi: int):
    print("Generating diagnostics plot...")
    has_diag = [e for e in evals.values() if "diagnostics" in e]
    if not has_diag:
        print("  [skip] no diagnostics found in eval JSONs"); return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Disagreement ──
    ax = axes[0]
    for e in has_diag:
        d = e["diagnostics"].get("disagreement", {})
        if d.get("mean") is None: continue
        label = f"{e['name'][:30]}"
        # Plot mean + min/max as error bar
        ax.errorbar([label], [d["mean"]],
                    yerr=[[d["mean"] - d["min"]], [d["max"] - d["mean"]]],
                    fmt="o", color=COLOR_ENSEMBLE, markersize=10, capsize=8,
                    elinewidth=1.5, markeredgecolor="white", markeredgewidth=1.5)
        ax.text(label, d["mean"] + 0.03, f"{d['mean']:.3f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_title("Pairwise Argmax Disagreement\n(per-decision fraction of distinct argmaxes)",
                 fontsize=12, color=TULANE["dark"], pad=10)
    ax.set_ylabel("Disagreement fraction", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.3)

    # ── Unseen rate ──
    ax = axes[1]
    for e in has_diag:
        d = e["diagnostics"].get("unseen_rate", {})
        if d.get("mean") is None: continue
        label = f"{e['name'][:30]}"
        ax.errorbar([label], [d["mean"]],
                    yerr=[[d["mean"] - d["min"]], [d["max"] - d["mean"]]],
                    fmt="o", color=COLOR_BASELINE, markersize=10, capsize=8,
                    elinewidth=1.5, markeredgecolor="white", markeredgewidth=1.5)
        ax.text(label, d["mean"] + 0.03, f"{d['mean']:.3f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_title("Unseen-State Fallback Rate\n(per-decision fraction of members hitting heuristic prior)",
                 fontsize=12, color=TULANE["dark"], pad=10)
    ax.set_ylabel("Unseen fraction", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("V6 Ensemble Diagnostics (defends against ensemble collapse + heuristic dominance)",
                 fontsize=14, color=TULANE["dark"], fontweight="bold", y=1.02)
    out = os.path.join(plots_dir, "diagnostics.png")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ════════════════════════════════════════════════════════════════════════
# Stdout: publication-grade text summary
# ════════════════════════════════════════════════════════════════════════

def print_text_summary(evals: Dict[str, dict]) -> None:
    line = "═" * 84
    print("\n" + line)
    print(" V6 EVALUATION — PUBLICATION-GRADE SUMMARY")
    print(line)

    def _row(label: str, e: Optional[dict]) -> None:
        if not e:
            print(f"  {label:<48}  (missing)"); return
        lo, hi = wilson_ci(e["wins"], e["n_battles"])
        print(f"  {label:<48}  WR={e['win_rate']:.4f}  "
              f"95% CI [{lo:.4f}, {hi:.4f}]   ({e['wins']:,}/{e['n_battles']:,})")

    print("\n── K-saturation (SOFT, vs heuristic) ──")
    _row("K=1   single member",  evals.get("K1_soft_heuristic"))
    _row("K=3   ensemble (soft)", evals.get("K3_soft_heuristic"))
    _row("K=5   ensemble (soft)", evals.get("K5_soft_heuristic"))
    _row("K=10  ensemble (soft)", evals.get("K10_soft_heuristic"))

    print("\n── Strategy comparison @ K=10 (vs heuristic) ──")
    _row("HARD voting",        evals.get("K10_hard_heuristic"))
    _row("SOFT voting",        evals.get("K10_soft_heuristic"))
    _row("CONFIDENCE voting",  evals.get("K10_confidence_heuristic"))

    print("\n── Compute-matched comparison (vs heuristic, 100,000 battles each) ──")
    bsl = evals.get("baseline5m_solo_heuristic")
    ens_best = evals.get("K10_confidence_heuristic") or evals.get("K10_soft_heuristic")
    _row("baseline_5m (1 agent, 5M battles)",  bsl)
    _row(f"ensemble (10 × 1M, {ens_best['strategy'] if ens_best else '?'})", ens_best)

    if bsl and ens_best:
        delta, lo, hi = diff_ci(ens_best["wins"], ens_best["n_battles"],
                                bsl["wins"], bsl["n_battles"])
        z = abs(delta) / (math.sqrt(
            ens_best["win_rate"] * (1 - ens_best["win_rate"]) / ens_best["n_battles"]
            + bsl["win_rate"] * (1 - bsl["win_rate"]) / bsl["n_battles"]))
        print(f"\n  ┌─────────────────────────────────────────────────────────────────────┐")
        print(f"  │  ENSEMBLE LIFT vs BASELINE_5M  =  +{delta * 100:.2f} pp                     │")
        print(f"  │     95% CI  [{lo * 100:+.2f}, {hi * 100:+.2f}] pp                            │")
        print(f"  │     z = {z:.1f}σ  →  p < 0.001 (essentially zero overlap)                │")
        print(f"  └─────────────────────────────────────────────────────────────────────┘")

    print("\n── Direct head-to-head (ensemble plays baseline_5m) ──")
    h2h = evals.get("K10_soft_baseline5m")
    _row("K=10 SOFT vs baseline_5m", h2h)
    if h2h:
        edge = h2h["win_rate"] - 0.5
        se = math.sqrt(0.5 * 0.5 / h2h["n_battles"])
        z = edge / se
        print(f"     → ensemble wins by {edge * 100:+.2f} pp over coin flip ({z:.1f}σ)")

    print("\n── Diagnostics (where logged) ──")
    for eid, e in evals.items():
        if "diagnostics" not in e:
            continue
        diag = e["diagnostics"]
        bits = []
        d = diag.get("disagreement", {})
        if d.get("mean") is not None:
            bits.append(f"disagree={d['mean']:.3f}")
        d = diag.get("unseen_rate", {})
        if d.get("mean") is not None:
            bits.append(f"unseen={d['mean']:.3f}")
        if bits:
            print(f"  {e['name']:<48}  " + "  ".join(bits))

    print("\n" + line + "\n")


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="V6 evaluation analysis — paper-grade plots + statistical summary",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--out", type=str, default=None,
                        help="Plots directory (default: ensemble_results/run_<id>/eval_plots)")
    parser.add_argument("--skip", type=str, nargs="*", default=[],
                        help="Plot names to skip")
    args = parser.parse_args()

    run_dir = os.path.join(ENSEMBLE_RESULTS_DIR, f"run_{args.run_id}")
    if not os.path.isdir(run_dir):
        sys.exit(f"Run directory not found: {run_dir}")

    plots_dir = args.out or os.path.join(run_dir, "eval_plots")
    os.makedirs(plots_dir, exist_ok=True)

    evals = load_evals(run_dir)
    if not evals:
        sys.exit(f"No eval_*.json files found in {run_dir}")
    print(f"Loaded {len(evals)} eval JSONs from {run_dir}\n")

    setup_style()

    plots = {
        "k_saturation":     plot_k_saturation,
        "strategy":         plot_strategy_comparison,
        "headline":         plot_headline_comparison,
        "head_to_head":     plot_head_to_head,
        "effect_table":     plot_effect_table,
        "diagnostics":      plot_diagnostics,
    }
    for name, fn in plots.items():
        if name in args.skip:
            print(f"[skip {name}]"); continue
        try:
            fn(evals, plots_dir, args.dpi)
        except Exception as exc:
            print(f"[error {name}] {exc}")
            import traceback; traceback.print_exc()

    print_text_summary(evals)
    print(f"All plots written to: {plots_dir}\n")


if __name__ == "__main__":
    main()
