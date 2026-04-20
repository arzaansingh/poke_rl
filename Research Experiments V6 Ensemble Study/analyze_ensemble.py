"""
V6 Ensemble Analysis — generates all paper/poster plots from trained members.

Reads:
  ensemble_results/run_{id}/member_{k}/logs/run_1.csv        (per-member learning curves)
  ensemble_results/run_{id}/eval_{strategy}_K{n}_{opp}.json  (per-eval summaries)
  ensemble_results/baseline_5m/logs/run_1.csv                (compute-matched control)

Produces:
  ensemble_results/run_{id}/plots/
    per_member_curves.png            30 learning-curve lines + mean±std band
    per_member_final_bar.png         Each member's solo rolling WR (weak-learner framing)
    k_saturation.png                 Ensemble WR vs K in {1,5,10,20,30}
    strategy_comparison.png          soft / hard / confidence bar chart
    ensemble_vs_baselines.png        Ensemble vs single-5M vs V5 M8 delta
    disagreement_hist.png            Pairwise-disagreement histogram
    unseen_rate_hist.png             Unseen-state fallback rate histogram
    ensemble_vs_members.png          Ensemble WR vs best-of-K vs mean-of-members

Usage:
    python analyze_ensemble.py --run-id 1
    python analyze_ensemble.py --run-id 1 --dpi 300 --out custom_plots/
"""

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                # noqa: E402
import numpy as np                             # noqa: E402

from shared.config import ENSEMBLE_RESULTS_DIR  # noqa: E402


# ────────────────────────────────────────────────────────────────────────
# Color theme (match V5 poster)
# ────────────────────────────────────────────────────────────────────────

TULANE_GREEN = "#006747"
TULANE_BLUE = "#418FDE"
TULANE_DARK = "#003D2B"
ACCENT_ORANGE = "#e67e22"
ACCENT_GRAY = "#95a5a6"
ACCENT_BLUE = "#2980b9"


# ────────────────────────────────────────────────────────────────────────
# Data loaders
# ────────────────────────────────────────────────────────────────────────

def _load_member_csv(csv_path):
    """Return (battles, rolling, overall) as numpy arrays.  Empty if missing."""
    battles, rolling, overall = [], [], []
    if not os.path.exists(csv_path):
        return np.array([]), np.array([]), np.array([])
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            battles.append(int(row["Battles"]))
            rolling.append(float(row["RollingWin"]))
            overall.append(float(row["OverallWin"]))
    return np.array(battles), np.array(rolling), np.array(overall)


def _collect_members(run_dir):
    """Yield (k, csv_path) for every member_k/ in run_dir, sorted by k."""
    member_dirs = sorted(glob.glob(os.path.join(run_dir, "member_*")),
                         key=lambda p: int(os.path.basename(p).split("_")[1]))
    for d in member_dirs:
        k = int(os.path.basename(d).split("_")[1])
        yield k, os.path.join(d, "logs", "run_1.csv")


def _collect_eval_jsons(run_dir):
    """Read every eval_*.json in run_dir.  Returns list of dicts."""
    out = []
    for p in glob.glob(os.path.join(run_dir, "eval_*.json")):
        try:
            with open(p) as f:
                out.append(json.load(f))
        except Exception:
            pass
    return out


# ────────────────────────────────────────────────────────────────────────
# Plot: per-member learning curves with mean±std band
# ────────────────────────────────────────────────────────────────────────

def plot_per_member_curves(run_dir, plots_dir, dpi):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

    all_battles = None
    all_rolling = []
    for k, csv_path in _collect_members(run_dir):
        battles, rolling, _ = _load_member_csv(csv_path)
        if len(battles) == 0:
            continue
        ax.plot(battles, rolling, color=TULANE_BLUE, alpha=0.25, linewidth=0.8)
        # Accumulate for mean/std (use first member's x-grid as reference)
        if all_battles is None:
            all_battles = battles
        # Interpolate to common grid if lengths differ
        if len(battles) == len(all_battles):
            all_rolling.append(rolling)
        else:
            # interpolate to all_battles grid
            all_rolling.append(np.interp(all_battles, battles, rolling))

    if all_rolling:
        arr = np.array(all_rolling)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.plot(all_battles, mean, color=TULANE_GREEN, linewidth=2.5,
                label=f"Mean (K={len(all_rolling)})")
        ax.fill_between(all_battles, mean - std, mean + std, color=TULANE_GREEN, alpha=0.2,
                        label="±1 std")

    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Training battles", fontsize=12)
    ax.set_ylabel("Rolling win rate (100 battles)", fontsize=12)
    ax.set_title("V6 Ensemble — Per-Member Learning Curves", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(plots_dir, "per_member_curves.png")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  wrote {out}")


# ────────────────────────────────────────────────────────────────────────
# Plot: per-member final rolling WR bar chart (weak-learner framing)
# ────────────────────────────────────────────────────────────────────────

def plot_per_member_final_bar(run_dir, plots_dir, dpi):
    ks, finals = [], []
    for k, csv_path in _collect_members(run_dir):
        battles, rolling, _ = _load_member_csv(csv_path)
        if len(rolling) == 0:
            continue
        ks.append(k)
        finals.append(rolling[-1])
    if not ks:
        print("  [skip] no per-member data")
        return

    fig, ax = plt.subplots(figsize=(12, 5), dpi=dpi)
    ax.bar(ks, finals, color=TULANE_BLUE, edgecolor=TULANE_DARK, linewidth=0.5)
    ax.axhline(np.mean(finals), color=TULANE_GREEN, linewidth=2,
               label=f"Mean = {np.mean(finals):.3f}")
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Member k", fontsize=12)
    ax.set_ylabel("Final rolling win rate", fontsize=12)
    ax.set_title("V6 — Per-Member Solo Win Rate (Weak-Learner View)",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(plots_dir, "per_member_final_bar.png")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  wrote {out}")


# ────────────────────────────────────────────────────────────────────────
# Plot: K-saturation curve
# ────────────────────────────────────────────────────────────────────────

def plot_k_saturation(run_dir, plots_dir, dpi):
    evals = _collect_eval_jsons(run_dir)
    rows = [(e["K"], e["strategy"], e["win_rate"])
            for e in evals if e.get("opponent") == "heuristic"]
    if not rows:
        print("  [skip] no heuristic eval_*.json found")
        return

    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
    by_strategy = {}
    for K, strat, wr in rows:
        by_strategy.setdefault(strat, []).append((K, wr))
    colors = {"soft": TULANE_GREEN, "hard": TULANE_BLUE, "confidence": ACCENT_ORANGE}
    for strat, pts in by_strategy.items():
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", color=colors.get(strat, "gray"), linewidth=2, label=strat)

    ax.set_xlabel("Ensemble size K", fontsize=12)
    ax.set_ylabel("Win rate vs SimpleHeuristicsPlayer", fontsize=12)
    ax.set_title("V6 — K-Saturation Curve", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(0.598, color="gray", linewidth=1, linestyle=":",
               label="V5 M8 (59.8%)", alpha=0.8)
    fig.tight_layout()
    out = os.path.join(plots_dir, "k_saturation.png")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  wrote {out}")


# ────────────────────────────────────────────────────────────────────────
# Plot: strategy comparison
# ────────────────────────────────────────────────────────────────────────

def plot_strategy_comparison(run_dir, plots_dir, dpi):
    evals = _collect_eval_jsons(run_dir)
    # Use only the full-K heuristic evals for the headline comparison.
    heuristic_evals = [e for e in evals if e.get("opponent") == "heuristic"]
    if not heuristic_evals:
        print("  [skip] no heuristic eval jsons")
        return
    max_K = max(e["K"] for e in heuristic_evals)
    full = [e for e in heuristic_evals if e["K"] == max_K]
    if not full:
        return

    strats = [e["strategy"] for e in full]
    wrs = [e["win_rate"] for e in full]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)
    colors = [{"soft": TULANE_GREEN, "hard": TULANE_BLUE,
               "confidence": ACCENT_ORANGE}.get(s, "gray") for s in strats]
    ax.bar(strats, wrs, color=colors, edgecolor=TULANE_DARK, linewidth=0.8)
    for i, wr in enumerate(wrs):
        ax.text(i, wr + 0.005, f"{wr:.3f}", ha="center", fontweight="bold")
    ax.axhline(0.598, color="gray", linewidth=1, linestyle=":")
    ax.text(len(strats) - 0.5, 0.603, "V5 M8 baseline (59.8%)", fontsize=9, color="gray")
    ax.set_ylabel("Win rate vs SimpleHeuristicsPlayer", fontsize=12)
    ax.set_title(f"V6 Voting Strategy Comparison  (K={max_K})",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(plots_dir, "strategy_comparison.png")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  wrote {out}")


# ────────────────────────────────────────────────────────────────────────
# Plot: ensemble vs baselines
# ────────────────────────────────────────────────────────────────────────

def plot_ensemble_vs_baselines(run_dir, plots_dir, dpi):
    """Bars: V5 M8 single (59.8%), baseline_5m single, ensemble best strategy."""
    evals = _collect_eval_jsons(run_dir)
    heuristic_evals = [e for e in evals if e.get("opponent") == "heuristic"]
    if not heuristic_evals:
        print("  [skip] no heuristic eval jsons")
        return

    max_K = max(e["K"] for e in heuristic_evals)
    full = [e for e in heuristic_evals if e["K"] == max_K]
    best = max(full, key=lambda e: e["win_rate"])

    # Baseline_5m WR (from its own CSV final rolling WR, OR from eval_* if any)
    baseline_csv = os.path.join(ENSEMBLE_RESULTS_DIR, "baseline_5m", "logs", "run_1.csv")
    _, br, _ = _load_member_csv(baseline_csv)
    baseline_wr = br[-1] if len(br) else None

    labels = ["V5 M8\n(50K, 59.8%)"]
    values = [0.598]
    if baseline_wr is not None:
        labels.append(f"V6 Single\n(5M, {baseline_wr:.1%})")
        values.append(baseline_wr)
    labels.append(f"V6 Ensemble\n(K={max_K}, {best['strategy']})")
    values.append(best["win_rate"])

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=dpi)
    colors = [ACCENT_GRAY, ACCENT_BLUE, TULANE_GREEN][:len(values)]
    ax.bar(labels, values, color=colors, edgecolor=TULANE_DARK, linewidth=0.8)
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Win rate vs SimpleHeuristicsPlayer", fontsize=12)
    ax.set_title("V6 — Headline Comparison", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(values) * 1.15)
    fig.tight_layout()
    out = os.path.join(plots_dir, "ensemble_vs_baselines.png")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  wrote {out}")


# ────────────────────────────────────────────────────────────────────────
# Plot: disagreement + unseen histograms (from --log-diagnostics eval JSONs)
# ────────────────────────────────────────────────────────────────────────

def plot_diagnostics_histograms(run_dir, plots_dir, dpi):
    evals = _collect_eval_jsons(run_dir)
    evals = [e for e in evals if "diagnostics" in e]
    if not evals:
        print("  [skip] no diagnostics in eval jsons (run with --log-diagnostics)")
        return

    # Summary stats only (we don't log per-decision in the JSON; just mean/min/max).
    # Render as bar chart of mean values across runs.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=dpi)

    for ax, key, title in [(axes[0], "disagreement", "Pairwise Argmax Disagreement"),
                           (axes[1], "unseen_rate", "Unseen-State Fallback Rate")]:
        labels, means, mins, maxs = [], [], [], []
        for e in evals:
            d = e["diagnostics"][key]
            if d["n"] == 0:
                continue
            labels.append(f"{e['strategy']}/K{e['K']}")
            means.append(d["mean"])
            mins.append(d["min"])
            maxs.append(d["max"])
        if not labels:
            ax.set_visible(False)
            continue
        xs = np.arange(len(labels))
        ax.bar(xs, means, yerr=[np.array(means) - np.array(mins), np.array(maxs) - np.array(means)],
               capsize=5, color=TULANE_BLUE, edgecolor=TULANE_DARK, linewidth=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Mean fraction", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(plots_dir, "diagnostics.png")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  wrote {out}")


# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────

def _build_parser():
    p = argparse.ArgumentParser(
        description="V6 Ensemble Analysis / Plotting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-id", type=int, default=1)
    p.add_argument("--out", type=str, default=None,
                   help="Plots directory (default: ensemble_results/run_{id}/plots)")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--skip", type=str, nargs="*", default=[],
                   help="Plot names to skip (e.g. k_saturation diagnostics)")
    return p


def main():
    args = _build_parser().parse_args()
    run_dir = os.path.join(ENSEMBLE_RESULTS_DIR, f"run_{args.run_id}")
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    plots_dir = args.out or os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"V6 Analyze — run {args.run_id}  →  {plots_dir}\n")

    plots = {
        "per_member_curves":   plot_per_member_curves,
        "per_member_final":    plot_per_member_final_bar,
        "k_saturation":        plot_k_saturation,
        "strategy_comparison": plot_strategy_comparison,
        "ensemble_vs_baselines": plot_ensemble_vs_baselines,
        "diagnostics":         plot_diagnostics_histograms,
    }
    for name, fn in plots.items():
        if name in args.skip:
            print(f"  [skip {name}]")
            continue
        try:
            fn(run_dir, plots_dir, args.dpi)
        except Exception as exc:
            print(f"  [error {name}] {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
