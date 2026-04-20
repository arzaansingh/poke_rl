"""
V5 Initialization Study Plotting
=================================
Plots learning curves for specific HP combos, averaging across runs
with variance bands. Works with partial/incomplete training data.

Usage:
  python shared/plot.py --combo hp_001                  # One combo, all 8 models
  python shared/plot.py --combo hp_001 hp_005 hp_010    # Compare combos
  python shared/plot.py --combo hp_001 --model model_4_hier_smart  # Single model
  python shared/plot.py --heatmap                       # Heatmap of best WR across grid
  python shared/plot.py --combo hp_001 --metric OverallWin  # Different y-axis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.config import (
    MODEL_NAMES, MODEL_LABELS, RUNS_PER_COMBO, BATTLES_PER_RUN,
    build_grid, GRID,
)

GRID_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "grid_results"
)
PLOTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots"
)

# Seaborn style
sns.set_theme(style="whitegrid", font_scale=1.1)
MODEL_COLORS = {
    "model_1_flat_zero":  "#1f77b4",
    "model_2_flat_smart": "#ff7f0e",
    "model_3_hier_zero":  "#2ca02c",
    "model_4_hier_smart": "#d62728",
    "model_5_flat_zero_fixed_eps": "#17becf",
    "model_6_flat_smart_fixed_eps": "#bcbd22",
    "model_7_hier_zero_fixed_eps": "#8c564b",
    "model_8_hier_smart_fixed_eps": "#e377c2",
}
MODEL_SHORT = {
    "model_1_flat_zero":  "Flat+Zero",
    "model_2_flat_smart": "Flat+Smart",
    "model_3_hier_zero":  "Hier+Zero",
    "model_4_hier_smart": "Hier+Smart",
    "model_5_flat_zero_fixed_eps":  "Flat+Zero+Fixed",
    "model_6_flat_smart_fixed_eps": "Flat+Smart+Fixed",
    "model_7_hier_zero_fixed_eps":  "Hier+Zero+Fixed",
    "model_8_hier_smart_fixed_eps": "Hier+Smart+Fixed",
}


def load_combo_data(combo_id):
    """Load all run CSVs for a combo. Returns {model_name: [df, df, ...]}."""
    combo_dir = os.path.join(GRID_RESULTS_DIR, combo_id)
    data = {}
    for model_name in MODEL_NAMES:
        dfs = []
        for r in range(1, RUNS_PER_COMBO + 1):
            csv_path = os.path.join(combo_dir, f"run_{r}", model_name, "logs", "run_1.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    for col in ['RollingWin', 'OverallWin']:
                        if col in df.columns and df[col].dtype == object:
                            df[col] = df[col].astype(float)
                    df['run'] = r
                    dfs.append(df)
                except Exception as e:
                    print(f"  Skipping {csv_path}: {e}")
        data[model_name] = dfs
    return data


def get_combo_params(combo_id):
    """Read params.json for a combo."""
    params_file = os.path.join(GRID_RESULTS_DIR, combo_id, "params.json")
    if os.path.exists(params_file):
        with open(params_file) as f:
            return json.load(f)
    # Fall back to computing from grid
    combos = build_grid()
    for c in combos:
        if c["combo_id"] == combo_id:
            return c
    return {"alpha": "?", "gamma": "?", "lam": "?"}


def interpolate_runs(dfs, x_col='Battles', y_col='RollingWin', n_points=200):
    """Interpolate multiple runs onto a common x-axis.

    Returns (x_common, mean, std, individual_curves).
    """
    if not dfs:
        return None, None, None, None

    # Find common x range (up to the shortest run's max)
    max_x_per_run = [df[x_col].max() for df in dfs if x_col in df.columns]
    if not max_x_per_run:
        return None, None, None, None

    x_max = min(max_x_per_run)  # Only average where ALL runs have data
    x_min = max(df[x_col].min() for df in dfs)
    x_common = np.linspace(x_min, x_max, n_points)

    curves = []
    for df in dfs:
        if x_col not in df.columns or y_col not in df.columns:
            continue
        y_interp = np.interp(x_common, df[x_col].values, df[y_col].values)
        curves.append(y_interp)

    if not curves:
        return None, None, None, None

    matrix = np.array(curves)
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    return x_common, mean, std, curves


def plot_combo(combo_id, models=None, metric='RollingWin', save=True):
    """Plot all V5 models for one HP combo: mean + std band, individual runs faint."""
    data = load_combo_data(combo_id)
    params = get_combo_params(combo_id)
    if models is None:
        models = MODEL_NAMES

    fig, ax = plt.subplots(figsize=(12, 7))

    title = (f"{combo_id}: "
             f"\u03b1={params.get('alpha', '?')}, "
             f"\u03b3={params.get('gamma', '?')}, "
             f"\u03bb={params.get('lam', '?')}")
    ax.set_title(title, fontsize=14, fontweight='bold')

    y_label = "Rolling Win Rate" if metric == "RollingWin" else metric
    ax.set_xlabel("Battles", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    for model_name in models:
        if model_name not in data:
            continue
        dfs = data[model_name]
        if not dfs:
            continue

        color = MODEL_COLORS[model_name]
        label = MODEL_SHORT[model_name]

        x, mean, std, curves = interpolate_runs(dfs, 'Battles', metric)
        if x is None:
            continue

        # Individual runs (faint)
        for curve in curves:
            ax.plot(x, curve, color=color, alpha=0.15, linewidth=0.7)

        # Mean + std band
        ax.plot(x, mean, color=color, linewidth=2.5, label=f"{label} (n={len(dfs)})")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4, label='50%')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, max(0.65, ax.get_ylim()[1]))

    plt.tight_layout()
    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        path = os.path.join(PLOTS_DIR, f"{combo_id}_{metric.lower()}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
        plt.close()
    else:
        plt.show()


def plot_compare_combos(combo_ids, model_name="model_4_hier_smart", metric='RollingWin', save=True):
    """Compare a specific model across multiple HP combos."""
    fig, ax = plt.subplots(figsize=(12, 7))

    palette = sns.color_palette("husl", len(combo_ids))

    ax.set_title(f"{MODEL_SHORT[model_name]} \u2014 Combo Comparison", fontsize=14, fontweight='bold')
    ax.set_xlabel("Battles", fontsize=12)
    y_label = "Rolling Win Rate" if metric == "RollingWin" else metric
    ax.set_ylabel(y_label, fontsize=12)

    for i, combo_id in enumerate(combo_ids):
        data = load_combo_data(combo_id)
        params = get_combo_params(combo_id)
        dfs = data.get(model_name, [])
        if not dfs:
            continue

        color = palette[i]
        label = (f"{combo_id} "
                 f"(\u03b1={params.get('alpha')}, "
                 f"\u03b3={params.get('gamma')}, "
                 f"\u03bb={params.get('lam')})")

        x, mean, std, _ = interpolate_runs(dfs, 'Battles', metric)
        if x is None:
            continue

        ax.plot(x, mean, color=color, linewidth=2.5, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()

    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        model_short = MODEL_SHORT[model_name].replace("+", "_").lower()
        combos_str = "_".join(combo_ids)
        path = os.path.join(PLOTS_DIR, f"compare_{model_short}_{combos_str}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
        plt.close()
    else:
        plt.show()


def plot_heatmap(metric='RollingWin', save=True):
    """Heatmap of best rolling WR per combo, broken down by model.

    Shows a grid with combos on y-axis and models on x-axis.
    Only includes combos that have data.
    """
    combos = build_grid()
    rows = []

    for combo in combos:
        combo_id = combo["combo_id"]
        combo_dir = os.path.join(GRID_RESULTS_DIR, combo_id)
        if not os.path.isdir(combo_dir):
            continue

        data = load_combo_data(combo_id)
        row = {
            "combo_id": combo_id,
            "label": f"\u03b1={combo['alpha']} \u03b3={combo['gamma']} \u03bb={combo['lam']}",
        }

        for model_name in MODEL_NAMES:
            dfs = data.get(model_name, [])
            if not dfs:
                row[MODEL_SHORT[model_name]] = np.nan
                continue
            # Best rolling WR across all runs (average of per-run best)
            bests = []
            for df in dfs:
                if metric in df.columns:
                    bests.append(df[metric].max())
            row[MODEL_SHORT[model_name]] = np.mean(bests) if bests else np.nan

        rows.append(row)

    if not rows:
        print("No data found for heatmap.")
        return

    df = pd.DataFrame(rows)
    df = df.set_index("label")
    model_cols = [MODEL_SHORT[m] for m in MODEL_NAMES]
    heatmap_data = df[model_cols].astype(float)

    # Add average column
    heatmap_data["Avg"] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values("Avg", ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(4, len(heatmap_data) * 0.5 + 2)))
    sns.heatmap(
        heatmap_data, annot=True, fmt=".1%", cmap="RdYlGn",
        vmin=0, vmax=0.6, linewidths=0.5,
        ax=ax, cbar_kws={"label": f"Best {metric}"}
    )
    ax.set_title("V5 Initialization Study \u2014 Best Rolling Win Rate by Combo", fontsize=13, fontweight='bold')
    ax.set_ylabel("")
    plt.tight_layout()

    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        path = os.path.join(PLOTS_DIR, f"heatmap_{metric.lower()}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
        plt.close()
    else:
        plt.show()


def plot_combo_panel(combo_id, save=True):
    """4-panel plot for a single combo: WinRate, TableSize, Epsilon, Reward."""
    data = load_combo_data(combo_id)
    params = get_combo_params(combo_id)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"{combo_id}: \u03b1={params.get('alpha')}, \u03b3={params.get('gamma')}, \u03bb={params.get('lam')}",
        fontsize=14, fontweight='bold'
    )

    panels = [
        (axes[0, 0], 'RollingWin', 'Rolling Win Rate'),
        (axes[0, 1], 'TableSize', 'Q-Table Size'),
        (axes[1, 0], 'Epsilon', 'Epsilon'),
        (axes[1, 1], 'AvgReward', 'Avg Reward'),
    ]

    for ax, y_col, title in panels:
        ax.set_title(title)
        ax.set_xlabel('Battles')
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.3)

        for model_name in MODEL_NAMES:
            dfs = data.get(model_name, [])
            if not dfs:
                continue
            color = MODEL_COLORS[model_name]
            label = MODEL_SHORT[model_name]

            x, mean, std, _ = interpolate_runs(dfs, 'Battles', y_col)
            if x is None:
                continue

            ax.plot(x, mean, color=color, linewidth=2, label=f"{label} (n={len(dfs)})")
            if y_col not in ('Epsilon',):
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)

        if y_col == 'RollingWin':
            ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
        if y_col == 'AvgReward':
            ax.axhline(0, color='gray', linestyle='--', alpha=0.4)

        ax.legend(fontsize=8, loc='best')

    plt.tight_layout()
    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        path = os.path.join(PLOTS_DIR, f"{combo_id}_panel.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V5 Initialization Study Plots")
    parser.add_argument("--combo", nargs="+", default=None,
                        help="HP combo(s) to plot (e.g. hp_001 hp_005)")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model for cross-combo comparison")
    parser.add_argument("--metric", type=str, default="RollingWin",
                        choices=["RollingWin", "OverallWin", "AvgReward", "TableSize"],
                        help="Metric to plot (default: RollingWin)")
    parser.add_argument("--heatmap", action="store_true",
                        help="Generate heatmap across all combos with data")
    parser.add_argument("--panel", action="store_true",
                        help="Generate 4-panel plot (WR, TableSize, Eps, Reward)")
    parser.add_argument("--no-save", action="store_true",
                        help="Show plot instead of saving")
    args = parser.parse_args()

    save = not args.no_save

    if args.heatmap:
        plot_heatmap(metric=args.metric, save=save)

    elif args.combo and len(args.combo) == 1 and not args.model:
        # Single combo: overlay all configured models
        if args.panel:
            plot_combo_panel(args.combo[0], save=save)
        else:
            plot_combo(args.combo[0], metric=args.metric, save=save)

    elif args.combo and len(args.combo) > 1:
        # Multiple combos: compare one model across combos
        model = args.model or "model_4_hier_smart"
        plot_compare_combos(args.combo, model_name=model, metric=args.metric, save=save)

    elif args.combo and args.model:
        # Single combo, single model
        plot_combo(args.combo[0], models=[args.model], metric=args.metric, save=save)

    else:
        # Default: plot all combos that have data
        combos = build_grid()
        plotted = 0
        for combo in combos:
            combo_dir = os.path.join(GRID_RESULTS_DIR, combo["combo_id"])
            if os.path.isdir(combo_dir):
                print(f"Plotting {combo['combo_id']}...")
                plot_combo(combo["combo_id"], metric=args.metric, save=save)
                plot_combo_panel(combo["combo_id"], save=save)
                plotted += 1
        if plotted > 0:
            plot_heatmap(metric=args.metric, save=save)
        else:
            print("No grid results found. Run the grid search first.")
