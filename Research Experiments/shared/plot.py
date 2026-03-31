"""
Per-model multi-run plotting.
Reads all CSVs for one model, overlays individual runs as thin lines,
draws mean as bold line with ±1 std shaded band.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime


def clean_percentage(val):
    if isinstance(val, str):
        return float(val.strip('%')) / 100.0
    return float(val)


def load_model_runs(model_dir):
    """Load all run CSVs for a model. Returns list of DataFrames."""
    log_dir = os.path.join(model_dir, "logs")
    files = sorted(glob.glob(os.path.join(log_dir, "run_*.csv")))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            for col in ['RollingWin', 'OverallWin']:
                if col in df.columns and df[col].dtype == object:
                    df[col] = df[col].apply(clean_percentage)
            dfs.append(df)
        except Exception as e:
            print(f"  Skipping {f}: {e}")
    return dfs


def _align_runs(dfs, x_col='Battles', y_col='RollingWin'):
    """Align multiple runs to common x-axis, return (x, matrix of y values)."""
    if not dfs:
        return None, None

    # Use the x values from the longest run
    all_x = set()
    for df in dfs:
        if x_col in df.columns:
            all_x.update(df[x_col].values)
    x_vals = np.array(sorted(all_x))

    matrix = np.full((len(dfs), len(x_vals)), np.nan)
    for i, df in enumerate(dfs):
        if x_col not in df.columns or y_col not in df.columns:
            continue
        for _, row in df.iterrows():
            idx = np.searchsorted(x_vals, row[x_col])
            if idx < len(x_vals) and x_vals[idx] == row[x_col]:
                matrix[i, idx] = row[y_col]

    return x_vals, matrix


def plot_model(model_dir, model_label, save_dir=None):
    """Generate 4-panel plot for one model across all runs."""
    dfs = load_model_runs(model_dir)
    if not dfs:
        print(f"  No data for {model_label}")
        return

    if save_dir is None:
        save_dir = os.path.join(model_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{model_label} — {len(dfs)} Runs", fontsize=14, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # --- Panel 1: Win Rate ---
    ax = axes[0, 0]
    x, matrix = _align_runs(dfs, 'Battles', 'RollingWin')
    if x is not None:
        for i in range(matrix.shape[0]):
            ax.plot(x, matrix[i], color=colors[i % len(colors)], alpha=0.3, linewidth=0.8)
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
        ax.plot(x, mean, color='black', linewidth=2, label='Mean')
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color='blue')
    ax.set_title('Rolling Win Rate')
    ax.set_xlabel('Battles')
    ax.set_ylabel('Win Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Table Size ---
    ax = axes[0, 1]
    x, matrix = _align_runs(dfs, 'Battles', 'TableSize')
    if x is not None:
        for i in range(matrix.shape[0]):
            ax.plot(x, matrix[i], color=colors[i % len(colors)], alpha=0.3, linewidth=0.8)
        mean = np.nanmean(matrix, axis=0)
        ax.plot(x, mean, color='black', linewidth=2, label='Mean')
    ax.set_title('Q-Table Size')
    ax.set_xlabel('Battles')
    ax.set_ylabel('Entries')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Epsilon ---
    ax = axes[1, 0]
    for i, df in enumerate(dfs):
        if 'Epsilon' in df.columns:
            ax.plot(df['Battles'], df['Epsilon'], color=colors[i % len(colors)], alpha=0.5, linewidth=0.8)
    ax.set_title('Epsilon Schedule')
    ax.set_xlabel('Battles')
    ax.set_ylabel('Epsilon')
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Average Reward ---
    ax = axes[1, 1]
    x, matrix = _align_runs(dfs, 'Battles', 'AvgReward')
    if x is not None:
        for i in range(matrix.shape[0]):
            ax.plot(x, matrix[i], color=colors[i % len(colors)], alpha=0.3, linewidth=0.8)
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
        ax.plot(x, mean, color='black', linewidth=2, label='Mean')
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color='red')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Average Reward')
    ax.set_xlabel('Battles')
    ax.set_ylabel('Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(save_dir, f"{os.path.basename(model_dir)}_{timestamp}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from shared.config import MODEL_NAMES, MODEL_LABELS

    experiment_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for model_name in MODEL_NAMES:
        model_dir = os.path.join(experiment_dir, model_name)
        print(f"Plotting {MODEL_LABELS[model_name]}...")
        plot_model(model_dir, MODEL_LABELS[model_name])
