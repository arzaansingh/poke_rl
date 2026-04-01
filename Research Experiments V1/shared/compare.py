"""
Cross-model comparison plots for the 2x2 factorial experiment.
Generates:
1. Learning curves: 4 bold lines (mean win rate) with CI bands
2. Final performance: Bar chart with error bars
3. Interaction plot: Zero vs Smart init × Flat vs Hierarchical
4. Table size comparison
5. Summary table printed to console
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.config import MODEL_NAMES, MODEL_LABELS, BATTLES_PER_RUN


MODEL_COLORS = {
    "model_1_flat_zero": "#1f77b4",   # Blue
    "model_2_flat_smart": "#ff7f0e",  # Orange
    "model_3_hier_zero": "#2ca02c",   # Green
    "model_4_hier_smart": "#d62728",  # Red
}

# Last N battles to average for "final performance"
FINAL_WINDOW = 2000


def clean_percentage(val):
    if isinstance(val, str):
        return float(val.strip('%')) / 100.0
    return float(val)


def load_all_models(experiment_dir):
    """Load all runs for all models. Returns {model_name: [list of DataFrames]}."""
    all_data = {}
    for model_name in MODEL_NAMES:
        log_dir = os.path.join(experiment_dir, model_name, "logs")
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
        all_data[model_name] = dfs
    return all_data


def _get_aligned_means(dfs, x_col='Battles', y_col='RollingWin'):
    """Return (x_vals, mean, std) aligned across runs."""
    if not dfs:
        return None, None, None
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

    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    return x_vals, mean, std


def _get_final_performance(dfs, y_col='RollingWin'):
    """Get mean and std of y_col over the last FINAL_WINDOW battles for each run."""
    final_vals = []
    for df in dfs:
        if 'Battles' not in df.columns or y_col not in df.columns:
            continue
        max_bat = df['Battles'].max()
        cutoff = max_bat - FINAL_WINDOW
        tail = df[df['Battles'] >= cutoff]
        if len(tail) > 0:
            final_vals.append(tail[y_col].mean())
    if not final_vals:
        return np.nan, np.nan
    return np.mean(final_vals), np.std(final_vals)


def generate_all_comparisons(experiment_dir):
    """Generate all comparison plots and save to comparison_plots/."""
    plot_dir = os.path.join(experiment_dir, "comparison_plots")
    os.makedirs(plot_dir, exist_ok=True)

    all_data = load_all_models(experiment_dir)

    # Check we have data
    has_data = any(len(dfs) > 0 for dfs in all_data.values())
    if not has_data:
        print("No data found for any model.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ===== PLOT 1: Learning Curves =====
    fig, ax = plt.subplots(figsize=(12, 7))
    for model_name in MODEL_NAMES:
        dfs = all_data[model_name]
        if not dfs:
            continue
        x, mean, std = _get_aligned_means(dfs, 'Battles', 'RollingWin')
        if x is None:
            continue
        color = MODEL_COLORS[model_name]
        label = MODEL_LABELS[model_name]
        ax.plot(x, mean, color=color, linewidth=2, label=label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_title('Learning Curves — Rolling Win Rate (Mean ± 1 SD)', fontsize=14)
    ax.set_xlabel('Battles', fontsize=12)
    ax.set_ylabel('Win Rate', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"learning_curves_{timestamp}.png"), dpi=150)
    plt.close()

    # ===== PLOT 2: Final Performance Bar Chart =====
    fig, ax = plt.subplots(figsize=(8, 6))
    names = []
    means = []
    stds = []
    colors = []
    for model_name in MODEL_NAMES:
        dfs = all_data[model_name]
        m, s = _get_final_performance(dfs)
        names.append(MODEL_LABELS[model_name])
        means.append(m)
        stds.append(s)
        colors.append(MODEL_COLORS[model_name])

    x_pos = np.arange(len(names))
    ax.bar(x_pos, means, yerr=stds, color=colors, capsize=5, alpha=0.8, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel('Win Rate (last 2K battles)')
    ax.set_title('Final Performance Comparison', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"final_performance_{timestamp}.png"), dpi=150)
    plt.close()

    # ===== PLOT 3: Interaction Plot =====
    fig, ax = plt.subplots(figsize=(8, 6))
    # Get final performance for each of the 4 conditions
    perf = {}
    for model_name in MODEL_NAMES:
        m, s = _get_final_performance(all_data[model_name])
        perf[model_name] = (m, s)

    # Flat line: model_1 (zero) → model_2 (smart)
    flat_zero_m, flat_zero_s = perf["model_1_flat_zero"]
    flat_smart_m, flat_smart_s = perf["model_2_flat_smart"]
    hier_zero_m, hier_zero_s = perf["model_3_hier_zero"]
    hier_smart_m, hier_smart_s = perf["model_4_hier_smart"]

    x_init = [0, 1]
    ax.errorbar(x_init, [flat_zero_m, flat_smart_m], yerr=[flat_zero_s, flat_smart_s],
                marker='o', markersize=8, linewidth=2, color='#1f77b4', label='Flat', capsize=5)
    ax.errorbar(x_init, [hier_zero_m, hier_smart_m], yerr=[hier_zero_s, hier_smart_s],
                marker='s', markersize=8, linewidth=2, color='#d62728', label='Hierarchical', capsize=5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Zero Init', 'Smart Init'], fontsize=12)
    ax.set_ylabel('Win Rate (last 2K battles)', fontsize=12)
    ax.set_title('Interaction Plot: Architecture × Initialization', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"interaction_plot_{timestamp}.png"), dpi=150)
    plt.close()

    # ===== PLOT 4: Table Size Comparison =====
    fig, ax = plt.subplots(figsize=(12, 7))
    for model_name in MODEL_NAMES:
        dfs = all_data[model_name]
        if not dfs:
            continue
        x, mean, std = _get_aligned_means(dfs, 'Battles', 'TableSize')
        if x is None:
            continue
        color = MODEL_COLORS[model_name]
        label = MODEL_LABELS[model_name]
        ax.plot(x, mean, color=color, linewidth=2, label=label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_title('Q-Table Size Growth', fontsize=14)
    ax.set_xlabel('Battles', fontsize=12)
    ax.set_ylabel('Table Entries', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"table_size_{timestamp}.png"), dpi=150)
    plt.close()

    # ===== Summary Table =====
    print(f"\n{'='*70}")
    print(f"{'Model':<22} {'Win Rate':>12} {'Table Size':>12} {'Runs':>6}")
    print(f"{'='*70}")
    for model_name in MODEL_NAMES:
        dfs = all_data[model_name]
        m, s = _get_final_performance(dfs)
        # Get final table size
        table_sizes = []
        for df in dfs:
            if 'TableSize' in df.columns and len(df) > 0:
                table_sizes.append(df['TableSize'].iloc[-1])
        ts_str = f"{np.mean(table_sizes):.0f}" if table_sizes else "N/A"
        wr_str = f"{m:.2%} ± {s:.2%}" if not np.isnan(m) else "N/A"
        print(f"{MODEL_LABELS[model_name]:<22} {wr_str:>12} {ts_str:>12} {len(dfs):>6}")
    print(f"{'='*70}")

    print(f"\nAll plots saved to: {plot_dir}")


def _parse_plot_filters(filter_args):
    """Parse --plot arguments into {model_name: [run_ids]} or {model_name: None} for all runs.

    Examples:
        []                    → all models, all runs
        ["model_1_flat_zero"] → just model_1, all runs
        ["model_1:2"]         → model_1, run 2 only
        ["model_1:1,3", "model_3"] → model_1 runs 1&3, model_3 all runs

    Also supports short names like "flat_zero", "hier_smart", etc.
    """
    SHORT_MAP = {
        "flat_zero": "model_1_flat_zero", "model_1": "model_1_flat_zero",
        "flat_smart": "model_2_flat_smart", "model_2": "model_2_flat_smart",
        "hier_zero": "model_3_hier_zero", "model_3": "model_3_hier_zero",
        "hier_smart": "model_4_hier_smart", "model_4": "model_4_hier_smart",
    }

    if not filter_args:
        return {m: None for m in MODEL_NAMES}

    result = {}
    for arg in filter_args:
        if ':' in arg:
            name_part, runs_part = arg.split(':', 1)
            run_ids = [int(r) for r in runs_part.split(',')]
        else:
            name_part = arg
            run_ids = None

        # Resolve short names
        model_name = SHORT_MAP.get(name_part, name_part)
        if model_name not in MODEL_NAMES:
            # Try partial match
            matches = [m for m in MODEL_NAMES if name_part in m]
            if len(matches) == 1:
                model_name = matches[0]
            else:
                print(f"Warning: unknown model '{name_part}', skipping")
                continue
        result[model_name] = run_ids

    return result


def plot_progress(experiment_dir, filter_args=None):
    """Plot current training progress. Can be called mid-training.

    If a model has only 1 run: plots that single run's curve.
    If multiple runs: plots the mean with confidence interval band.
    Supports filtering to specific models and runs.

    Args:
        experiment_dir: Path to the experiment directory
        filter_args: List of filter strings (e.g. ["model_1", "model_3:2"]) or empty list for all
    """
    filters = _parse_plot_filters(filter_args)

    if not filters:
        print("No models selected.")
        return

    print(f"Plotting progress for: {', '.join(MODEL_LABELS.get(m, m) for m in filters)}")

    # Load data per filter
    plot_data = {}  # {label: [list of DataFrames]}
    for model_name, run_ids in filters.items():
        log_dir = os.path.join(experiment_dir, model_name, "logs")
        files = sorted(glob.glob(os.path.join(log_dir, "run_*.csv")))
        if not files:
            print(f"  No data for {MODEL_LABELS.get(model_name, model_name)}")
            continue

        dfs = []
        for f in files:
            # Extract run_id from filename
            basename = os.path.basename(f)
            try:
                rid = int(basename.replace("run_", "").replace(".csv", ""))
            except ValueError:
                continue
            if run_ids is not None and rid not in run_ids:
                continue
            try:
                df = pd.read_csv(f)
                for col in ['RollingWin', 'OverallWin']:
                    if col in df.columns and df[col].dtype == object:
                        df[col] = df[col].apply(clean_percentage)
                df._run_id = rid  # Tag with run_id
                dfs.append(df)
            except Exception as e:
                print(f"  Skipping {f}: {e}")

        if not dfs:
            continue

        label = MODEL_LABELS.get(model_name, model_name)
        if run_ids is not None and len(run_ids) == 1:
            label += f" (Run {run_ids[0]})"

        plot_data[model_name] = (label, dfs)

    if not plot_data:
        print("No data to plot.")
        return

    # === FIGURE 1: Learning Curves ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Win rate plot
    ax = axes[0]
    for model_name, (label, dfs) in plot_data.items():
        color = MODEL_COLORS.get(model_name, '#333333')
        if len(dfs) == 1:
            # Single run: just plot the line
            df = dfs[0]
            if 'Battles' in df.columns and 'RollingWin' in df.columns:
                ax.plot(df['Battles'], df['RollingWin'], color=color, linewidth=1.5, label=label, alpha=0.9)
        else:
            # Multiple runs: mean + CI
            x, mean, std = _get_aligned_means(dfs, 'Battles', 'RollingWin')
            if x is not None:
                ax.plot(x, mean, color=color, linewidth=2, label=label)
                ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_title('Rolling Win Rate', fontsize=14)
    ax.set_xlabel('Battles', fontsize=12)
    ax.set_ylabel('Win Rate', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Table size plot
    ax2 = axes[1]
    for model_name, (label, dfs) in plot_data.items():
        color = MODEL_COLORS.get(model_name, '#333333')
        if len(dfs) == 1:
            df = dfs[0]
            if 'Battles' in df.columns and 'TableSize' in df.columns:
                ax2.plot(df['Battles'], df['TableSize'], color=color, linewidth=1.5, label=label, alpha=0.9)
        else:
            x, mean, std = _get_aligned_means(dfs, 'Battles', 'TableSize')
            if x is not None:
                ax2.plot(x, mean, color=color, linewidth=2, label=label)
                ax2.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)

    ax2.set_title('Q-Table Size', fontsize=14)
    ax2.set_xlabel('Battles', fontsize=12)
    ax2.set_ylabel('Table Entries', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Training Progress (in-progress)', fontsize=16, y=1.02)
    plt.tight_layout()

    # Save and show
    plot_dir = os.path.join(experiment_dir, "comparison_plots")
    os.makedirs(plot_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(plot_dir, f"progress_{timestamp}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.show()
    plt.close()

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Model':<22} {'Battles':>10} {'Win Rate':>10} {'Table Size':>12} {'Runs':>6}")
    print(f"{'='*70}")
    for model_name, (label, dfs) in plot_data.items():
        total_battles = max(df['Battles'].max() for df in dfs if 'Battles' in df.columns)
        m, s = _get_final_performance(dfs)
        table_sizes = []
        for df in dfs:
            if 'TableSize' in df.columns and len(df) > 0:
                table_sizes.append(df['TableSize'].iloc[-1])
        ts_str = f"{np.mean(table_sizes):.0f}" if table_sizes else "N/A"
        wr_str = f"{m:.2%}" if not np.isnan(m) else "N/A"
        if len(dfs) > 1 and not np.isnan(s):
            wr_str += f" ± {s:.2%}"
        print(f"{label:<22} {total_battles:>10,.0f} {wr_str:>10} {ts_str:>12} {len(dfs):>6}")
    print(f"{'='*70}")


if __name__ == "__main__":
    import sys as _sys
    experiment_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if len(_sys.argv) > 1:
        plot_progress(experiment_dir, _sys.argv[1:])
    else:
        generate_all_comparisons(experiment_dir)
