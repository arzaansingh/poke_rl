"""
Poster-quality figures for V5 Initialization Study.

Generates publication-ready plots using seaborn/matplotlib with unified
Tulane + Pokemon color theme. Pulls data directly from experiment CSVs.

Usage:
  python shared/poster_plots.py                    # Generate all 5 figures
  python shared/poster_plots.py --plot heatmap     # Single plot
  python shared/poster_plots.py --plot curves      # Single plot
  python shared/poster_plots.py --dpi 600          # Higher resolution
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.config import MODEL_NAMES, MODEL_LABELS, build_grid
from shared.plot import load_combo_data, interpolate_runs, get_combo_params

# ═══════════════════════════════════════════════
#  UNIFIED POSTER COLOR PALETTE
# ═══════════════════════════════════════════════
TULANE = {'green': '#006747', 'blue': '#418FDE', 'kelly': '#43B02A', 'dark': '#003D2B'}

# 4 base model colors — same color for Decay/Fixed, distinguished by linestyle
_BASE = {
    'flat_zero': '#95a5a6',
    'flat_smart': '#e67e22',
    'hier_zero': '#2980b9',
    'hier_smart': '#006747',
}

POSTER_COLORS = {
    'model_1_flat_zero': _BASE['flat_zero'],
    'model_2_flat_smart': _BASE['flat_smart'],
    'model_3_hier_zero': _BASE['hier_zero'],
    'model_4_hier_smart': _BASE['hier_smart'],
    'model_5_flat_zero_fixed_eps': _BASE['flat_zero'],
    'model_6_flat_smart_fixed_eps': _BASE['flat_smart'],
    'model_7_hier_zero_fixed_eps': _BASE['hier_zero'],
    'model_8_hier_smart_fixed_eps': _BASE['hier_smart'],
}

POSTER_LINESTYLES = {
    'model_1_flat_zero': '-',
    'model_2_flat_smart': '-',
    'model_3_hier_zero': '-',
    'model_4_hier_smart': '-',
    'model_5_flat_zero_fixed_eps': '--',
    'model_6_flat_smart_fixed_eps': '--',
    'model_7_hier_zero_fixed_eps': '--',
    'model_8_hier_smart_fixed_eps': '--',
}

POSTER_SHORT = {
    'model_1_flat_zero': 'Flat+Zero',
    'model_2_flat_smart': 'Flat+Smart',
    'model_3_hier_zero': 'Hier+Zero',
    'model_4_hier_smart': 'Hier+Smart',
    'model_5_flat_zero_fixed_eps': 'Flat+Zero+Fix',
    'model_6_flat_smart_fixed_eps': 'Flat+Smart+Fix',
    'model_7_hier_zero_fixed_eps': 'Hier+Zero+Fix',
    'model_8_hier_smart_fixed_eps': 'Hier+Smart+Fix',
}

# ── Paths ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID_RESULTS_DIR = os.path.join(PROJECT_ROOT, "grid_results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "poster", "figures")
SUMMARY_CSV = os.path.join(GRID_RESULTS_DIR, "summary.csv")


def setup_style():
    sns.set_theme(style="whitegrid", font_scale=1.3)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.titleweight': 'bold',
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
    })


def save_fig(fig, name, dpi=300):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════
#  1. HEATMAP — Best Rolling Win Rate
# ═══════════════════════════════════════════════
def plot_heatmap(dpi=300):
    print("Generating heatmap (best rolling win rate)...")
    df = pd.read_csv(SUMMARY_CSV)

    from matplotlib.colors import LinearSegmentedColormap

    model_order = [
        'best_Flat_ZeroInit', 'best_Flat_SmartInit',
        'best_Hier_ZeroInit', 'best_Hier_SmartInit',
        'best_Flat_ZeroInit_FixedEps', 'best_Flat_SmartInit_FixedEps',
        'best_Hier_ZeroInit_FixedEps', 'best_Hier_SmartInit_FixedEps',
    ]
    model_order = [c for c in model_order if c in df.columns]

    short_names = {
        'best_Flat_ZeroInit': 'Flat+Zero',
        'best_Flat_SmartInit': 'Flat+Smart',
        'best_Hier_ZeroInit': 'Hier+Zero',
        'best_Hier_SmartInit': 'Hier+Smart',
        'best_Flat_ZeroInit_FixedEps': 'Flat+Zero+Fix',
        'best_Flat_SmartInit_FixedEps': 'Flat+Smart+Fix',
        'best_Hier_ZeroInit_FixedEps': 'Hier+Zero+Fix',
        'best_Hier_SmartInit_FixedEps': 'Hier+Smart+Fix',
    }

    combo_labels = []
    for _, row in df.iterrows():
        combo_labels.append(f"α={row['alpha']}  γ={row['gamma']}  λ={row['lam']}")

    heat_data = df[model_order].copy()
    heat_data.index = combo_labels
    heat_data.columns = [short_names.get(c, c) for c in model_order]

    heat_data['Avg'] = heat_data.mean(axis=1)
    heat_data = heat_data.sort_values('Avg', ascending=False)
    heat_data.drop(columns='Avg', inplace=True)

    # Red → white → green
    tulane_cmap = LinearSegmentedColormap.from_list('bold_rg', [
        (0.0, '#CC4433'),    # strong red
        (0.5, '#FFFFFF'),    # white midpoint
        (1.0, '#2E8B4A'),    # strong green
    ])

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(
        heat_data, annot=True, fmt='.1%', cmap=tulane_cmap,
        vmin=0.05, vmax=0.65, linewidths=0.8, ax=ax,
        cbar_kws={'label': 'Best Rolling Win Rate', 'shrink': 0.9, 'aspect': 15, 'fraction': 0.06, 'pad': 0.03},
        annot_kws={'size': 14, 'weight': 'bold'},
    )

    ax.set_title('Best Rolling Win Rate — All HP Combos', fontsize=18, pad=12, color=TULANE['dark'])
    ax.set_ylabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right', fontsize=14)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16, width=2, length=6)
    cbar.set_label('Best Rolling Win Rate', fontsize=17, fontweight='bold')
    cbar.outline.set_linewidth(2)

    save_fig(fig, 'heatmap_rollingwin.png', dpi)


# ═══════════════════════════════════════════════
#  1b. BEST CONFIG — 4-model focused results
# ═══════════════════════════════════════════════
# The 4 "natural" models: zero+decay, smart+fixed
FOCUS_MODELS = [
    'model_1_flat_zero',
    'model_6_flat_smart_fixed_eps',
    'model_3_hier_zero',
    'model_8_hier_smart_fixed_eps',
]
FOCUS_LABELS = {
    'model_1_flat_zero': 'Flat + Zero',
    'model_6_flat_smart_fixed_eps': 'Flat + Smart',
    'model_3_hier_zero': 'Hier + Zero',
    'model_8_hier_smart_fixed_eps': 'Hier + Smart',
}
# Pastel/sleek versions matching the SVG diagram palette
FOCUS_COLORS = {
    'model_1_flat_zero': '#B0BEC5',      # soft steel gray
    'model_6_flat_smart_fixed_eps': '#FFAB76',  # soft peach orange
    'model_3_hier_zero': '#81B4E0',      # soft sky blue
    'model_8_hier_smart_fixed_eps': '#6DBF8B',  # soft mint green
}


def plot_best_config(dpi=300, combo_id='hp_001'):
    """Generate single unified 2x2 figure for the best-config results block."""
    print(f"Generating best-config plots for {combo_id}...")
    data = load_combo_data(combo_id)
    params = get_combo_params(combo_id)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11),
                             gridspec_kw={'hspace': 0.32, 'wspace': 0.25})
    ax_roll, ax_overall = axes[0]
    ax_violin, ax_qtable = axes[1]

    # ── Top row: Rolling Win + Overall Win ──
    for model_name in FOCUS_MODELS:
        dfs = data.get(model_name, [])
        if not dfs:
            continue
        color = FOCUS_COLORS[model_name]
        label = FOCUS_LABELS[model_name]

        x, mean, std, curves = interpolate_runs(dfs, 'Battles', 'RollingWin')
        if x is not None:
            ax_roll.plot(x, mean, color=color, linewidth=2.8, label=label)
            ax_roll.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)

        x2, mean2, std2, curves2 = interpolate_runs(dfs, 'Battles', 'OverallWin')
        if x2 is not None:
            ax_overall.plot(x2, mean2, color=color, linewidth=2.8, label=label)
            ax_overall.fill_between(x2, mean2 - std2, mean2 + std2, color=color, alpha=0.12)

    for ax, title in [(ax_roll, 'Rolling Win Rate (1K window)'), (ax_overall, 'Overall Win Rate')]:
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
        ax.set_xlabel('Battles', fontsize=13)
        ax.set_title(title, fontsize=15, color=TULANE['dark'])
        ax.set_ylim(0, 0.72)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x / 1000)}K'))
        ax.grid(True, alpha=0.3)

    ax_roll.set_ylabel('Win Rate', fontsize=13)

    # ── Shared horizontal legend at top ──
    handles, labels = ax_roll.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='upper center', ncol=4,
                     fontsize=15, framealpha=0.9, handlelength=2.5,
                     handletextpad=0.8, columnspacing=2.0,
                     borderpad=0.6, bbox_to_anchor=(0.5, 1.005))
    for line in leg.get_lines():
        line.set_linewidth(4.0)
    for text in leg.get_texts():
        text.set_fontweight('bold')

    # ── Bottom-left: Violin ──
    records = []
    for model_name in FOCUS_MODELS:
        dfs = data.get(model_name, [])
        for df in dfs:
            if 'RollingWin' in df.columns:
                records.append({
                    'Model': FOCUS_LABELS[model_name],
                    'Final Win Rate': df['RollingWin'].iloc[-1],
                })
    plot_df = pd.DataFrame(records)
    model_order = [FOCUS_LABELS[m] for m in FOCUS_MODELS]
    palette = {FOCUS_LABELS[m]: FOCUS_COLORS[m] for m in FOCUS_MODELS}

    sns.violinplot(data=plot_df, x='Model', y='Final Win Rate', hue='Model',
                   order=model_order, palette=palette, inner='box',
                   ax=ax_violin, cut=0, legend=False)
    ax_violin.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
    ax_violin.set_xlabel('')
    ax_violin.set_ylabel('Final Rolling Win Rate', fontsize=13)
    ax_violin.set_title('Win Rate Distribution (10 runs)', fontsize=15, color=TULANE['dark'])
    ax_violin.set_ylim(0, 0.72)
    ax_violin.set_xticklabels(model_order, fontsize=11, fontweight='bold')
    ax_violin.grid(axis='y', alpha=0.3)

    # ── Bottom-right: Q-Table Size ──
    table_data = []
    for model_name in FOCUS_MODELS:
        dfs = data.get(model_name, [])
        sizes = [df['TableSize'].iloc[-1] for df in dfs if 'TableSize' in df.columns]
        if sizes:
            table_data.append({
                'Model': FOCUS_LABELS[model_name],
                'Q-Table Entries': np.mean(sizes),
                'color': FOCUS_COLORS[model_name],
            })

    bars = ax_qtable.bar(
        range(len(table_data)),
        [d['Q-Table Entries'] / 1e6 for d in table_data],
        color=[d['color'] for d in table_data],
        edgecolor='white', linewidth=1.5, width=0.6,
    )
    for bar, d in zip(bars, table_data):
        val = d['Q-Table Entries'] / 1e6
        ax_qtable.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                       f'{val:.1f}M', ha='center', fontsize=13, fontweight='bold',
                       color=TULANE['dark'])

    ax_qtable.set_ylabel('Q-Table Entries (millions)', fontsize=13)
    ax_qtable.set_title('Q-Table Size at 50K Battles', fontsize=15, color=TULANE['dark'])
    ax_qtable.set_ylim(0, max(d['Q-Table Entries'] for d in table_data) / 1e6 * 1.2)
    ax_qtable.grid(axis='y', alpha=0.3)
    ax_qtable.set_xticks(range(len(table_data)))
    ax_qtable.set_xticklabels([d['Model'] for d in table_data], rotation=20, ha='right', fontsize=11, fontweight='bold')

    fig.subplots_adjust(top=0.90, bottom=0.07, left=0.07, right=0.97)
    save_fig(fig, f'{combo_id}_combined.png', dpi)


# ═══════════════════════════════════════════════
#  2. LEARNING CURVES (hp_001)
# ═══════════════════════════════════════════════
def plot_learning_curves(dpi=300, combo_id='hp_001'):
    print(f"Generating learning curves for {combo_id}...")
    data = load_combo_data(combo_id)
    params = get_combo_params(combo_id)

    fig, ax = plt.subplots(figsize=(11, 6))

    for model_name in MODEL_NAMES:
        dfs = data.get(model_name, [])
        if not dfs:
            continue

        color = POSTER_COLORS[model_name]
        ls = POSTER_LINESTYLES[model_name]
        label = POSTER_SHORT[model_name]

        x, mean, std, curves = interpolate_runs(dfs, 'Battles', 'RollingWin')
        if x is None:
            continue

        for curve in curves:
            ax.plot(x, curve, color=color, alpha=0.08, linewidth=0.5, linestyle=ls)

        ax.plot(x, mean, color=color, linewidth=2.8, linestyle=ls, label=f"{label} (n={len(dfs)})")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)

    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1.5, label='50%')
    ax.set_xlabel('Battles', fontsize=14)
    ax.set_ylabel('Rolling Win Rate (1K window)', fontsize=14)
    ax.set_title(
        f'Learning Curves — {combo_id} (α={params.get("alpha")}, γ={params.get("gamma")}, λ={params.get("lam")})',
        fontsize=16, color=TULANE['dark']
    )
    ax.set_ylim(0, 0.70)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x / 1000)}K'))
    ax.legend(loc='lower right', fontsize=9, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    save_fig(fig, f'{combo_id}_learning_curves.png', dpi)


# ═══════════════════════════════════════════════
#  3. VIOLIN + STRIP PLOT (hp_001)
# ═══════════════════════════════════════════════
def plot_violin(dpi=300, combo_id='hp_001'):
    print(f"Generating violin plot for {combo_id}...")
    data = load_combo_data(combo_id)

    records = []
    for model_name in MODEL_NAMES:
        dfs = data.get(model_name, [])
        for df in dfs:
            if 'RollingWin' in df.columns:
                final_wr = df['RollingWin'].iloc[-1]
                records.append({
                    'Model': POSTER_SHORT[model_name],
                    'Final Win Rate': final_wr,
                    'color': POSTER_COLORS[model_name],
                    'model_key': model_name,
                })

    plot_df = pd.DataFrame(records)
    if plot_df.empty:
        print("  No data found.")
        return

    model_order = [POSTER_SHORT[m] for m in MODEL_NAMES]
    palette = {POSTER_SHORT[m]: POSTER_COLORS[m] for m in MODEL_NAMES}

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.violinplot(
        data=plot_df, x='Model', y='Final Win Rate', hue='Model', order=model_order,
        palette=palette, inner=None, alpha=0.3, ax=ax, cut=0, legend=False,
    )
    sns.stripplot(
        data=plot_df, x='Model', y='Final Win Rate', hue='Model', order=model_order,
        palette=palette, size=7, jitter=0.15, alpha=0.8, ax=ax,
        edgecolor='white', linewidth=0.8, legend=False,
    )

    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('')
    ax.set_ylabel('Final Rolling Win Rate', fontsize=14)
    ax.set_title(f'Distribution of Final Win Rates — {combo_id} (10 runs each)', fontsize=16, color=TULANE['dark'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=11)
    ax.set_ylim(0, 0.72)
    ax.grid(axis='y', alpha=0.3)

    save_fig(fig, f'{combo_id}_violin.png', dpi)


# ═══════════════════════════════════════════════
#  4. INTERACTION PLOT
# ═══════════════════════════════════════════════
def plot_interaction(dpi=300):
    print("Generating interaction plot...")
    df = pd.read_csv(SUMMARY_CSV)

    means = {
        'Flat+Zero': df['final_Flat_ZeroInit'].mean(),
        'Flat+Smart': df['final_Flat_SmartInit'].mean(),
        'Hier+Zero': df['final_Hier_ZeroInit'].mean(),
        'Hier+Smart': df['final_Hier_SmartInit'].mean(),
        'Flat+Zero+Fix': df['final_Flat_ZeroInit_FixedEps'].mean(),
        'Flat+Smart+Fix': df['final_Flat_SmartInit_FixedEps'].mean(),
        'Hier+Zero+Fix': df['final_Hier_ZeroInit_FixedEps'].mean(),
        'Hier+Smart+Fix': df['final_Hier_SmartInit_FixedEps'].mean(),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    # Panel 1: Architecture × Initialization (averaged over epsilon)
    x_labels = ['Zero Init', 'Smart Init']
    flat_vals = [
        np.mean([means['Flat+Zero'], means['Flat+Zero+Fix']]),
        np.mean([means['Flat+Smart'], means['Flat+Smart+Fix']]),
    ]
    hier_vals = [
        np.mean([means['Hier+Zero'], means['Hier+Zero+Fix']]),
        np.mean([means['Hier+Smart'], means['Hier+Smart+Fix']]),
    ]

    ax1.plot(x_labels, flat_vals, 'o-', color=_BASE['flat_smart'], linewidth=3, markersize=12, label='Flat', markeredgecolor='white', markeredgewidth=2)
    ax1.plot(x_labels, hier_vals, 's-', color=_BASE['hier_smart'], linewidth=3, markersize=12, label='Hierarchical', markeredgecolor='white', markeredgewidth=2)

    for i, (fv, hv) in enumerate(zip(flat_vals, hier_vals)):
        ax1.annotate(f'{fv:.1%}', (i, fv), textcoords='offset points', xytext=(15, -5), fontsize=11, color=_BASE['flat_smart'], fontweight='bold')
        ax1.annotate(f'{hv:.1%}', (i, hv), textcoords='offset points', xytext=(15, 5), fontsize=11, color=_BASE['hier_smart'], fontweight='bold')

    ax1.set_title('Architecture × Initialization', fontsize=14, color=TULANE['dark'])
    ax1.set_ylabel('Mean Final Win Rate', fontsize=13)
    ax1.legend(fontsize=12, loc='upper left')
    ax1.set_ylim(0, 0.60)
    ax1.axhline(0.5, color='gray', linestyle=':', alpha=0.4)
    ax1.grid(axis='y', alpha=0.3)

    # Panel 2: Initialization × Epsilon (averaged over architecture)
    decay_vals = [
        np.mean([means['Flat+Zero'], means['Hier+Zero']]),
        np.mean([means['Flat+Smart'], means['Hier+Smart']]),
    ]
    fixed_vals = [
        np.mean([means['Flat+Zero+Fix'], means['Hier+Zero+Fix']]),
        np.mean([means['Flat+Smart+Fix'], means['Hier+Smart+Fix']]),
    ]

    ax2.plot(x_labels, decay_vals, 'o-', color=TULANE['blue'], linewidth=3, markersize=12, label='Decay ε', markeredgecolor='white', markeredgewidth=2)
    ax2.plot(x_labels, fixed_vals, 's--', color=TULANE['kelly'], linewidth=3, markersize=12, label='Fixed ε', markeredgecolor='white', markeredgewidth=2)

    for i, (dv, xv) in enumerate(zip(decay_vals, fixed_vals)):
        ax2.annotate(f'{dv:.1%}', (i, dv), textcoords='offset points', xytext=(15, -5), fontsize=11, color=TULANE['blue'], fontweight='bold')
        ax2.annotate(f'{xv:.1%}', (i, xv), textcoords='offset points', xytext=(15, 5), fontsize=11, color=TULANE['kelly'], fontweight='bold')

    ax2.set_title('Initialization × Epsilon Schedule', fontsize=14, color=TULANE['dark'])
    ax2.legend(fontsize=12, loc='upper left')
    ax2.axhline(0.5, color='gray', linestyle=':', alpha=0.4)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Interaction Effects — V5 Initialization Study', fontsize=17, fontweight='bold', color=TULANE['dark'], y=1.02)
    plt.tight_layout()
    save_fig(fig, 'interaction_plot.png', dpi)


# ═══════════════════════════════════════════════
#  5. EFFECT SIZES (HORIZONTAL BAR)
# ═══════════════════════════════════════════════
def plot_effect_sizes(dpi=300):
    print("Generating effect sizes...")
    df = pd.read_csv(SUMMARY_CSV)

    all_final = {}
    col_map = {
        'model_1_flat_zero': 'final_Flat_ZeroInit',
        'model_2_flat_smart': 'final_Flat_SmartInit',
        'model_3_hier_zero': 'final_Hier_ZeroInit',
        'model_4_hier_smart': 'final_Hier_SmartInit',
        'model_5_flat_zero_fixed_eps': 'final_Flat_ZeroInit_FixedEps',
        'model_6_flat_smart_fixed_eps': 'final_Flat_SmartInit_FixedEps',
        'model_7_hier_zero_fixed_eps': 'final_Hier_ZeroInit_FixedEps',
        'model_8_hier_smart_fixed_eps': 'final_Hier_SmartInit_FixedEps',
    }
    for model_name, col in col_map.items():
        all_final[model_name] = df[col].mean()

    # Compute marginal effects
    smart_flat = np.mean([all_final['model_2_flat_smart'], all_final['model_6_flat_smart_fixed_eps']]) - \
                 np.mean([all_final['model_1_flat_zero'], all_final['model_5_flat_zero_fixed_eps']])

    smart_hier = np.mean([all_final['model_4_hier_smart'], all_final['model_8_hier_smart_fixed_eps']]) - \
                 np.mean([all_final['model_3_hier_zero'], all_final['model_7_hier_zero_fixed_eps']])

    # Lambda effect (0.7 vs 0.9) — from summary.csv row averages
    lam07 = df[df['lam'] == 0.7]['avg_final'].mean()
    lam09 = df[df['lam'] == 0.9]['avg_final'].mean()
    lam_effect = lam07 - lam09

    # Alpha effect (0.1 vs 0.2)
    a01 = df[df['alpha'] == 0.1]['avg_final'].mean()
    a02 = df[df['alpha'] == 0.2]['avg_final'].mean()
    alpha_effect = a01 - a02

    # Hierarchy effect
    flat_mean = np.mean([all_final['model_1_flat_zero'], all_final['model_2_flat_smart'],
                         all_final['model_5_flat_zero_fixed_eps'], all_final['model_6_flat_smart_fixed_eps']])
    hier_mean = np.mean([all_final['model_3_hier_zero'], all_final['model_4_hier_smart'],
                         all_final['model_7_hier_zero_fixed_eps'], all_final['model_8_hier_smart_fixed_eps']])
    hier_effect = hier_mean - flat_mean

    # Fixed epsilon effect
    decay_mean = np.mean([all_final['model_1_flat_zero'], all_final['model_2_flat_smart'],
                          all_final['model_3_hier_zero'], all_final['model_4_hier_smart']])
    fixed_mean = np.mean([all_final['model_5_flat_zero_fixed_eps'], all_final['model_6_flat_smart_fixed_eps'],
                          all_final['model_7_hier_zero_fixed_eps'], all_final['model_8_hier_smart_fixed_eps']])
    eps_effect = fixed_mean - decay_mean

    effects = [
        ('Smart Init\n(Flat models)', smart_flat, _BASE['flat_smart']),
        ('Smart Init\n(Hier models)', smart_hier, _BASE['hier_smart']),
        ('λ = 0.7 vs 0.9', lam_effect, TULANE['blue']),
        ('α = 0.1 vs 0.2', alpha_effect, TULANE['kelly']),
        ('Hierarchy\n(vs Flat)', hier_effect, _BASE['hier_zero']),
        ('Fixed ε\n(vs Decay)', eps_effect, TULANE['dark']),
    ]

    effects.sort(key=lambda e: e[1], reverse=True)
    labels, values, colors = zip(*effects)

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.barh(range(len(labels)), [v * 100 for v in values], color=colors, height=0.6, edgecolor='white', linewidth=1.5)

    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'+{v * 100:.1f}pp', va='center', fontsize=12, fontweight='bold', color=TULANE['dark'])

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel('Effect Size (percentage points)', fontsize=13)
    ax.set_title('Factor Effect Sizes on Win Rate', fontsize=16, color=TULANE['dark'], pad=12)
    ax.axvline(0, color='gray', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    ax.set_xlim(-2, max(v * 100 for _, v, _ in effects) + 8)

    save_fig(fig, 'effect_sizes.png', dpi)


# ═══════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════
PLOT_FUNCTIONS = {
    'heatmap': plot_heatmap,
    'bestconfig': plot_best_config,
    'curves': plot_learning_curves,
    'violin': plot_violin,
    'interaction': plot_interaction,
    'effects': plot_effect_sizes,
}


def main():
    parser = argparse.ArgumentParser(description="V5 Poster Figures")
    parser.add_argument('--plot', type=str, default=None,
                        choices=list(PLOT_FUNCTIONS.keys()),
                        help='Generate a specific plot (default: all)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Output DPI (default: 300)')
    parser.add_argument('--combo', type=str, default='hp_001',
                        help='HP combo for curves/violin (default: hp_001)')
    args = parser.parse_args()

    setup_style()

    if args.plot:
        fn = PLOT_FUNCTIONS[args.plot]
        if args.plot in ('curves', 'violin', 'bestconfig'):
            fn(dpi=args.dpi, combo_id=args.combo)
        else:
            fn(dpi=args.dpi)
    else:
        print(f"Generating all poster figures (dpi={args.dpi})...")
        plot_heatmap(dpi=args.dpi)
        plot_learning_curves(dpi=args.dpi, combo_id=args.combo)
        plot_violin(dpi=args.dpi, combo_id=args.combo)
        plot_interaction(dpi=args.dpi)
        plot_effect_sizes(dpi=args.dpi)
        print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == '__main__':
    main()
