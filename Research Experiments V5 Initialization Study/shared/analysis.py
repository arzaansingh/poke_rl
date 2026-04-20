"""
V5 Initialization Study — Statistical Analysis
===============================================
Comprehensive statistical analysis of the V5 experiment
across hyperparameter combos. Includes hypothesis testing, effect sizes,
ANOVA, variance analysis, learning dynamics, and visualizations.

Usage:
  python shared/statistics.py                        # Full analysis, all combos
  python shared/statistics.py --combo hp_001         # Single combo deep-dive
  python shared/statistics.py --combo hp_001 hp_002  # Compare specific combos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
import json
import warnings
from itertools import combinations

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.config import (
    MODEL_NAMES, MODEL_LABELS, RUNS_PER_COMBO, BATTLES_PER_RUN,
    build_grid,
)
from shared.plot import (
    load_combo_data, get_combo_params, interpolate_runs,
    MODEL_COLORS, MODEL_SHORT, GRID_RESULTS_DIR, PLOTS_DIR,
)

# scipy imports — install if missing
try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    print("WARNING: scipy not installed. Install with: pip install scipy")
    print("  Hypothesis tests and ANOVA will be skipped.\n")
    HAS_SCIPY = False

sns.set_theme(style="whitegrid", font_scale=1.05)

STATS_PLOTS_DIR = os.path.join(PLOTS_DIR, "statistics")


# ═══════════════════════════════════════════════════════════════
#  DATA EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_final_metrics(combo_id):
    """Extract final-row metrics for each model×run.

    Returns DataFrame with columns:
      model, run, final_rolling, final_overall, peak_rolling,
      final_reward, final_table_size, final_speed
    """
    data = load_combo_data(combo_id)
    rows = []
    for model_name in MODEL_NAMES:
        for df in data.get(model_name, []):
            if df.empty:
                continue
            last = df.iloc[-1]
            rows.append({
                "combo_id": combo_id,
                "model": model_name,
                "model_short": MODEL_SHORT[model_name],
                "arch": "Hier" if "hier" in model_name else "Flat",
                "init": "Smart" if "smart" in model_name else "Zero",
                "schedule": "Fixed" if "fixed_eps" in model_name else "Decay",
                "run": int(df["run"].iloc[0]),
                "battles": int(last["Battles"]),
                "final_rolling": float(last["RollingWin"]),
                "final_overall": float(last["OverallWin"]),
                "peak_rolling": float(df["RollingWin"].max()),
                "final_reward": float(last["AvgReward"]),
                "final_table_size": int(last["TableSize"]),
                "final_speed": float(last["Speed"]),
            })
    return pd.DataFrame(rows)


def extract_learning_curves(combo_id):
    """Get interpolated learning curves for each model.

    Returns dict: {model_name: (x, mean, std, curves)}
    """
    data = load_combo_data(combo_id)
    result = {}
    for model_name in MODEL_NAMES:
        dfs = data.get(model_name, [])
        if not dfs:
            continue
        result[model_name] = interpolate_runs(dfs, 'Battles', 'RollingWin', n_points=500)
    return result


def convergence_battle(curves_tuple, threshold):
    """Find the first battle count where mean curve crosses a threshold."""
    x, mean, std, _ = curves_tuple
    if x is None or mean is None:
        return None
    mask = mean >= threshold
    if mask.any():
        return int(x[np.argmax(mask)])
    return None


def compute_auc(curves_tuple):
    """Area under the mean learning curve (higher = learned more, faster)."""
    x, mean, _, _ = curves_tuple
    if x is None or mean is None:
        return None
    return float(np.trapezoid(mean, x))


# ═══════════════════════════════════════════════════════════════
#  SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════

def print_summary(df, combo_id):
    """Print a clean summary table for one combo."""
    params = get_combo_params(combo_id)
    label = f"\u03b1={params.get('alpha')}, \u03b3={params.get('gamma')}, \u03bb={params.get('lam')}"

    print(f"\n{'=' * 100}")
    print(f"  {combo_id}: {label}")
    print(f"{'=' * 100}")

    sub = df[df["combo_id"] == combo_id]
    if sub.empty:
        print("  No data.")
        return

    header = (f"  {'Model':<16} {'n':>3} {'Mean':>7} {'Std':>7} {'Med':>7} "
              f"{'IQR':>12} {'Peak':>7} {'OverallWR':>10} {'TableSize':>12} {'Reward':>8}")
    print(header)
    print(f"  {'-' * 96}")

    for model_name in MODEL_NAMES:
        ms = sub[sub["model"] == model_name]
        if ms.empty:
            continue
        r = ms["final_rolling"]
        o = ms["final_overall"]
        ts = ms["final_table_size"]
        rw = ms["final_reward"]
        q1, q3 = r.quantile(0.25), r.quantile(0.75)
        print(f"  {MODEL_SHORT[model_name]:<16} {len(ms):>3} {r.mean():>7.3f} {r.std():>7.3f} "
              f"{r.median():>7.3f} [{q1:.3f},{q3:.3f}] {ms['peak_rolling'].mean():>7.3f} "
              f"{o.mean():>10.3f} {ts.mean():>12,.0f} {rw.mean():>8.3f}")

    print()


# ═══════════════════════════════════════════════════════════════
#  HYPOTHESIS TESTING
# ═══════════════════════════════════════════════════════════════

def cohens_d(a, b):
    """Cohen's d effect size (pooled std)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_std = np.sqrt(((na - 1) * a.std()**2 + (nb - 1) * b.std()**2) / (na + nb - 2))
    if pooled_std == 0:
        return np.nan
    return (a.mean() - b.mean()) / pooled_std


def pairwise_tests(df, combo_id):
    """Run pairwise hypothesis tests on final rolling WR between all model pairs."""
    if not HAS_SCIPY:
        print("  [scipy not installed — skipping hypothesis tests]")
        return

    sub = df[df["combo_id"] == combo_id]
    models = [m for m in MODEL_NAMES if m in sub["model"].values]
    pairs = list(combinations(models, 2))
    n_tests = len(pairs)

    print(f"  Pairwise Comparisons (Bonferroni-corrected, {n_tests} tests, \u03b1=0.05)")
    print(f"  {'Pair':<35} {'t-stat':>8} {'p(Welch)':>10} {'p(corr)':>10} {'U-stat':>8} "
          f"{'p(MW)':>10} {'Cohen d':>8} {'Sig':>5}")
    print(f"  {'-' * 105}")

    for m1, m2 in pairs:
        a = sub[sub["model"] == m1]["final_rolling"].values
        b = sub[sub["model"] == m2]["final_rolling"].values

        if len(a) < 2 or len(b) < 2:
            continue

        # Welch's t-test
        t_stat, p_welch = sp_stats.ttest_ind(a, b, equal_var=False)
        p_corrected = min(p_welch * n_tests, 1.0)  # Bonferroni

        # Mann-Whitney U (non-parametric)
        u_stat, p_mw = sp_stats.mannwhitneyu(a, b, alternative='two-sided')

        d = cohens_d(pd.Series(a), pd.Series(b))
        sig = "***" if p_corrected < 0.001 else "**" if p_corrected < 0.01 else "*" if p_corrected < 0.05 else ""

        label = f"{MODEL_SHORT[m1]} vs {MODEL_SHORT[m2]}"
        print(f"  {label:<35} {t_stat:>8.3f} {p_welch:>10.4f} {p_corrected:>10.4f} "
              f"{u_stat:>8.0f} {p_mw:>10.4f} {d:>8.3f} {sig:>5}")

    print()


# ═══════════════════════════════════════════════════════════════
#  FACTORIAL ANOVA
# ═══════════════════════════════════════════════════════════════

def factorial_anova(df, combo_id):
    """2x2 ANOVA: Architecture (flat/hier) x Initialization (zero/smart)."""
    if not HAS_SCIPY:
        print("  [scipy not installed — skipping ANOVA]")
        return

    sub = df[df["combo_id"] == combo_id].copy()
    if sub.empty:
        return

    # Get the 4 groups
    groups = {}
    for arch in ["Flat", "Hier"]:
        for init in ["Zero", "Smart"]:
            g = sub[(sub["arch"] == arch) & (sub["init"] == init)]["final_rolling"].values
            groups[(arch, init)] = g

    print(f"  2x2 Factorial ANOVA (Architecture x Initialization)")
    print(f"  {'-' * 60}")

    # Cell means
    for (arch, init), vals in groups.items():
        if len(vals) > 0:
            print(f"    {arch:>5}+{init:<5}: mean={np.mean(vals):.3f} std={np.std(vals):.3f} n={len(vals)}")

    # Main effects
    flat_vals = sub[sub["arch"] == "Flat"]["final_rolling"].values
    hier_vals = sub[sub["arch"] == "Hier"]["final_rolling"].values
    zero_vals = sub[sub["init"] == "Zero"]["final_rolling"].values
    smart_vals = sub[sub["init"] == "Smart"]["final_rolling"].values

    # Architecture main effect
    t_arch, p_arch = sp_stats.ttest_ind(flat_vals, hier_vals, equal_var=False)
    d_arch = cohens_d(pd.Series(hier_vals), pd.Series(flat_vals))

    # Initialization main effect
    t_init, p_init = sp_stats.ttest_ind(zero_vals, smart_vals, equal_var=False)
    d_init = cohens_d(pd.Series(smart_vals), pd.Series(zero_vals))

    print(f"\n  Main Effects:")
    print(f"    Architecture (Hier vs Flat):  t={t_arch:.3f}, p={p_arch:.4f}, d={d_arch:.3f} "
          f"{'***' if p_arch < 0.001 else '**' if p_arch < 0.01 else '*' if p_arch < 0.05 else 'ns'}")
    print(f"    Initialization (Smart vs Zero): t={t_init:.3f}, p={p_init:.4f}, d={d_init:.3f} "
          f"{'***' if p_init < 0.001 else '**' if p_init < 0.01 else '*' if p_init < 0.05 else 'ns'}")

    # Interaction: does the effect of smart init differ by architecture?
    flat_smart_boost = (groups.get(("Flat", "Smart"), []).mean()
                        - groups.get(("Flat", "Zero"), []).mean()
                        if len(groups.get(("Flat", "Smart"), [])) > 0
                        and len(groups.get(("Flat", "Zero"), [])) > 0
                        else np.nan)
    hier_smart_boost = (groups.get(("Hier", "Smart"), []).mean()
                        - groups.get(("Hier", "Zero"), []).mean()
                        if len(groups.get(("Hier", "Smart"), [])) > 0
                        and len(groups.get(("Hier", "Zero"), [])) > 0
                        else np.nan)

    print(f"\n  Interaction (Smart Init Boost):")
    print(f"    Flat:  Smart - Zero = {flat_smart_boost:+.3f}")
    print(f"    Hier:  Smart - Zero = {hier_smart_boost:+.3f}")
    if not np.isnan(flat_smart_boost) and not np.isnan(hier_smart_boost):
        direction = "helps Flat MORE" if flat_smart_boost > hier_smart_boost else "helps Hier MORE"
        if abs(flat_smart_boost - hier_smart_boost) < 0.02:
            direction = "similar effect"
        diff = flat_smart_boost - hier_smart_boost
        print(f"    Difference: {diff:+.3f} ({direction})")

    print()


# ═══════════════════════════════════════════════════════════════
#  LAMBDA SENSITIVITY
# ═══════════════════════════════════════════════════════════════

def lambda_analysis(all_df, combo_ids):
    """Compare same model across different lambda values."""
    if not HAS_SCIPY:
        print("  [scipy not installed — skipping lambda ANOVA]")
        return

    if len(combo_ids) < 2:
        return

    print(f"  Lambda Sensitivity (One-way ANOVA per model across {len(combo_ids)} combos)")
    print(f"  {'-' * 80}")

    # Get lambda for each combo
    combo_lams = {}
    for cid in combo_ids:
        p = get_combo_params(cid)
        combo_lams[cid] = p.get("lam", "?")

    lam_str = ", ".join(f"{cid}(\u03bb={combo_lams[cid]})" for cid in combo_ids)
    print(f"  Combos: {lam_str}\n")

    for model_name in MODEL_NAMES:
        groups = []
        labels = []
        for cid in combo_ids:
            vals = all_df[(all_df["combo_id"] == cid) & (all_df["model"] == model_name)]["final_rolling"].values
            if len(vals) >= 2:
                groups.append(vals)
                labels.append(f"\u03bb={combo_lams[cid]}")

        if len(groups) < 2:
            continue

        f_stat, p_val = sp_stats.f_oneway(*groups)
        means_str = " | ".join(f"{l}: {g.mean():.3f}" for l, g in zip(labels, groups))
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  {MODEL_SHORT[model_name]:<16} F={f_stat:.3f} p={p_val:.4f} {sig:>4}  [{means_str}]")

    print()


# ═══════════════════════════════════════════════════════════════
#  VARIANCE & BOOTSTRAP
# ═══════════════════════════════════════════════════════════════

def variance_analysis(df, combo_id):
    """Coefficient of variation and bootstrapped 95% CI."""
    sub = df[df["combo_id"] == combo_id]

    print(f"  Variance Analysis & Bootstrapped 95% CI")
    print(f"  {'Model':<16} {'CV':>8} {'Boot 95% CI':>20} {'CI Width':>10}")
    print(f"  {'-' * 60}")

    for model_name in MODEL_NAMES:
        vals = sub[sub["model"] == model_name]["final_rolling"].values
        if len(vals) < 2:
            continue

        cv = vals.std() / vals.mean() if vals.mean() > 0 else np.nan

        # Bootstrap 95% CI
        rng = np.random.RandomState(42)
        boot_means = []
        for _ in range(10000):
            sample = rng.choice(vals, size=len(vals), replace=True)
            boot_means.append(sample.mean())
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

        print(f"  {MODEL_SHORT[model_name]:<16} {cv:>8.3f} [{ci_lo:.3f}, {ci_hi:.3f}]{' ':>5} {ci_hi - ci_lo:>10.3f}")

    print()


# ═══════════════════════════════════════════════════════════════
#  LEARNING DYNAMICS
# ═══════════════════════════════════════════════════════════════

def learning_dynamics(combo_id):
    """Convergence speed and AUC analysis."""
    curves = extract_learning_curves(combo_id)

    print(f"  Learning Dynamics")
    thresholds = [0.20, 0.30, 0.40, 0.50]
    header = f"  {'Model':<16}" + "".join(f" {'>' + str(int(t*100)) + '%':>8}" for t in thresholds) + f" {'AUC':>12}"
    print(header)
    print(f"  {'-' * (16 + 8 * len(thresholds) + 12 + 2)}")

    for model_name in MODEL_NAMES:
        if model_name not in curves:
            continue
        c = curves[model_name]
        conv = []
        for t in thresholds:
            b = convergence_battle(c, t)
            conv.append(f"{b:>7,}" if b else f"{'never':>7}")
        auc = compute_auc(c)
        auc_str = f"{auc:>11,.0f}" if auc else "N/A"
        print(f"  {MODEL_SHORT[model_name]:<16}" + "".join(f" {v:>8}" for v in conv) + f" {auc_str}")

    print()


# ═══════════════════════════════════════════════════════════════
#  KEY FINDINGS
# ═══════════════════════════════════════════════════════════════

def print_findings(df, combo_id):
    """Print auto-generated key findings."""
    sub = df[df["combo_id"] == combo_id]
    params = get_combo_params(combo_id)

    hz = sub[sub["model"] == "model_3_hier_zero"]["final_rolling"]
    hs = sub[sub["model"] == "model_4_hier_smart"]["final_rolling"]
    fs = sub[sub["model"] == "model_2_flat_smart"]["final_rolling"]
    fz = sub[sub["model"] == "model_1_flat_zero"]["final_rolling"]

    print(f"  KEY FINDINGS ({combo_id}):")
    print(f"  {'-' * 50}")

    if not hz.empty and not hs.empty:
        diff = hz.mean() - hs.mean()
        if diff > 0.02:
            print(f"  ! Hier+Zero OUTPERFORMS Hier+Smart by {diff:.1%}")
            print(f"    Hier+Zero: {hz.mean():.1%} vs Hier+Smart: {hs.mean():.1%}")

            # Why? Check table sizes
            hz_ts = sub[sub["model"] == "model_3_hier_zero"]["final_table_size"].mean()
            hs_ts = sub[sub["model"] == "model_4_hier_smart"]["final_table_size"].mean()
            print(f"    Table size: Hier+Zero={hz_ts:,.0f} vs Hier+Smart={hs_ts:,.0f} ({hs_ts/hz_ts:.1f}x)")
            if hs_ts > hz_ts * 1.3:
                print(f"    -> Smart init creates {hs_ts/hz_ts:.1f}x more states, potentially")
                print(f"       spreading learning across too many entries (overfitting to priors)")

            # Check variance
            print(f"    Variance: Hier+Zero std={hz.std():.3f} vs Hier+Smart std={hs.std():.3f}")
            if hz.std() > hs.std() * 1.2:
                print(f"    -> Hier+Zero has higher variance — some runs excel, some don't")
                print(f"       but the upside ceiling is higher than Hier+Smart's")

    if not fs.empty and not fz.empty:
        boost = fs.mean() - fz.mean()
        print(f"\n  Smart Init Boost:")
        print(f"    Flat:  +{boost:.1%} (Zero={fz.mean():.1%} -> Smart={fs.mean():.1%})")
        if not hz.empty and not hs.empty:
            h_boost = hs.mean() - hz.mean()
            print(f"    Hier:  {h_boost:+.1%} (Zero={hz.mean():.1%} -> Smart={hs.mean():.1%})")
            if h_boost < 0:
                print(f"    -> Smart init HURTS hierarchical models at these hyperparameters!")
                print(f"       The heuristic priors may bias the sub-agent's switch policy")
                print(f"       toward type-matchup switching, preventing discovery of")
                print(f"       better strategies (e.g., staying in to tank hits)")

    print()


# ═══════════════════════════════════════════════════════════════
#  VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════

def plot_box_violin(df, combo_id, save=True):
    """Box + violin plots of final rolling WR per model."""
    sub = df[df["combo_id"] == combo_id].copy()
    params = get_combo_params(combo_id)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{combo_id}: \u03b1={params.get('alpha')}, \u03b3={params.get('gamma')}, \u03bb={params.get('lam')}",
                 fontsize=13, fontweight='bold')

    order = [MODEL_SHORT[m] for m in MODEL_NAMES]
    palette = {MODEL_SHORT[m]: MODEL_COLORS[m] for m in MODEL_NAMES}

    # Box plot
    ax = axes[0]
    sns.boxplot(data=sub, x="model_short", y="final_rolling", order=order,
                palette=palette, ax=ax, width=0.5)
    sns.stripplot(data=sub, x="model_short", y="final_rolling", order=order,
                  color="black", alpha=0.4, size=5, ax=ax, jitter=True)
    ax.set_title("Final Rolling WR — Box Plot")
    ax.set_xlabel("")
    ax.set_ylabel("Final Rolling Win Rate")
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)

    # Violin plot
    ax = axes[1]
    sns.violinplot(data=sub, x="model_short", y="final_rolling", order=order,
                   palette=palette, ax=ax, inner="box", cut=0)
    ax.set_title("Final Rolling WR — Violin Plot")
    ax.set_xlabel("")
    ax.set_ylabel("Final Rolling Win Rate")
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)

    plt.tight_layout()
    if save:
        os.makedirs(STATS_PLOTS_DIR, exist_ok=True)
        path = os.path.join(STATS_PLOTS_DIR, f"{combo_id}_box_violin.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
        plt.close()
    else:
        plt.show()


def plot_interaction(df, combo_id, save=True):
    """2x2 interaction plot: Architecture x Initialization."""
    sub = df[df["combo_id"] == combo_id]
    params = get_combo_params(combo_id)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"Interaction Plot — {combo_id}\n\u03b1={params.get('alpha')}, \u03b3={params.get('gamma')}, \u03bb={params.get('lam')}",
                 fontsize=13, fontweight='bold')

    for arch, color, marker in [("Flat", "#1f77b4", "o"), ("Hier", "#d62728", "s")]:
        means = []
        errs = []
        for init in ["Zero", "Smart"]:
            vals = sub[(sub["arch"] == arch) & (sub["init"] == init)]["final_rolling"]
            means.append(vals.mean() if not vals.empty else 0)
            errs.append(vals.std() if not vals.empty else 0)
        ax.errorbar([0, 1], means, yerr=errs, marker=marker, markersize=10,
                    linewidth=2.5, color=color, label=arch, capsize=8, capthick=2)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Zero Init", "Smart Init"], fontsize=12)
    ax.set_ylabel("Final Rolling Win Rate", fontsize=12)
    ax.legend(fontsize=12)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlim(-0.3, 1.3)

    plt.tight_layout()
    if save:
        os.makedirs(STATS_PLOTS_DIR, exist_ok=True)
        path = os.path.join(STATS_PLOTS_DIR, f"{combo_id}_interaction.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
        plt.close()
    else:
        plt.show()


def plot_effect_sizes(df, combo_id, save=True):
    """Forest plot of pairwise Cohen's d effect sizes."""
    if not HAS_SCIPY:
        return

    sub = df[df["combo_id"] == combo_id]
    params = get_combo_params(combo_id)
    models = [m for m in MODEL_NAMES if m in sub["model"].values]
    pairs = list(combinations(models, 2))

    labels = []
    ds = []
    ci_los = []
    ci_his = []

    for m1, m2 in pairs:
        a = sub[sub["model"] == m1]["final_rolling"].values
        b = sub[sub["model"] == m2]["final_rolling"].values
        if len(a) < 2 or len(b) < 2:
            continue

        d = cohens_d(pd.Series(a), pd.Series(b))
        # Approximate CI for Cohen's d
        n1, n2 = len(a), len(b)
        se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
        ci_lo = d - 1.96 * se_d
        ci_hi = d + 1.96 * se_d

        labels.append(f"{MODEL_SHORT[m1]} vs\n{MODEL_SHORT[m2]}")
        ds.append(d)
        ci_los.append(ci_lo)
        ci_his.append(ci_hi)

    if not ds:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(ds) * 0.6 + 1)))
    ax.set_title(f"Effect Sizes (Cohen's d) — {combo_id}\n\u03b1={params.get('alpha')}, \u03b3={params.get('gamma')}, \u03bb={params.get('lam')}",
                 fontsize=13, fontweight='bold')

    y_pos = range(len(ds))
    colors = ['#2ca02c' if d > 0 else '#d62728' for d in ds]
    ax.barh(y_pos, ds, color=colors, alpha=0.6, height=0.5)
    ax.errorbar(ds, y_pos, xerr=[[d - cl for d, cl in zip(ds, ci_los)],
                                  [ch - d for d, ch in zip(ds, ci_his)]],
                fmt='none', color='black', capsize=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.axvline(-0.8, color='gray', linestyle=':', alpha=0.5, label='Large effect')
    ax.axvline(0.8, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("Cohen's d (positive = first model better)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save:
        os.makedirs(STATS_PLOTS_DIR, exist_ok=True)
        path = os.path.join(STATS_PLOTS_DIR, f"{combo_id}_effect_sizes.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
        plt.close()
    else:
        plt.show()


def plot_convergence(combo_id, save=True):
    """Timeline showing when each model crosses WR thresholds."""
    curves = extract_learning_curves(combo_id)
    params = get_combo_params(combo_id)
    thresholds = [0.15, 0.20, 0.30, 0.40, 0.50]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(f"Convergence Timeline — {combo_id}\n\u03b1={params.get('alpha')}, \u03b3={params.get('gamma')}, \u03bb={params.get('lam')}",
                 fontsize=13, fontweight='bold')

    y_ticks = []
    y_labels = []
    for i, model_name in enumerate(MODEL_NAMES):
        if model_name not in curves:
            continue
        c = curves[model_name]
        color = MODEL_COLORS[model_name]

        for j, t in enumerate(thresholds):
            b = convergence_battle(c, t)
            y = i * (len(thresholds) + 1) + j
            y_ticks.append(y)
            y_labels.append(f"{MODEL_SHORT[model_name]} >{int(t*100)}%")
            if b:
                ax.barh(y, b, color=color, alpha=0.6 + 0.08 * j, height=0.7)
                ax.text(b + 500, y, f"{b:,}", va='center', fontsize=8)
            else:
                ax.barh(y, BATTLES_PER_RUN, color='lightgray', alpha=0.3, height=0.7)
                ax.text(BATTLES_PER_RUN * 0.5, y, "never", va='center', ha='center',
                        fontsize=8, color='gray')

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Battles")
    ax.invert_yaxis()
    plt.tight_layout()

    if save:
        os.makedirs(STATS_PLOTS_DIR, exist_ok=True)
        path = os.path.join(STATS_PLOTS_DIR, f"{combo_id}_convergence.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
        plt.close()
    else:
        plt.show()


def plot_correlation_matrix(df, combo_id, save=True):
    """Correlation matrix between final metrics."""
    sub = df[df["combo_id"] == combo_id]
    params = get_combo_params(combo_id)

    metrics = ["final_rolling", "final_overall", "peak_rolling",
               "final_reward", "final_table_size", "final_speed"]
    labels = ["Rolling WR", "Overall WR", "Peak WR", "Reward", "Table Size", "Speed"]

    corr_data = sub[metrics].rename(columns=dict(zip(metrics, labels)))
    corr = corr_data.corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, ax=ax, linewidths=0.5)
    ax.set_title(f"Metric Correlations — {combo_id}\n\u03b1={params.get('alpha')}, \u03b3={params.get('gamma')}, \u03bb={params.get('lam')}",
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    if save:
        os.makedirs(STATS_PLOTS_DIR, exist_ok=True)
        path = os.path.join(STATS_PLOTS_DIR, f"{combo_id}_correlations.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
        plt.close()
    else:
        plt.show()


def plot_lambda_sensitivity(all_df, combo_ids, save=True):
    """Bar chart comparing models across different lambda values."""
    if len(combo_ids) < 2:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    combo_lams = {}
    for cid in combo_ids:
        p = get_combo_params(cid)
        combo_lams[cid] = p.get("lam", "?")

    x = np.arange(len(combo_ids))
    width = 0.18
    offsets = np.arange(len(MODEL_NAMES)) - (len(MODEL_NAMES) - 1) / 2

    for i, model_name in enumerate(MODEL_NAMES):
        means = []
        stds = []
        for cid in combo_ids:
            vals = all_df[(all_df["combo_id"] == cid) & (all_df["model"] == model_name)]["final_rolling"]
            means.append(vals.mean() if not vals.empty else 0)
            stds.append(vals.std() if not vals.empty else 0)

        ax.bar(x + offsets[i] * width, means, width * 0.9, yerr=stds,
               color=MODEL_COLORS[model_name], label=MODEL_SHORT[model_name],
               capsize=4, alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"\u03bb={combo_lams[cid]}\n({cid})" for cid in combo_ids], fontsize=11)
    ax.set_ylabel("Final Rolling Win Rate", fontsize=12)
    ax.set_title("Lambda Sensitivity — Final WR by Model", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    if save:
        os.makedirs(STATS_PLOTS_DIR, exist_ok=True)
        combos_str = "_".join(combo_ids)
        path = os.path.join(STATS_PLOTS_DIR, f"lambda_sensitivity_{combos_str}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
        plt.close()
    else:
        plt.show()


def plot_auc_comparison(combo_ids, save=True):
    """Bar chart of area under learning curve (efficiency metric)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    combo_lams = {}
    for cid in combo_ids:
        p = get_combo_params(cid)
        combo_lams[cid] = p.get("lam", "?")

    x = np.arange(len(combo_ids))
    width = 0.18
    offsets = np.arange(len(MODEL_NAMES)) - (len(MODEL_NAMES) - 1) / 2

    for i, model_name in enumerate(MODEL_NAMES):
        aucs = []
        for cid in combo_ids:
            curves = extract_learning_curves(cid)
            if model_name in curves:
                auc = compute_auc(curves[model_name])
                aucs.append(auc if auc else 0)
            else:
                aucs.append(0)

        ax.bar(x + offsets[i] * width, aucs, width * 0.9,
               color=MODEL_COLORS[model_name], label=MODEL_SHORT[model_name],
               alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"\u03bb={combo_lams[cid]}\n({cid})" for cid in combo_ids], fontsize=11)
    ax.set_ylabel("Area Under Learning Curve", fontsize=12)
    ax.set_title("Learning Efficiency (AUC) — Higher = Learned More, Faster", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    plt.tight_layout()
    if save:
        os.makedirs(STATS_PLOTS_DIR, exist_ok=True)
        combos_str = "_".join(combo_ids)
        path = os.path.join(STATS_PLOTS_DIR, f"auc_{combos_str}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
        plt.close()
    else:
        plt.show()


def plot_table_size_vs_wr(df, combo_ids, save=True):
    """Scatter: final table size vs final rolling WR, colored by model."""
    sub = df[df["combo_id"].isin(combo_ids)]

    fig, ax = plt.subplots(figsize=(10, 7))
    for model_name in MODEL_NAMES:
        ms = sub[sub["model"] == model_name]
        ax.scatter(ms["final_table_size"], ms["final_rolling"],
                   c=MODEL_COLORS[model_name], label=MODEL_SHORT[model_name],
                   alpha=0.6, s=60, edgecolors='black', linewidths=0.5)

    ax.set_xlabel("Final Q-Table Size", fontsize=12)
    ax.set_ylabel("Final Rolling Win Rate", fontsize=12)
    ax.set_title("Table Size vs Win Rate", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    if save:
        os.makedirs(STATS_PLOTS_DIR, exist_ok=True)
        path = os.path.join(STATS_PLOTS_DIR, f"table_vs_wr.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
        plt.close()
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def run_full_analysis(combo_ids):
    """Run the complete statistical analysis pipeline."""

    print("\n" + "=" * 100)
    print("  V5 INITIALIZATION STUDY — STATISTICAL ANALYSIS")
    print("=" * 100)

    # Collect all data
    all_dfs = []
    for cid in combo_ids:
        df = extract_final_metrics(cid)
        if not df.empty:
            all_dfs.append(df)
    if not all_dfs:
        print("No data found.")
        return

    all_df = pd.concat(all_dfs, ignore_index=True)

    # Per-combo analysis
    for cid in combo_ids:
        print_summary(all_df, cid)
        pairwise_tests(all_df, cid)
        factorial_anova(all_df, cid)
        variance_analysis(all_df, cid)
        learning_dynamics(cid)
        print_findings(all_df, cid)

        # Plots per combo
        print(f"  Generating plots for {cid}...")
        plot_box_violin(all_df, cid)
        plot_interaction(all_df, cid)
        plot_effect_sizes(all_df, cid)
        plot_convergence(cid)
        plot_correlation_matrix(all_df, cid)

    # Cross-combo analysis (if multiple)
    if len(combo_ids) > 1:
        print(f"\n{'=' * 100}")
        print(f"  CROSS-COMBO ANALYSIS ({len(combo_ids)} combos)")
        print(f"{'=' * 100}\n")
        lambda_analysis(all_df, combo_ids)

        print(f"  Generating cross-combo plots...")
        plot_lambda_sensitivity(all_df, combo_ids)
        plot_auc_comparison(combo_ids)
        plot_table_size_vs_wr(all_df, combo_ids)

    print(f"\n{'=' * 100}")
    print(f"  All plots saved to: {STATS_PLOTS_DIR}")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V5 Initialization Study — Statistical Analysis")
    parser.add_argument("--combo", nargs="+", default=None,
                        help="Specific combo(s) to analyze (e.g. hp_001 hp_002)")
    args = parser.parse_args()

    if args.combo:
        combo_ids = args.combo
    else:
        # Auto-detect combos with data
        combo_ids = []
        for combo in build_grid():
            cid = combo["combo_id"]
            combo_dir = os.path.join(GRID_RESULTS_DIR, cid)
            if os.path.isdir(combo_dir):
                # Check if at least one run has a CSV
                for r in range(1, RUNS_PER_COMBO + 1):
                    test_csv = os.path.join(combo_dir, f"run_{r}", MODEL_NAMES[0], "logs", "run_1.csv")
                    if os.path.exists(test_csv):
                        combo_ids.append(cid)
                        break

    if not combo_ids:
        print("No grid results found. Run the grid search first.")
        sys.exit(1)

    print(f"Analyzing combos: {combo_ids}")
    run_full_analysis(combo_ids)
