"""
Central configuration for V4 — Hyperparameter Grid Search.

Runs all 4 models across a grid of hyperparameter combinations.
Each combo runs RUNS_PER_COMBO times, where each run uses a different
randomly-sampled 9-pokemon pool (but all 4 models in a run share the same pool).

Grid dimensions: ALPHA × GAMMA × LAMBDA
Fixed: EPS_DECAY=100K, BATTLES_PER_RUN=200K, POOL_SIZE=9
"""

import os
import random
import itertools

# --- Full Pokemon Pool (20 OU-legal species from sets.json) ---
FULL_POKEMON_POOL = [
    "tyranitar", "metagross", "infernape", "lucario", "togekiss",
    "gengar", "starmie", "scizor", "gyarados", "dragonite",
    "blissey", "skarmory", "hippowdon", "roserade", "swampert",
    "jolteon", "flygon", "bronzong", "gliscor", "empoleon",
]

# ── Pool size per run (randomly sampled from FULL_POKEMON_POOL) ──
POOL_SIZE = 9

# --- Battle Config ---
BATTLE_FORMAT = "gen4ou"
BATTLE_TIMEOUT = 1

# --- Training Config ---
BATTLES_PER_RUN = 100_000    # 200K per run — fast iterations
BATTLES_PER_LOG = 100
SAVE_FREQ = 5_000
RUNS_PER_COMBO = 10          # 10 runs per HP combo (each with different pool)

# --- Epsilon Schedule (fixed, not grid-searched) ---
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_BATTLES = 100_000  # Fast decay — 50% of training

# --- Hyperparameter Grid ---
GRID = {
    "alpha": [0.05, 0.1, 0.2],
    "gamma": [0.9, 0.99, 0.999],
    "lam":   [0.5, 0.7, 0.9],
}

# Default HPs (used when not grid-searching)
ALPHA = 0.1
GAMMA = 0.99
LAMBDA = 0.7


def build_grid():
    """Generate all hyperparameter combinations as list of dicts.

    Returns:
        List of dicts like:
          [{"alpha": 0.05, "gamma": 0.95, "lam": 0.5, "combo_id": "hp_001"}, ...]
    """
    keys = list(GRID.keys())
    values = [GRID[k] for k in keys]
    combos = []
    for i, vals in enumerate(itertools.product(*values), start=1):
        combo = dict(zip(keys, vals))
        combo["combo_id"] = f"hp_{i:03d}"
        combos.append(combo)
    return combos


def get_pool_for_run(run_id):
    """Get the deterministic 9-pokemon pool for a given run.

    Each run_id gets a different random sample of 9 from the full 20,
    but the same run_id always produces the same pool (seeded by run_id).
    All 4 models in the same run share this pool.
    """
    rng = random.Random(run_id)
    pool = rng.sample(FULL_POKEMON_POOL, POOL_SIZE)
    return sorted(pool)  # Sort for consistency


def get_epsilon(battles_so_far, eps_decay=None):
    """Linear decay from EPS_START to EPS_END over eps_decay battles."""
    if eps_decay is None:
        eps_decay = EPS_DECAY_BATTLES
    if battles_so_far >= eps_decay:
        return EPS_END
    frac = battles_so_far / eps_decay
    return EPS_START + (EPS_END - EPS_START) * frac


# --- Paths ---
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHOWDOWN_DIR = os.path.join(_PROJECT_ROOT, "pokemon-showdown")
SHOWDOWN_CLI_PATH = os.path.join(SHOWDOWN_DIR, "pokemon-showdown")
SHOWDOWN_DATA_PATH = os.path.join(SHOWDOWN_DIR, "data", "random-battles", "gen4", "sets.json")

# --- Models ---
MODEL_NAMES = [
    "model_1_flat_zero",
    "model_2_flat_smart",
    "model_3_hier_zero",
    "model_4_hier_smart",
]

MODEL_LABELS = {
    "model_1_flat_zero": "Flat + Zero Init",
    "model_2_flat_smart": "Flat + Smart Init",
    "model_3_hier_zero": "Hier + Zero Init",
    "model_4_hier_smart": "Hier + Smart Init",
}

# Backward compat — POKEMON_POOL for team_builder default (uses full pool)
POKEMON_POOL = FULL_POKEMON_POOL[:POOL_SIZE]
