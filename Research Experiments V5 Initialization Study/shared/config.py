"""
Central configuration for V5 — same as V4, plus standardized smart init
and four fixed-epsilon comparison models.
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
POOL_SIZE = 8

# --- Battle Config ---
BATTLE_FORMAT = "gen4ou"
BATTLE_TIMEOUT = 1

# --- Training Config ---
BATTLES_PER_RUN = 50_000
BATTLES_PER_LOG = 100
SAVE_FREQ = 5_000
RUNS_PER_COMBO = 10

# --- Epsilon Schedules ---
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_BATTLES = BATTLES_PER_RUN/2
FIXED_EPSILON = EPS_END

# --- Smart Initialization Standardization ---
# Calibrated from V4 zero-init learned Q-values using a robust
# central-range estimate rather than raw min/max outliers.
SMART_INIT_Q_MIN = -0.01
SMART_INIT_Q_MAX = 0.01

# --- Hyperparameter Grid ---
GRID = {
    "alpha": [0.1, 0.2],
    "gamma": [0.99, 0.999],
    "lam":   [0.7, 0.9],
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


def resolve_epsilon(battles_so_far, mode="decay", fixed_epsilon=None):
    """Resolve epsilon for a given training step and schedule mode."""
    if mode == "fixed":
        return FIXED_EPSILON if fixed_epsilon is None else fixed_epsilon
    return get_epsilon(battles_so_far)


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
    "model_5_flat_zero_fixed_eps",
    "model_6_flat_smart_fixed_eps",
    "model_7_hier_zero_fixed_eps",
    "model_8_hier_smart_fixed_eps",
]

MODEL_LABELS = {
    "model_1_flat_zero": "Flat + Zero Init",
    "model_2_flat_smart": "Flat + Smart Init",
    "model_3_hier_zero": "Hier + Zero Init",
    "model_4_hier_smart": "Hier + Smart Init",
    "model_5_flat_zero_fixed_eps": "Flat + Zero Init + Fixed Eps",
    "model_6_flat_smart_fixed_eps": "Flat + Smart Init + Fixed Eps",
    "model_7_hier_zero_fixed_eps": "Hier + Zero Init + Fixed Eps",
    "model_8_hier_smart_fixed_eps": "Hier + Smart Init + Fixed Eps",
}

MODEL_EPSILON_MODE = {
    "model_1_flat_zero": "decay",
    "model_2_flat_smart": "decay",
    "model_3_hier_zero": "decay",
    "model_4_hier_smart": "decay",
    "model_5_flat_zero_fixed_eps": "fixed",
    "model_6_flat_smart_fixed_eps": "fixed",
    "model_7_hier_zero_fixed_eps": "fixed",
    "model_8_hier_smart_fixed_eps": "fixed",
}

# Backward compat — POKEMON_POOL for team_builder default (uses full pool)
POKEMON_POOL = FULL_POKEMON_POOL[:POOL_SIZE]
