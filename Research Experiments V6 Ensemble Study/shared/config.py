"""
V6 Ensemble Study — central configuration.

Self-contained: V6 does not import from V5. This module merges V5's battle /
pool / reward / heuristic constants (copied verbatim) with V6-specific
experiment parameters (K=30 ensemble members, 1M battles per member, 5M
compute-matched baseline).
"""

import os
import random
import itertools


# =============================================================================
# POKEMON POOL
# =============================================================================

# 20 OU-legal species drawn from Pokemon Showdown's Gen 4 sets.json.
FULL_POKEMON_POOL = [
    "tyranitar", "metagross", "infernape", "lucario", "togekiss",
    "gengar", "starmie", "scizor", "gyarados", "dragonite",
    "blissey", "skarmory", "hippowdon", "roserade", "swampert",
    "jolteon", "flygon", "bronzong", "gliscor", "empoleon",
]

# V5 used POOL_SIZE=8 of 20 per run.  V6 uses the FULL pool for every member.
POOL_SIZE = len(FULL_POKEMON_POOL)   # 20
POKEMON_POOL = list(FULL_POKEMON_POOL)


# =============================================================================
# BATTLE CONFIG
# =============================================================================

BATTLE_FORMAT = "gen4ou"
BATTLE_TIMEOUT = 1


# =============================================================================
# TRAINING CONFIG
# =============================================================================

# V6 ensemble: K=30 members, 1M battles each
K_MEMBERS = 30
BATTLES_PER_MEMBER = 1_000_000

# Compute-matched single-member baseline (literature-required control).
# 5M = 1/6 of the 30M ensemble compute — probes whether concentrating budget
# in one agent beats spreading it across K members.
BASELINE_SINGLE_BATTLES = 5_000_000

# Logging cadence (inherited from V5; unchanged)
BATTLES_PER_LOG = 100
SAVE_FREQ = 5_000     # Checkpoint to disk every N battles (enables resumption)


# =============================================================================
# EPSILON SCHEDULES
# =============================================================================

# V6 members all use FIXED epsilon = 0.05 (matches V5 winner M8).
# Decay schedule retained for completeness / baseline comparisons.
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_BATTLES = BATTLES_PER_MEMBER // 2
FIXED_EPSILON = EPS_END


def get_epsilon(battles_so_far, eps_decay=None):
    """Linear decay from EPS_START to EPS_END over eps_decay battles."""
    if eps_decay is None:
        eps_decay = EPS_DECAY_BATTLES
    if battles_so_far >= eps_decay:
        return EPS_END
    frac = battles_so_far / eps_decay
    return EPS_START + (EPS_END - EPS_START) * frac


def resolve_epsilon(battles_so_far, mode="fixed", fixed_epsilon=None):
    """Resolve epsilon for a given step and schedule mode.

    V6 defaults to 'fixed' because the winning V5 config (M8) uses fixed ε=0.05.
    """
    if mode == "fixed":
        return FIXED_EPSILON if fixed_epsilon is None else fixed_epsilon
    return get_epsilon(battles_so_far)


# =============================================================================
# SMART INITIALIZATION (for HierSmartPlayer's heuristic Q-priors)
# =============================================================================

# Calibrated from V4 zero-init learned Q-values using a robust central-range
# estimate rather than raw min/max outliers.  Unchanged from V5.
SMART_INIT_Q_MIN = -0.01
SMART_INIT_Q_MAX = 0.01


# =============================================================================
# BEST HYPERPARAMETERS (V5 winner hp_001, model M8 = Hier+Smart+FixedEps)
# =============================================================================

ALPHA = 0.1         # Learning rate
GAMMA = 0.99        # Discount factor
LAMBDA = 0.7        # Eligibility trace decay
# (Epsilon handled by resolve_epsilon() / FIXED_EPSILON above.)

# Aliases for V6 readability:
HP_ALPHA = ALPHA
HP_GAMMA = GAMMA
HP_LAM = LAMBDA
FIXED_EPS = FIXED_EPSILON


# =============================================================================
# V5 HYPERPARAMETER GRID (kept for compatibility with V5 tooling — unused in V6)
# =============================================================================

GRID = {
    "alpha": [0.1, 0.2],
    "gamma": [0.99, 0.999],
    "lam":   [0.7, 0.9],
}


def build_grid():
    """Generate all hyperparameter combinations as list of dicts (V5 legacy)."""
    keys = list(GRID.keys())
    values = [GRID[k] for k in keys]
    combos = []
    for i, vals in enumerate(itertools.product(*values), start=1):
        combo = dict(zip(keys, vals))
        combo["combo_id"] = f"hp_{i:03d}"
        combos.append(combo)
    return combos


def get_pool_for_run(run_id):
    """V5 legacy: sampled 8 of 20. V6 always uses full 20 regardless of run_id."""
    return list(FULL_POKEMON_POOL)


# =============================================================================
# ORCHESTRATION DEFAULTS
# =============================================================================

BASE_SEED = 10_000               # member_k uses BASE_SEED + k
DEFAULT_MAX_PARALLEL = 10        # concurrent training subprocesses
DEFAULT_BASE_PORT = 9000         # slot-i uses port DEFAULT_BASE_PORT + i


# =============================================================================
# PATHS
# =============================================================================

_THIS_FILE = os.path.abspath(__file__)
_V6_DIR = os.path.dirname(os.path.dirname(_THIS_FILE))              # .../Research Experiments V6 Ensemble Study
_PROJECT_ROOT = os.path.dirname(_V6_DIR)                             # .../PokeAgent

V6_DIR = _V6_DIR
PROJECT_ROOT = _PROJECT_ROOT
ENSEMBLE_RESULTS_DIR = os.path.join(V6_DIR, "ensemble_results")

# Pokemon Showdown simulator (shared with V5, lives at project root)
SHOWDOWN_DIR = os.path.join(_PROJECT_ROOT, "pokemon-showdown")
SHOWDOWN_CLI_PATH = os.path.join(SHOWDOWN_DIR, "pokemon-showdown")
SHOWDOWN_DATA_PATH = os.path.join(SHOWDOWN_DIR, "data", "random-battles", "gen4", "sets.json")


# =============================================================================
# MODEL METADATA (single model in V6: the HierSmartPlayer from V5 M8)
# =============================================================================

MODEL_NAME = "hier_smart_fixed_eps"
MODEL_LABEL = "Hier + Smart Init + Fixed Eps"
MODEL_EPSILON_MODE = "fixed"
