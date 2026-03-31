"""
Central configuration for the 2x2 factorial research experiment (V2 — Smaller Pool).
All models share these hyperparameters to ensure fair comparison.

V2 changes from V1:
  - Configurable pool size (default 13, was 20)
  - 30 runs per model (was 5) for statistical significance
  - Optimized hyperparameters from RL literature
  - Reduced battles/run to make 30 runs feasible
"""

import os

# --- Full Pokemon Pool (20 OU-legal species from sets.json) ---
# Ordered by competitive viability. Slice with POOL_SIZE to get subsets.
FULL_POKEMON_POOL = [
    "tyranitar", "metagross", "infernape", "lucario", "togekiss",
    "gengar", "starmie", "scizor", "gyarados", "dragonite",
    "blissey", "skarmory", "hippowdon", "roserade", "swampert",
    "jolteon", "flygon", "bronzong", "gliscor", "empoleon",
]

# ── Configurable pool size (6-20). Teams are always 6 from this pool. ──
# Recommended BATTLES_PER_RUN by pool size:
#   Pool  6 →   500,000  (C(6,6)=1 composition)
#   Pool  8 → 1,000,000  (C(8,6)=28)
#   Pool 10 → 1,500,000  (C(10,6)=210)
#   Pool 13 → 1,000,000  (C(13,6)=1,716)  ← default
#   Pool 15 → 2,500,000  (C(15,6)=5,005)
#   Pool 20 → 5,000,000  (C(20,6)=38,760)
POOL_SIZE = 13
POKEMON_POOL = FULL_POKEMON_POOL[:POOL_SIZE]

# --- Battle Config ---
BATTLE_FORMAT = "gen4ou"
BATTLE_TIMEOUT = 1

# --- Training Config ---
BATTLES_PER_RUN = 1_000_000
BATTLES_PER_LOG = 1_000
SAVE_FREQ = 5_000
RUNS_PER_MODEL = 30
RANDOM_SEEDS = list(range(1, 31))  # [1, 2, ..., 30]

# --- Q-Learning Hyperparameters (optimized from RL literature) ---
ALPHA = 0.1       # Learning rate — standard for tabular Q-learning (Sutton & Barto 2018)
GAMMA = 0.99      # Discount factor — better for 20-40 turn episodes (0.99^30=0.74)
LAMBDA = 0.9      # Trace decay — λ=0.9 empirically optimal for TD(λ) (Singh & Sutton 1996)

# --- Epsilon Schedule ---
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_BATTLES = 500_000  # 50% of BATTLES_PER_RUN

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


def get_epsilon(battles_so_far):
    """Linear decay from EPS_START to EPS_END over EPS_DECAY_BATTLES."""
    if battles_so_far >= EPS_DECAY_BATTLES:
        return EPS_END
    frac = battles_so_far / EPS_DECAY_BATTLES
    return EPS_START + (EPS_END - EPS_START) * frac
