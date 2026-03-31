"""
Central configuration for the 2x2 factorial research experiment.
All models share these hyperparameters to ensure fair comparison.
"""

import os

# --- Pokemon Pool (20 OU-legal species from sets.json) ---
POKEMON_POOL = [
    "tyranitar", "metagross", "infernape", "lucario", "togekiss",
    "gengar", "starmie", "scizor", "gyarados", "dragonite",
    "blissey", "skarmory", "hippowdon", "roserade", "swampert",
    "jolteon", "flygon", "bronzong", "gliscor", "empoleon",
]

# --- Battle Config ---
BATTLE_FORMAT = "gen4ou"
BATTLE_TIMEOUT = 1

# --- Training Config ---
BATTLES_PER_RUN = 10_000_000
BATTLES_PER_LOG = 1_000
SAVE_FREQ = 5_000
RUNS_PER_MODEL = 5
RANDOM_SEEDS = [42, 137, 256, 67, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# --- Q-Learning Hyperparameters ---
ALPHA = 0.1
GAMMA = 0.995
LAMBDA = 0.6967

# --- Epsilon Schedule ---
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_BATTLES = 5_000_000

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
