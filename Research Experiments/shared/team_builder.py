"""
Gen4OU Pool Team Builder for Research Experiments
==================================================
Builds random teams from a fixed pool of 20 OU-legal Pokemon.
Both learner and opponent use the same pool for controlled comparison.

Reuses move selection, role-based EVs/natures/items from v17gen4/teams_v17gen4.py.

No subprocess validation — all 20 pool species are verified OU-legal from sets.json,
and known problematic move combinations are excluded via _MOVE_EXCLUSIONS.
Team generation is pure Python (~0ms per team).
"""

import json
import random as _random
from poke_env.teambuilder import Teambuilder
from poke_env.data import GenData

from shared.config import POKEMON_POOL, SHOWDOWN_DATA_PATH

# =============================================================================
# LOAD GEN 4 REFERENCE DATA
# =============================================================================

_gen4_data = GenData.from_gen(4)
_gen4_moves = _gen4_data.moves
_gen4_pokedex = _gen4_data.pokedex


def _load_sets_json():
    with open(SHOWDOWN_DATA_PATH) as f:
        return json.load(f)


_SETS_DATA = _load_sets_json()


# =============================================================================
# HIDDEN POWER IVS (from v17gen4)
# =============================================================================

_HIDDEN_POWER_IVS = {
    "fighting": "IVs: 30 Def / 30 SpA / 30 SpD / 30 Spe",
    "flying":   "IVs: 30 SpA / 30 SpD / 30 Spe",
    "poison":   "IVs: 30 Def / 30 SpA / 30 SpD",
    "ground":   "IVs: 30 SpA / 30 SpD",
    "rock":     "IVs: 30 Def / 30 SpD / 30 Spe",
    "bug":      "IVs: 30 SpD / 30 Spe",
    "ghost":    "IVs: 30 Def / 30 SpD",
    "steel":    "IVs: 30 SpD",
    "fire":     "IVs: 30 Def / 30 SpA / 30 Spe",
    "water":    "IVs: 30 SpA / 30 Spe",
    "grass":    "IVs: 30 Def / 30 SpA",
    "electric": "IVs: 30 SpA",
    "psychic":  "IVs: 30 Def / 30 Spe",
    "ice":      "IVs: 30 Spe",
    "dragon":   "IVs: 30 Def",
    "dark":     "",
}


# =============================================================================
# MOVE LEGALITY FIXES (from v17gen4)
# =============================================================================

_MOVE_EXCLUSIONS = {
    "pikachu": {"surf"},
    "raichu": {"surf"},
    # Togekiss + Baton Pass + stat-boosting move = banned by Baton Pass Stat Trap Clause
    "togekiss": {"batonpass"},
}

_INCOMPATIBLE_MOVE_PAIRS = {
    "roserade": [("leafstorm", "spikes"), ("sleeppowder", "spikes")],
}


def _select_moves(species_id, movepool, rng, count=4):
    """Select up to count moves respecting all Gen 4 legality rules."""
    excluded = _MOVE_EXCLUSIONS.get(species_id, set())
    pool = [m for m in movepool if m not in excluded]

    hp_moves = [m for m in pool if m.startswith("hiddenpower")]
    non_hp_moves = [m for m in pool if not m.startswith("hiddenpower")]

    candidates = list(non_hp_moves)
    if hp_moves:
        candidates.append(rng.choice(hp_moves))

    pairs = _INCOMPATIBLE_MOVE_PAIRS.get(species_id, [])
    shuffled = list(candidates)
    rng.shuffle(shuffled)

    selected = []
    for move in shuffled:
        if len(selected) >= count:
            break
        conflict = False
        for m_a, m_b in pairs:
            if (move == m_a and m_b in selected) or (move == m_b and m_a in selected):
                conflict = True
                break
        if not conflict:
            selected.append(move)

    return selected


# =============================================================================
# DISPLAY NAME HELPERS
# =============================================================================

def _get_move_display_name(move_id):
    clean = move_id.replace(" ", "").lower()
    return _gen4_moves.get(clean, {}).get("name", move_id.capitalize())


def _get_species_display_name(species_id):
    entry = _gen4_pokedex.get(species_id, {})
    return entry.get("name", species_id.capitalize())


def _is_physical_pokemon(species_id):
    entry = _gen4_pokedex.get(species_id, {})
    stats = entry.get("baseStats", {})
    return stats.get("atk", 0) >= stats.get("spa", 0)


def _get_hidden_power_type(move_id):
    if move_id.startswith("hiddenpower") and len(move_id) > len("hiddenpower"):
        return move_id[len("hiddenpower"):]
    return None


# =============================================================================
# ROLE-BASED BUILD PARAMETERS (from v17gen4)
# =============================================================================

_ROLE_ITEMS = {
    "Setup Sweeper":  {"phys": ["Life Orb", "Lum Berry", "Leftovers"],
                       "spec": ["Life Orb", "Lum Berry", "Leftovers"]},
    "Fast Attacker":  {"phys": ["Life Orb", "Choice Band", "Expert Belt"],
                       "spec": ["Life Orb", "Choice Specs", "Expert Belt"]},
    "Wallbreaker":    {"phys": ["Choice Band", "Life Orb", "Expert Belt"],
                       "spec": ["Choice Specs", "Life Orb", "Expert Belt"]},
    "Bulky Attacker": {"phys": ["Leftovers", "Life Orb", "Choice Band"],
                       "spec": ["Leftovers", "Life Orb", "Choice Specs"]},
    "Bulky Support":  {"phys": ["Leftovers"], "spec": ["Leftovers"]},
    "Staller":        {"phys": ["Leftovers"], "spec": ["Leftovers"]},
    "Spinner":        {"phys": ["Leftovers"], "spec": ["Leftovers"]},
    "Fast Support":   {"phys": ["Focus Sash", "Leftovers"], "spec": ["Focus Sash", "Leftovers"]},
    "Bulky Setup":    {"phys": ["Leftovers", "Lum Berry"], "spec": ["Leftovers", "Lum Berry"]},
    "AV Pivot":       {"phys": ["Leftovers"], "spec": ["Leftovers"]},
}

_ROLE_EVS = {
    "Setup Sweeper":  {"phys": "252 Atk / 4 SpD / 252 Spe", "spec": "252 SpA / 4 SpD / 252 Spe"},
    "Fast Attacker":  {"phys": "252 Atk / 4 SpD / 252 Spe", "spec": "252 SpA / 4 SpD / 252 Spe"},
    "Wallbreaker":    {"phys": "252 Atk / 4 SpD / 252 Spe", "spec": "252 SpA / 4 SpD / 252 Spe"},
    "Bulky Attacker": {"phys": "252 HP / 252 Atk / 4 SpD",  "spec": "252 HP / 252 SpA / 4 SpD"},
    "Bulky Support":  {"phys": "252 HP / 252 Def / 4 SpD",  "spec": "252 HP / 252 Def / 4 SpD"},
    "Staller":        {"phys": "252 HP / 252 Def / 4 SpD",  "spec": "252 HP / 252 Def / 4 SpD"},
    "Spinner":        {"phys": "252 HP / 252 Def / 4 SpD",  "spec": "252 HP / 252 Def / 4 SpD"},
    "Fast Support":   {"phys": "252 HP / 4 Def / 252 Spe",  "spec": "252 HP / 4 Def / 252 Spe"},
    "Bulky Setup":    {"phys": "252 HP / 128 Def / 128 SpD","spec": "252 HP / 128 Def / 128 SpD"},
    "AV Pivot":       {"phys": "252 HP / 128 Def / 128 SpD","spec": "252 HP / 128 Def / 128 SpD"},
}

_ROLE_NATURES = {
    "Setup Sweeper":  {"phys": ["Adamant", "Jolly"],  "spec": ["Timid", "Modest"]},
    "Fast Attacker":  {"phys": ["Adamant", "Jolly"],  "spec": ["Timid", "Modest"]},
    "Wallbreaker":    {"phys": ["Adamant", "Jolly"],  "spec": ["Modest", "Timid"]},
    "Bulky Attacker": {"phys": ["Adamant"],            "spec": ["Modest"]},
    "Bulky Support":  {"phys": ["Impish", "Careful"],  "spec": ["Bold", "Calm"]},
    "Staller":        {"phys": ["Impish", "Careful"],  "spec": ["Bold", "Calm"]},
    "Spinner":        {"phys": ["Impish"],             "spec": ["Bold"]},
    "Fast Support":   {"phys": ["Jolly"],              "spec": ["Timid"]},
    "Bulky Setup":    {"phys": ["Careful", "Adamant"], "spec": ["Calm", "Modest"]},
    "AV Pivot":       {"phys": ["Careful"],            "spec": ["Calm"]},
}

# Fallback items for resolving item clause conflicts
_FALLBACK_ITEMS = [
    "Leftovers", "Life Orb", "Choice Band", "Choice Specs", "Choice Scarf",
    "Expert Belt", "Lum Berry", "Focus Sash", "Shed Shell", "Wide Lens",
    "Muscle Band", "Wise Glasses", "Scope Lens", "Shell Bell", "Razor Claw",
]


# =============================================================================
# BUILD SINGLE POKEMON STRING
# =============================================================================

def _build_pokemon_str(species_id, rng=None, used_items=None):
    """
    Build a Showdown paste string for one Pokemon from sets.json data.
    gen4ou uses level 100 (default), so no Level line is included.
    """
    if rng is None:
        rng = _random
    if used_items is None:
        used_items = set()

    species_data = _SETS_DATA.get(species_id)
    if not species_data:
        return ""

    sets = species_data.get("sets", [])
    if not sets:
        return ""

    chosen_set = rng.choice(sets)
    role = chosen_set["role"]
    movepool = list(chosen_set["movepool"])
    abilities = chosen_set.get("abilities", [])

    moves = _select_moves(species_id, movepool, rng)
    ability = rng.choice(abilities) if abilities else ""

    is_phys = _is_physical_pokemon(species_id)
    orient = "phys" if is_phys else "spec"

    # Pick item, respecting item clause (no duplicates)
    item_dict = _ROLE_ITEMS.get(role, {"phys": ["Leftovers"], "spec": ["Leftovers"]})
    item_candidates = [i for i in item_dict[orient] if i not in used_items]
    if not item_candidates:
        item_candidates = [i for i in _FALLBACK_ITEMS if i not in used_items]
    if not item_candidates:
        item_candidates = ["Leftovers"]
    item = rng.choice(item_candidates)
    used_items.add(item)

    ev_dict = _ROLE_EVS.get(role, {"phys": "252 HP / 4 Def / 252 Spe", "spec": "252 HP / 4 Def / 252 Spe"})
    evs = ev_dict[orient]

    nature_dict = _ROLE_NATURES.get(role, {"phys": ["Hardy"], "spec": ["Hardy"]})
    nature = rng.choice(nature_dict[orient])

    display_name = _get_species_display_name(species_id)

    hp_type = None
    for m in moves:
        hp_type = _get_hidden_power_type(m)
        if hp_type:
            break

    lines = []
    lines.append(f"{display_name} @ {item}")
    if ability:
        lines.append(f"Ability: {ability}")
    lines.append(f"EVs: {evs}")
    lines.append(f"{nature} Nature")
    if hp_type and _HIDDEN_POWER_IVS.get(hp_type):
        lines.append(_HIDDEN_POWER_IVS[hp_type])
    for m in moves:
        lines.append(f"- {_get_move_display_name(m)}")

    return "\n".join(lines)


# =============================================================================
# BUILD RANDOM TEAM FROM POOL
# =============================================================================

def _build_random_team(pool, rng):
    """
    Build a random 6-Pokemon team from the pool. Pure Python, no subprocess.

    All 20 pool species are verified OU-legal from sets.json, and known
    problematic move combinations are excluded via _MOVE_EXCLUSIONS.
    """
    team_species = rng.sample(pool, min(6, len(pool)))
    used_items = set()

    team_parts = []
    for species_id in team_species:
        sub_rng = _random.Random(f"{rng.random()}_{species_id}")
        pokemon_str = _build_pokemon_str(species_id, rng=sub_rng, used_items=used_items)
        if pokemon_str:
            team_parts.append(pokemon_str)

    return "\n\n".join(team_parts)


# =============================================================================
# TEAMBUILDER CLASS (for poke-env)
# =============================================================================

class Gen4OUPoolTeambuilder(Teambuilder):
    """
    Teambuilder that generates random teams from a fixed pool of 20 gen4ou Pokemon.
    Each call to yield_team() generates a fresh random team (pure Python, ~0ms).
    """

    def __init__(self, pool=None, seed=None):
        self.pool = pool or POKEMON_POOL
        self.rng = _random.Random(seed)

    def yield_team(self):
        team_paste = _build_random_team(self.pool, self.rng)
        parsed = self.parse_showdown_team(team_paste)
        return self.join_team(parsed)
