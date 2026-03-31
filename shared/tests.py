"""
Comprehensive test suite for the 2×2 factorial research experiment.
Run via: python run_all.py --tests

Tests cover:
  1. Feature extraction (state tuple shapes, bench sorting, deduplication)
  2. Player logic (Q-table updates, trace decay, action masking, switch hashing)
  3. State representation correctness (flat=35, hier master=20, hier sub=17)
  4. Smart init (heuristic seeding, softmax normalization)
  5. Config consistency (seeds, epsilon schedule, model names)
  6. Rewards (dense reward calculation, snapshot correctness)
  7. Team builder (pool size, team generation, no duplicate items)
  8. Save/load round-trip (pickle correctness for all 4 models)
  9. Integration (end-to-end short battle for all 4 models)
"""

import os
import sys
import math
import zlib
import pickle
import tempfile
import traceback
import random
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Counts ──
_passed = 0
_failed = 0
_errors = []


def _test(name, condition, detail=""):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  ✓ {name}")
    else:
        _failed += 1
        msg = f"  ✗ {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        _errors.append(msg)


def _section(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# =============================================================================
# MOCK OBJECTS (simulate poke_env battle objects without a server)
# =============================================================================

class MockStatus:
    def __init__(self, value=0):
        self.value = value


class MockMove:
    def __init__(self, id, base_power=80, type_name="normal", category=None,
                 accuracy=1.0, expected_hits=1.0):
        from poke_env.battle.move_category import MoveCategory
        self.id = id
        self.base_power = base_power
        self.type = type_name
        self.category = category or MoveCategory.SPECIAL
        self.accuracy = accuracy
        self.expected_hits = expected_hits


class MockPokemon:
    def __init__(self, species, current_hp=100, max_hp=100, status=None,
                 ability="pressure", base_stats=None, fainted=False, boosts=None, types=None):
        self.species = species
        self.current_hp = current_hp
        self.max_hp = max_hp
        self.current_hp_fraction = current_hp / max_hp if max_hp > 0 else 0
        self.status = status
        self.ability = ability
        self.base_stats = base_stats or {"hp": 100, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100}
        self.fainted = fainted
        self.boosts = boosts or {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}
        self.speed = base_stats.get("spe", 100) if base_stats else 100
        self.types = types or ["normal"]

    def damage_multiplier(self, t):
        return 1.0


class MockSideCondition:
    SPIKES = "SPIKES"
    STEALTH_ROCK = "STEALTH_ROCK"
    STICKY_WEB = "STICKY_WEB"
    TOXIC_SPIKES = "TOXIC_SPIKES"


class MockBattle:
    """Simulates a poke_env Battle object with enough fields for feature extraction."""
    def __init__(self, team, opp_team, active_idx=0, opp_active_idx=0, side_conditions=None):
        self._team = team
        self._opp_team = opp_team
        self.active_pokemon = team[active_idx] if team else None
        self.opponent_active_pokemon = opp_team[opp_active_idx] if opp_team else None
        self.side_conditions = side_conditions or {}
        self.force_switch = False
        self.available_moves = []
        self.available_switches = [m for i, m in enumerate(team) if i != active_idx and not m.fainted]
        self._finished = False
        self.n_finished_battles = 0
        self.won = False
        self.battle_tag = "test_battle_1"

    @property
    def team(self):
        return {f"p1:{m.species}": m for m in self._team}

    @property
    def opponent_team(self):
        return {f"p2:{m.species}": m for m in self._opp_team}


def _make_test_battle(my_species=None, opp_species=None, my_hp=None, fainted_indices=None):
    """Create a standard test battle with 6v6 from the pool."""
    from shared.config import FULL_POKEMON_POOL

    if my_species is None:
        my_species = FULL_POKEMON_POOL[:6]
    if opp_species is None:
        opp_species = FULL_POKEMON_POOL[6:12]
    if my_hp is None:
        my_hp = [100] * 6
    if fainted_indices is None:
        fainted_indices = []

    my_team = []
    for i, sp in enumerate(my_species):
        fainted = i in fainted_indices
        hp = 0 if fainted else my_hp[i]
        my_team.append(MockPokemon(sp, current_hp=hp, max_hp=100, fainted=fainted,
                                   base_stats={"hp": 100, "atk": 80, "def": 80, "spa": 80, "spd": 80, "spe": 80 + i * 5}))

    opp_team = []
    for i, sp in enumerate(opp_species):
        opp_team.append(MockPokemon(sp, current_hp=100, max_hp=100,
                                    base_stats={"hp": 100, "atk": 80, "def": 80, "spa": 80, "spd": 80, "spe": 75 + i * 5}))

    battle = MockBattle(my_team, opp_team)
    battle.available_moves = [MockMove("thunderbolt", 90), MockMove("icebeam", 90)]
    return battle


# =============================================================================
# TEST GROUPS
# =============================================================================

def test_config():
    _section("CONFIG")
    from shared.config import (
        POKEMON_POOL, FULL_POKEMON_POOL, POOL_SIZE,
        BATTLES_PER_RUN, RUNS_PER_COMBO,
        EPS_START, EPS_END, EPS_DECAY_BATTLES, BATTLE_FORMAT,
        ALPHA, GAMMA, LAMBDA, MODEL_NAMES, MODEL_LABELS, get_epsilon,
        GRID, build_grid, get_pool_for_run,
    )

    _test("Full pool has 20 Pokemon", len(FULL_POKEMON_POOL) == 20, f"got {len(FULL_POKEMON_POOL)}")
    _test("Pool size >= 6 (need full teams)", POOL_SIZE >= 6, f"got {POOL_SIZE}")
    _test("Pool size <= full pool", POOL_SIZE <= len(FULL_POKEMON_POOL), f"{POOL_SIZE} > {len(FULL_POKEMON_POOL)}")
    _test("Runs per combo >= 1", RUNS_PER_COMBO >= 1, f"got {RUNS_PER_COMBO}")
    _test("Battles per run >= 1000", BATTLES_PER_RUN >= 1000, f"got {BATTLES_PER_RUN}")
    _test("Eps decay > 0", EPS_DECAY_BATTLES > 0, f"got {EPS_DECAY_BATTLES}")
    _test("Eps decay <= battles per run", EPS_DECAY_BATTLES <= BATTLES_PER_RUN, f"{EPS_DECAY_BATTLES} > {BATTLES_PER_RUN}")
    _test("Epsilon start=1.0", EPS_START == 1.0)
    _test("Epsilon end=0.05", EPS_END == 0.05)
    _test("Epsilon at 0 battles = 1.0", abs(get_epsilon(0) - 1.0) < 1e-6)
    _test("Epsilon at decay end = 0.05", abs(get_epsilon(EPS_DECAY_BATTLES) - 0.05) < 1e-6)
    _test("Epsilon at 2x decay = 0.05 (clamped)", abs(get_epsilon(EPS_DECAY_BATTLES * 2) - 0.05) < 1e-6)
    _test("Epsilon monotonically decreasing",
          get_epsilon(0) > get_epsilon(EPS_DECAY_BATTLES // 2) > get_epsilon(EPS_DECAY_BATTLES))
    _test("4 model names", len(MODEL_NAMES) == 4)
    _test("4 model labels", len(MODEL_LABELS) == 4)
    _test("Battle format is gen4ou", BATTLE_FORMAT == "gen4ou")
    _test("Alpha in (0, 1]", 0 < ALPHA <= 1)
    _test("Gamma in (0, 1]", 0 < GAMMA <= 1)
    _test("Lambda in [0, 1]", 0 <= LAMBDA <= 1)

    # Grid search tests
    _test("Grid has alpha key", "alpha" in GRID)
    _test("Grid has gamma key", "gamma" in GRID)
    _test("Grid has lam key", "lam" in GRID)
    _test("Grid alpha values in (0, 1]", all(0 < v <= 1 for v in GRID["alpha"]))
    _test("Grid gamma values in (0, 1]", all(0 < v <= 1 for v in GRID["gamma"]))
    _test("Grid lam values in [0, 1]", all(0 <= v <= 1 for v in GRID["lam"]))

    combos = build_grid()
    expected_count = 1
    for v in GRID.values():
        expected_count *= len(v)
    _test(f"build_grid produces {expected_count} combos", len(combos) == expected_count,
          f"got {len(combos)}")
    _test("All combo_ids unique", len(set(c["combo_id"] for c in combos)) == len(combos))
    _test("Combo IDs start with hp_", all(c["combo_id"].startswith("hp_") for c in combos))
    _test("First combo has all keys", all(k in combos[0] for k in ["alpha", "gamma", "lam", "combo_id"]))

    # Pool generation tests
    pool1 = get_pool_for_run(1)
    pool1b = get_pool_for_run(1)
    pool2 = get_pool_for_run(2)
    _test(f"Pool for run 1 has {POOL_SIZE} pokemon", len(pool1) == POOL_SIZE)
    _test("Pool is deterministic (same run_id = same pool)", pool1 == pool1b)
    _test("Different run_ids produce different pools", pool1 != pool2)
    _test("Pool species come from full pool", all(s in FULL_POKEMON_POOL for s in pool1))
    _test("Pool species are unique", len(set(pool1)) == POOL_SIZE)
    _test("Pool is sorted", pool1 == sorted(pool1))
    all_pools_valid = True
    for r in range(1, RUNS_PER_COMBO + 1):
        p = get_pool_for_run(r)
        if len(p) != POOL_SIZE or len(set(p)) != POOL_SIZE:
            all_pools_valid = False
            break
    _test(f"All {RUNS_PER_COMBO} run pools valid ({POOL_SIZE} unique species)", all_pools_valid)


def test_features():
    _section("FEATURES")
    from shared.features import AdvancedFeatureExtractor

    ext = AdvancedFeatureExtractor()
    battle = _make_test_battle()

    # --- get_battle_state (20-tuple) ---
    bs = ext.get_battle_state(battle)
    _test("get_battle_state returns tuple", isinstance(bs, tuple))
    _test("get_battle_state length = 20", len(bs) == 20, f"got {len(bs)}")
    _test("battle_state[0] = active species", bs[0] == battle.active_pokemon.species)
    _test("battle_state[7] = opp species", bs[7] == battle.opponent_active_pokemon.species)

    # --- get_bench_detail (15-tuple) ---
    bd = ext.get_bench_detail(battle)
    _test("get_bench_detail returns tuple", isinstance(bd, tuple))
    _test("get_bench_detail length = 15", len(bd) == 15, f"got {len(bd)}")

    # Check sorting: species at positions 0, 3, 6, 9, 12 should be alphabetical
    bench_species = [bd[i] for i in range(0, 15, 3)]
    _test("bench_detail species are sorted", bench_species == sorted(bench_species),
          f"got {bench_species}")

    # Active pokemon should NOT be in bench detail
    active_sp = battle.active_pokemon.species
    _test("active pokemon not in bench_detail", active_sp not in bench_species,
          f"active={active_sp}, bench={bench_species}")

    # --- get_sub_state (17-tuple) ---
    ss = ext.get_sub_state(battle)
    _test("get_sub_state returns tuple", isinstance(ss, tuple))
    _test("get_sub_state length = 17", len(ss) == 17, f"got {len(ss)}")
    _test("sub_state[0] = opp_species", ss[0] == battle.opponent_active_pokemon.species)
    _test("sub_state[1] = opp_hp bucket", ss[1] in (0, 1, 2))
    _test("sub_state[2:] = bench_detail", ss[2:] == bd)

    # --- get_flat_state (35-tuple) ---
    fs = ext.get_flat_state(battle)
    _test("get_flat_state returns tuple", isinstance(fs, tuple))
    _test("get_flat_state length = 35", len(fs) == 35, f"got {len(fs)}")
    _test("flat_state[:20] = battle_state", fs[:20] == bs)
    _test("flat_state[20:] = bench_detail", fs[20:] == bd)

    # --- HP bucketing ---
    _test("HP bucket: 0 hp → 0", ext.get_hp_bucket(0, 100) == 0)
    _test("HP bucket: 15% → 0 (low)", ext.get_hp_bucket(15, 100) == 0)
    _test("HP bucket: 35% → 1 (mid)", ext.get_hp_bucket(35, 100) == 1)
    _test("HP bucket: 80% → 2 (high)", ext.get_hp_bucket(80, 100) == 2)

    # --- Fainted mons in bench detail ---
    battle_fainted = _make_test_battle(fainted_indices=[1, 3])
    bd_f = ext.get_bench_detail(battle_fainted)
    _test("bench_detail with fainted still has 15 elements", len(bd_f) == 15)
    # Fainted mons should have hp_bucket=0
    bench_hps = [bd_f[i] for i in range(1, 15, 3)]
    _test("fainted bench mons have hp_bucket=0",
          sum(1 for h in bench_hps if h == 0) >= 2,
          f"hp buckets: {bench_hps}")

    # --- Different active → different bench detail ---
    battle2 = _make_test_battle()
    battle2.active_pokemon = battle2._team[1]  # Different active
    battle2.available_switches = [m for i, m in enumerate(battle2._team) if i != 1 and not m.fainted]
    bd2 = ext.get_bench_detail(battle2)
    _test("different active → different bench_detail", bd != bd2)


def test_bench_detail_sorting():
    _section("BENCH DETAIL SORTING (ORDER INVARIANCE)")
    from shared.features import AdvancedFeatureExtractor

    ext = AdvancedFeatureExtractor()

    # Create two battles with same team but different internal order
    species_a = ["tyranitar", "metagross", "infernape", "lucario", "togekiss", "gengar"]
    species_b = ["tyranitar", "gengar", "togekiss", "lucario", "infernape", "metagross"]

    team_a = [MockPokemon(sp, current_hp=80, max_hp=100,
                          base_stats={"hp": 100, "atk": 80, "def": 80, "spa": 80, "spd": 80, "spe": 80})
              for sp in species_a]
    team_b = [MockPokemon(sp, current_hp=80, max_hp=100,
                          base_stats={"hp": 100, "atk": 80, "def": 80, "spa": 80, "spd": 80, "spe": 80})
              for sp in species_b]

    opp = [MockPokemon("starmie", base_stats={"hp": 60, "atk": 75, "def": 85, "spa": 100, "spd": 85, "spe": 115})]

    battle_a = MockBattle(team_a, opp)
    battle_b = MockBattle(team_b, opp)

    bd_a = ext.get_bench_detail(battle_a)
    bd_b = ext.get_bench_detail(battle_b)

    _test("same bench (diff order) → same bench_detail", bd_a == bd_b,
          f"\na={bd_a}\nb={bd_b}")


def test_state_uniqueness():
    _section("STATE SPACE UNIQUENESS")
    from shared.features import AdvancedFeatureExtractor
    from shared.config import POKEMON_POOL

    ext = AdvancedFeatureExtractor()

    # Run many random battles and check state diversity
    flat_states = set()
    battle_states = set()
    sub_states = set()

    rng = random.Random(42)
    for _ in range(200):
        my_team_sp = rng.sample(POKEMON_POOL, 6)
        opp_team_sp = rng.sample(POKEMON_POOL, 6)
        my_hps = [rng.randint(0, 100) for _ in range(6)]
        fainted = [i for i in range(6) if my_hps[i] == 0]

        battle = _make_test_battle(my_species=my_team_sp, opp_species=opp_team_sp,
                                   my_hp=my_hps, fainted_indices=fainted)

        fs = ext.get_flat_state(battle)
        bs = ext.get_battle_state(battle)
        ss = ext.get_sub_state(battle)

        flat_states.add(fs)
        battle_states.add(bs)
        sub_states.add(ss)

    _test(f"200 random battles → {len(flat_states)} unique flat states (expect ~200)",
          len(flat_states) >= 150)
    _test(f"200 random battles → {len(battle_states)} unique battle states",
          len(battle_states) >= 100)
    _test(f"flat states >= battle states (35-tuple > 20-tuple)",
          len(flat_states) >= len(battle_states))


def test_switch_hash():
    _section("SWITCH HASH vs MOVE HASH")

    # Ensure switch hashes never collide with move hashes
    from shared.config import POKEMON_POOL

    move_ids = ["thunderbolt", "icebeam", "flamethrower", "earthquake", "closecombat",
                "swordsdance", "stealthrock", "uturn", "surf", "psychic",
                "stoneedge", "ironhead", "dragonclaw", "toxic", "recover"]

    move_hashes = {zlib.adler32(m.encode()) for m in move_ids}
    switch_hashes = {zlib.adler32(("switch_" + sp).encode()) for sp in POKEMON_POOL}

    _test("switch hashes don't collide with move hashes",
          len(move_hashes & switch_hashes) == 0,
          f"collisions: {move_hashes & switch_hashes}")

    from shared.config import POOL_SIZE as _PS
    _test(f"all {_PS} switch hashes are unique",
          len(switch_hashes) == _PS, f"got {len(switch_hashes)}")

    _test("switch hash(-1) not in switch_hashes",
          -1 not in switch_hashes)


def test_flat_zero_player():
    _section("MODEL 1: FLAT + ZERO INIT")
    from model_1_flat_zero.player import FlatZeroPlayer

    p = FlatZeroPlayer.__new__(FlatZeroPlayer)
    # Manually init without connecting to server
    p.extractor = __import__("shared.features", fromlist=["AdvancedFeatureExtractor"]).AdvancedFeatureExtractor()
    p.q_table = {}
    p.active_traces = {}
    p.alpha = 0.1
    p.gamma = 0.995
    p.lam = 0.6967
    p.epsilon = 0.5
    p.last_state_key = None
    p.last_action_hash = None
    p.last_reward_snapshot = None
    p.step_buffer = []

    # Test state extraction
    battle = _make_test_battle()
    state = p.extractor.get_flat_state(battle)
    _test("flat player uses 35-tuple state", len(state) == 35)

    # Test Q-value default
    _test("Q-value default is 0.0", p.get_q_value(state, 12345) == 0.0)

    # Test Q-table update via traces
    p.last_state_key = state
    p.last_action_hash = 12345
    p.active_traces[(state, 12345)] = 1.0
    p._update_traces_and_q(reward=1.0, max_next_q=0.0, next_action_is_greedy=True)
    _test("Q-value updated after trace update",
          p.q_table.get((state, 12345), 0.0) != 0.0)

    # Test action building
    actions = p._build_actions(battle)
    move_actions = [a for a in actions if not a[2]]
    switch_actions = [a for a in actions if a[2]]
    _test(f"_build_actions: {len(move_actions)} moves + {len(switch_actions)} switches",
          len(move_actions) == 2 and len(switch_actions) == 5)

    # Test save/load round-trip
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        tmp_path = f.name
    try:
        p.q_table = {("test_state", 1): 0.5, ("test_state", 2): -0.3}
        p.save_table(tmp_path)
        p.q_table = {}
        p.load_table(tmp_path)
        _test("save/load preserves Q-table",
              len(p.q_table) == 2 and abs(p.q_table[("test_state", 1)] - 0.5) < 1e-6)
    finally:
        os.unlink(tmp_path)


def test_flat_smart_player():
    _section("MODEL 2: FLAT + SMART INIT")
    from model_2_flat_smart.player import FlatSmartPlayer

    p = FlatSmartPlayer.__new__(FlatSmartPlayer)
    p.extractor = __import__("shared.features", fromlist=["AdvancedFeatureExtractor"]).AdvancedFeatureExtractor()
    p.q_table = {}
    p.active_traces = {}
    p.alpha = 0.1
    p.gamma = 0.995
    p.lam = 0.6967
    p.epsilon = 0.5
    p.last_state_key = None
    p.last_action_hash = None
    p.last_reward_snapshot = None
    p.step_buffer = []

    battle = _make_test_battle()
    state = p.extractor.get_flat_state(battle)
    actions = p._build_actions(battle)

    # Before init: all Q=0
    q_before = [p.get_q_value(state, a[0]) for a in actions]
    _test("before init: all Q-values = 0", all(q == 0.0 for q in q_before))

    # After init: all Q-values should be set and sum to ~1 (softmax normalized)
    p._initialize_state_if_needed(battle, state, actions)
    q_after = [p.get_q_value(state, a[0]) for a in actions]
    _test("after init: Q-values are non-zero", all(q != 0.0 for q in q_after))
    _test("after init: Q-values sum ≈ 1.0 (softmax)",
          abs(sum(q_after) - 1.0) < 0.01, f"sum={sum(q_after):.4f}")
    _test("after init: all Q-values > 0 (softmax)", all(q > 0 for q in q_after))

    # Second call should not overwrite
    old_q = dict(p.q_table)
    p._initialize_state_if_needed(battle, state, actions)
    _test("re-init does not overwrite existing Q-values",
          all(abs(p.q_table[k] - old_q[k]) < 1e-10 for k in old_q))


def test_hier_zero_player():
    _section("MODEL 3: HIER + ZERO INIT")
    from model_3_hier_zero.player import HierZeroPlayer

    p = HierZeroPlayer.__new__(HierZeroPlayer)
    p.extractor = __import__("shared.features", fromlist=["AdvancedFeatureExtractor"]).AdvancedFeatureExtractor()
    p.q_table = {}
    p.switch_table = {}
    p.active_traces = {}
    p.switch_traces = {}
    p.alpha = 0.1
    p.gamma = 0.995
    p.lam = 0.6967
    p.epsilon = 0.5
    p.last_state_key = None
    p.last_action_hash = None
    p.last_switch_context = None
    p.last_switch_action_was_greedy = False
    p.last_reward_snapshot = None
    p.step_buffer = []

    battle = _make_test_battle()

    # Master state should be 20-tuple
    master_state = p.extractor.get_battle_state(battle)
    _test("hier master uses 20-tuple state", len(master_state) == 20)

    # Sub state should be 17-tuple
    sub_state = p.extractor.get_sub_state(battle)
    _test("hier sub uses 17-tuple state", len(sub_state) == 17)

    # Switch table keyed by (sub_state, switch_hash)
    switch_hash = zlib.adler32(("switch_" + "metagross").encode())
    key = (sub_state, switch_hash)
    _test("switch_table default is 0.0", p.get_switch_value(sub_state, switch_hash) == 0.0)

    # Set and read back
    p.switch_table[key] = 0.42
    _test("switch_table set/get works", abs(p.get_switch_value(sub_state, switch_hash) - 0.42) < 1e-6)

    # Test trace update with switch context
    p.last_state_key = master_state
    p.last_action_hash = -1  # switch
    p.last_switch_context = key
    p.last_switch_action_was_greedy = True
    p.active_traces[(master_state, -1)] = 1.0
    p.switch_traces[key] = 1.0

    p._update_traces_and_q(reward=0.5, max_next_q=0.0, next_action_is_greedy=True)
    _test("master Q updated via traces",
          p.q_table.get((master_state, -1), 0.0) != 0.0)
    _test("switch Q updated via traces",
          p.switch_table.get(key, 0.0) != 0.42)  # Should have changed from original

    # Save/load with both tables
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        tmp_path = f.name
    try:
        p.q_table = {("ms", 1): 0.5}
        p.switch_table = {("ss", 2): -0.3}
        p.save_table(tmp_path)
        p.q_table = {}
        p.switch_table = {}
        p.load_table(tmp_path)
        _test("save/load preserves both tables",
              len(p.q_table) == 1 and len(p.switch_table) == 1)
    finally:
        os.unlink(tmp_path)


def test_hier_smart_player():
    _section("MODEL 4: HIER + SMART INIT")
    from model_4_hier_smart.player import HierSmartPlayer

    p = HierSmartPlayer.__new__(HierSmartPlayer)
    p.extractor = __import__("shared.features", fromlist=["AdvancedFeatureExtractor"]).AdvancedFeatureExtractor()
    p.q_table = {}
    p.switch_table = {}
    p.active_traces = {}
    p.switch_traces = {}
    p.alpha = 0.1
    p.gamma = 0.995
    p.lam = 0.6967
    p.epsilon = 0.5
    p.last_state_key = None
    p.last_action_hash = None
    p.last_switch_context = None
    p.last_switch_action_was_greedy = False
    p.last_reward_snapshot = None
    p.step_buffer = []

    battle = _make_test_battle()
    state = p.extractor.get_battle_state(battle)

    # Master init
    actions = [(zlib.adler32(m.id.encode()), m) for m in battle.available_moves]
    actions.append((-1, None))

    p._initialize_master_if_needed(battle, state, actions)
    q_vals = [p.get_q_value(state, h) for h, _ in actions]
    _test("master smart init: all actions initialized", all(q != 0.0 for q in q_vals))
    _test("master smart init: values sum ≈ 1.0", abs(sum(q_vals) - 1.0) < 0.01)

    # Switch init
    sub_state = p.extractor.get_sub_state(battle)
    candidates = battle.available_switches
    p._initialize_switch_if_needed(battle, sub_state, candidates)

    switch_vals = []
    for mon in candidates:
        sh = zlib.adler32(("switch_" + mon.species).encode())
        switch_vals.append(p.get_switch_value(sub_state, sh))
    _test("switch smart init: all candidates initialized", all(v != 0.0 for v in switch_vals))
    _test("switch smart init: values sum ≈ 1.0", abs(sum(switch_vals) - 1.0) < 0.01)


def test_rewards():
    _section("REWARDS")
    from shared.rewards import get_dense_reward_snapshot, calculate_step_reward

    battle = _make_test_battle()
    snap = get_dense_reward_snapshot(battle)

    _test("snapshot has my_hp", "my_hp" in snap)
    _test("snapshot has opp_hp", "opp_hp" in snap)
    _test("snapshot has my_fainted", "my_fainted" in snap)
    _test("snapshot has opp_fainted", "opp_fainted" in snap)

    # Null prev → 0 reward
    _test("null prev snapshot → 0 reward", calculate_step_reward(None, snap) == 0.0)

    # Same snapshots → 0 reward
    _test("identical snapshots → 0 reward", calculate_step_reward(snap, snap) == 0.0)

    # Opponent loses HP → positive reward
    snap2 = dict(snap)
    snap2["opp_hp"] = snap["opp_hp"] - 1.0
    r = calculate_step_reward(snap, snap2)
    _test("opponent HP loss → positive reward", r > 0, f"reward={r}")

    # Our mon faints → negative reward
    snap3 = dict(snap)
    snap3["my_fainted"] = snap["my_fainted"] + 1
    r2 = calculate_step_reward(snap, snap3)
    _test("our faint → negative reward", r2 < 0, f"reward={r2}")


def test_team_builder():
    _section("TEAM BUILDER")
    from shared.team_builder import Gen4OUPoolTeambuilder
    from shared.config import POKEMON_POOL

    tb = Gen4OUPoolTeambuilder(seed=42)
    team = tb.yield_team()

    _test("yield_team returns a string", isinstance(team, str))
    _test("yield_team returns non-empty", len(team) > 0)

    # Generate 10 teams and check they're different
    teams = set()
    tb2 = Gen4OUPoolTeambuilder(seed=42)
    for _ in range(10):
        teams.add(tb2.yield_team())
    _test("team builder produces varied teams", len(teams) >= 5,
          f"got {len(teams)} unique teams out of 10")

    # Same seed → same first team
    tb3 = Gen4OUPoolTeambuilder(seed=42)
    tb4 = Gen4OUPoolTeambuilder(seed=42)
    _test("same seed → reproducible teams", tb3.yield_team() == tb4.yield_team())


def test_indexed_team_builder():
    _section("INDEXED TEAM BUILDER (V3 deterministic per-battle teams)")
    from shared.team_builder import IndexedTeambuilder

    # Same base_seed + same battle_index → identical team
    tb_a = IndexedTeambuilder(base_seed=42)
    tb_b = IndexedTeambuilder(base_seed=42)
    tb_a.battle_index = 0
    tb_b.battle_index = 0
    _test("same seed+index → same team", tb_a.yield_team() == tb_b.yield_team())

    # Different battle_index → different teams
    tb_a.battle_index = 0
    tb_b.battle_index = 1
    _test("different index → different team", tb_a.yield_team() != tb_b.yield_team())

    # Different base_seed → different teams (opponent vs learner)
    tb_opp = IndexedTeambuilder(base_seed=1042)
    tb_lrn = IndexedTeambuilder(base_seed=2042)
    tb_opp.battle_index = 5
    tb_lrn.battle_index = 5
    _test("different base_seed → different team (opp vs learner)",
          tb_opp.yield_team() != tb_lrn.yield_team())

    # Order-independent: setting index 100 directly gives same result
    # whether or not we generated indices 0-99 first
    tb_fresh = IndexedTeambuilder(base_seed=42)
    tb_fresh.battle_index = 100
    team_direct = tb_fresh.yield_team()

    tb_used = IndexedTeambuilder(base_seed=42)
    for i in range(100):
        tb_used.battle_index = i
        tb_used.yield_team()
    tb_used.battle_index = 100
    team_after_usage = tb_used.yield_team()
    _test("index 100 same whether fresh or after 0-99", team_direct == team_after_usage)

    # Cross-model determinism: 4 different IndexedTeambuilders with same seed
    # all produce identical teams for each battle index (the core V3 guarantee)
    model_tbs = [IndexedTeambuilder(base_seed=42) for _ in range(4)]
    for battle_idx in [0, 50, 999]:
        teams = []
        for tb in model_tbs:
            tb.battle_index = battle_idx
            teams.append(tb.yield_team())
        _test(f"4 models get same team at index {battle_idx}",
              len(set(teams)) == 1)

    # Variety: different indices produce varied teams
    tb_var = IndexedTeambuilder(base_seed=42)
    unique_teams = set()
    for i in range(20):
        tb_var.battle_index = i
        unique_teams.add(tb_var.yield_team())
    _test("indexed builder produces varied teams across indices",
          len(unique_teams) >= 10, f"got {len(unique_teams)} unique out of 20")


def test_information_parity():
    _section("INFORMATION PARITY (flat vs hier see same info)")
    from shared.features import AdvancedFeatureExtractor

    ext = AdvancedFeatureExtractor()
    battle = _make_test_battle()

    flat_state = ext.get_flat_state(battle)      # 35-tuple
    battle_state = ext.get_battle_state(battle)   # 20-tuple
    sub_state = ext.get_sub_state(battle)          # 17-tuple
    bench_detail = ext.get_bench_detail(battle)    # 15-tuple

    # Flat = battle_state + bench_detail
    _test("flat = battle_state ++ bench_detail",
          flat_state == battle_state + bench_detail)

    # Sub = (opp_species, opp_hp) + bench_detail
    _test("sub starts with opp_species from battle_state",
          sub_state[0] == battle_state[7])  # opp_species is at index 7 in battle_state
    _test("sub starts with opp_hp from battle_state",
          sub_state[1] == battle_state[8])  # opp_hp is at index 8
    _test("sub[2:] = bench_detail", sub_state[2:] == bench_detail)

    # Both models access the SAME total information
    flat_info = set(flat_state)
    hier_info = set(battle_state) | set(sub_state)
    # Not set equality (tuples have positional meaning), but content overlap
    _test("flat state contains all battle_state features",
          flat_state[:20] == battle_state)
    _test("flat state contains all bench_detail features",
          flat_state[20:] == bench_detail)
    _test("hier sub contains all bench_detail features",
          sub_state[2:] == bench_detail)


def test_trace_decay():
    _section("ELIGIBILITY TRACE DECAY")

    # Test that traces decay correctly on greedy actions and clear on exploratory
    from model_1_flat_zero.player import FlatZeroPlayer

    p = FlatZeroPlayer.__new__(FlatZeroPlayer)
    p.extractor = __import__("shared.features", fromlist=["AdvancedFeatureExtractor"]).AdvancedFeatureExtractor()
    p.q_table = {}
    p.active_traces = {}
    p.alpha = 0.1
    p.gamma = 0.995
    p.lam = 0.5
    p.epsilon = 0.5
    p.last_state_key = "state_A"
    p.last_action_hash = 1

    # Add a trace and do greedy update
    # _update_traces_and_q first increments trace by 1.0 (accumulating),
    # then decays all traces by γλ. So: (1.0 + 1.0) * γλ = 2.0 * 0.995 * 0.5
    p.active_traces[("state_A", 1)] = 1.0
    p._update_traces_and_q(reward=0.0, max_next_q=0.0, next_action_is_greedy=True)

    expected_decay = (1.0 + 1.0) * p.gamma * p.lam  # accumulate then decay
    actual = p.active_traces.get(("state_A", 1), 0.0)
    _test("trace accumulates then decays by γλ on greedy", abs(actual - expected_decay) < 0.01,
          f"expected ~{expected_decay:.3f}, got {actual:.3f}")

    # Now do non-greedy: traces should clear
    p.last_state_key = "state_B"
    p.last_action_hash = 2
    p.active_traces[("state_B", 2)] = 1.0
    p._update_traces_and_q(reward=0.0, max_next_q=0.0, next_action_is_greedy=False)
    _test("traces cleared on exploratory action", len(p.active_traces) == 0)


def test_action_masking():
    _section("ACTION MASKING")

    # Test that fainted mons can't be switched to
    battle = _make_test_battle(fainted_indices=[2, 4])
    _test("available_switches excludes fainted mons",
          len(battle.available_switches) == 3,
          f"got {len(battle.available_switches)} (expected 3 of 5 bench alive)")

    # Force switch: no moves available
    battle2 = _make_test_battle()
    battle2.force_switch = True
    battle2.available_moves = []  # No moves during force switch

    from model_1_flat_zero.player import FlatZeroPlayer
    p = FlatZeroPlayer.__new__(FlatZeroPlayer)
    p.extractor = __import__("shared.features", fromlist=["AdvancedFeatureExtractor"]).AdvancedFeatureExtractor()
    p.q_table = {}
    actions = p._build_actions(battle2)
    move_actions = [a for a in actions if not a[2]]
    switch_actions = [a for a in actions if a[2]]
    _test("force_switch: no move actions", len(move_actions) == 0)
    _test("force_switch: switch actions available", len(switch_actions) == 5)


def test_train_common():
    _section("TRAIN COMMON UTILITIES")
    from shared.train_common import get_table_size, write_live_status

    # Test get_table_size with mock player
    class MockPlayer:
        def __init__(self):
            self.q_table = {i: 0 for i in range(100)}

    class MockHierPlayer:
        def __init__(self):
            self.q_table = {i: 0 for i in range(100)}
            self.switch_table = {i: 0 for i in range(50)}

    _test("get_table_size flat", get_table_size(MockPlayer()) == 100)
    _test("get_table_size hier (q + switch)", get_table_size(MockHierPlayer()) == 150)

    # Test atomic write
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
        tmp_path = f.name
    try:
        write_live_status(tmp_path, {"battles": 1000, "speed": 20.5})
        import json
        with open(tmp_path) as f:
            data = json.load(f)
        _test("write_live_status writes valid JSON", data["battles"] == 1000)
        _test("write_live_status has speed field", abs(data["speed"] - 20.5) < 0.01)
    finally:
        os.unlink(tmp_path)
        tmp_file = tmp_path + ".tmp"
        if os.path.exists(tmp_file):
            os.unlink(tmp_file)


def test_integration_short_battle():
    """Run a quick 5-battle integration test for each model against SimpleHeuristics."""
    _section("INTEGRATION (5 battles per model, requires Showdown on port 9000)")

    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('localhost', 9000)) != 0:
            print("  ⚠ Showdown not running on port 9000 — skipping integration tests")
            print("  ⚠ Start it with: node pokemon-showdown start --no-security 9000")
            return

    import asyncio
    from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
    from poke_env.player import SimpleHeuristicsPlayer
    from shared.team_builder import Gen4OUPoolTeambuilder

    models = [
        ("model_1_flat_zero", "FlatZeroPlayer"),
        ("model_2_flat_smart", "FlatSmartPlayer"),
        ("model_3_hier_zero", "HierZeroPlayer"),
        ("model_4_hier_smart", "HierSmartPlayer"),
    ]

    for model_dir, class_name in models:
        try:
            mod = __import__(f"{model_dir}.player", fromlist=[class_name])
            PlayerClass = getattr(mod, class_name)

            import uuid
            uid = uuid.uuid4().hex[:6]
            # Create unique subclass to avoid name collision on server
            UniquePlayer = type(f"Test_{class_name}_{uid}", (PlayerClass,), {})
            UniqueOpp = type(f"TestOpp_{uid}", (SimpleHeuristicsPlayer,), {})

            player = UniquePlayer(
                battle_format="gen4ou",
                server_configuration=LocalhostServerConfiguration,
                max_concurrent_battles=1,
                team=Gen4OUPoolTeambuilder(seed=42),
            )
            opponent = UniqueOpp(
                battle_format="gen4ou",
                server_configuration=LocalhostServerConfiguration,
                max_concurrent_battles=1,
                team=Gen4OUPoolTeambuilder(seed=43),
            )

            asyncio.get_event_loop().run_until_complete(
                asyncio.wait_for(
                    player.battle_against(opponent, n_battles=5),
                    timeout=60,
                )
            )

            wins = player.n_won_battles
            total = player.n_finished_battles
            q_size = len(player.q_table)
            switch_size = len(player.switch_table) if hasattr(player, 'switch_table') else 0

            _test(f"{class_name}: 5 battles completed ({wins}/{total} wins, Q={q_size}, S={switch_size})",
                  total == 5)

        except Exception as e:
            _test(f"{class_name}: integration test", False, f"error: {e}")
            traceback.print_exc()


def test_grid_search():
    _section("GRID SEARCH")
    import csv
    import json
    import tempfile
    import shutil

    from shared.config import build_grid, GRID, MODEL_NAMES, RUNS_PER_COMBO, get_pool_for_run

    # --- Port allocation ---
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from run_grid import _get_port

    # Test slot-based port allocation
    _test("grid port slot 0 = 9000", _get_port(0) == 9000)
    _test("grid port slot 7 = 9007", _get_port(7) == 9007)

    # 8 slots = 8 unique ports
    slots = set(_get_port(i) for i in range(8))
    _test("8 slots = 8 unique ports", len(slots) == 8)
    _test("all grid ports >= 9000", all(p >= 9000 for p in slots))

    # --- Grid combo generation ---
    combos = build_grid()
    _test("combos have sequential IDs",
          [c["combo_id"] for c in combos[:3]] == ["hp_001", "hp_002", "hp_003"])

    # Verify each combo has valid HP ranges
    for combo in combos:
        ok = (combo["alpha"] in GRID["alpha"] and
              combo["gamma"] in GRID["gamma"] and
              combo["lam"] in GRID["lam"])
        if not ok:
            _test(f"combo {combo['combo_id']} has valid grid values", False, str(combo))
            break
    else:
        _test("all combos have valid grid values", True)

    # --- Progress reading ---
    from run_grid import _get_actual_progress, _get_final_rolling_wr

    tmpdir = tempfile.mkdtemp()
    try:
        # Create fake run directory structure: run_dir/model_name/logs/run_1.csv
        model_log_dir = os.path.join(tmpdir, "model_1_flat_zero", "logs")
        os.makedirs(model_log_dir)
        csv_path = os.path.join(model_log_dir, "run_1.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Battles", "RollingWin", "OverallWin", "Epsilon", "Speed", "AvgReward", "TableSize"])
            writer.writerow([1000, "0.3500", "0.3500", "0.9900", "10.0", "-0.3000", 50000])
            writer.writerow([2000, "0.4200", "0.3850", "0.9800", "10.5", "-0.1500", 80000])

        battles, wins = _get_actual_progress(tmpdir, "model_1_flat_zero")
        _test("progress reads 2000 battles", battles == 2000)
        _test("progress computes wins from overall_wr", wins == int(round(0.385 * 2000)))

        wr = _get_final_rolling_wr(tmpdir, "model_1_flat_zero")
        _test("final rolling WR = 0.42", abs(wr - 0.42) < 0.001, f"got {wr}")

        # No logs = 0 progress
        b, w = _get_actual_progress(tmpdir, "model_2_flat_smart")
        _test("missing model returns 0 progress", b == 0 and w == 0)
    finally:
        shutil.rmtree(tmpdir)

    # --- train_common accepts HP overrides + pool ---
    from shared.train_common import parse_train_args
    old_argv = sys.argv
    sys.argv = ["train.py", "--alpha", "0.05", "--gamma_val", "0.95", "--lam", "0.7",
                "--output_dir", "/tmp/test_output",
                "--pool", "tyranitar,metagross,infernape,lucario,togekiss,gengar,starmie,scizor,gyarados"]
    args = parse_train_args()
    sys.argv = old_argv
    _test("parse_train_args reads --alpha", args.alpha == 0.05)
    _test("parse_train_args reads --gamma_val", args.gamma_val == 0.95)
    _test("parse_train_args reads --lam", args.lam == 0.7)
    _test("parse_train_args reads --output_dir", args.output_dir == "/tmp/test_output")
    _test("parse_train_args reads --pool", args.pool == "tyranitar,metagross,infernape,lucario,togekiss,gengar,starmie,scizor,gyarados")
    _test("pool splits to correct species count", len(args.pool.split(",")) == 9)  # 9 in test input string


# =============================================================================
# RUNNER
# =============================================================================

def run_all_tests(skip_integration=False):
    global _passed, _failed, _errors
    _passed = 0
    _failed = 0
    _errors = []

    print("\n" + "=" * 60)
    print("  RESEARCH EXPERIMENT V4 — GRID SEARCH TEST SUITE")
    print("=" * 60)

    start = time.time()

    test_config()
    test_features()
    test_bench_detail_sorting()
    test_state_uniqueness()
    test_switch_hash()
    test_flat_zero_player()
    test_flat_smart_player()
    test_hier_zero_player()
    test_hier_smart_player()
    test_rewards()
    test_team_builder()
    test_indexed_team_builder()
    test_information_parity()
    test_trace_decay()
    test_action_masking()
    test_train_common()
    test_grid_search()

    if not skip_integration:
        test_integration_short_battle()
    else:
        print("\n  ⚠ Integration tests skipped (--skip-integration)")

    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {_passed} passed, {_failed} failed  ({elapsed:.1f}s)")
    print(f"{'=' * 60}")

    if _errors:
        print("\n  FAILURES:")
        for e in _errors:
            print(f"    {e}")

    return _failed == 0
