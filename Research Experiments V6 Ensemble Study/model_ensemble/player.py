"""
V6 EnsemblePlayer — combines K trained Q-tables at inference time.

Composition design: the EnsemblePlayer is a single `poke_env.Player` subclass
(one websocket, one Showdown account) that holds K lightweight `MemberQTable`
containers. Each member is just {q_table, switch_table} loaded from disk —
no Player machinery per member. This keeps memory sane and avoids spawning
K websockets.

At choose_move time:
  1. Extract the battle state ONCE using V5's AdvancedFeatureExtractor.
  2. Enumerate possible actions.
  3. For each member, seed any unseen (state, action) priors using V5's
     HeuristicEngine (generalized optimistic initialization — literature
     flags this as a methodological micro-novelty).
  4. Build a K × |actions| Q-matrix, combine per the configured strategy:
       - 'soft':        argmax over mean Q across members
       - 'hard':        plurality vote over per-member argmaxes (tie -> mean Q)
       - 'confidence':  weight member k by (max_k - mean_k) spread, then mean
  5. If master picks -1 (switch), recurse into the same ensembling over the
     switch_table for sub-agent selection.

Optional diagnostics (enabled via flags) log per-decision:
  - pairwise argmax disagreement (defends against ensemble collapse)
  - unseen-state fallback rate (how often heuristic priors fire)
"""

import os
import sys
import random
import pickle
import zlib
import gc
import logging
from collections import Counter
from typing import List, Tuple

# Ensure V6's root is on sys.path so `from shared.X import ...` resolves to
# V6's own shared/ package.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_V6_DIR = os.path.dirname(_THIS_DIR)
if _V6_DIR not in sys.path:
    sys.path.insert(0, _V6_DIR)

from poke_env.player.player import Player                      # noqa: E402
from poke_env.battle.pokemon import Pokemon                    # noqa: E402

from shared.features import AdvancedFeatureExtractor           # noqa: E402
from shared.heuristics import HeuristicEngine                  # noqa: E402


# Same Pokemon gen1/4 move patch that V5's HierSmartPlayer applies.
_original_available_moves = Pokemon.available_moves_from_request
def _patched_available_moves(self, request):
    try:
        return _original_available_moves(self, request)
    except AssertionError:
        return []
Pokemon.available_moves_from_request = _patched_available_moves


def _switch_hash(species: str) -> int:
    """Must match V5's HierSmartPlayer._switch_hash."""
    return zlib.adler32(("switch_" + species).encode())


# ──────────────────────────────────────────────────────────────────────
# MemberQTable: lightweight container for one trained member's tables.
# ──────────────────────────────────────────────────────────────────────

class MemberQTable:
    """Holds (q_table, switch_table) from one trained ensemble member.

    Not a poke_env.Player.  Just a data container + convenience lookups.
    """

    __slots__ = ("q_table", "switch_table", "path")

    def __init__(self, q_table: dict, switch_table: dict, path: str = ""):
        self.q_table = q_table
        self.switch_table = switch_table
        self.path = path

    @classmethod
    def load(cls, path: str) -> "MemberQTable":
        gc.disable()
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            q = data.get("q", {})
            s = data.get("switch", {})
        finally:
            gc.enable()
        logging.info(f"[EnsemblePlayer] loaded {path}  Q={len(q)}  Switch={len(s)}")
        return cls(q_table=q, switch_table=s, path=path)

    def get_q(self, state_key, action_hash) -> float:
        return self.q_table.get((state_key, action_hash), 0.0)

    def get_switch(self, sub_state_key, switch_hash) -> float:
        return self.switch_table.get((sub_state_key, switch_hash), 0.0)


# ──────────────────────────────────────────────────────────────────────
# EnsemblePlayer
# ──────────────────────────────────────────────────────────────────────

class EnsemblePlayer(Player):
    """Single poke_env.Player backed by K frozen MemberQTable instances."""

    VALID_STRATEGIES = ("soft", "hard", "confidence")

    def __init__(
        self,
        member_paths: List[str],
        strategy: str = "soft",
        log_disagreement: bool = False,
        log_unseen_rate: bool = False,
        epsilon: float = 0.0,
        battle_format: str = "gen4ou",
        **kwargs,
    ):
        super().__init__(battle_format=battle_format, **kwargs)

        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"strategy={strategy!r} not in {self.VALID_STRATEGIES}"
            )

        self.strategy = strategy
        self.epsilon = epsilon
        self.extractor = AdvancedFeatureExtractor()   # one shared extractor
        self.members: List[MemberQTable] = [
            MemberQTable.load(p) for p in member_paths
        ]
        self.K = len(self.members)
        if self.K == 0:
            raise ValueError("EnsemblePlayer requires at least one member")

        # Diagnostics
        self.log_disagreement = log_disagreement
        self.log_unseen_rate = log_unseen_rate
        self.disagreement_log: List[float] = []   # per-decision argmax-disagreement fraction
        self.unseen_log: List[float] = []         # per-decision fraction of members hitting fallback

    # ────────────────────────────────────────────────────────────────
    # Core: choose_move
    # ────────────────────────────────────────────────────────────────

    def choose_move(self, battle):
        state_key = self.extractor.get_battle_state(battle)

        # Enumerate actions exactly like HierSmartPlayer does.
        possible_actions: List[Tuple[int, object]] = []
        if (not battle.force_switch
                and battle.active_pokemon
                and not battle.active_pokemon.fainted):
            for move in battle.available_moves:
                possible_actions.append((zlib.adler32(move.id.encode()), move))
        if battle.available_switches:
            possible_actions.append((-1, None))

        if not possible_actions:
            return self.choose_random_move(battle)

        # Per-member: seed priors for any unseen (state, action) pairs and
        # count how many members hit the heuristic fallback for this state.
        n_unseen_members = self._seed_master_priors(battle, state_key, possible_actions)
        if self.log_unseen_rate:
            self.unseen_log.append(n_unseen_members / self.K)

        # Build K × |actions| Q-matrix.
        q_matrix = [
            [m.get_q(state_key, a_hash) for a_hash, _ in possible_actions]
            for m in self.members
        ]

        # Log pairwise disagreement (fraction of distinct argmax actions).
        if self.log_disagreement:
            argmaxes = [row.index(max(row)) for row in q_matrix]
            self.disagreement_log.append(len(set(argmaxes)) / self.K)

        # Combine and pick action.
        chosen_idx = self._combine_master(q_matrix)

        # ε-greedy at ensemble level (default 0.0 at eval).
        if self.epsilon > 0 and random.random() < self.epsilon:
            chosen_idx = random.randint(0, len(possible_actions) - 1)

        chosen_hash, chosen_obj = possible_actions[chosen_idx]

        if chosen_hash == -1:
            return self._sub_agent_switch_ensemble(battle)
        return self.create_order(chosen_obj)

    # ────────────────────────────────────────────────────────────────
    # Sub-agent switch (same ensembling dance on switch_table)
    # ────────────────────────────────────────────────────────────────

    def _sub_agent_switch_ensemble(self, battle):
        candidates = battle.available_switches
        if not candidates:
            return self.choose_random_move(battle)

        sub_state = self.extractor.get_sub_state(battle)

        # Seed priors per-member for unseen switches.
        self._seed_switch_priors(battle, sub_state, candidates)

        # Build K × |candidates| matrix of switch-Q values.
        switch_hashes = [_switch_hash(mon.species) for mon in candidates]
        q_matrix = [
            [m.get_switch(sub_state, h) for h in switch_hashes]
            for m in self.members
        ]

        chosen_idx = self._combine_master(q_matrix)   # same combination rule
        return self.create_order(candidates[chosen_idx])

    # ────────────────────────────────────────────────────────────────
    # Per-member prior seeding (generalized optimistic init)
    # ────────────────────────────────────────────────────────────────

    def _seed_master_priors(self, battle, state_key, possible_actions) -> int:
        """For each member, fill in any missing (state, action) Q-values with
        heuristic priors.  Returns number of members that hit the fallback
        (i.e. had at least one missing action for this state)."""
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        # Compute raw heuristic scores ONCE — they don't depend on which
        # member we're seeding (same formula, same battle state).
        raw_scores = []
        for action_hash, move_obj in possible_actions:
            if action_hash == -1:
                score = HeuristicEngine.get_master_switch_score(battle)
            else:
                score = HeuristicEngine.get_move_score(battle, move_obj, active, opponent)
            raw_scores.append(score)
        priors = HeuristicEngine.build_q_priors(raw_scores)

        n_unseen_members = 0
        for m in self.members:
            missing = False
            for i, (a_hash, _) in enumerate(possible_actions):
                if (state_key, a_hash) not in m.q_table:
                    m.q_table[(state_key, a_hash)] = priors[i]
                    missing = True
            if missing:
                n_unseen_members += 1
        return n_unseen_members

    def _seed_switch_priors(self, battle, sub_state, candidates) -> int:
        opponent = battle.opponent_active_pokemon
        raw_scores = [
            HeuristicEngine.get_switch_score(battle, mon, opponent)
            for mon in candidates
        ]
        priors = HeuristicEngine.build_q_priors(raw_scores)

        n_unseen_members = 0
        for m in self.members:
            missing = False
            for i, mon in enumerate(candidates):
                key = (sub_state, _switch_hash(mon.species))
                if key not in m.switch_table:
                    m.switch_table[key] = priors[i]
                    missing = True
            if missing:
                n_unseen_members += 1
        return n_unseen_members

    # ────────────────────────────────────────────────────────────────
    # Combination strategies
    # ────────────────────────────────────────────────────────────────

    def _combine_master(self, q_matrix: List[List[float]]) -> int:
        """Return the chosen action index given a K × |actions| Q-matrix."""
        n_actions = len(q_matrix[0])

        if self.strategy == "soft":
            mean_q = [
                sum(row[i] for row in q_matrix) / self.K
                for i in range(n_actions)
            ]
            return self._argmax_with_random_tiebreak(mean_q)

        if self.strategy == "hard":
            argmaxes = [self._argmax_with_random_tiebreak(row) for row in q_matrix]
            vote_counts = Counter(argmaxes)
            top_count = max(vote_counts.values())
            winners = [a for a, c in vote_counts.items() if c == top_count]
            if len(winners) == 1:
                return winners[0]
            # Tie-break on mean Q.
            mean_q_sub = {
                a: sum(row[a] for row in q_matrix) / self.K
                for a in winners
            }
            top_mean = max(mean_q_sub.values())
            top_winners = [a for a, q in mean_q_sub.items() if q == top_mean]
            return random.choice(top_winners)

        if self.strategy == "confidence":
            # Weight member k by (max_k - mean_k): rewards peaky, decisive members.
            weights = []
            for row in q_matrix:
                row_max = max(row)
                row_mean = sum(row) / n_actions
                weights.append(max(row_max - row_mean, 0.0))
            w_sum = sum(weights)
            if w_sum <= 0.0:
                # All members flat — fall back to uniform (soft) voting.
                weighted = [
                    sum(row[i] for row in q_matrix) / self.K
                    for i in range(n_actions)
                ]
            else:
                weighted = [
                    sum(q_matrix[k][i] * weights[k] for k in range(self.K)) / w_sum
                    for i in range(n_actions)
                ]
            return self._argmax_with_random_tiebreak(weighted)

        raise ValueError(f"Unknown strategy {self.strategy}")

    @staticmethod
    def _argmax_with_random_tiebreak(values: List[float]) -> int:
        best = max(values)
        best_indices = [i for i, v in enumerate(values) if v == best]
        return random.choice(best_indices) if len(best_indices) > 1 else best_indices[0]

    # ────────────────────────────────────────────────────────────────
    # Diagnostics helpers
    # ────────────────────────────────────────────────────────────────

    def reset_diagnostics(self):
        self.disagreement_log.clear()
        self.unseen_log.clear()

    def diagnostics_summary(self) -> dict:
        def stats(xs):
            if not xs:
                return {"n": 0, "mean": None, "min": None, "max": None}
            return {
                "n": len(xs),
                "mean": sum(xs) / len(xs),
                "min": min(xs),
                "max": max(xs),
            }
        return {
            "K": self.K,
            "strategy": self.strategy,
            "disagreement": stats(self.disagreement_log),
            "unseen_rate": stats(self.unseen_log),
        }
