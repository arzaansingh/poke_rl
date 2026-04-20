"""
HierSmartPlayer — Hierarchical + Smart Init + Watkins Q(lambda) learner.

Copied verbatim from V5's `model_4_hier_smart/player.py` (the winning V5
configuration) so V6 has no dependency on V5. Only the sys.path / shared.*
imports are adjusted to point at V6's own `shared/` package.

Master agent: 20-tuple state (battle_state), actions: {move hashes, -1=switch}
Sub-agent:    17-tuple state (opp context + sorted bench detail),
              actions: {switch species hashes}
Standardized heuristic priors for both tables.
"""

import os
import sys
import random
import pickle
import zlib
import gc
import logging

# Ensure V6's root is on sys.path so `from shared.X import ...` resolves to
# V6's own shared/ package.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_V6_DIR = os.path.dirname(_THIS_DIR)
if _V6_DIR not in sys.path:
    sys.path.insert(0, _V6_DIR)

from poke_env.player.player import Player           # noqa: E402
from poke_env.battle.pokemon import Pokemon          # noqa: E402

from shared.features import AdvancedFeatureExtractor                         # noqa: E402
from shared.heuristics import HeuristicEngine                                # noqa: E402
from shared.rewards import get_dense_reward_snapshot, calculate_step_reward  # noqa: E402


_original_available_moves = Pokemon.available_moves_from_request
def patched_available_moves(self, request):
    try:
        return _original_available_moves(self, request)
    except AssertionError:
        return []
Pokemon.available_moves_from_request = patched_available_moves


def _switch_hash(species):
    return zlib.adler32(("switch_" + species).encode())


class HierSmartPlayer(Player):
    def __init__(self, battle_format="gen4ou", alpha=0.1, gamma=0.995, lam=0.6967, epsilon=0.1, **kwargs):
        super().__init__(battle_format=battle_format, **kwargs)
        self.extractor = AdvancedFeatureExtractor()

        self.q_table = {}
        self.switch_table = {}

        self.active_traces = {}
        self.switch_traces = {}

        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon

        self.last_state_key = None
        self.last_action_hash = None
        self.last_switch_context = None      # (sub_state, switch_hash)
        self.last_switch_action_was_greedy = False

        self.last_reward_snapshot = None
        self.step_buffer = []

    # --- HEURISTIC INITIALIZATION ---

    def _initialize_master_if_needed(self, battle, state_key, possible_actions):
        missing = [h for h, _ in possible_actions if (state_key, h) not in self.q_table]
        if not missing:
            return

        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        raw_scores = []
        for action_hash, move_obj in possible_actions:
            if action_hash == -1:
                score = HeuristicEngine.get_master_switch_score(battle)
            else:
                score = HeuristicEngine.get_move_score(battle, move_obj, active, opponent)
            raw_scores.append(score)

        priors = HeuristicEngine.build_q_priors(raw_scores)

        for i, (action_hash, _) in enumerate(possible_actions):
            if (state_key, action_hash) not in self.q_table:
                self.q_table[(state_key, action_hash)] = priors[i]

    def _initialize_switch_if_needed(self, battle, sub_state, candidates):
        """Seed unseen switches with heuristic priors standardized to [-0.01, 0.01]."""
        missing = [mon for mon in candidates
                   if (sub_state, _switch_hash(mon.species)) not in self.switch_table]
        if not missing:
            return

        opponent = battle.opponent_active_pokemon
        raw_scores = []
        for mon in candidates:
            score = HeuristicEngine.get_switch_score(battle, mon, opponent)
            raw_scores.append(score)

        priors = HeuristicEngine.build_q_priors(raw_scores)

        for i, mon in enumerate(candidates):
            key = (sub_state, _switch_hash(mon.species))
            if key not in self.switch_table:
                self.switch_table[key] = priors[i]

    # --- Q-LEARNING ---

    def pop_step_rewards(self):
        out = self.step_buffer
        self.step_buffer = []
        return out

    def get_q_value(self, state_key, action_hash):
        return self.q_table.get((state_key, action_hash), 0.0)

    def get_switch_value(self, sub_state_key, switch_hash):
        return self.switch_table.get((sub_state_key, switch_hash), 0.0)

    def _update_traces_and_q(self, reward, max_next_q, next_action_is_greedy):
        old_q = self.get_q_value(self.last_state_key, self.last_action_hash)
        delta = reward + self.gamma * max_next_q - old_q

        # --- Switch traces ---
        if self.last_switch_context:
            self.switch_traces[self.last_switch_context] = \
                self.switch_traces.get(self.last_switch_context, 0.0) + 1.0

        switch_to_remove = []
        for s_key, e_val in self.switch_traces.items():
            self.switch_table[s_key] = self.switch_table.get(s_key, 0.0) + self.alpha * delta * e_val
            new_e = e_val * self.gamma * self.lam \
                if (next_action_is_greedy and self.last_switch_action_was_greedy) else 0.0
            if new_e < 0.001:
                switch_to_remove.append(s_key)
            else:
                self.switch_traces[s_key] = new_e
        for k in switch_to_remove:
            del self.switch_traces[k]

        # --- Master traces ---
        trace_key = (self.last_state_key, self.last_action_hash)
        self.active_traces[trace_key] = self.active_traces.get(trace_key, 0.0) + 1.0

        keys_to_remove = []
        for key, e_val in self.active_traces.items():
            self.q_table[key] = self.q_table.get(key, 0.0) + self.alpha * delta * e_val
            new_e = e_val * self.gamma * self.lam if next_action_is_greedy else 0.0
            if new_e < 0.001:
                keys_to_remove.append(key)
            else:
                self.active_traces[key] = new_e
        for k in keys_to_remove:
            del self.active_traces[k]

        if not next_action_is_greedy:
            self.active_traces.clear()
            self.switch_traces.clear()
            self.last_switch_action_was_greedy = False
            self.last_switch_context = None

    def choose_move(self, battle):
        current_snapshot = get_dense_reward_snapshot(battle)
        step_reward = calculate_step_reward(self.last_reward_snapshot, current_snapshot)
        if step_reward != 0:
            self.step_buffer.append(step_reward)
        self.last_reward_snapshot = current_snapshot

        state_key = self.extractor.get_battle_state(battle)

        possible_actions = []
        if not battle.force_switch and battle.active_pokemon and not battle.active_pokemon.fainted:
            for move in battle.available_moves:
                possible_actions.append((zlib.adler32(move.id.encode()), move))
        if battle.available_switches:
            possible_actions.append((-1, None))

        if not possible_actions:
            return self.choose_random_move(battle)

        self._initialize_master_if_needed(battle, state_key, possible_actions)

        q_values = [self.get_q_value(state_key, a_hash) for a_hash, _ in possible_actions]

        max_q = max(q_values) if q_values else 0.0
        best_indices = [i for i, q in enumerate(q_values) if q == max_q]
        greedy_idx = random.choice(best_indices)

        if random.random() < self.epsilon:
            chosen_idx = random.randint(0, len(possible_actions) - 1)
            is_greedy = (q_values[chosen_idx] == max_q)
        else:
            chosen_idx = greedy_idx
            is_greedy = True

        chosen_hash, chosen_obj = possible_actions[chosen_idx]

        if self.last_state_key is not None:
            self._update_traces_and_q(step_reward, max_q, is_greedy)

        self.last_state_key = state_key
        self.last_action_hash = chosen_hash

        if chosen_hash == -1:
            self.last_switch_context = None
            return self._sub_agent_switch(battle, is_greedy)
        else:
            self.last_switch_context = None
            return self.create_order(chosen_obj)

    def _sub_agent_switch(self, battle, parent_greedy):
        available = battle.available_switches
        if not available:
            return self.choose_random_move(battle)

        sub_state = self.extractor.get_sub_state(battle)
        self._initialize_switch_if_needed(battle, sub_state, available)

        candidates = []
        for mon in available:
            s_hash = _switch_hash(mon.species)
            q_val = self.get_switch_value(sub_state, s_hash)
            candidates.append((mon, s_hash, q_val))

        best_q = max(c[2] for c in candidates)
        best_indices = [i for i, c in enumerate(candidates) if c[2] == best_q]
        greedy_idx = random.choice(best_indices)

        if random.random() < self.epsilon:
            chosen_idx = random.randint(0, len(candidates) - 1)
            is_sub_greedy = (candidates[chosen_idx][2] == best_q)
        else:
            chosen_idx = greedy_idx
            is_sub_greedy = True

        chosen_mon, chosen_hash, _ = candidates[chosen_idx]
        self.last_switch_context = (sub_state, chosen_hash)
        self.last_switch_action_was_greedy = is_sub_greedy and parent_greedy
        return self.create_order(chosen_mon)

    def _battle_finished(self, battle, won):
        current_snapshot = get_dense_reward_snapshot(battle)
        step_reward = calculate_step_reward(self.last_reward_snapshot, current_snapshot)
        final_reward = step_reward + (1.0 if won else -1.0)

        if self.last_switch_context:
            old_v = self.switch_table.get(self.last_switch_context, 0.0)
            self.switch_table[self.last_switch_context] = old_v + self.alpha * (final_reward - old_v)

        if self.last_state_key is not None:
            self._update_traces_and_q(final_reward, 0.0, True)

        self.last_state_key = None
        self.last_action_hash = None
        self.active_traces.clear()
        self.switch_traces.clear()
        self.last_switch_context = None
        self.last_switch_action_was_greedy = False
        self.last_reward_snapshot = None
        self._n_finished_battles += 1
        if won:
            self._n_won_battles += 1

    def save_table(self, path):
        gc.disable()
        tmp = path + ".tmp"
        try:
            snapshot = {'q': dict(self.q_table), 'switch': dict(self.switch_table)}
            with open(tmp, 'wb') as f:
                pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
        except Exception as e:
            print(f"Save error: {e}")
            if os.path.exists(tmp):
                os.remove(tmp)
        finally:
            gc.enable()

    def load_table(self, path):
        gc.disable()
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data.get('q', {})
                self.switch_table = data.get('switch', {})
            logging.critical(f"Loaded Q ({len(self.q_table)}) Switch ({len(self.switch_table)})")
        except Exception as e:
            logging.critical(f"Fresh start: {e}")
        finally:
            gc.enable()
