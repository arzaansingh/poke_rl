"""
Model 2: Flat + Smart Init
============================
Single Q-table, 35-tuple state (battle_20 + bench_detail_15),
heuristic softmax initialization for unseen states.
"""

import random
import pickle
import os
import zlib
import math
import gc
import logging
from poke_env.player.player import Player
from poke_env.battle.pokemon import Pokemon

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.features import AdvancedFeatureExtractor
from shared.heuristics import HeuristicEngine
from shared.rewards import get_dense_reward_snapshot, calculate_step_reward

_original_available_moves = Pokemon.available_moves_from_request
def patched_available_moves(self, request):
    try: return _original_available_moves(self, request)
    except AssertionError: return []
Pokemon.available_moves_from_request = patched_available_moves


def _switch_hash(species):
    return zlib.adler32(("switch_" + species).encode())


class FlatSmartPlayer(Player):
    def __init__(self, battle_format="gen4ou", alpha=0.1, gamma=0.995, lam=0.6967, epsilon=0.1, **kwargs):
        super().__init__(battle_format=battle_format, **kwargs)
        self.extractor = AdvancedFeatureExtractor()
        self.q_table = {}
        self.active_traces = {}
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.last_state_key = None
        self.last_action_hash = None
        self.last_reward_snapshot = None
        self.step_buffer = []

    def _initialize_state_if_needed(self, battle, state_key, actions):
        """Heuristic softmax init for unseen (state, action) pairs."""
        missing = [a_hash for a_hash, _, _ in actions if (state_key, a_hash) not in self.q_table]
        if not missing:
            return

        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        raw_scores = []
        for _, action_obj, is_switch in actions:
            if is_switch:
                score = HeuristicEngine.get_switch_score(battle, action_obj, opponent) * 50
            else:
                score = HeuristicEngine.get_move_score(battle, action_obj, active, opponent)
            raw_scores.append(score)

        max_s = max(raw_scores) if raw_scores else 0
        exp_scores = [math.exp((s - max_s) / 10.0) for s in raw_scores]
        sum_exp = sum(exp_scores)
        normalized = [e / sum_exp for e in exp_scores]

        for i, (a_hash, _, _) in enumerate(actions):
            if (state_key, a_hash) not in self.q_table:
                self.q_table[(state_key, a_hash)] = normalized[i]

    def pop_step_rewards(self):
        out = self.step_buffer
        self.step_buffer = []
        return out

    def get_q_value(self, state_key, action_hash):
        return self.q_table.get((state_key, action_hash), 0.0)

    def _update_traces_and_q(self, reward, max_next_q, next_action_is_greedy):
        old_q = self.get_q_value(self.last_state_key, self.last_action_hash)
        delta = reward + self.gamma * max_next_q - old_q

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

    def _build_actions(self, battle):
        actions = []
        if not battle.force_switch and battle.active_pokemon and not battle.active_pokemon.fainted:
            for move in battle.available_moves:
                actions.append((zlib.adler32(move.id.encode()), move, False))
        for mon in battle.available_switches:
            actions.append((_switch_hash(mon.species), mon, True))
        return actions

    def choose_move(self, battle):
        current_snapshot = get_dense_reward_snapshot(battle)
        step_reward = calculate_step_reward(self.last_reward_snapshot, current_snapshot)
        if step_reward != 0:
            self.step_buffer.append(step_reward)
        self.last_reward_snapshot = current_snapshot

        state_key = self.extractor.get_flat_state(battle)
        actions = self._build_actions(battle)

        if not actions:
            return self.choose_random_move(battle)

        self._initialize_state_if_needed(battle, state_key, actions)

        q_values = [self.get_q_value(state_key, a_hash) for a_hash, _, _ in actions]

        max_q = max(q_values)
        best_indices = [i for i, q in enumerate(q_values) if q == max_q]
        greedy_idx = random.choice(best_indices)

        if random.random() < self.epsilon:
            chosen_idx = random.randint(0, len(actions) - 1)
            is_greedy = (q_values[chosen_idx] == max_q)
        else:
            chosen_idx = greedy_idx
            is_greedy = True

        chosen_hash, chosen_obj, is_switch = actions[chosen_idx]

        if self.last_state_key is not None:
            self._update_traces_and_q(step_reward, max_q, is_greedy)

        self.last_state_key = state_key
        self.last_action_hash = chosen_hash
        return self.create_order(chosen_obj)

    def _battle_finished(self, battle, won):
        current_snapshot = get_dense_reward_snapshot(battle)
        step_reward = calculate_step_reward(self.last_reward_snapshot, current_snapshot)
        final_reward = step_reward + (1.0 if won else -1.0)

        if self.last_state_key is not None:
            self._update_traces_and_q(final_reward, 0.0, True)

        self.last_state_key = None
        self.last_action_hash = None
        self.active_traces.clear()
        self.last_reward_snapshot = None
        self._n_finished_battles += 1
        if won:
            self._n_won_battles += 1

    def save_table(self, path):
        gc.disable()
        tmp = path + ".tmp"
        try:
            with open(tmp, 'wb') as f:
                pickle.dump({'q': self.q_table}, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
        except Exception as e:
            print(f"Save error: {e}")
            if os.path.exists(tmp): os.remove(tmp)
        finally:
            gc.enable()

    def load_table(self, path):
        gc.disable()
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data.get('q', {})
            logging.critical(f"Loaded Q ({len(self.q_table)})")
        except Exception as e:
            logging.critical(f"Fresh start: {e}")
        finally:
            gc.enable()
