"""
HeuristicEngine — estimates move/switch quality for smart initialization.
Extracted from v16gen4/player_v16.py.
"""

from poke_env.battle.move_category import MoveCategory

from shared.config import SMART_INIT_Q_MIN, SMART_INIT_Q_MAX


class HeuristicEngine:
    SPEED_TIER_COEFICIENT = 0.1
    HP_FRACTION_COEFICIENT = 0.4

    @staticmethod
    def _stat_estimation(mon, stat):
        if stat not in mon.boosts:
            return mon.base_stats.get(stat, 0)
        boost = mon.boosts[stat]
        if boost > 1:
            multiplier = (2 + boost) / 2
        else:
            multiplier = 2 / (2 - boost)
        return ((2 * mon.base_stats.get(stat, 100) + 31) + 5) * multiplier

    @staticmethod
    def _estimate_matchup(mon, opponent):
        if not opponent:
            return 0
        score = max([opponent.damage_multiplier(t) for t in mon.types if t is not None])
        score -= max([mon.damage_multiplier(t) for t in opponent.types if t is not None])
        if mon.base_stats.get("spe", 0) > opponent.base_stats.get("spe", 0):
            score += HeuristicEngine.SPEED_TIER_COEFICIENT
        elif opponent.base_stats.get("spe", 0) > mon.base_stats.get("spe", 0):
            score -= HeuristicEngine.SPEED_TIER_COEFICIENT
        score += mon.current_hp_fraction * HeuristicEngine.HP_FRACTION_COEFICIENT
        score -= opponent.current_hp_fraction * HeuristicEngine.HP_FRACTION_COEFICIENT
        return score

    @staticmethod
    def get_move_score(battle, move, active, opponent):
        if not opponent or not active:
            return 0.0
        if move.category == MoveCategory.PHYSICAL:
            atk = HeuristicEngine._stat_estimation(active, "atk")
            defn = HeuristicEngine._stat_estimation(opponent, "def")
        else:
            atk = HeuristicEngine._stat_estimation(active, "spa")
            defn = HeuristicEngine._stat_estimation(opponent, "spd")
        ratio = atk / defn if defn > 0 else 1.0
        stab = 1.5 if move.type in active.types else 1.0
        type_eff = opponent.damage_multiplier(move.type)
        score = move.base_power * stab * ratio * move.accuracy * move.expected_hits * type_eff
        return score

    @staticmethod
    def get_switch_score(battle, candidate, opponent):
        return HeuristicEngine._estimate_matchup(candidate, opponent)

    @staticmethod
    def get_master_switch_score(battle):
        opponent = battle.opponent_active_pokemon
        if not opponent or not battle.available_switches:
            return 0.0
        scores = [
            HeuristicEngine.get_switch_score(battle, candidate, opponent) * 50.0
            for candidate in battle.available_switches
        ]
        return max(scores) if scores else 0.0

    @staticmethod
    def build_q_priors(raw_scores, min_q=SMART_INIT_Q_MIN, max_q=SMART_INIT_Q_MAX):
        """Map heuristic scores directly into standardized Q priors.

        V4 used softmax probabilities directly as Q-values. V5 instead preserves
        the heuristic ranking while standardizing unseen-action priors into a
        bounded range so the learner gets a head start without a hard policy.
        """
        if not raw_scores:
            return []
        if len(raw_scores) == 1:
            return [0.0]

        low = min(raw_scores)
        high = max(raw_scores)
        if high <= low:
            return [0.0 for _ in raw_scores]

        width = max_q - min_q
        return [min_q + ((score - low) / (high - low)) * width for score in raw_scores]
