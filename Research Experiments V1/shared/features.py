"""
Feature extraction for the research experiment.

State representations:
  - get_battle_state(battle) → 20-tuple  (used by BOTH models as master/primary state)
  - get_bench_detail(battle) → 15-tuple  (sorted per-mon bench info)
  - get_sub_state(battle)    → 17-tuple  (opp context + bench detail, hier sub-agent)
  - get_flat_state(battle)   → 35-tuple  (battle_state + bench_detail, flat models)

Design principle: flat and hierarchical see the SAME information,
only the structure (one big table vs decomposed tables) differs.
"""

import zlib
import logging

try:
    from poke_env.battle.status import Status
except ImportError:
    try:
        from poke_env.battle import Status
    except ImportError:
        Status = None

try:
    from poke_env.battle.side_condition import SideCondition
except ImportError:
    try:
        from poke_env.battle import SideCondition
    except ImportError:
        SideCondition = None


class AdvancedFeatureExtractor:
    def __init__(self):
        if Status is None:
            logging.warning("Status enum not found. Features will be 0.")
        if SideCondition is None:
            logging.warning("SideCondition enum not found. Features will be 0.")

    def get_hp_bucket(self, current_hp, max_hp):
        if max_hp == 0 or current_hp == 0:
            return 0
        ratio = current_hp / max_hp
        if ratio > 0.5:
            return 2  # High
        if ratio > 0.2:
            return 1  # Mid
        return 0  # Low

    def get_status_int(self, status):
        if status is None or Status is None:
            return 0
        return status.value

    def get_ability_hash(self, ability):
        if ability is None:
            return 0
        return zlib.adler32(ability.encode())

    def get_speed_check(self, my_mon, opp_mon):
        if not my_mon or not opp_mon:
            return 0
        try:
            my_speed = getattr(my_mon, "speed", my_mon.base_stats.get('spe', 0))
            opp_speed = getattr(opp_mon, "speed", opp_mon.base_stats.get('spe', 0))
            return 1 if my_speed > opp_speed else 0
        except AttributeError:
            return 0

    def get_boost_flags(self, mon):
        if not mon:
            return (0, 0)
        boosts = mon.boosts
        has_any = 0
        has_max = 0
        for stat, stage in boosts.items():
            if stage > 0:
                has_any = 1
            if stage == 6:
                has_max = 1
        return (has_any, has_max)

    def get_hazards_tuple(self, battle):
        if SideCondition is None:
            return (0, 0, 0, 0)
        sc = battle.side_conditions
        has_spikes = 1 if SideCondition.SPIKES in sc else 0
        has_rocks = 1 if SideCondition.STEALTH_ROCK in sc else 0
        has_web = 1 if SideCondition.STICKY_WEB in sc else 0
        has_tspikes = 1 if SideCondition.TOXIC_SPIKES in sc else 0
        return (has_spikes, has_rocks, has_web, has_tspikes)

    # ── Internal: 12-tuple active matchup ──

    def _get_matchup_state(self, battle):
        """12-tuple: active Pokemon matchup features."""
        my_mon = battle.active_pokemon
        if my_mon:
            my_species = my_mon.species
            my_hp = self.get_hp_bucket(my_mon.current_hp, my_mon.max_hp)
            my_status = self.get_status_int(my_mon.status)
            my_ability = self.get_ability_hash(my_mon.ability)
            my_boosted, my_max_boosted = self.get_boost_flags(my_mon)
        else:
            my_species, my_hp, my_status, my_ability = "None", 0, 0, 0
            my_boosted, my_max_boosted = 0, 0

        opp_mon = battle.opponent_active_pokemon
        if opp_mon:
            opp_species = opp_mon.species
            opp_hp = self.get_hp_bucket(opp_mon.current_hp, opp_mon.max_hp)
            opp_status = self.get_status_int(opp_mon.status)
            opp_boosted, opp_max_boosted = self.get_boost_flags(opp_mon)
        else:
            opp_species, opp_hp, opp_status = "None", 0, 0
            opp_boosted, opp_max_boosted = 0, 0

        is_faster = self.get_speed_check(my_mon, opp_mon)

        return (
            my_species, my_hp, my_status, my_ability, my_boosted, my_max_boosted, is_faster,
            opp_species, opp_hp, opp_status, opp_boosted, opp_max_boosted
        )

    # ── Internal: 8-tuple bench aggregates ──

    def _get_bench_aggregates(self, battle):
        """8-tuple: aggregate bench + hazard features."""
        bench_alive = 0
        bench_healthy = 0
        bench_statused = 0
        has_healthy_switch = 0
        active = battle.active_pokemon
        for mon in battle.team.values():
            if mon == active or mon.fainted:
                continue
            bench_alive += 1
            hp_ratio = mon.current_hp / mon.max_hp if mon.max_hp > 0 else 0
            if hp_ratio > 0.5:
                bench_healthy += 1
            if mon.status is not None:
                bench_statused += 1
            if hp_ratio > 0.5 and mon.status is None:
                has_healthy_switch = 1

        hazards = self.get_hazards_tuple(battle)
        return (bench_alive, bench_healthy, bench_statused,
                hazards[0], hazards[1], hazards[2], hazards[3],
                has_healthy_switch)

    # ── Public: bench detail (15-tuple) ──

    def get_bench_detail(self, battle):
        """
        15-tuple: sorted (species, hp_bucket, status) × 5 bench mons.
        Sorted alphabetically by species for order-invariance.
        Fainted mons included with hp_bucket=0.
        """
        active = battle.active_pokemon
        bench = []
        for mon in battle.team.values():
            if mon == active:
                continue
            species = mon.species
            hp = 0 if mon.fainted else self.get_hp_bucket(mon.current_hp, mon.max_hp)
            status = self.get_status_int(mon.status)
            bench.append((species, hp, status))

        bench.sort(key=lambda x: x[0])

        # Pad if < 5 (shouldn't happen with 6-mon teams)
        while len(bench) < 5:
            bench.append(("none", 0, 0))

        # Flatten to 15-tuple
        result = ()
        for species, hp, status in bench[:5]:
            result += (species, hp, status)
        return result

    # ── Public API ──

    def get_battle_state(self, battle):
        """
        20-tuple: used by BOTH models as the primary/master state.
        = matchup_12 + bench_aggregates_8
        """
        return self._get_matchup_state(battle) + self._get_bench_aggregates(battle)

    def get_sub_state(self, battle):
        """
        17-tuple: used by hierarchical sub-agent for switch decisions.
        = (opp_species, opp_hp) + bench_detail_15
        """
        opp = battle.opponent_active_pokemon
        if opp:
            opp_species = opp.species
            opp_hp = self.get_hp_bucket(opp.current_hp, opp.max_hp)
        else:
            opp_species = "None"
            opp_hp = 0
        return (opp_species, opp_hp) + self.get_bench_detail(battle)

    def get_flat_state(self, battle):
        """
        35-tuple: used by flat models — all info in one state.
        = battle_state_20 + bench_detail_15
        """
        return self.get_battle_state(battle) + self.get_bench_detail(battle)

    # Keep old name as alias for backward compat with heuristics
    def get_master_state(self, battle):
        return self._get_matchup_state(battle)
