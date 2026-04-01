"""
Dense reward calculation for all experiment models.
Extracted from v16gen4/player_v16.py.
"""


def get_dense_reward_snapshot(battle):
    """Capture current battle state for reward calculation."""
    my_hp = sum(mon.current_hp_fraction for mon in battle.team.values())
    opp_hp = sum(mon.current_hp_fraction for mon in battle.opponent_team.values())
    my_fainted = sum(1 for mon in battle.team.values() if mon.fainted)
    opp_fainted = sum(1 for mon in battle.opponent_team.values() if mon.fainted)
    my_status = sum(1 for mon in battle.team.values() if mon.status)
    opp_status = sum(1 for mon in battle.opponent_team.values() if mon.status)
    my_boosts = 0
    if battle.active_pokemon:
        my_boosts = sum(battle.active_pokemon.boosts.values())
    return {
        'my_hp': my_hp, 'opp_hp': opp_hp,
        'my_fainted': my_fainted, 'opp_fainted': opp_fainted,
        'my_status': my_status, 'opp_status': opp_status,
        'my_boosts': my_boosts,
    }


def calculate_step_reward(prev_snapshot, curr_snapshot):
    """Calculate dense step reward from two consecutive snapshots."""
    if prev_snapshot is None:
        return 0.0
    prev = prev_snapshot
    curr = curr_snapshot
    reward = 0.0
    reward += (curr['opp_fainted'] - prev['opp_fainted']) * 0.1
    reward -= (curr['my_fainted'] - prev['my_fainted']) * 0.1
    opp_hp_lost = prev['opp_hp'] - curr['opp_hp']
    my_hp_lost = prev['my_hp'] - curr['my_hp']
    reward += 0.05 * (opp_hp_lost - my_hp_lost)
    new_opp_status = curr['opp_status'] - prev['opp_status']
    new_my_status = curr['my_status'] - prev['my_status']
    reward += 0.01 * (new_opp_status - new_my_status)
    boost_change = curr['my_boosts'] - prev['my_boosts']
    reward += 0.01 * boost_change
    return reward
