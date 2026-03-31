"""
Common training loop for all research experiment models.
Each model's train.py just imports this and passes its player class.
"""

import asyncio
import os
import csv
import json
import time
import logging
import uuid
import sys
import argparse
import traceback
from collections import deque
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration, ServerConfiguration
from poke_env.player import SimpleHeuristicsPlayer

from shared.config import (
    BATTLE_FORMAT, BATTLE_TIMEOUT, BATTLES_PER_LOG, SAVE_FREQ,
    ALPHA, GAMMA, LAMBDA,
)
from shared.team_builder import IndexedTeambuilder

logging.basicConfig(level=logging.CRITICAL)
logging.getLogger("poke_env").setLevel(logging.CRITICAL)


def log_stats(filename, battles, rolling_win, overall_win, epsilon, speed, avg_rew, table_size):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Battles', 'RollingWin', 'OverallWin', 'Epsilon', 'Speed', 'AvgReward', 'TableSize'])
        writer.writerow([battles, f"{rolling_win:.4f}", f"{overall_win:.4f}", f"{epsilon:.4f}", f"{speed:.1f}", f"{avg_rew:.4f}", table_size])


def write_live_status(status_file, data):
    """Atomically write live status JSON for the dashboard to read."""
    tmp = status_file + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(data, f)
    os.replace(tmp, status_file)


_IS_TTY = sys.stdout.isatty()


def print_live_progress(current_count, log_window_size, current_speed):
    if not _IS_TTY:
        return  # Skip progress bars when piped to file
    progress_fraction = min(current_count / log_window_size, 1.0)
    bar_len = 20
    filled = int(progress_fraction * bar_len)
    bar = f"[{'=' * filled}{'-' * (bar_len - filled)}]"
    sys.stdout.write(f"\rProgress {bar} {progress_fraction:.0%} ({current_count}/{log_window_size}, {current_speed:.1f}/s)")
    sys.stdout.flush()


def get_unique_player_class(base_class, prefix, run_uuid):
    return type(f"{prefix}_{run_uuid}", (base_class,), {})


def get_table_size(learner):
    """Get total Q-table size across all tables the player has."""
    size = len(learner.q_table)
    if hasattr(learner, 'switch_table'):
        size += len(learner.switch_table)
    return size


def _get_server_config(port):
    """Get a ServerConfiguration for the given port. Falls back to default 9000."""
    if port == 8000:
        return LocalhostServerConfiguration
    return ServerConfiguration(
        f"ws://localhost:{port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )


async def run_training(player_class, args, model_dir):
    """
    Main training loop.

    Args:
        player_class: The Player subclass to train
        args: argparse Namespace with run_id, seed, epsilon, batch_size, historic_battles, historic_wins, port
        model_dir: Directory for this model (e.g., "model_1_flat_zero")
    """
    run_uuid = uuid.uuid4().hex[:8]
    server_config = _get_server_config(getattr(args, 'port', 8000))

    # Create directories
    log_dir = os.path.join(model_dir, "logs")
    model_save_dir = os.path.join(model_dir, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"run_{args.run_id}.csv")
    model_file = os.path.join(model_save_dir, f"run_{args.run_id}.pkl")

    # Deterministic team builders — battle N always produces the same teams
    # across all 4 models, making the only variable the model itself.
    # Opponent and learner use different base seeds so they get different teams.
    opp_tb = IndexedTeambuilder(base_seed=args.seed + 1000)
    learner_tb = IndexedTeambuilder(base_seed=args.seed + 2000)

    # Opponent
    OppClass = get_unique_player_class(SimpleHeuristicsPlayer, "Opp", run_uuid)
    opponent = OppClass(
        battle_format=BATTLE_FORMAT,
        server_configuration=server_config,
        max_concurrent_battles=1,
        team=opp_tb,
    )

    # Learner
    LearnerClass = get_unique_player_class(player_class, "Learner", run_uuid)
    learner = LearnerClass(
        battle_format=BATTLE_FORMAT,
        server_configuration=server_config,
        max_concurrent_battles=1,
        alpha=ALPHA, gamma=GAMMA, lam=LAMBDA, epsilon=args.epsilon,
        team=learner_tb,
    )

    # Load existing model if continuing
    if os.path.exists(model_file):
        learner.load_table(model_file)

    battles_collected = 0
    start_time = time.time()
    initial_learner_wins = learner.n_won_battles
    session_outcomes = deque(maxlen=BATTLES_PER_LOG)
    accumulated_total_reward = 0.0
    log_window_start_time = time.time()
    current_log_progress = 0
    consecutive_timeouts = 0

    # Live status file — updated every 10 battles for the dashboard
    # Per-run file so parallel runs don't overwrite each other
    status_file = os.path.join(model_dir, f"live_status_run_{args.run_id}.json")

    # Track last logged values for the status file
    last_rolling_wr = 0.0
    last_overall_wr = 0.0
    last_avg_rew = 0.0
    last_speed = 0.0
    last_table_size = 0

    if args.historic_battles == 0:
        print(f"--- {player_class.__name__} Run {args.run_id} (Batch: {args.batch_size}, Eps: {args.epsilon:.3f}) ---")

    while battles_collected < args.batch_size:
        try:
            if battles_collected > 0 and battles_collected % SAVE_FREQ == 0:
                learner.save_table(model_file)

            # Set battle index so both teams are deterministic for this battle
            battle_idx = args.historic_battles + battles_collected
            opp_tb.battle_index = battle_idx
            learner_tb.battle_index = battle_idx

            wins_before = learner.n_won_battles

            await asyncio.wait_for(
                learner.battle_against(opponent, n_battles=1),
                timeout=BATTLE_TIMEOUT
            )

            consecutive_timeouts = 0
            is_win = learner.n_won_battles > wins_before
            session_outcomes.append(1 if is_win else 0)

            outcome_reward = 1.0 if is_win else -1.0
            accumulated_total_reward += outcome_reward
            step_rewards = learner.pop_step_rewards()
            accumulated_total_reward += sum(step_rewards)

            battles_collected += 1
            current_log_progress += 1

            # Write live status every 10 battles for precise dashboard tracking
            if battles_collected % 10 == 0:
                elapsed = time.time() - start_time
                live_speed = battles_collected / elapsed if elapsed > 0 else 0.0
                current_session_wins = learner.n_won_battles - initial_learner_wins
                total_battles = args.historic_battles + battles_collected
                total_wins = args.historic_wins + current_session_wins
                live_rolling = sum(session_outcomes) / len(session_outcomes) if session_outcomes else 0.0
                live_overall = total_wins / total_battles if total_battles > 0 else 0.0
                try:
                    write_live_status(status_file, {
                        "run_id": args.run_id,
                        "battles": total_battles,
                        "battles_in_batch": battles_collected,
                        "batch_size": args.batch_size,
                        "wins": total_wins,
                        "rolling_wr": round(live_rolling, 4),
                        "overall_wr": round(live_overall, 4),
                        "epsilon": round(learner.epsilon, 4),
                        "speed": round(live_speed, 1),
                        "table_size": get_table_size(learner),
                        "avg_reward": round(accumulated_total_reward / max(current_log_progress, 1), 4),
                        "progress_in_window": current_log_progress,
                        "window_size": BATTLES_PER_LOG,
                    })
                except Exception:
                    pass  # Don't let status write failures stop training

            if current_log_progress % 10 == 0 and current_log_progress < BATTLES_PER_LOG:
                elapsed_progress = time.time() - log_window_start_time
                current_speed = current_log_progress / elapsed_progress if elapsed_progress > 0 else 0
                print_live_progress(current_log_progress, BATTLES_PER_LOG, current_speed)

            if battles_collected % BATTLES_PER_LOG == 0:
                if _IS_TTY:
                    sys.stdout.write("\r" + " " * 80 + "\r")
                    sys.stdout.flush()

                rolling_wr = sum(session_outcomes) / len(session_outcomes) if session_outcomes else 0.0
                current_session_wins = learner.n_won_battles - initial_learner_wins
                total_battles = args.historic_battles + battles_collected
                total_wins = args.historic_wins + current_session_wins
                overall_wr = total_wins / total_battles if total_battles > 0 else 0.0

                elapsed = time.time() - start_time
                speed = battles_collected / elapsed if elapsed > 0 else 0.0
                table_size = get_table_size(learner)
                avg_rew = accumulated_total_reward / BATTLES_PER_LOG

                print(f"Bat {total_battles}: Roll {rolling_wr:.2%} | Overall {overall_wr:.2%} | AvgRew {avg_rew:.3f} | Eps {learner.epsilon:.3f} | States {table_size} | {speed:.1f}/s")

                log_stats(log_file, total_battles, rolling_wr, overall_wr, learner.epsilon, speed, avg_rew, table_size)

                last_rolling_wr = rolling_wr
                last_overall_wr = overall_wr
                last_avg_rew = avg_rew
                last_speed = speed
                last_table_size = table_size

                accumulated_total_reward = 0.0
                current_log_progress = 0
                log_window_start_time = time.time()

        except asyncio.TimeoutError:
            consecutive_timeouts += 1
            if consecutive_timeouts >= 5:
                print(f"\n5 consecutive timeouts. Saving and exiting.")
                learner.save_table(model_file)
                sys.exit(1)
            time.sleep(0.1)
            continue
        except Exception:
            traceback.print_exc()
            continue

    learner.save_table(model_file)

    # Clean up status file
    try:
        os.remove(status_file)
    except OSError:
        pass

    # Print final stats
    total_battles = args.historic_battles + battles_collected
    current_session_wins = learner.n_won_battles - initial_learner_wins
    total_wins = args.historic_wins + current_session_wins
    print(f"Finished: {total_battles} battles, {total_wins}/{total_battles} wins ({total_wins/total_battles:.2%})")

    return battles_collected, current_session_wins


def parse_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--historic_battles", type=int, default=0)
    parser.add_argument("--historic_wins", type=int, default=0)
    parser.add_argument("--port", type=int, default=9000)
    return parser.parse_args()
