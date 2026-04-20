"""
V6 Ensemble Evaluation CLI.

Loads K trained ensemble members and runs N battles vs a chosen opponent,
combining agents via a configurable voting strategy (soft / hard / confidence).

Usage:
    # Primary: evaluate ensemble vs SimpleHeuristicsPlayer
    python play_ensemble.py --members ensemble_results/run_1 --strategy soft --n-battles 500

    # Vary the voting strategy
    python play_ensemble.py --members ensemble_results/run_1 --strategy hard --n-battles 500
    python play_ensemble.py --members ensemble_results/run_1 --strategy confidence --n-battles 500

    # K-saturation curve (free post-hoc ablation):
    for n in 1 5 10 20 30; do
      python play_ensemble.py --members ensemble_results/run_1 --subset $n --n-battles 500
    done

    # Compute-matched baseline comparison
    python play_ensemble.py --members ensemble_results/run_1 \\
        --opponent baseline_5m --n-battles 500

    # Human play mode: wait for browser challenges
    python play_ensemble.py --members ensemble_results/run_1 --human --port 8000
"""

import argparse
import asyncio
import glob
import json
import logging
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from poke_env.ps_client.server_configuration import (                     # noqa: E402
    LocalhostServerConfiguration,
    ServerConfiguration,
)
from poke_env.ps_client import AccountConfiguration                       # noqa: E402
from poke_env.player import SimpleHeuristicsPlayer, RandomPlayer          # noqa: E402

from shared.config import BATTLE_FORMAT, FULL_POKEMON_POOL, ENSEMBLE_RESULTS_DIR   # noqa: E402
from shared.team_builder import IndexedTeambuilder                        # noqa: E402
from model_ensemble.player import EnsemblePlayer                          # noqa: E402
from model_ensemble.hier_smart_player import HierSmartPlayer              # noqa: E402


logging.basicConfig(level=logging.CRITICAL)
logging.getLogger("poke_env").setLevel(logging.CRITICAL)


# ────────────────────────────────────────────────────────────────────────
# Member pkl discovery
# ────────────────────────────────────────────────────────────────────────

def _resolve_member_paths(members_dir: str, subset: int = None):
    """Return list of member_k pkl paths, sorted by k.  If `subset` given,
    returns the first `subset` paths (member_1, member_2, ...)."""
    if not os.path.isdir(members_dir):
        raise FileNotFoundError(f"Members directory not found: {members_dir}")

    # Try manifest.json first (authoritative)
    manifest_path = os.path.join(members_dir, "manifest.json")
    paths = []
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        for name in sorted(manifest["members"], key=lambda s: int(s.split("_")[1])):
            rel = manifest["members"][name]["pkl"]
            paths.append(os.path.join(members_dir, rel))
    else:
        # Fallback: glob member_*/models/run_1.pkl
        candidates = glob.glob(os.path.join(members_dir, "member_*", "models", "run_1.pkl"))
        def _k(p):
            parts = p.split(os.sep)
            member_seg = [s for s in parts if s.startswith("member_")][0]
            return int(member_seg.split("_")[1])
        paths = sorted(candidates, key=_k)

    if not paths:
        raise RuntimeError(f"No trained members found in {members_dir}")

    if subset is not None:
        if subset < 1 or subset > len(paths):
            raise ValueError(f"--subset {subset} out of range (1..{len(paths)})")
        paths = paths[:subset]

    return paths


def _server_config(port: int):
    if port == 8000:
        return LocalhostServerConfiguration
    return ServerConfiguration(
        f"ws://localhost:{port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )


# ────────────────────────────────────────────────────────────────────────
# Automated eval vs bot opponent
# ────────────────────────────────────────────────────────────────────────

async def evaluate_vs_opponent(
    ensemble: EnsemblePlayer,
    opponent,
    n_battles: int,
):
    """Run n_battles head-to-head, return stats dict."""
    start = time.time()
    wins_before = ensemble.n_won_battles
    await ensemble.battle_against(opponent, n_battles=n_battles)
    wins = ensemble.n_won_battles - wins_before
    elapsed = time.time() - start
    return {
        "n_battles": n_battles,
        "wins": wins,
        "win_rate": wins / n_battles if n_battles else 0.0,
        "elapsed_s": elapsed,
        "speed_bps": n_battles / elapsed if elapsed else 0.0,
    }


def _build_opponent(kind: str, battle_format: str, server_config, pool_seed: int,
                    single_model_path: str = None):
    """Build the chosen opponent.  Pools use IndexedTeambuilder (deterministic
    teams given seed+battle index)."""
    tb = IndexedTeambuilder(pool=list(FULL_POKEMON_POOL), base_seed=pool_seed)

    if kind == "heuristic":
        return SimpleHeuristicsPlayer(
            battle_format=battle_format,
            server_configuration=server_config,
            account_configuration=AccountConfiguration("V6HeuOpp", None),
            max_concurrent_battles=1,
            team=tb,
        )
    if kind == "random":
        return RandomPlayer(
            battle_format=battle_format,
            server_configuration=server_config,
            account_configuration=AccountConfiguration("V6RandOpp", None),
            max_concurrent_battles=1,
            team=tb,
        )
    if kind == "baseline_5m":
        path = single_model_path or os.path.join(
            ENSEMBLE_RESULTS_DIR, "baseline_5m", "models", "run_1.pkl"
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"baseline_5m model not found: {path}")
        opp = HierSmartPlayer(
            battle_format=battle_format,
            server_configuration=server_config,
            account_configuration=AccountConfiguration("V6Baseln5M", None),
            max_concurrent_battles=1,
            epsilon=0.0,
            team=tb,
        )
        opp.load_table(path)
        return opp
    if kind == "single":
        if not single_model_path:
            raise ValueError("--opponent single requires --single-model PATH")
        opp = HierSmartPlayer(
            battle_format=battle_format,
            server_configuration=server_config,
            account_configuration=AccountConfiguration("V6SingleOpp", None),
            max_concurrent_battles=1,
            epsilon=0.0,
            team=tb,
        )
        opp.load_table(single_model_path)
        return opp
    raise ValueError(f"Unknown opponent kind: {kind}")


# ────────────────────────────────────────────────────────────────────────
# Human play mode
# ────────────────────────────────────────────────────────────────────────

async def run_human_play(ensemble: EnsemblePlayer, n_challenges: int = 100):
    print(f"\nAGENT ONLINE:  {ensemble.username}")
    print(f"Strategy: {ensemble.strategy}  │  Members: {ensemble.K}")
    print(f"Open http://127.0.0.1:8000, search '{ensemble.username}', challenge [Gen 4] OU")
    print("(Ctrl+C to stop)\n")
    await ensemble.accept_challenges(None, n_challenges)


# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────

def _build_parser():
    p = argparse.ArgumentParser(
        description="V6 Ensemble Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--members", type=str, required=True,
                   help="Directory containing member_*/models/run_1.pkl (e.g. ensemble_results/run_1)")
    p.add_argument("--subset", type=int, default=None,
                   help="Use only the first N members (for K-saturation curve)")
    p.add_argument("--strategy", choices=list(EnsemblePlayer.VALID_STRATEGIES),
                   default="soft", help="Combination rule")
    p.add_argument("--opponent", choices=["heuristic", "random", "baseline_5m", "single"],
                   default="heuristic", help="Opponent type")
    p.add_argument("--single-model", type=str, default=None,
                   help="Path to a single-agent .pkl (required for --opponent single)")
    p.add_argument("--n-battles", type=int, default=500, help="Number of eval battles")
    p.add_argument("--port", type=int, default=9000, help="Pokemon Showdown port")
    p.add_argument("--pool-seed", type=int, default=99999,
                   help="Seed for opponent's team builder (independent from training seeds)")
    p.add_argument("--log-diagnostics", action="store_true",
                   help="Log pairwise disagreement + unseen-state-rate per decision")
    p.add_argument("--out", type=str, default=None,
                   help="Path to write eval summary JSON (default: <members>/eval_<strategy>_K<N>.json)")
    p.add_argument("--human", action="store_true",
                   help="Human play mode: accept browser challenges instead of auto-eval")
    p.add_argument("--epsilon", type=float, default=0.0,
                   help="Ensemble epsilon at inference (default 0.0)")
    return p


async def main_async(args):
    members_dir = os.path.abspath(args.members)
    member_paths = _resolve_member_paths(members_dir, subset=args.subset)
    K = len(member_paths)
    print(f"Loading {K} ensemble members from {members_dir}")

    server = _server_config(args.port)

    ensemble = EnsemblePlayer(
        member_paths=member_paths,
        strategy=args.strategy,
        log_disagreement=args.log_diagnostics,
        log_unseen_rate=args.log_diagnostics,
        epsilon=args.epsilon,
        battle_format=BATTLE_FORMAT,
        server_configuration=server,
        account_configuration=AccountConfiguration(f"V6E{args.strategy[:3].title()}{K}", None),
        max_concurrent_battles=1,
        team=IndexedTeambuilder(pool=list(FULL_POKEMON_POOL), base_seed=args.pool_seed + 1),
    )

    if args.human:
        await run_human_play(ensemble)
        return

    opponent = _build_opponent(
        kind=args.opponent,
        battle_format=BATTLE_FORMAT,
        server_config=server,
        pool_seed=args.pool_seed,
        single_model_path=args.single_model,
    )

    print(f"Opponent: {args.opponent}  │  Strategy: {args.strategy}  │  "
          f"K={K}  │  n_battles={args.n_battles}\n")

    stats = await evaluate_vs_opponent(ensemble, opponent, args.n_battles)
    stats.update({
        "K": K,
        "strategy": args.strategy,
        "opponent": args.opponent,
        "members_dir": members_dir,
        "battle_format": BATTLE_FORMAT,
        "subset": args.subset,
    })

    if args.log_diagnostics:
        stats["diagnostics"] = ensemble.diagnostics_summary()

    print(f"\nRESULT:  {stats['win_rate']:.3f} "
          f"({stats['wins']}/{stats['n_battles']})  "
          f"[{stats['speed_bps']:.1f} battles/s]")
    if args.log_diagnostics:
        diag = stats["diagnostics"]
        if diag["disagreement"]["n"]:
            print(f"  disagreement: mean {diag['disagreement']['mean']:.3f}  "
                  f"[{diag['disagreement']['min']:.2f}..{diag['disagreement']['max']:.2f}]")
        if diag["unseen_rate"]["n"]:
            print(f"  unseen rate:  mean {diag['unseen_rate']['mean']:.3f}  "
                  f"[{diag['unseen_rate']['min']:.2f}..{diag['unseen_rate']['max']:.2f}]")

    out_path = args.out or os.path.join(
        members_dir, f"eval_{args.strategy}_K{K}_{args.opponent}.json"
    )
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nWrote summary → {out_path}")


def main():
    args = _build_parser().parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
