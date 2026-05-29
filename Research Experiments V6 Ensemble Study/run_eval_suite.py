"""
V6 Eval Suite Orchestrator with Live Dashboard.

Single-process runner that executes all 8 evaluation conditions sequentially:
  1. K_max SOFT vs heuristic
  2. K_max HARD vs heuristic
  3. K_max CONFIDENCE vs heuristic
  4. K=1 SOFT vs heuristic
  5. K=3 SOFT vs heuristic
  6. K=5 SOFT vs heuristic
  7. K_max SOFT vs baseline_5m  (head-to-head)
  8. baseline_5m solo vs heuristic

Why a single process:
  - Each Q-table loads from disk once (cached). Multiple ensemble players
    at different K values share the same in-memory tables. Saves ~15 min
    of redundant disk loads across 8 evals.
  - Live dashboard refreshes every batch — no more "stare at silent terminal
    for 2 hours per eval".
  - Each completed eval saves its JSON immediately, so a mid-suite crash
    doesn't lose finished results.
  - Single Ctrl+C cleanly shuts down everything.

Usage:
    python run_eval_suite.py --run-id 1 --k-max 10 --n-battles 100000 --port 9000
    python run_eval_suite.py --resume                       # skip evals whose JSON exists
    python run_eval_suite.py --skip-baseline                # don't run evals that need baseline_5m
"""

import argparse
import asyncio
import glob
import json
import logging
import os
import shutil
import sys
import time
import uuid

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from poke_env.ps_client.server_configuration import (
    LocalhostServerConfiguration, ServerConfiguration,
)
from poke_env.ps_client import AccountConfiguration
from poke_env.player import SimpleHeuristicsPlayer

from shared.config import (
    BATTLE_FORMAT, FULL_POKEMON_POOL, ENSEMBLE_RESULTS_DIR,
)
from shared.team_builder import IndexedTeambuilder


# Eval needs DIVERSE matchups across the 100k battles per eval (not the same
# fixed team 100k times). The plain IndexedTeambuilder keeps battle_index=0
# forever unless something externally advances it; during training,
# train_common.py manually sets battle_index before each battle. In a batched
# `battle_against(n_battles=N)` call we cannot interleave that, so we
# subclass to auto-advance after each yield. Same (base_seed, battle_index)
# semantics as training, just with the index advancing on its own.
class _AutoAdvancingIndexedTeambuilder(IndexedTeambuilder):
    def yield_team(self):
        result = super().yield_team()
        self.battle_index += 1
        return result
from model_ensemble.player import EnsemblePlayer, MemberQTable
from model_ensemble.hier_smart_player import HierSmartPlayer

logging.basicConfig(level=logging.CRITICAL)
logging.getLogger("poke_env").setLevel(logging.CRITICAL)


# ─── Helpers ──────────────────────────────────────────────────────────

def _fmt_dur(s):
    if s is None:
        return "?"
    s = max(0, int(s))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    if s < 86400:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    return f"{s // 86400}d{(s % 86400) // 3600:02d}h"


def _fmt_count(n):
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 10_000:
        return f"{n / 1_000:.1f}k"
    if n >= 1_000:
        return f"{n / 1_000:.2f}k"
    return str(n)


def _server_config(port):
    if port == 8000:
        return LocalhostServerConfiguration
    return ServerConfiguration(
        f"ws://localhost:{port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )


# ─── Dashboard rendering ──────────────────────────────────────────────

_IS_TTY = sys.stdout.isatty()
DASHBOARD_REFRESH_S = 1.0   # seconds between TTY redraws


def _hide_cursor():
    sys.stdout.write("\033[?25l")
def _show_cursor():
    sys.stdout.write("\033[?25h")
def _clear():
    sys.stdout.write("\033[2J\033[H")


class SuiteState:
    def __init__(self, configs, n_battles):
        self.configs = configs
        self.n_battles = n_battles
        self.results = [None] * len(configs)
        self.current_idx = -1
        self.current_completed = 0
        self.current_wins = 0
        self.current_start_time = None
        self.suite_start_time = time.time()
        self._last_render = 0.0
        self._last_plain_print = 0.0


def _render_dashboard(state, force=False):
    now = time.time()
    if not force and now - state._last_render < DASHBOARD_REFRESH_S:
        return
    state._last_render = now
    if not _IS_TTY:
        return

    width = max(100, shutil.get_terminal_size((140, 40)).columns - 1)
    _clear()

    elapsed = now - state.suite_start_time
    n_done = sum(1 for r in state.results if r is not None)
    n_total = len(state.configs)
    n_active = 1 if (state.current_idx >= 0 and state.results[state.current_idx] is None) else 0

    avg_eval_time = (sum(r["elapsed_s"] for r in state.results if r) / n_done) if n_done else None
    cur_remaining = 0.0
    if n_active and state.current_start_time:
        elapsed_eval = now - state.current_start_time
        speed = state.current_completed / max(elapsed_eval, 0.001)
        if speed > 0:
            cur_remaining = (state.n_battles - state.current_completed) / speed
    pending = n_total - n_done - n_active
    suite_eta = (pending * avg_eval_time + cur_remaining) if avg_eval_time else None

    bar = "═" * width
    print(bar)
    print(f"V6 EVALUATION SUITE  │  {state.n_battles:,} battles/eval  │  Eval {n_done + n_active}/{n_total}"[:width])
    print(f"  Elapsed: {_fmt_dur(elapsed):<10}  ETA: {_fmt_dur(suite_eta):<10}"[:width])
    print(bar)
    header = (
        f"  {'#':<2} {'Name':<37}  {'Battles':>17}  {'WR':>5}  "
        f"{'Speed':>9}  {'Time':>9}  {'Status':<6}"
    )
    print(header[:width])
    print(("  " + "─" * (width - 4))[:width])

    for i, cfg in enumerate(state.configs):
        result = state.results[i]
        if result:
            line = (
                f"  {i + 1:<2} {cfg['name'][:37]:<37}  "
                f"{result['n_battles']:>8,}/{result['n_battles']:<8,}  "
                f"{result['win_rate']:>5.3f}  "
                f"{'-':>9}  {_fmt_dur(result['elapsed_s']):>9}  {'DONE':<6}"
            )
        elif i == state.current_idx:
            elapsed_eval = now - state.current_start_time if state.current_start_time else 0
            speed = state.current_completed / max(elapsed_eval, 0.001)
            wr = state.current_wins / state.current_completed if state.current_completed else 0
            line = (
                f"  {i + 1:<2} {cfg['name'][:37]:<37}  "
                f"{state.current_completed:>8,}/{state.n_battles:<8,}  "
                f"{wr:>5.3f}  "
                f"{speed:>5.1f} b/s  {_fmt_dur(elapsed_eval):>9}  {'RUN':<6}"
            )
        else:
            line = (
                f"  {i + 1:<2} {cfg['name'][:37]:<37}  "
                f"{'0':>8}/{state.n_battles:<8,}  "
                f"{'-':>5}  "
                f"{'-':>9}  {'-':>9}  {'WAIT':<6}"
            )
        print(line[:width])

    print(("  " + "─" * (width - 4))[:width])
    print("  Ctrl+C to stop. Each completed eval is saved to JSON before the next starts."[:width])
    print(bar)
    sys.stdout.flush()


def _plain_tick(state, name):
    now = time.time()
    if now - state._last_plain_print < 30.0:
        return
    state._last_plain_print = now
    elapsed = now - state.current_start_time if state.current_start_time else 0
    speed = state.current_completed / max(elapsed, 0.001)
    wr = state.current_wins / state.current_completed if state.current_completed else 0
    eta = (state.n_battles - state.current_completed) / speed if speed > 0 else 0
    sys.stdout.write(
        f"[{_fmt_dur(now - state.suite_start_time)}] [{state.current_idx + 1}/{len(state.configs)}] "
        f"{name}: {state.current_completed:,}/{state.n_battles:,} "
        f"({100 * state.current_completed / state.n_battles:5.1f}%)  "
        f"WR={wr:.3f}  speed={speed:5.1f}b/s  ETA={_fmt_dur(eta)}\n"
    )
    sys.stdout.flush()


# ─── Build players ─────────────────────────────────────────────────────

def _make_team_builder(seed):
    return _AutoAdvancingIndexedTeambuilder(pool=list(FULL_POKEMON_POOL), base_seed=seed)


def _build_ensemble(member_paths, strategy, server, account_name, log_diagnostics, pool_seed):
    return EnsemblePlayer(
        member_paths=member_paths,
        strategy=strategy,
        log_disagreement=log_diagnostics,
        log_unseen_rate=log_diagnostics,
        epsilon=0.0,
        battle_format=BATTLE_FORMAT,
        server_configuration=server,
        account_configuration=AccountConfiguration(account_name, None),
        max_concurrent_battles=1,
        team=_make_team_builder(pool_seed + 1),
    )


def _build_opponent(kind, server, pool_seed, baseline_pkl=None, account_suffix=""):
    tb = _make_team_builder(pool_seed)
    if kind == "heuristic":
        return SimpleHeuristicsPlayer(
            battle_format=BATTLE_FORMAT,
            server_configuration=server,
            account_configuration=AccountConfiguration(f"V6Heu{account_suffix}", None),
            max_concurrent_battles=1,
            team=tb,
        )
    if kind == "baseline_5m":
        opp = HierSmartPlayer(
            battle_format=BATTLE_FORMAT,
            server_configuration=server,
            account_configuration=AccountConfiguration(f"V6Bsl{account_suffix}", None),
            max_concurrent_battles=1,
            epsilon=0.0,
            team=tb,
        )
        opp.load_table(baseline_pkl)
        return opp
    raise ValueError(f"Unknown opponent kind: {kind}")


# ─── Eval execution loop ───────────────────────────────────────────────

def _free_battle_buffers(*players):
    """Clear poke-env's _battles dict on each player to free memory between
    batches. Without this, Player._battles accumulates Battle objects
    (~100-500 KB each) across the entire eval, causing OOM by ~30k battles."""
    for p in players:
        if p is None:
            continue
        try:
            p._battles.clear()
        except AttributeError:
            pass


async def _run_one_eval(state, idx, ensemble, opponent, n_battles, batch_size):
    state.current_idx = idx
    state.current_completed = 0
    state.current_wins = 0
    state.current_start_time = time.time()
    state._last_render = 0.0
    state._last_plain_print = 0.0

    name = state.configs[idx]["name"]
    # We track wins ourselves because we clear _battles between batches —
    # ensemble.n_won_battles is derived from _battles and would reset to 0
    # after each clear. Each batch_against fills _battles with N fresh
    # battles; we read n_won_battles immediately before clearing.
    while state.current_completed < n_battles:
        n = min(batch_size, n_battles - state.current_completed)
        await ensemble.battle_against(opponent, n_battles=n)
        # Snapshot this batch's wins BEFORE clearing
        batch_wins = ensemble.n_won_battles
        # Free poke-env's per-battle memory now that the batch is done
        _free_battle_buffers(ensemble, opponent)
        state.current_completed += n
        state.current_wins += batch_wins
        if _IS_TTY:
            _render_dashboard(state)
        else:
            _plain_tick(state, name)

    elapsed = time.time() - state.current_start_time
    return {
        "id": state.configs[idx]["id"],
        "name": state.configs[idx]["name"],
        "n_battles": n_battles,
        "wins": state.current_wins,
        "win_rate": state.current_wins / n_battles if n_battles else 0.0,
        "elapsed_s": elapsed,
        "speed_bps": n_battles / elapsed if elapsed > 0 else 0.0,
        "subset": state.configs[idx]["subset"],
        "strategy": state.configs[idx]["strategy"],
        "opponent": state.configs[idx]["opponent"],
    }


# ─── Suite orchestrator ────────────────────────────────────────────────

def _eval_configs(k_max):
    """The standard 8-eval suite parameterized by K_max."""
    return [
        {"id": f"K{k_max}_soft_heuristic",
         "name": f"K={k_max} SOFT vs heuristic",
         "strategy": "soft", "subset": k_max, "opponent": "heuristic"},
        {"id": f"K{k_max}_hard_heuristic",
         "name": f"K={k_max} HARD vs heuristic",
         "strategy": "hard", "subset": k_max, "opponent": "heuristic"},
        {"id": f"K{k_max}_confidence_heuristic",
         "name": f"K={k_max} CONFIDENCE vs heuristic",
         "strategy": "confidence", "subset": k_max, "opponent": "heuristic"},
        {"id": "K1_soft_heuristic",
         "name": "K=1 SOFT vs heuristic",
         "strategy": "soft", "subset": 1, "opponent": "heuristic"},
        {"id": "K3_soft_heuristic",
         "name": "K=3 SOFT vs heuristic",
         "strategy": "soft", "subset": 3, "opponent": "heuristic"},
        {"id": "K5_soft_heuristic",
         "name": "K=5 SOFT vs heuristic",
         "strategy": "soft", "subset": 5, "opponent": "heuristic"},
        {"id": f"K{k_max}_soft_baseline5m",
         "name": f"K={k_max} SOFT vs baseline_5m",
         "strategy": "soft", "subset": k_max, "opponent": "baseline_5m"},
        {"id": "baseline5m_solo_heuristic",
         "name": "baseline_5m solo vs heuristic",
         "strategy": "soft", "subset": "baseline", "opponent": "heuristic"},
    ]


def _eval_json_path(run_dir, cfg):
    return os.path.join(run_dir, f"eval_{cfg['id']}.json")


async def _main(args):
    run_dir = os.path.join(ENSEMBLE_RESULTS_DIR, f"run_{args.run_id}")
    if not os.path.isdir(run_dir):
        sys.exit(f"Run directory not found: {run_dir}")

    # Discover member paths (all 30 trained pkls — we'll subset later)
    all_member_paths = sorted(
        glob.glob(os.path.join(run_dir, "member_*", "models", "run_1.pkl")),
        key=lambda p: int(os.path.basename(os.path.dirname(os.path.dirname(p))).split("_")[1])
    )
    if len(all_member_paths) < args.k_max:
        sys.exit(f"Only {len(all_member_paths)} members found, need {args.k_max}")
    used_paths = all_member_paths[:args.k_max]

    # Baseline pkl (for opponent build + solo eval)
    baseline_pkl = os.path.join(ENSEMBLE_RESULTS_DIR, "baseline_5m", "models", "run_1.pkl")
    has_baseline = os.path.exists(baseline_pkl)

    print(f"V6 EVAL SUITE — run_id={args.run_id}  k_max={args.k_max}  n_battles={args.n_battles}")
    print(f"  Run dir       : {run_dir}")
    print(f"  Members found : {len(all_member_paths)} (using first {len(used_paths)})")
    print(f"  Baseline_5m   : {'found' if has_baseline else 'MISSING'} at {baseline_pkl}")

    # Pre-load all needed Q-tables into the cache (one-time cost)
    print(f"\nPre-loading {len(used_paths)} member Q-tables (one-time, cached)...", flush=True)
    t0 = time.time()
    for i, p in enumerate(used_paths, start=1):
        MemberQTable.load(p)
        print(f"  loaded {i}/{len(used_paths)}: {os.path.basename(os.path.dirname(os.path.dirname(p)))}", flush=True)
    if has_baseline:
        print(f"  loading baseline_5m...", flush=True)
        # We don't cache the baseline as a MemberQTable here; HierSmartPlayer.load_table
        # reads it as a regular pkl when needed.
    print(f"Pre-load complete in {_fmt_dur(time.time() - t0)}\n", flush=True)

    # Build configs
    configs = _eval_configs(args.k_max)
    if args.skip_baseline:
        configs = [c for c in configs if c["opponent"] != "baseline_5m" and c["subset"] != "baseline"]
    if not has_baseline:
        configs = [c for c in configs if c["opponent"] != "baseline_5m" and c["subset"] != "baseline"]
        print("⚠️  baseline_5m pkl missing; auto-skipping evals 7-8.\n")

    server = _server_config(args.port)
    state = SuiteState(configs, args.n_battles)
    suffix = uuid.uuid4().hex[:4]   # short unique tag for account names

    # Resume: skip evals whose JSON exists
    if args.resume:
        for i, cfg in enumerate(configs):
            jpath = _eval_json_path(run_dir, cfg)
            if os.path.exists(jpath):
                try:
                    with open(jpath) as f:
                        state.results[i] = json.load(f)
                    print(f"[resume] eval {i+1}/{len(configs)} already complete: {cfg['name']}")
                except Exception:
                    pass

    if _IS_TTY:
        _hide_cursor()

    try:
        # Build a baseline-as-K1 wrapper if eval 8 is in the suite
        baseline_wrapper_dir = "/tmp/baseline_as_k1_eval"
        if any(c["subset"] == "baseline" for c in configs):
            wrap_pkl_dir = os.path.join(baseline_wrapper_dir, "member_1", "models")
            os.makedirs(wrap_pkl_dir, exist_ok=True)
            wrap_link = os.path.join(wrap_pkl_dir, "run_1.pkl")
            if not os.path.exists(wrap_link):
                os.symlink(baseline_pkl, wrap_link)

        for idx, cfg in enumerate(configs):
            if state.results[idx] is not None:
                continue   # resumed

            # Pick member paths for this eval's K
            if cfg["subset"] == "baseline":
                paths = [os.path.join(baseline_wrapper_dir, "member_1", "models", "run_1.pkl")]
            else:
                paths = used_paths[:cfg["subset"]]

            # Build players (fresh per eval to ensure clean account/connection)
            ensemble = _build_ensemble(
                member_paths=paths,
                strategy=cfg["strategy"],
                server=server,
                account_name=f"V6E{idx + 1}{suffix}",
                log_diagnostics=(cfg["opponent"] == "heuristic" and cfg["subset"] != "baseline"),
                pool_seed=99999,
            )
            opponent = _build_opponent(
                kind=cfg["opponent"],
                server=server,
                pool_seed=99999,
                baseline_pkl=baseline_pkl if cfg["opponent"] == "baseline_5m" else None,
                account_suffix=f"{idx + 1}{suffix}",
            )

            # Initial dashboard render so the row goes into RUN status immediately
            state.current_idx = idx
            _render_dashboard(state, force=True)

            result = await _run_one_eval(state, idx, ensemble, opponent, args.n_battles, args.batch_size)

            # Add diagnostics if available
            if hasattr(ensemble, "diagnostics_summary") and ensemble.disagreement_log:
                result["diagnostics"] = ensemble.diagnostics_summary()

            state.results[idx] = result

            # Persist JSON immediately
            with open(_eval_json_path(run_dir, cfg), "w") as f:
                json.dump(result, f, indent=2)

            _render_dashboard(state, force=True)

    except KeyboardInterrupt:
        if _IS_TTY:
            _show_cursor()
        print("\n\nKeyboardInterrupt — completed eval JSONs are persisted.")
        return
    finally:
        if _IS_TTY:
            _show_cursor()

    # Final summary
    _render_dashboard(state, force=True)
    print("\n=== FINAL SUMMARY ===")
    for r in state.results:
        if r:
            print(f"  [{r['id']}] WR={r['win_rate']:.4f} ({r['wins']:,}/{r['n_battles']:,})  "
                  f"time={_fmt_dur(r['elapsed_s'])}  speed={r['speed_bps']:.1f} b/s")


def _build_parser():
    p = argparse.ArgumentParser(
        description="V6 evaluation suite orchestrator with live dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-id", type=int, default=1)
    p.add_argument("--k-max", type=int, default=10,
                   help="Max ensemble size (must fit in RAM — 5.5 GB/member)")
    p.add_argument("--n-battles", type=int, default=100_000,
                   help="Battles per eval")
    p.add_argument("--port", type=int, default=9000,
                   help="Pokemon Showdown port")
    p.add_argument("--batch-size", type=int, default=100,
                   help="Dashboard refresh rate in battles")
    p.add_argument("--resume", action="store_true",
                   help="Skip evals whose JSON already exists")
    p.add_argument("--skip-baseline", action="store_true",
                   help="Skip evals that require baseline_5m")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        if _IS_TTY:
            _show_cursor()
        print("\nShutting down.")
