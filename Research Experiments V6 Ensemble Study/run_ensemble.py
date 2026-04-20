"""
V6 Ensemble Study — Parallel Training Orchestrator with live dashboard.

Launches K ensemble members (default: K=30, 1M battles each) on a fixed pool
of 20 Pokémon.  Diversity source: random seed only (member_k uses seed
BASE_SEED + k).  All members share V5 M8's best hyperparameters (hp_001:
alpha=0.1, gamma=0.99, lambda=0.7, fixed epsilon=0.05).

Each member trains in 5k-battle batches via repeated `train_member.py`
subprocess calls.  V5's SAVE_FREQ=5000 persistence means interrupted runs
resume cleanly from the last checkpoint.

DASHBOARD: Live per-member table (battles / % / rolling WR / speed / ETA),
auto-refreshes every 2 seconds in a TTY.  In non-TTY (piped) contexts, prints
a plain status line every 30 seconds.  Subprocess stdout is redirected to
per-member log files under ensemble_results/run_<id>/member_<k>/train_stdout.log,
so the terminal stays clean.

Usage:
    python run_ensemble.py --max-parallel 16                     # defaults: K=30, 1M battles
    python run_ensemble.py --max-parallel 16 --resume            # resume an interrupted run
    python run_ensemble.py --run-baseline --base-port 9020       # 5M single-member control
    python run_ensemble.py --k 2 --battles 500 --max-parallel 2  # smoke
    python run_ensemble.py --status --run-id 1                   # one-shot snapshot
    python run_ensemble.py --reset --run-id 1                    # DELETE run 1's logs+models
    python run_ensemble.py --reset-baseline                      # DELETE baseline logs+models
"""

import argparse
import csv
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

# V6 root must be on sys.path before shared.* imports.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from shared.config import (   # noqa: E402
    K_MEMBERS,
    BATTLES_PER_MEMBER,
    BASELINE_SINGLE_BATTLES,
    BATTLES_PER_LOG,
    FULL_POKEMON_POOL,
    BASE_SEED,
    DEFAULT_MAX_PARALLEL,
    DEFAULT_BASE_PORT,
    HP_ALPHA,
    HP_GAMMA,
    HP_LAM,
    FIXED_EPS,
    SHOWDOWN_DIR,
    ENSEMBLE_RESULTS_DIR,
    V6_DIR,
)

TRAIN_MEMBER_SCRIPT = os.path.join(V6_DIR, "model_ensemble", "train_member.py")
TRAIN_SINGLE_SCRIPT = os.path.join(V6_DIR, "baseline_single", "train_single.py")

# Each subprocess call trains this many battles before returning. Smaller =
# more resilient to crashes; larger = less subprocess overhead. Matches V5.
BATCH_SIZE = 5_000

# Dashboard refresh rate (TTY) and plain-text tick rate (non-TTY)
DASHBOARD_REFRESH_S = 2.0
PLAIN_TICK_S = 30.0


# ────────────────────────────────────────────────────────────────────────
# Port helpers
# ────────────────────────────────────────────────────────────────────────

def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def _port_for_slot(slot_idx: int, base_port: int = DEFAULT_BASE_PORT) -> int:
    return base_port + slot_idx


def _find_free_port_range(start: int, count: int, scan_limit: int = 200) -> int:
    """Find the first `start'` >= start such that ports [start'..start'+count-1]
    are ALL free.  Used to avoid colliding with a concurrently-running
    ensemble or baseline.  Raises if no range found within `scan_limit`."""
    for offset in range(0, scan_limit):
        candidate = start + offset
        if all(not _is_port_in_use(candidate + i) for i in range(count)):
            return candidate
    raise RuntimeError(
        f"No free {count}-port range found starting from {start}."
    )


def _find_free_port(start: int, scan_limit: int = 200) -> int:
    """Return the first free single port >= start.  Raises if none found."""
    return _find_free_port_range(start, count=1, scan_limit=scan_limit)


# ────────────────────────────────────────────────────────────────────────
# Pokémon Showdown server lifecycle
# ────────────────────────────────────────────────────────────────────────

def _start_showdown_server(port: int, log_dir: str, quiet: bool = False):
    """Spawn a Pokemon Showdown server on `port`. Returns (proc, log_f, port)."""
    if _is_port_in_use(port):
        if not quiet:
            _plain(f"[showdown] port {port} already in use — reusing existing server")
        return (None, None, port)

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"showdown_{port}.log")
    log_f = open(log_path, "w")
    showdown_cmd = os.path.join(SHOWDOWN_DIR, "pokemon-showdown")
    if not os.path.exists(showdown_cmd):
        raise FileNotFoundError(
            f"Pokemon Showdown not found at {showdown_cmd}.\n"
            "Install at <project>/pokemon-showdown/ (matches V5 setup)."
        )

    proc = subprocess.Popen(
        ["node", showdown_cmd, "start", "--no-security", "--skip-build", str(port)],
        cwd=SHOWDOWN_DIR,
        stdout=log_f, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    for _ in range(40):
        if _is_port_in_use(port):
            break
        time.sleep(0.25)
    if not quiet:
        _plain(f"[showdown] started on :{port}  (pid={proc.pid})")
    return (proc, log_f, port)


def _stop_servers(servers):
    for proc, log_f, _port in servers:
        if proc is None:
            continue
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        if log_f:
            log_f.close()


# ────────────────────────────────────────────────────────────────────────
# Progress helpers (CSV + live_status JSON)
# ────────────────────────────────────────────────────────────────────────

def _read_last_csv_row(log_file):
    """Return (battles, rolling_wr, overall_wr) from last row."""
    if not os.path.exists(log_file):
        return 0, 0.0, 0.0
    try:
        with open(log_file, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            last = None
            for row in reader:
                if row:
                    last = row
            if last is None:
                return 0, 0.0, 0.0
            return int(last[0]), float(last[1]), float(last[2])
    except Exception:
        return 0, 0.0, 0.0


def _member_dir(run_dir, k):
    return os.path.join(run_dir, f"member_{k}")


def _member_log_file(run_dir, k):
    return os.path.join(_member_dir(run_dir, k), "logs", "run_1.csv")


def _member_live_status(run_dir, k):
    """Read the live_status JSON (written every 10 battles by train_common)."""
    path = os.path.join(_member_dir(run_dir, k), "live_status_run_1.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _member_progress(run_dir, k):
    """(battles, wins, rolling_wr) from CSV — authoritative at checkpoints."""
    battles, rolling_wr, overall_wr = _read_last_csv_row(_member_log_file(run_dir, k))
    return battles, int(round(overall_wr * battles)), rolling_wr


def _member_live_view(run_dir, k):
    """Fine-grained view for dashboard: prefers live_status (updates every 10
    battles) over CSV (updates every 100). Returns dict with keys:
    battles, rolling_wr, overall_wr, speed, table_size, avg_reward, epsilon."""
    live = _member_live_status(run_dir, k)
    if live:
        return {
            "battles": live.get("battles", 0),
            "rolling_wr": live.get("rolling_wr", 0.0),
            "overall_wr": live.get("overall_wr", 0.0),
            "speed": live.get("speed", 0.0),
            "table_size": live.get("table_size", 0),
            "avg_reward": live.get("avg_reward", 0.0),
            "epsilon": live.get("epsilon", 0.0),
        }
    # Fall back to CSV (e.g., between batch subprocess spawns)
    battles, rolling_wr, overall_wr = _read_last_csv_row(_member_log_file(run_dir, k))
    return {
        "battles": battles,
        "rolling_wr": rolling_wr,
        "overall_wr": overall_wr,
        "speed": 0.0,
        "table_size": 0,
        "avg_reward": 0.0,
        "epsilon": 0.0,
    }


# ────────────────────────────────────────────────────────────────────────
# Dashboard rendering
# ────────────────────────────────────────────────────────────────────────

_IS_TTY = sys.stdout.isatty()


def _plain(msg):
    """Print without disturbing the dashboard — goes to stderr in TTY mode so
    a tee'd log captures it but the dashboard's ANSI updates stay clean."""
    stream = sys.stderr if _IS_TTY else sys.stdout
    print(msg, file=stream, flush=True)


def _fmt_count(n):
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 10_000:
        return f"{n / 1_000:.1f}k"
    if n >= 1_000:
        return f"{n / 1_000:.2f}k"
    return str(n)


def _fmt_duration(seconds):
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    if h < 24:
        return f"{h}h{m:02d}m"
    d = h // 24
    h = h % 24
    return f"{d}d{h:02d}h"


def _mini_bar(fraction, width=10):
    fraction = max(0.0, min(1.0, fraction))
    filled = int(fraction * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _clear_and_home():
    sys.stdout.write("\033[2J\033[H")


def _hide_cursor():
    sys.stdout.write("\033[?25l")


def _show_cursor():
    sys.stdout.write("\033[?25h")


def _fmt_delta(delta):
    """Format a signed delta: +1.2k, -34, +0."""
    sign = "+" if delta >= 0 else "-"
    abs_d = abs(int(delta))
    if abs_d >= 1_000_000:
        return f"{sign}{abs_d / 1_000_000:.1f}M"
    if abs_d >= 1_000:
        return f"{sign}{abs_d / 1_000:.1f}k"
    return f"{sign}{abs_d}"


# Per-member Q-size snapshot across renders, for computing ΔQ.
_prev_table_sizes: dict = {}


def _render_dashboard(run_dir, run_id, K, battles_per_member, start_time,
                      slots, initial_total_battles):
    """Render the full dashboard frame. Called every DASHBOARD_REFRESH_S.

    Columns: Mem | Battles | Prog% | RollWR | OverWR | Speed | Q-size | ΔQ | ETA | bar | Status
    """
    term_w = shutil.get_terminal_size(fallback=(140, 40)).columns
    width = max(80, term_w - 1)
    _clear_and_home()

    elapsed = time.time() - start_time

    # Collect per-member stats + compute ΔQ since last render
    rows = []
    total_battles = 0
    total_speed = 0.0
    total_q = 0
    total_dq = 0
    sum_rolling = 0.0
    sum_overall = 0.0
    n_with_rolling = 0
    for k in range(1, K + 1):
        v = _member_live_view(run_dir, k)
        battles = v["battles"]
        total_battles += battles
        total_speed += v["speed"]

        cur_q = v["table_size"]
        prev_q = _prev_table_sizes.get(k, cur_q)
        dq = cur_q - prev_q
        _prev_table_sizes[k] = cur_q
        total_q += cur_q
        total_dq += dq
        v["dq"] = dq

        if battles >= BATTLES_PER_LOG:
            sum_rolling += v["rolling_wr"]
            sum_overall += v["overall_wr"]
            n_with_rolling += 1
        rows.append((k, v))

    # ── Header ──
    active_slots = sum(1 for s in slots.values() if s is not None)
    batch_total = K * battles_per_member
    done_frac = total_battles / batch_total if batch_total else 0.0
    remaining = max(0, batch_total - total_battles)
    overall_eta_s = remaining / total_speed if total_speed > 0 else None
    avg_rolling = sum_rolling / n_with_rolling if n_with_rolling else 0.0
    avg_overall = sum_overall / n_with_rolling if n_with_rolling else 0.0

    bar = "═" * width
    print(bar)
    print(
        (f"V6 ENSEMBLE  │  Run {run_id}  │  K={K} × {battles_per_member:,} battles"
         f"  │  active: {active_slots}/{K}")[:width]
    )
    line2 = (
        f"  Elapsed: {_fmt_duration(elapsed):<8}"
        f"  Battles: {_fmt_count(total_battles):>6}/{_fmt_count(batch_total):<6}"
        f" ({done_frac * 100:5.1f}%)"
        f"  Roll WR: {avg_rolling:.3f}"
        f"  Over WR: {avg_overall:.3f}"
        f"  Throughput: {total_speed:6.1f} b/s"
        f"  Q-total: {_fmt_count(total_q):>6} ({_fmt_delta(total_dq)})"
    )
    if overall_eta_s is not None:
        line2 += f"  ETA: {_fmt_duration(overall_eta_s)}"
    print(line2[:width])
    print(bar)

    # ── Column header ──
    header = (
        f"  {'Mem':>3}  {'Battles':>9}  {'Prog':>5}  {'RollWR':>6}  {'OverWR':>6}"
        f"  {'Speed':>9}  {'Q-size':>7}  {'ΔQ':>7}  {'ETA':>7}  {'Progress':<12}  {'Status':<6}"
    )
    print(header[:width])
    print(("  " + "─" * (width - 4))[:width])

    # ── Per-member rows ──
    for k, v in rows:
        battles = v["battles"]
        rolling = v["rolling_wr"]
        overall = v["overall_wr"]
        speed = v["speed"]
        table_size = v["table_size"]
        dq = v["dq"]
        prog_frac = battles / battles_per_member if battles_per_member else 0.0

        if battles >= battles_per_member:
            status, eta_str = "DONE", "-"
            bar_glyph = _mini_bar(1.0, width=10)
        elif speed > 0:
            eta_s = max(0, (battles_per_member - battles) / speed)
            status, eta_str = "RUN", _fmt_duration(eta_s)
            bar_glyph = _mini_bar(prog_frac, width=10)
        elif battles > 0:
            status, eta_str = "IDLE", "-"
            bar_glyph = _mini_bar(prog_frac, width=10)
        else:
            status, eta_str = "WAIT", "-"
            bar_glyph = _mini_bar(0.0, width=10)

        line = (
            f"  {k:>3}  {_fmt_count(battles):>9}  {prog_frac * 100:>4.0f}%"
            f"  {rolling:>6.3f}  {overall:>6.3f}"
            f"  {speed:>5.1f} b/s  {_fmt_count(table_size):>7}  {_fmt_delta(dq):>7}"
            f"  {eta_str:>7}  {bar_glyph:<12}  {status:<6}"
        )
        print(line[:width])

    print(("  " + "─" * (width - 4))[:width])
    footer = (
        f"  Ctrl+C to stop (safe — auto-resumes).  "
        f"Per-member full log: tail -f {os.path.relpath(run_dir)}/member_N/train_stdout.log"
    )
    print(footer[:width])
    print(bar)
    sys.stdout.flush()


def _render_plain_tick(run_dir, K, battles_per_member, start_time):
    """Non-TTY fallback: one-line status, printed every PLAIN_TICK_S."""
    total_battles = 0
    total_speed = 0.0
    active = 0
    for k in range(1, K + 1):
        v = _member_live_view(run_dir, k)
        total_battles += v["battles"]
        total_speed += v["speed"]
        if v["speed"] > 0:
            active += 1
    batch_total = K * battles_per_member
    frac = total_battles / batch_total if batch_total else 0.0
    elapsed = time.time() - start_time
    remaining = max(0, batch_total - total_battles)
    eta = _fmt_duration(remaining / total_speed) if total_speed > 0 else "?"
    sys.stdout.write(
        f"[{_fmt_duration(elapsed)}] total {total_battles:,}/{batch_total:,} "
        f"({frac * 100:.1f}%)  active {active}/{K}  speed {total_speed:.1f} b/s  ETA {eta}\n"
    )
    sys.stdout.flush()


# ────────────────────────────────────────────────────────────────────────
# Metadata / manifest
# ────────────────────────────────────────────────────────────────────────

def _write_run_metadata(run_dir, K, battles_per_member, base_seed):
    os.makedirs(run_dir, exist_ok=True)
    params = {
        "K": K,
        "battles_per_member": battles_per_member,
        "alpha": HP_ALPHA, "gamma": HP_GAMMA, "lam": HP_LAM,
        "fixed_epsilon": FIXED_EPS, "epsilon_mode": "fixed",
        "battle_format": "gen4ou",
        "base_seed": base_seed,
        "pool_size": len(FULL_POKEMON_POOL),
    }
    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)
    with open(os.path.join(run_dir, "pool.json"), "w") as f:
        json.dump(list(FULL_POKEMON_POOL), f, indent=2)


def _write_manifest(run_dir, K):
    members = {}
    for k in range(1, K + 1):
        pkl = os.path.join(_member_dir(run_dir, k), "models", "run_1.pkl")
        if os.path.exists(pkl):
            battles, wins, _ = _member_progress(run_dir, k)
            members[f"member_{k}"] = {
                "pkl": os.path.relpath(pkl, run_dir),
                "battles": battles,
                "wins": wins,
            }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump({"K": K, "members": members}, f, indent=2)


# ────────────────────────────────────────────────────────────────────────
# Shepherd: one python subprocess trains one member to completion.
# The parent orchestrator spawns one shepherd per occupied slot.
# ────────────────────────────────────────────────────────────────────────

def _launch_shepherd(k, run_dir, port, base_seed, battles_total):
    """Launch this same script with --internal-shepherd to supervise one
    member.  Returns the Popen handle; stdout/stderr go to a per-member log
    file so the parent's terminal stays clean for the dashboard."""
    stdout_log = os.path.join(_member_dir(run_dir, k), "train_stdout.log")
    os.makedirs(os.path.dirname(stdout_log), exist_ok=True)
    log_f = open(stdout_log, "a", buffering=1)
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--internal-shepherd",
        "--k", str(k),
        "--run-id", str(os.path.basename(run_dir).split("_", 1)[1]) if "_" in os.path.basename(run_dir) else "1",
        "--battles", str(battles_total),
        "--port", str(port),
        "--base-seed", str(base_seed),
    ]
    # NOTE: the shepherd's actual run_id is recovered from run_dir path; the
    # --run-id value here is purely for the shepherd's own arg parser.
    proc = subprocess.Popen(
        cmd, cwd=V6_DIR,
        stdout=log_f, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    proc._log_f = log_f   # attach for later close
    return proc


def _stop_shepherd(proc, save_timeout_s=60):
    """Send SIGTERM to the shepherd's process group and WAIT up to
    save_timeout_s seconds for the child's training loop to save its Q-table
    before escalating to SIGKILL."""
    if proc is None:
        return
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        # Give the shepherd + train_member subprocess time to save the
        # Q-table (train_common's SIGTERM handler does this).
        try:
            proc.wait(timeout=save_timeout_s)
        except subprocess.TimeoutExpired:
            # Escalate: force kill if still alive after the grace period.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    log_f = getattr(proc, "_log_f", None)
    if log_f:
        log_f.close()


def _run_member_batch(k, run_dir, port, base_seed, battles_total, battles_done, wins):
    """Fire one 5k-battle subprocess; return new (battles_done, wins)."""
    output_dir = _member_dir(run_dir, k)
    os.makedirs(output_dir, exist_ok=True)
    remaining = battles_total - battles_done
    batch = min(BATCH_SIZE, remaining)
    seed = base_seed + k
    pool_str = ",".join(FULL_POKEMON_POOL)
    cmd = [
        sys.executable, TRAIN_MEMBER_SCRIPT,
        "--run_id", "1",
        "--seed", str(seed),
        "--epsilon", str(FIXED_EPS),
        "--batch_size", str(batch),
        "--historic_battles", str(battles_done),
        "--historic_wins", str(wins),
        "--port", str(port),
        "--alpha", str(HP_ALPHA),
        "--gamma_val", str(HP_GAMMA),
        "--lam", str(HP_LAM),
        "--output_dir", output_dir,
        "--pool", pool_str,
        "--epsilon_mode", "fixed",
        "--fixed_epsilon", str(FIXED_EPS),
    ]
    # Note: in shepherd mode, this subprocess's stdout is inherited from the
    # shepherd's stdout, which is the per-member log file — so its prints go
    # to disk, not the parent terminal.
    subprocess.run(cmd, cwd=V6_DIR)
    battles, wins, _ = _member_progress(run_dir, k)
    return battles, wins


def _internal_shepherd_main(args):
    """Runs inside a shepherd subprocess.  Loops batches until member_k is
    at args.battles."""
    # The parent passes --run-id as part of the command; we use it directly.
    run_dir = os.path.join(ENSEMBLE_RESULTS_DIR, f"run_{args.run_id}")
    battles, wins, _ = _member_progress(run_dir, args.k)
    print(f"[shepherd k={args.k}] starting on port {args.port}  "
          f"(already done: {battles:,} / {args.battles:,})", flush=True)
    while battles < args.battles:
        battles, wins = _run_member_batch(
            args.k, run_dir, args.port, args.base_seed, args.battles, battles, wins
        )
    print(f"[shepherd k={args.k}] complete ({battles:,} battles, {wins:,} wins)",
          flush=True)


# ────────────────────────────────────────────────────────────────────────
# Top-level ensemble run (slot-pool model, with dashboard)
# ────────────────────────────────────────────────────────────────────────

def run_ensemble(K, battles_per_member, run_id, max_parallel, base_seed,
                 base_port):
    """Auto-resumes: members with saved checkpoints at or past battles_per_member
    are skipped; partial members resume from their last 5k-battle save.
    Use --reset to start from scratch."""
    run_dir = os.path.join(ENSEMBLE_RESULTS_DIR, f"run_{run_id}")
    _write_run_metadata(run_dir, K, battles_per_member, base_seed)

    # Build queue of members that still need work (auto-resume).
    queue = []
    already_complete = 0
    for k in range(1, K + 1):
        battles_done, _, _ = _member_progress(run_dir, k)
        if battles_done >= battles_per_member:
            already_complete += 1
            continue
        queue.append(k)

    initial_total = sum(_member_progress(run_dir, k)[0] for k in range(1, K + 1))

    if not queue:
        _plain(f"All {K} members already trained in {run_dir}. Writing manifest.")
        _write_manifest(run_dir, K)
        return

    _plain(f"\nV6 Ensemble Run {run_id}  │  K={K}  │  {battles_per_member:,} battles/member"
           f"  │  queue={len(queue)}  (already complete: {already_complete})"
           f"  │  max_parallel={max_parallel}")

    # Auto-scan for a free consecutive port range starting from base_port.
    try:
        effective_base_port = _find_free_port_range(base_port, max_parallel)
    except RuntimeError as e:
        _plain(f"ERROR: {e}")
        return
    if effective_base_port != base_port:
        _plain(f"  [ports] {base_port}..{base_port + max_parallel - 1} in use — "
               f"using {effective_base_port}..{effective_base_port + max_parallel - 1} instead")
    _plain(f"  [ports] ensemble: {effective_base_port}..{effective_base_port + max_parallel - 1}\n")

    # Spawn Showdown servers once, sized to max_parallel.
    servers = []
    for slot_idx in range(max_parallel):
        port = _port_for_slot(slot_idx, effective_base_port)
        servers.append(_start_showdown_server(port, os.path.join(run_dir, "showdown_logs")))
    time.sleep(2)

    # Slot pool: slot_idx -> (shepherd Popen, member_k) or None
    slots = {i: None for i in range(max_parallel)}
    start_time = time.time()
    q_idx = 0
    last_plain = 0.0

    if _IS_TTY:
        _hide_cursor()

    def assign_next(slot_idx):
        nonlocal q_idx
        if q_idx >= len(queue):
            return
        k = queue[q_idx]
        q_idx += 1
        port = _port_for_slot(slot_idx, effective_base_port)
        proc = _launch_shepherd(k, run_dir, port, base_seed, battles_per_member)
        slots[slot_idx] = (proc, k)

    try:
        # Fill all slots up to queue size
        for slot_idx in range(max_parallel):
            assign_next(slot_idx)

        while True:
            # Reap finished shepherds; reassign the slot.
            for slot_idx, val in list(slots.items()):
                if val is None:
                    continue
                proc, k = val
                if proc.poll() is not None:
                    log_f = getattr(proc, "_log_f", None)
                    if log_f:
                        log_f.close()
                    slots[slot_idx] = None
                    assign_next(slot_idx)

            # All done?
            if q_idx >= len(queue) and all(v is None for v in slots.values()):
                break

            # Render
            if _IS_TTY:
                _render_dashboard(run_dir, run_id, K, battles_per_member,
                                  start_time, slots, initial_total)
                time.sleep(DASHBOARD_REFRESH_S)
            else:
                now = time.time()
                if now - last_plain >= PLAIN_TICK_S:
                    _render_plain_tick(run_dir, K, battles_per_member, start_time)
                    last_plain = now
                time.sleep(1.0)
    except KeyboardInterrupt:
        _plain("\n\nKeyboardInterrupt — stopping shepherds and Showdown servers.")
        for val in slots.values():
            if val is not None:
                _stop_shepherd(val[0])
    finally:
        if _IS_TTY:
            _show_cursor()
        _stop_servers(servers)

    # One final dashboard render to show terminal state
    if _IS_TTY:
        _render_dashboard(run_dir, run_id, K, battles_per_member, start_time,
                          slots, initial_total)
    _write_manifest(run_dir, K)
    _plain(f"\nManifest written to {os.path.join(run_dir, 'manifest.json')}")


# ────────────────────────────────────────────────────────────────────────
# 5M single-member baseline (same dashboard look, K=1)
# ────────────────────────────────────────────────────────────────────────

def run_baseline(run_id, base_seed, port):
    """Auto-resumes from the last 5k-battle checkpoint.  If `port` is in use
    (e.g., the ensemble is running on it), auto-scans for a free one."""
    baseline_dir = os.path.join(ENSEMBLE_RESULTS_DIR, "baseline_5m")
    os.makedirs(baseline_dir, exist_ok=True)

    log_file = os.path.join(baseline_dir, "logs", "run_1.csv")
    battles_done, rolling_wr, overall_wr = _read_last_csv_row(log_file)
    wins = int(round(overall_wr * battles_done))

    if battles_done >= BASELINE_SINGLE_BATTLES:
        _plain(f"Baseline already complete ({battles_done:,} battles).")
        return

    # Auto-scan for a free port — avoids collision with a concurrent ensemble.
    try:
        effective_port = _find_free_port(port)
    except RuntimeError as e:
        _plain(f"ERROR: {e}")
        return
    if effective_port != port:
        _plain(f"  [ports] {port} in use — using {effective_port} instead")

    _plain(f"\nV6 Baseline  │  single HierSmartPlayer  │  "
           f"{BASELINE_SINGLE_BATTLES:,} battles  │  port {effective_port}")
    if battles_done > 0:
        _plain(f"  [resume] already done: {battles_done:,} / {BASELINE_SINGLE_BATTLES:,}"
               f"  ({100 * battles_done / BASELINE_SINGLE_BATTLES:.1f}%)")
    _plain("")

    port = effective_port   # shadow so the rest of the function uses it
    servers = [_start_showdown_server(port, os.path.join(baseline_dir, "showdown_logs"))]
    time.sleep(2)

    start_time = time.time()
    last_plain = 0.0
    if _IS_TTY:
        _hide_cursor()

    # Per-render Q-size snapshot for computing ΔQ
    prev_q_ref = [None]   # mutable closure cell

    def _render():
        width = max(80, shutil.get_terminal_size(fallback=(120, 40)).columns - 1)
        _clear_and_home()
        live = {}
        path = os.path.join(baseline_dir, "live_status_run_1.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    live = json.load(f)
            except Exception:
                live = {}

        b = live.get("battles", battles_done)
        roll = live.get("rolling_wr", rolling_wr)
        over = live.get("overall_wr", overall_wr)
        speed = live.get("speed", 0.0)
        q_size = live.get("table_size", 0)
        avg_rew = live.get("avg_reward", 0.0)
        eps = live.get("epsilon", FIXED_EPS)

        if prev_q_ref[0] is None:
            prev_q_ref[0] = q_size
        dq = q_size - prev_q_ref[0]
        prev_q_ref[0] = q_size

        frac = b / BASELINE_SINGLE_BATTLES if BASELINE_SINGLE_BATTLES else 0.0
        remaining = max(0, BASELINE_SINGLE_BATTLES - b)
        eta = _fmt_duration(remaining / speed) if speed > 0 else "?"
        elapsed = _fmt_duration(time.time() - start_time)
        bar_char = "═" * width

        print(bar_char)
        print(f"V6 BASELINE  │  Run {run_id}  │  1 × {BASELINE_SINGLE_BATTLES:,} battles"[:width])
        print(bar_char)
        # Line 1: progress & WR
        print((
            f"  Elapsed: {elapsed:<8}"
            f"  Battles: {b:,}/{BASELINE_SINGLE_BATTLES:,}"
            f" ({frac * 100:5.1f}%)"
            f"  Roll WR: {roll:.3f}"
            f"  Over WR: {over:.3f}"
            f"  Speed: {speed:5.1f} b/s"
            f"  ETA: {eta}"
        )[:width])
        # Line 2: Q-table stats + epsilon + avg reward
        print((
            f"  Q-size: {q_size:>9,} entries  ({_fmt_delta(dq)}/refresh)"
            f"  ε: {eps:.3f}"
            f"  AvgRew: {avg_rew:+.3f}"
        )[:width])
        print(bar_char)
        # Progress bar (single line, dynamic width)
        bar_w = min(60, max(20, width - 20))
        print(f"  {_mini_bar(frac, width=bar_w)}  {frac * 100:5.1f}%"[:width])
        print(bar_char)
        print(
            f"  Ctrl+C is safe (auto-resumes).  "
            f"tail -f {os.path.relpath(baseline_dir)}/train_stdout.log for full log"[:width]
        )
        print(bar_char)
        sys.stdout.flush()

    pool_str = ",".join(FULL_POKEMON_POOL)
    try:
        while battles_done < BASELINE_SINGLE_BATTLES:
            remaining = BASELINE_SINGLE_BATTLES - battles_done
            batch = min(BATCH_SIZE, remaining)
            stdout_log = os.path.join(baseline_dir, "train_stdout.log")
            # Popen + poll so dashboard can update every DASHBOARD_REFRESH_S
            # while the 5k-battle batch runs (instead of staying silent for
            # ~5 minutes per batch).  Subprocess stdout goes to the log file.
            log_f = open(stdout_log, "a", buffering=1)
            proc = subprocess.Popen(
                [sys.executable, TRAIN_SINGLE_SCRIPT,
                 "--run_id", "1",
                 "--seed", str(base_seed),
                 "--epsilon", str(FIXED_EPS),
                 "--batch_size", str(batch),
                 "--historic_battles", str(battles_done),
                 "--historic_wins", str(wins),
                 "--port", str(port),
                 "--alpha", str(HP_ALPHA),
                 "--gamma_val", str(HP_GAMMA),
                 "--lam", str(HP_LAM),
                 "--output_dir", baseline_dir,
                 "--pool", pool_str,
                 "--epsilon_mode", "fixed",
                 "--fixed_epsilon", str(FIXED_EPS)],
                cwd=V6_DIR,
                stdout=log_f, stderr=subprocess.STDOUT,
            )
            # Render dashboard live while the batch runs.
            while proc.poll() is None:
                if _IS_TTY:
                    _render()
                    time.sleep(DASHBOARD_REFRESH_S)
                else:
                    now = time.time()
                    if now - last_plain >= PLAIN_TICK_S:
                        # Read live_status for the most up-to-date stats.
                        live_path = os.path.join(baseline_dir, "live_status_run_1.json")
                        live = {}
                        if os.path.exists(live_path):
                            try:
                                with open(live_path) as f:
                                    live = json.load(f)
                            except Exception:
                                live = {}
                        b = live.get("battles", battles_done)
                        roll = live.get("rolling_wr", rolling_wr)
                        speed = live.get("speed", 0.0)
                        frac = b / BASELINE_SINGLE_BATTLES
                        eta = _fmt_duration((BASELINE_SINGLE_BATTLES - b) / speed) if speed > 0 else "?"
                        sys.stdout.write(
                            f"[{_fmt_duration(time.time() - start_time)}] "
                            f"baseline {b:,}/{BASELINE_SINGLE_BATTLES:,} "
                            f"({frac * 100:.1f}%)  WR {roll:.3f}  {speed:.1f} b/s  ETA {eta}\n"
                        )
                        sys.stdout.flush()
                        last_plain = now
                    time.sleep(1.0)
            log_f.close()
            battles_done, rolling_wr, overall_wr = _read_last_csv_row(log_file)
            wins = int(round(overall_wr * battles_done))

            # Render between batches
            if _IS_TTY:
                _render()
            else:
                now = time.time()
                if now - last_plain >= PLAIN_TICK_S:
                    frac = battles_done / BASELINE_SINGLE_BATTLES
                    sys.stdout.write(
                        f"[{_fmt_duration(time.time() - start_time)}] "
                        f"baseline {battles_done:,}/{BASELINE_SINGLE_BATTLES:,} "
                        f"({frac * 100:.1f}%)  WR {rolling_wr:.3f}\n"
                    )
                    sys.stdout.flush()
                    last_plain = now

        _plain(f"Baseline complete ({battles_done:,} battles, {wins:,} wins)")
    except KeyboardInterrupt:
        _plain("\nInterrupted — safe to re-run (checkpoints persist every 5k battles).")
    finally:
        if _IS_TTY:
            _show_cursor()
        _stop_servers(servers)


# ────────────────────────────────────────────────────────────────────────
# Status snapshot (one-shot, no dashboard)
# ────────────────────────────────────────────────────────────────────────

def _print_status(run_id, K, battles_per_member):
    run_dir = os.path.join(ENSEMBLE_RESULTS_DIR, f"run_{run_id}")
    print(f"\nV6 Ensemble Run {run_id}  │  target: K={K} × {battles_per_member:,} battles\n")
    print(f"  {'Mem':>3}  {'Battles':>12}  {'% done':>7}  {'Rolling WR':>10}  {'Speed':>9}")
    print(f"  {'-' * 3}  {'-' * 12}  {'-' * 7}  {'-' * 10}  {'-' * 9}")
    total_done = 0
    for k in range(1, K + 1):
        v = _member_live_view(run_dir, k)
        total_done += v["battles"]
        pct = 100.0 * v["battles"] / battles_per_member if battles_per_member else 0.0
        print(f"  {k:>3}  {v['battles']:>12,}  {pct:>6.1f}%  {v['rolling_wr']:>9.3f}  "
              f"{v['speed']:>6.1f} b/s")
    total_target = K * battles_per_member
    pct_overall = 100.0 * total_done / total_target if total_target else 0.0
    print(f"\n  TOTAL      {total_done:>12,}  {pct_overall:>6.1f}%\n")

    # Baseline status too
    bl = os.path.join(ENSEMBLE_RESULTS_DIR, "baseline_5m", "logs", "run_1.csv")
    if os.path.exists(bl):
        b, r, _ = _read_last_csv_row(bl)
        print(f"  Baseline (5M single):  {b:,}  ({100*b/BASELINE_SINGLE_BATTLES:.1f}%)  "
              f"Rolling WR: {r:.3f}\n")


# ────────────────────────────────────────────────────────────────────────
# Reset (destructive — delete logs + models)
# ────────────────────────────────────────────────────────────────────────

def _confirm(prompt):
    sys.stdout.write(f"\n{prompt}  type 'yes' to confirm: ")
    sys.stdout.flush()
    try:
        return input().strip().lower() == "yes"
    except EOFError:
        return False


def _reset_run(run_id, skip_confirm=False):
    run_dir = os.path.join(ENSEMBLE_RESULTS_DIR, f"run_{run_id}")
    if not os.path.isdir(run_dir):
        print(f"Run directory does not exist: {run_dir}")
        return
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, files in os.walk(run_dir) for f in files)
    print(f"About to DELETE {run_dir}  ({size / 1e6:.1f} MB)")
    if not skip_confirm and not _confirm("Are you sure?"):
        print("Aborted.")
        return
    shutil.rmtree(run_dir)
    print(f"Deleted {run_dir}")


def _reset_baseline(skip_confirm=False):
    bl_dir = os.path.join(ENSEMBLE_RESULTS_DIR, "baseline_5m")
    if not os.path.isdir(bl_dir):
        print(f"Baseline directory does not exist: {bl_dir}")
        return
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, files in os.walk(bl_dir) for f in files)
    print(f"About to DELETE {bl_dir}  ({size / 1e6:.1f} MB)")
    if not skip_confirm and not _confirm("Are you sure?"):
        print("Aborted.")
        return
    shutil.rmtree(bl_dir)
    print(f"Deleted {bl_dir}")


# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────

_BASELINE_DEFAULT_PORT = 9100    # far enough from ensemble's 9000-range


def _build_parser():
    p = argparse.ArgumentParser(
        description="V6 Ensemble Training Orchestrator. "
                    "Auto-resumes from saved checkpoints; auto-scans for a free port range.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--k", type=int, default=K_MEMBERS,
                   help="Ensemble size")
    p.add_argument("--battles", type=int, default=BATTLES_PER_MEMBER,
                   help="Battles per member")
    p.add_argument("--run-id", type=int, default=1,
                   help="Run ID; different IDs produce disjoint seed ranges")
    p.add_argument("--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL,
                   help="Max concurrent training slots")
    p.add_argument("--base-seed", type=int, default=BASE_SEED,
                   help="Base seed (member_k uses BASE_SEED + k)")
    p.add_argument("--base-port", type=int, default=None,
                   help="Base Showdown port. Default: 9000 for ensemble, 9100 for baseline. "
                        "Auto-scans upward if in use.")
    p.add_argument("--status", action="store_true",
                   help="Print one-shot progress table and exit")
    p.add_argument("--run-baseline", action="store_true",
                   help="Train the 5M compute-matched single-member baseline")
    p.add_argument("--reset", action="store_true",
                   help="DELETE all logs and models for --run-id (prompts for 'yes')")
    p.add_argument("--reset-baseline", action="store_true",
                   help="DELETE baseline_5m logs and models (prompts for 'yes')")
    p.add_argument("--reset-all", action="store_true",
                   help="DELETE all ensemble_results/ contents (prompts for 'yes')")
    p.add_argument("--yes", action="store_true",
                   help="Skip confirmation prompts for --reset*")
    # Internal shepherd mode (not for users)
    p.add_argument("--internal-shepherd", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--port", type=int, default=DEFAULT_BASE_PORT, help=argparse.SUPPRESS)
    return p


def main():
    args = _build_parser().parse_args()

    if args.internal_shepherd:
        _internal_shepherd_main(args)
        return

    if args.reset_all:
        if os.path.isdir(ENSEMBLE_RESULTS_DIR):
            size = sum(os.path.getsize(os.path.join(dp, f))
                       for dp, _, files in os.walk(ENSEMBLE_RESULTS_DIR) for f in files)
            print(f"About to DELETE all of {ENSEMBLE_RESULTS_DIR}  ({size / 1e6:.1f} MB)")
            if args.yes or _confirm("WIPE EVERYTHING?"):
                shutil.rmtree(ENSEMBLE_RESULTS_DIR)
                os.makedirs(ENSEMBLE_RESULTS_DIR, exist_ok=True)
                print("Done.")
        else:
            print(f"{ENSEMBLE_RESULTS_DIR} does not exist.")
        return

    if args.reset:
        _reset_run(args.run_id, skip_confirm=args.yes)
        return

    if args.reset_baseline:
        _reset_baseline(skip_confirm=args.yes)
        return

    if args.status:
        _print_status(args.run_id, args.k, args.battles)
        return

    if args.run_baseline:
        # Default baseline port is 9100 (clear of the ensemble's 9000-range).
        port = args.base_port if args.base_port is not None else _BASELINE_DEFAULT_PORT
        run_baseline(run_id=args.run_id, base_seed=args.base_seed, port=port)
        return

    # Default ensemble base-port is 9000; auto-scan upward if in use.
    base_port = args.base_port if args.base_port is not None else DEFAULT_BASE_PORT
    run_ensemble(
        K=args.k,
        battles_per_member=args.battles,
        run_id=args.run_id,
        max_parallel=args.max_parallel,
        base_seed=args.base_seed,
        base_port=base_port,
    )


if __name__ == "__main__":
    main()
