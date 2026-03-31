"""
Research Experiment Orchestrator
=================================
Runs all 4 models x 5 runs in parallel with linear epsilon decay.
Each model runs as its own subprocess. A live dashboard polls CSV logs
to show progress for all 4 models simultaneously.
"""

import subprocess
import sys
import os
import time
import glob
import csv
import json
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import signal

from shared.config import (
    MODEL_NAMES, MODEL_LABELS,
    BATTLES_PER_RUN, BATTLES_PER_LOG, RUNS_PER_MODEL, RANDOM_SEEDS,
    SHOWDOWN_DIR,
    get_epsilon,
)

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_SCRIPTS = {
    "model_1_flat_zero": os.path.join(EXPERIMENT_DIR, "model_1_flat_zero", "train.py"),
    "model_2_flat_smart": os.path.join(EXPERIMENT_DIR, "model_2_flat_smart", "train.py"),
    "model_3_hier_zero": os.path.join(EXPERIMENT_DIR, "model_3_hier_zero", "train.py"),
    "model_4_hier_smart": os.path.join(EXPERIMENT_DIR, "model_4_hier_smart", "train.py"),
}

BATCH_SIZE = 2_000  # Battles per subprocess invocation

# Each model gets its own Showdown server for max throughput
MODEL_PORTS = {
    "model_1_flat_zero": 9000,
    "model_2_flat_smart": 9001,
    "model_3_hier_zero": 9002,
    "model_4_hier_smart": 9003,
}

# For --parallel-runs: each (model, run) gets its own port
# model_1 runs 1-5 → ports 9000-9004, model_2 → 9005-9009, etc.
def _get_parallel_run_port(model_idx, run_idx):
    """Port for (model_idx 0-3, run_idx 0-4) in parallel-runs mode."""
    return 9000 + model_idx * RUNS_PER_MODEL + run_idx


def _is_port_in_use(port):
    """Check if a port is already bound."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def _start_showdown_servers():
    """Start Showdown server instances on ports 9000-9003. Skips ports already in use."""
    servers = []
    showdown_cmd = os.path.join(SHOWDOWN_DIR, "pokemon-showdown")
    ports_started = []
    for model_name in MODEL_NAMES:
        port = MODEL_PORTS[model_name]
        if _is_port_in_use(port):
            print(f"  Port {port} already in use — reusing existing server")
            servers.append((None, None, port))
            continue
        log_dir = os.path.join(EXPERIMENT_DIR, model_name, "stdout_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_f = open(os.path.join(log_dir, f"showdown_{port}.log"), "w")
        proc = subprocess.Popen(
            ["node", showdown_cmd, "start", "--no-security", "--skip-build", str(port)],
            cwd=SHOWDOWN_DIR,
            stdout=log_f, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        servers.append((proc, log_f, port))
        ports_started.append(port)
    if ports_started:
        print(f"  Started servers on ports: {ports_started}")
        print("  Waiting for servers to initialize...")
        time.sleep(5)
    return servers


def _stop_showdown_servers(servers):
    """Stop Showdown server processes we started (skip pre-existing ones)."""
    for proc, log_f, port in servers:
        if proc is None:
            continue  # Was pre-existing, don't touch
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        if log_f:
            log_f.close()


def _get_actual_progress(model_name, run_id):
    """Read actual battle count and wins from the CSV log. Robust to crashes."""
    log_file = os.path.join(EXPERIMENT_DIR, model_name, "logs", f"run_{run_id}.csv")
    if not os.path.exists(log_file):
        return 0, 0
    try:
        with open(log_file, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            last_row = None
            for row in reader:
                if row:
                    last_row = row
            if last_row:
                battles = int(last_row[0])
                overall_wr = float(last_row[2])
                wins = int(round(overall_wr * battles))
                return battles, wins
    except Exception:
        pass
    return 0, 0


def run_single_model_all_runs(model_name, port=None):
    """Run all 5 runs for one model sequentially. Called from parallel launcher."""
    if port is None:
        port = MODEL_PORTS.get(model_name, 9000)
    for run_idx in range(RUNS_PER_MODEL):
        run_id = run_idx + 1
        seed = RANDOM_SEEDS[run_idx]
        _run_one_experiment(model_name, run_id, seed, port=port)


def _run_one_experiment(model_name, run_id, seed, port=None):
    """Run one model for one full run using batched subprocesses.

    Reads actual progress from CSV after each batch — robust to crashes
    and restarts (always resumes from where training actually got to).
    """
    train_script = TRAIN_SCRIPTS[model_name]
    label = MODEL_LABELS[model_name]
    if port is None:
        port = MODEL_PORTS.get(model_name, 9000)

    # Read actual progress (handles resume after crash)
    battles_done, wins = _get_actual_progress(model_name, run_id)

    if battles_done == 0:
        print(f"\n{'='*60}")
        print(f" {label} — Run {run_id} (seed={seed}, port={port})")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f" {label} — Run {run_id} RESUMING from {battles_done:,} battles")
        print(f"{'='*60}")

    while battles_done < BATTLES_PER_RUN:
        remaining = BATTLES_PER_RUN - battles_done
        batch = min(BATCH_SIZE, remaining)
        epsilon = get_epsilon(battles_done)

        cmd = [
            sys.executable, train_script,
            "--run_id", str(run_id),
            "--seed", str(seed),
            "--epsilon", str(epsilon),
            "--batch_size", str(batch),
            "--historic_battles", str(battles_done),
            "--historic_wins", str(wins),
            "--port", str(port),
        ]

        result = subprocess.run(cmd, cwd=EXPERIMENT_DIR)

        if result.returncode != 0:
            print(f"WARNING: {model_name} run {run_id} batch failed (exit {result.returncode})")

        # Always read actual progress from CSV — never assume batch completed fully
        battles_done, wins = _get_actual_progress(model_name, run_id)


def _get_live_status(model_name):
    """Read live status for this model. In parallel-runs mode, returns AVERAGED values
    across all active runs. In sequential mode, returns the single active run's status."""
    model_dir = os.path.join(EXPERIMENT_DIR, model_name)
    # Try per-run files first (parallel-runs mode), fall back to legacy single file
    all_statuses = []
    for f_name in glob.glob(os.path.join(model_dir, "live_status_run_*.json")):
        try:
            with open(f_name, 'r') as f:
                data = json.load(f)
            all_statuses.append(data)
        except (json.JSONDecodeError, OSError):
            continue

    if all_statuses:
        if not _PARALLEL_RUNS_MODE or len(all_statuses) == 1:
            # Sequential mode or single run: return the one with most battles
            return max(all_statuses, key=lambda d: d.get('battles', 0))
        # Parallel mode: average across all active runs
        n = len(all_statuses)
        total_battles = sum(d.get('battles', 0) for d in all_statuses)
        avg = {
            'run_id': 'AVG',
            'battles': total_battles,
            'rolling_wr': sum(d.get('rolling_wr', 0) for d in all_statuses) / n,
            'overall_wr': sum(d.get('overall_wr', 0) for d in all_statuses) / n,
            'avg_reward': sum(d.get('avg_reward', 0) for d in all_statuses) / n,
            'epsilon': sum(d.get('epsilon', 0) for d in all_statuses) / n,
            'table_size': sum(d.get('table_size', 0) for d in all_statuses) // n,
            'speed': sum(d.get('speed', 0) for d in all_statuses),  # total throughput
            'progress_in_window': sum(d.get('progress_in_window', 0) for d in all_statuses) // n,
            'window_size': all_statuses[0].get('window_size', BATTLES_PER_LOG),
        }
        return avg

    # Legacy fallback
    status_file = os.path.join(model_dir, "live_status.json")
    try:
        with open(status_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _get_all_run_statuses(model_name):
    """Read live status for ALL runs of a model. Returns list of (run_id, status_dict)."""
    model_dir = os.path.join(EXPERIMENT_DIR, model_name)
    results = []
    for f_name in sorted(glob.glob(os.path.join(model_dir, "live_status_run_*.json"))):
        try:
            with open(f_name, 'r') as f:
                data = json.load(f)
            results.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return results


def _get_run_count(model_name):
    """Count CSV log files to determine run progress."""
    log_dir = os.path.join(EXPERIMENT_DIR, model_name, "logs")
    files = glob.glob(os.path.join(log_dir, "run_*.csv"))
    return len(files)


def _fmt_elapsed(seconds):
    """Format elapsed time."""
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h{m:02d}m"


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _mini_bar(fraction, width=10):
    """Small progress bar: [████░░░░░░]"""
    filled = int(fraction * width)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


# ── Dashboard layout ──
# Line 1: ═══ top border
# Line 2: header stats (elapsed, total%, ETA)
# Line 3: ═══ border
# Line 4: column headers
# Line 5: ─── separator
# Lines 6-9: model rows (one per model)
# Line 10: ─── separator
# Line 11: runs summary (only in --parallel-runs mode, else footer)
# Line 12: ═══ bottom border
# Line 13: footer
_HEADER_LINE = 2      # 1-indexed line for the stats header
_DATA_START = 6       # 1-indexed line for first model row
_RUNS_LINE = 11       # 1-indexed line for runs summary
_BOTTOM_LINE = 12
_FOOTER_LINE = 13
_TOTAL_LINES = 13
_W = 130              # Table width
_PARALLEL_RUNS_MODE = False  # Set by run_all_parallel_runs()


def _draw_table_frame():
    """Draw the static table frame once. Called at dashboard init."""
    sys.stdout.write("\033[2J\033[H")  # Clear screen
    # Line 1
    print(f"{'═' * _W}")
    # Line 2 (placeholder, will be overwritten)
    print(f"  {'RESEARCH EXPERIMENT':^{_W - 4}}")
    # Line 3
    print(f"{'═' * _W}")
    # Line 4: column headers
    h = (f"  {'Model':<20} {'':>4} {'Run':>5} {'Battles':>10} {'Done':>7}"
         f" │ {'Roll%':>6} {'Avg%':>6} {'AvgRew':>7} {'Eps':>6} {'States':>9} {'Bat/s':>7} {'Next':>12}")
    print(h)
    # Line 5
    print(f"  {'─' * (_W - 4)}")
    # Lines 6-9: empty rows
    for _ in range(len(MODEL_NAMES)):
        print()
    # Line 10: separator
    print(f"  {'─' * (_W - 4)}")
    # Line 11: runs summary (blank initially)
    print()
    # Line 12
    print(f"{'═' * _W}")
    # Line 13
    print(f"  Ctrl+C to stop all models.")
    sys.stdout.flush()


def _move_to(line):
    """Move cursor to the start of a 1-indexed line."""
    sys.stdout.write(f"\033[{line};1H")


def _write_line(line_num, text):
    """Overwrite a specific line with text, padding to full width."""
    _move_to(line_num)
    padded = text.ljust(_W)
    sys.stdout.write(padded)


def _update_dashboard(processes, start_time):
    """
    Update only the changing lines in the table using live status files.
    All data comes directly from the training loop — no estimates.
    """
    elapsed = time.time() - start_time
    elapsed_str = _fmt_elapsed(elapsed)
    total_per_model = BATTLES_PER_RUN * RUNS_PER_MODEL

    total_overall_battles = 0
    model_rows = []

    for model_name in MODEL_NAMES:
        label = MODEL_LABELS[model_name]
        proc = processes.get(model_name)

        if proc and proc.poll() is not None:
            status = "DONE" if proc.returncode == 0 else "FAIL"
        elif proc:
            status = "AVG" if _PARALLEL_RUNS_MODE else "RUN"
        else:
            status = "WAIT"

        # Compute true total battles across ALL runs (completed + active)
        # This ensures ETA stays accurate as runs complete and clean up status files
        model_total_battles = 0
        for r_idx in range(RUNS_PER_MODEL):
            b, _ = _get_actual_progress(model_name, r_idx + 1)
            model_total_battles += b

        live = _get_live_status(model_name)
        if live:
            # Add battles from active runs not yet in CSV
            # (live status may be ahead of CSV by up to BATTLES_PER_LOG)
            live_battles = live.get('battles', 0)
            # Use the larger of CSV-based total and live-status total
            battles = max(model_total_battles, live_battles)

            run_id = live.get('run_id', '?')
            total_runs = _get_run_count(model_name) or 1
            speed = live.get('speed', 0.0)

            rolling_str = f"{live.get('rolling_wr', 0):.1%}"
            overall_str = f"{live.get('overall_wr', 0):.1%}"
            rew_str = f"{live.get('avg_reward', 0):+.3f}"
            eps_str = f"{live.get('epsilon', 0):.3f}"
            table_size = live.get('table_size', 0)
            if table_size >= 1_000_000:
                states_str = f"{table_size/1e6:.2f}M"
            elif table_size >= 1_000:
                states_str = f"{table_size/1e3:.1f}K"
            else:
                states_str = str(table_size)
            speed_str = f"{speed:.1f}/s"

            # Progress bar: exact progress within current 1K window
            progress_in_window = live.get('progress_in_window', 0)
            window_size = live.get('window_size', BATTLES_PER_LOG)
            window_frac = progress_in_window / window_size if window_size > 0 else 0
            bar = _mini_bar(window_frac)

            # Overall % done across all runs
            total_overall_battles += battles
            pct = min(battles / total_per_model * 100, 100.0)

            run_str = "AVG" if (_PARALLEL_RUNS_MODE and run_id == 'AVG') else f"{run_id}/{RUNS_PER_MODEL}"
            line = (f"  {label:<20} {status:>4} {run_str:>5} {battles:>10,} {pct:>6.1f}%"
                    f" │ {rolling_str:>6} {overall_str:>6} {rew_str:>7} {eps_str:>6} {states_str:>9} {speed_str:>7} {bar}")
        else:
            # Even without live status, count completed CSV battles for ETA
            total_overall_battles += model_total_battles
            if model_total_battles > 0:
                pct = min(model_total_battles / total_per_model * 100, 100.0)
                bar = _mini_bar(0)
                line = (f"  {label:<20} {status:>4} {'—':>5} {model_total_battles:>10,} {pct:>6.1f}%"
                        f" │ {'—':>6} {'—':>6} {'—':>7} {'—':>6} {'—':>9} {'—':>7} {bar}")
            else:
                bar = _mini_bar(0)
                line = (f"  {label:<20} {status:>4} {'—':>5} {'—':>10} {'0.0%':>7}"
                        f" │ {'—':>6} {'—':>6} {'—':>7} {'—':>6} {'—':>9} {'—':>7} {bar}")

        model_rows.append(line)

    grand_total = total_per_model * len(MODEL_NAMES)
    grand_pct = total_overall_battles / grand_total * 100 if grand_total > 0 else 0

    if elapsed > 60 and total_overall_battles > 0:
        rate = total_overall_battles / elapsed
        remaining = (grand_total - total_overall_battles) / rate if rate > 0 else 0
        eta_str = _fmt_elapsed(remaining)
    else:
        eta_str = "..."

    # Update header line (line 2)
    header = (f"  RESEARCH EXPERIMENT   │   Elapsed: {elapsed_str}   │"
              f"   Total: {grand_pct:.1f}% ({total_overall_battles:,}/{grand_total:,})   │   ETA: {eta_str}")
    _write_line(_HEADER_LINE, header)

    # Refresh column headers (line 4)
    col_header = (f"  {'Model':<20} {'':>4} {'Run':>5} {'Battles':>10} {'Done':>7}"
                  f" │ {'Roll%':>6} {'Avg%':>6} {'AvgRew':>7} {'Eps':>6} {'States':>9} {'Bat/s':>7} {'Next':>12}")
    _write_line(4, col_header)

    # Update each model row (lines 6-9)
    for i, row_text in enumerate(model_rows):
        _write_line(_DATA_START + i, row_text)

    # Runs summary line (line 11) — compact format for 30 runs
    if _PARALLEL_RUNS_MODE:
        run_parts = []
        for model_name in MODEL_NAMES:
            short = MODEL_LABELS[model_name].replace(" + ", "+").replace("Init", "")[:6]
            completed = 0
            active = 0
            for r_idx in range(RUNS_PER_MODEL):
                b, _ = _get_actual_progress(model_name, r_idx + 1)
                if b >= BATTLES_PER_RUN:
                    completed += 1
                elif b > 0:
                    active += 1
            pct = completed / RUNS_PER_MODEL if RUNS_PER_MODEL > 0 else 0
            bar = _mini_bar(pct, width=8)
            run_parts.append(f"{short} {completed:>2}/{RUNS_PER_MODEL} {bar}")
        runs_line = "  Runs: " + "  ".join(run_parts)
        _write_line(_RUNS_LINE, runs_line)
    else:
        # Sequential mode: show current run progress
        run_parts = []
        for model_name in MODEL_NAMES:
            short = MODEL_LABELS[model_name].replace(" + ", "+").replace("Init", "")[:6]
            completed = 0
            for r_idx in range(RUNS_PER_MODEL):
                b, _ = _get_actual_progress(model_name, r_idx + 1)
                if b >= BATTLES_PER_RUN:
                    completed += 1
            if completed > 0 or _get_run_count(model_name) > 0:
                run_parts.append(f"{short} {completed}/{RUNS_PER_MODEL}")
        if run_parts:
            _write_line(_RUNS_LINE, "  Runs: " + "  │  ".join(run_parts))
        else:
            _write_line(_RUNS_LINE, "")

    # Park cursor below the table
    _move_to(_TOTAL_LINES + 1)
    sys.stdout.flush()

    all_done = all(
        (processes.get(m) and processes[m].poll() is not None)
        for m in MODEL_NAMES
    )
    return all_done


def run_all_parallel():
    """
    Run all 4 models in parallel, each with its own Showdown server.
    Dashboard updates in-place (no re-drawing).
    """
    total_start = time.time()

    # Start 4 separate Showdown servers (one per model)
    print("Starting 4 Showdown servers (ports 9000-9003)...")
    showdown_servers = _start_showdown_servers()
    print("Servers ready. Launching models...")

    processes = {}
    log_file_handles = {}

    for model_name in MODEL_NAMES:
        port = MODEL_PORTS[model_name]
        log_dir = os.path.join(EXPERIMENT_DIR, model_name, "stdout_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "parallel_run.log")

        cmd = [
            sys.executable, __file__,
            "--model", model_name,
            "--port", str(port),
        ]

        log_f = open(log_path, "w")
        proc = subprocess.Popen(
            cmd, cwd=EXPERIMENT_DIR,
            stdout=log_f, stderr=subprocess.STDOUT,
        )
        processes[model_name] = proc
        log_file_handles[model_name] = log_f

    # Draw static frame once, then update data rows in-place
    _draw_table_frame()

    try:
        while True:
            all_done = _update_dashboard(processes, total_start)
            if all_done:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        _move_to(_TOTAL_LINES + 2)
        print("\nInterrupted! Killing all subprocesses...")
        for proc in processes.values():
            if proc.poll() is None:
                proc.kill()
        for f in log_file_handles.values():
            f.close()
        _stop_showdown_servers(showdown_servers)
        sys.exit(1)

    # Cleanup
    for f in log_file_handles.values():
        f.close()
    _stop_showdown_servers(showdown_servers)

    _move_to(_TOTAL_LINES + 2)
    total_elapsed = time.time() - total_start
    print(f"\nALL EXPERIMENTS COMPLETE — {total_elapsed/60:.1f} minutes total")
    _generate_plots()


def run_all_parallel_runs(max_parallel=0):
    """
    Run ALL experiments (4 models × N runs) in parallel.
    Each (model, run) gets its own Showdown server and subprocess.
    Dashboard shows per-model progress + compact run completion line.

    Args:
        max_parallel: Max concurrent experiments. 0 = all at once (default).
    """
    global _PARALLEL_RUNS_MODE
    _PARALLEL_RUNS_MODE = True
    total_start = time.time()
    total_experiments = len(MODEL_NAMES) * RUNS_PER_MODEL

    # Build the full queue of (model_idx, model_name, run_idx)
    # Interleave models: run 1 for all models, then run 2, etc.
    # This ensures at least one run per model is always active.
    experiment_queue = []
    for r_idx in range(RUNS_PER_MODEL):
        for m_idx, model_name in enumerate(MODEL_NAMES):
            experiment_queue.append((m_idx, model_name, r_idx))

    # If max_parallel is 0 or >= total, run everything at once
    if max_parallel <= 0 or max_parallel >= total_experiments:
        max_parallel = total_experiments

    batch_size = min(max_parallel, total_experiments)
    print(f"Running {total_experiments} experiments, max {batch_size} at a time")

    # Start Showdown servers for the ports we'll need
    ports_needed = set()
    for m_idx, model_name, r_idx in experiment_queue:
        ports_needed.add(_get_parallel_run_port(m_idx, r_idx))

    print(f"Starting {len(ports_needed)} Showdown servers (ports {min(ports_needed)}-{max(ports_needed)})...")
    showdown_cmd = os.path.join(SHOWDOWN_DIR, "pokemon-showdown")
    servers = []
    ports_started = []

    for m_idx, model_name in enumerate(MODEL_NAMES):
        for r_idx in range(RUNS_PER_MODEL):
            port = _get_parallel_run_port(m_idx, r_idx)
            if _is_port_in_use(port):
                servers.append((None, None, port))
                continue
            log_dir = os.path.join(EXPERIMENT_DIR, model_name, "stdout_logs")
            os.makedirs(log_dir, exist_ok=True)
            log_f = open(os.path.join(log_dir, f"showdown_{port}.log"), "w")
            proc = subprocess.Popen(
                ["node", showdown_cmd, "start", "--no-security", "--skip-build", str(port)],
                cwd=SHOWDOWN_DIR,
                stdout=log_f, stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
            servers.append((proc, log_f, port))
            ports_started.append(port)

    if ports_started:
        print(f"  Started {len(ports_started)} servers (ports {min(ports_started)}-{max(ports_started)})")
        print("  Waiting for servers to initialize...")
        time.sleep(8)
    print("Servers ready. Launching experiments...")

    # Track all launched processes
    processes = {}       # keyed by model_name (first active run's process, for dashboard)
    all_procs = []       # all (model_name, run_id, proc, log_f) for cleanup
    log_file_handles = []
    active_procs = []    # currently running (model_name, run_id, proc, log_f)
    queue_idx = 0        # next experiment to launch from experiment_queue

    def _launch_experiment(m_idx, model_name, r_idx):
        """Launch a single experiment subprocess."""
        run_id = r_idx + 1
        port = _get_parallel_run_port(m_idx, r_idx)

        log_dir = os.path.join(EXPERIMENT_DIR, model_name, "stdout_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"parallel_run_{run_id}.log")

        cmd = [
            sys.executable, __file__,
            "--model", model_name,
            "--run", str(run_id),
            "--port", str(port),
        ]

        log_f = open(log_path, "w")
        proc = subprocess.Popen(
            cmd, cwd=EXPERIMENT_DIR,
            stdout=log_f, stderr=subprocess.STDOUT,
        )
        entry = (model_name, run_id, proc, log_f)
        all_procs.append(entry)
        active_procs.append(entry)
        log_file_handles.append(log_f)

        # Dashboard tracks first active run's process per model
        if model_name not in processes or processes[model_name].poll() is not None:
            processes[model_name] = proc

    # Launch initial batch
    while queue_idx < len(experiment_queue) and len(active_procs) < batch_size:
        m_idx, model_name, r_idx = experiment_queue[queue_idx]
        _launch_experiment(m_idx, model_name, r_idx)
        queue_idx += 1

    # Draw dashboard and poll
    _draw_table_frame()

    try:
        while True:
            _update_dashboard(processes, total_start)

            # Check for completed processes, launch new ones
            still_active = []
            for entry in active_procs:
                model_name, run_id, proc, log_f = entry
                if proc.poll() is not None:
                    # Finished — launch next from queue if available
                    while queue_idx < len(experiment_queue) and len(still_active) < batch_size:
                        m_idx, mn, r_idx = experiment_queue[queue_idx]
                        _launch_experiment(m_idx, mn, r_idx)
                        still_active.append(all_procs[-1])
                        queue_idx += 1
                        break
                else:
                    still_active.append(entry)
            active_procs[:] = still_active

            # All done?
            if not active_procs and queue_idx >= len(experiment_queue):
                break
            time.sleep(5)
    except KeyboardInterrupt:
        _move_to(_TOTAL_LINES + 2)
        print(f"\nInterrupted! Killing all {len(all_procs)} subprocesses...")
        for _, _, proc, _ in all_procs:
            if proc.poll() is None:
                proc.kill()
        for f in log_file_handles:
            f.close()
        _stop_showdown_servers(servers)
        sys.exit(1)

    # Cleanup
    for f in log_file_handles:
        f.close()
    _stop_showdown_servers(servers)

    _move_to(_TOTAL_LINES + 2)
    total_elapsed = time.time() - total_start
    print(f"\nALL {total_experiments} EXPERIMENTS COMPLETE — {total_elapsed/60:.1f} minutes total")
    _generate_plots()


def _reset_all_logs():
    """Delete all logs, models, and status files for every model. Requires confirmation."""
    # Show what will be deleted
    print("This will DELETE all data for every model:")
    for model_name in MODEL_NAMES:
        model_dir = os.path.join(EXPERIMENT_DIR, model_name)
        for subdir in ["logs", "models", "stdout_logs"]:
            path = os.path.join(model_dir, subdir)
            if os.path.isdir(path):
                count = len(os.listdir(path))
                print(f"  {model_name}/{subdir}/ ({count} files)")
    plots_dir = os.path.join(EXPERIMENT_DIR, "comparison_plots")
    if os.path.isdir(plots_dir):
        print(f"  comparison_plots/")

    confirm = input("\nType RESET to confirm: ")
    if confirm.strip() != "RESET":
        print("Aborted.")
        sys.exit(1)

    for model_name in MODEL_NAMES:
        model_dir = os.path.join(EXPERIMENT_DIR, model_name)
        for subdir in ["logs", "models", "stdout_logs"]:
            path = os.path.join(model_dir, subdir)
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"  Deleted {model_name}/{subdir}/")
        # Clean up all status files (legacy + per-run)
        for status in glob.glob(os.path.join(model_dir, "live_status*.json")):
            os.remove(status)

    if os.path.isdir(plots_dir):
        shutil.rmtree(plots_dir)
        print("  Deleted comparison_plots/")

    print("All logs, models, and plots reset.")


def _generate_plots():
    print("\nGenerating comparison plots...")
    try:
        from shared.compare import generate_all_comparisons
        generate_all_comparisons(EXPERIMENT_DIR)
        print("Plots saved to comparison_plots/")
    except Exception as e:
        print(f"Plot generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Run only this model (all 5 runs sequentially)")
    parser.add_argument("--run", type=int, default=None,
                        help="Run only this run ID (1-5), requires --model")
    parser.add_argument("--port", type=int, default=None,
                        help="Showdown server port (default: auto per model)")
    parser.add_argument("--sequential", action="store_true",
                        help="Run all models sequentially instead of parallel")
    parser.add_argument("--parallel-runs", action="store_true",
                        help="Run ALL experiments (4 models × N runs) fully in parallel with 20 servers")
    parser.add_argument("--max-parallel", type=int, default=0,
                        help="Max concurrent experiments in --parallel-runs mode (0 = all at once)")
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip training, just generate plots")
    parser.add_argument("--plot", nargs="*", default=None,
                        help="Plot current progress. No args=all models averaged. "
                             "Specify models/runs like: model_1 model_3:2 model_2:1,3")
    parser.add_argument("--reset", action="store_true",
                        help="Delete all logs, models, and plots (asks for confirmation)")
    parser.add_argument("--tests", action="store_true",
                        help="Run comprehensive test suite")
    parser.add_argument("--skip-integration", action="store_true",
                        help="Skip integration tests (no Showdown needed)")
    args = parser.parse_args()

    if args.tests:
        from shared.tests import run_all_tests
        success = run_all_tests(skip_integration=args.skip_integration)
        sys.exit(0 if success else 1)

    if args.reset:
        _reset_all_logs()
        sys.exit(0)

    if args.plot is not None:
        from shared.compare import plot_progress
        plot_progress(EXPERIMENT_DIR, args.plot)
        sys.exit(0)

    if args.plots_only:
        from shared.compare import generate_all_comparisons
        generate_all_comparisons(EXPERIMENT_DIR)
        sys.exit(0)

    port = args.port or (MODEL_PORTS.get(args.model, 9000) if args.model else 9000)

    if args.model and args.run:
        seed = RANDOM_SEEDS[args.run - 1]
        _run_one_experiment(args.model, args.run, seed, port=port)
    elif args.model:
        run_single_model_all_runs(args.model, port=port)
    elif args.parallel_runs:
        run_all_parallel_runs(max_parallel=args.max_parallel)
    elif args.sequential:
        for model_name in MODEL_NAMES:
            run_single_model_all_runs(model_name)
        _generate_plots()
    else:
        # Default: parallel models, sequential runs (starts 4 servers)
        run_all_parallel()
