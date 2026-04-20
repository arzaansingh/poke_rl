"""
V5 Initialization Study Orchestrator
=====================================
Runs all 8 models across every HP combination × multiple runs.
Each run uses a different randomly-sampled 9-pokemon pool, but
all models in a run share the same pool.

Directory structure:
  grid_results/
    hp_001/                              ← one HP combo
      params.json                        ← {alpha, gamma, lam}
      run_1/                             ← run 1 (pool seeded by run_id=1)
        pool.json                        ← the 9 pokemon for this run
        model_1_flat_zero/logs/run_1.csv
        model_2_flat_smart/logs/run_1.csv
        ...
      run_2/
        ...
    summary.csv                          ← ranked results

Usage:
  python run_grid.py                            # Run full grid (8 parallel default)
  python run_grid.py --max-parallel 16          # More parallelism
  python run_grid.py --combo hp_001             # Run single combo (all runs)
  python run_grid.py --combo hp_001 --run 3     # Single combo, single run
  python run_grid.py --summary                  # Print ranked results
  python run_grid.py --list-grid                # Show all HP combos
  python run_grid.py --tests --skip-integration
"""

import subprocess
import sys
import os
import time
import glob
import csv
import json
import shutil
import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared.config import (
    MODEL_NAMES, MODEL_LABELS, BATTLES_PER_RUN, BATTLES_PER_LOG,
    SHOWDOWN_DIR, RUNS_PER_COMBO, MODEL_EPSILON_MODE,
    build_grid, resolve_epsilon, get_pool_for_run,
)

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
GRID_RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "grid_results")

TRAIN_SCRIPTS = {
    "model_1_flat_zero": os.path.join(EXPERIMENT_DIR, "model_1_flat_zero", "train.py"),
    "model_2_flat_smart": os.path.join(EXPERIMENT_DIR, "model_2_flat_smart", "train.py"),
    "model_3_hier_zero": os.path.join(EXPERIMENT_DIR, "model_3_hier_zero", "train.py"),
    "model_4_hier_smart": os.path.join(EXPERIMENT_DIR, "model_4_hier_smart", "train.py"),
    "model_5_flat_zero_fixed_eps": os.path.join(EXPERIMENT_DIR, "model_5_flat_zero_fixed_eps", "train.py"),
    "model_6_flat_smart_fixed_eps": os.path.join(EXPERIMENT_DIR, "model_6_flat_smart_fixed_eps", "train.py"),
    "model_7_hier_zero_fixed_eps": os.path.join(EXPERIMENT_DIR, "model_7_hier_zero_fixed_eps", "train.py"),
    "model_8_hier_smart_fixed_eps": os.path.join(EXPERIMENT_DIR, "model_8_hier_smart_fixed_eps", "train.py"),
}

BATCH_SIZE = 5_000  # Battles per subprocess invocation


# ── Port allocation ──
# Each concurrent experiment gets a unique port.
# We assign: 9000 + flat_index where flat_index is the experiment's position in the queue.
# Since we limit concurrency via max_parallel, we only need max_parallel ports.
def _get_port(slot_idx):
    """Port for a given concurrency slot (0-based)."""
    return 9000 + slot_idx


def _is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def _order_combos(combos, reverse_order=False):
    """Return combos in the requested traversal order."""
    return list(reversed(combos)) if reverse_order else list(combos)


def _model_code(model_name):
    """Compact dashboard code for a model name."""
    label = MODEL_LABELS[model_name]
    arch = "F" if "Flat" in label else "H"
    init = "S" if "Smart" in label else "Z"
    schedule = "Fx" if MODEL_EPSILON_MODE[model_name] == "fixed" else "Dc"
    return f"{arch}{init}-{schedule}"


# ── Progress tracking ──

def _get_actual_progress(run_dir, model_name):
    """Read actual battle count from CSV log for a run+model."""
    log_file = os.path.join(run_dir, model_name, "logs", "run_1.csv")
    if not os.path.exists(log_file):
        return 0, 0
    try:
        with open(log_file, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
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


def _get_final_rolling_wr(run_dir, model_name):
    """Get the final rolling win rate from the last CSV row."""
    log_file = os.path.join(run_dir, model_name, "logs", "run_1.csv")
    if not os.path.exists(log_file):
        return None
    try:
        with open(log_file, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            last_row = None
            for row in reader:
                if row:
                    last_row = row
            if last_row:
                return float(last_row[1])
    except Exception:
        pass
    return None


def _get_best_rolling_wr(run_dir, model_name):
    """Get the best rolling win rate from the entire CSV."""
    log_file = os.path.join(run_dir, model_name, "logs", "run_1.csv")
    if not os.path.exists(log_file):
        return None
    try:
        best = 0.0
        with open(log_file, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    val = float(row[1])
                    if val > best:
                        best = val
        return best if best > 0 else None
    except Exception:
        pass
    return None


def _get_avg_rolling_wr(combo_dir, model_name):
    """Average the final rolling WR across all runs for a combo+model."""
    wrs = []
    for r in range(1, RUNS_PER_COMBO + 1):
        run_dir = os.path.join(combo_dir, f"run_{r}")
        wr = _get_final_rolling_wr(run_dir, model_name)
        if wr is not None:
            wrs.append(wr)
    return sum(wrs) / len(wrs) if wrs else None


def _get_avg_best_rolling_wr(combo_dir, model_name):
    """Average the best rolling WR across all runs for a combo+model."""
    wrs = []
    for r in range(1, RUNS_PER_COMBO + 1):
        run_dir = os.path.join(combo_dir, f"run_{r}")
        wr = _get_best_rolling_wr(run_dir, model_name)
        if wr is not None:
            wrs.append(wr)
    return sum(wrs) / len(wrs) if wrs else None


# ── Single experiment runner ──

def _run_one_experiment(combo, run_id, model_name, port):
    """Run a single model for a single HP combo + run, batched with resume support."""
    combo_id = combo["combo_id"]
    combo_dir = os.path.join(GRID_RESULTS_DIR, combo_id)
    run_dir = os.path.join(combo_dir, f"run_{run_id}")
    output_dir = os.path.join(run_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save combo params
    params_file = os.path.join(combo_dir, "params.json")
    if not os.path.exists(params_file):
        os.makedirs(combo_dir, exist_ok=True)
        with open(params_file, 'w') as f:
            json.dump(combo, f, indent=2)

    # Save pool for this run
    pool = get_pool_for_run(run_id)
    pool_file = os.path.join(run_dir, "pool.json")
    if not os.path.exists(pool_file):
        with open(pool_file, 'w') as f:
            json.dump(pool, f, indent=2)

    train_script = TRAIN_SCRIPTS[model_name]
    pool_str = ",".join(pool)
    epsilon_mode = MODEL_EPSILON_MODE[model_name]

    battles_done, wins = _get_actual_progress(run_dir, model_name)

    while battles_done < BATTLES_PER_RUN:
        remaining = BATTLES_PER_RUN - battles_done
        batch = min(BATCH_SIZE, remaining)
        epsilon = resolve_epsilon(battles_done, mode=epsilon_mode)

        cmd = [
            sys.executable, train_script,
            "--run_id", "1",
            "--seed", str(run_id),
            "--epsilon", str(epsilon),
            "--batch_size", str(batch),
            "--historic_battles", str(battles_done),
            "--historic_wins", str(wins),
            "--port", str(port),
            "--alpha", str(combo["alpha"]),
            "--gamma_val", str(combo["gamma"]),
            "--lam", str(combo["lam"]),
            "--output_dir", output_dir,
            "--pool", pool_str,
            "--epsilon_mode", epsilon_mode,
        ]
        if epsilon_mode == "fixed":
            cmd.extend(["--fixed_epsilon", str(epsilon)])

        result = subprocess.run(cmd, cwd=EXPERIMENT_DIR)
        if result.returncode != 0:
            print(f"WARNING: {combo_id}/run_{run_id}/{model_name} batch failed (exit {result.returncode})")

        battles_done, wins = _get_actual_progress(run_dir, model_name)


# ── Showdown servers ──

def _start_showdown_server(port, log_dir):
    if _is_port_in_use(port):
        return (None, None, port)
    showdown_cmd = os.path.join(SHOWDOWN_DIR, "pokemon-showdown")
    os.makedirs(log_dir, exist_ok=True)
    log_f = open(os.path.join(log_dir, f"showdown_{port}.log"), "w")
    proc = subprocess.Popen(
        ["node", showdown_cmd, "start", "--no-security", "--skip-build", str(port)],
        cwd=SHOWDOWN_DIR,
        stdout=log_f, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return (proc, log_f, port)


def _stop_servers(servers):
    for proc, log_f, port in servers:
        if proc is None:
            continue
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        if log_f:
            log_f.close()


# ── Dashboard ──

def _fmt_elapsed(seconds):
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h{m:02d}m"

def _mini_bar(fraction, width=8):
    filled = int(fraction * width)
    return f"[{'█' * filled}{'░' * (width - filled)}]"

_W = 120
_DATA_START = 6
_TOTAL_LINES = 0
_MODEL_COL_W = 6
_LAST_DASHBOARD_SIGNATURE = None


def _get_terminal_size():
    """Current terminal size with a safe fallback for nonstandard PTYs."""
    size = shutil.get_terminal_size(fallback=(140, 24))
    width = max(40, size.columns - 1)  # keep one spare column to prevent wrap
    height = max(8, size.lines)
    return width, height


def _compute_dashboard_layout(combos):
    """Choose a dashboard layout that fits the current terminal."""
    width, height = _get_terminal_size()

    if width >= 130:
        mode = "full"
    elif width >= 90:
        mode = "compact"
    else:
        mode = "mini"

    if mode == "mini":
        data_start = 4
        overhead = 7
    else:
        data_start = 6
        overhead = 9

    max_data_rows = max(1, height - overhead)
    if len(combos) > max_data_rows:
        visible_combo_count = max(0, max_data_rows - 1)
        hidden_count = len(combos) - visible_combo_count
    else:
        visible_combo_count = len(combos)
        hidden_count = 0

    data_rows = visible_combo_count + (1 if hidden_count else 0)
    divider_line = data_start + data_rows
    summary_line = divider_line + 1
    bottom_line = summary_line + 1 if mode != "mini" else None
    ctrl_line = (bottom_line + 1) if bottom_line is not None else (summary_line + 1)

    return {
        "width": width,
        "height": height,
        "mode": mode,
        "data_start": data_start,
        "visible_combo_count": visible_combo_count,
        "hidden_count": hidden_count,
        "data_rows": data_rows,
        "divider_line": divider_line,
        "summary_line": summary_line,
        "bottom_line": bottom_line,
        "ctrl_line": ctrl_line,
        "cursor_line": ctrl_line + 2,
        "signature": (width, height, mode, visible_combo_count, hidden_count),
    }


def _apply_dashboard_layout(layout):
    global _W, _DATA_START, _TOTAL_LINES
    _W = layout["width"]
    _DATA_START = layout["data_start"]
    _TOTAL_LINES = layout["divider_line"]


def _fit_line(text):
    """Clip and pad a line so cursor-based redraws never wrap."""
    return text[:_W].ljust(_W)


def _fmt_count_short(value):
    """Compact battle counts for the live dashboard."""
    value = float(value)
    abs_v = abs(value)
    if abs_v >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_v >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{int(value)}"

def _move_to(line):
    """Move cursor to the start of a 1-indexed line."""
    sys.stdout.write(f"\033[{line};1H")

def _write_line(line_num, text):
    """Overwrite a specific line with text, padding to full width."""
    _move_to(line_num)
    sys.stdout.write(_fit_line(text))

def _render_dashboard_header(layout, total_combos, elapsed=None, initializing=False):
    status = "Initializing..." if initializing else f"Elapsed: {elapsed}"
    if layout["mode"] == "full":
        return (
            f"  V5 INITIALIZATION STUDY │ {total_combos} combos × {RUNS_PER_COMBO} runs × "
            f"{len(MODEL_NAMES)} models × {BATTLES_PER_RUN:,} battles │ {status}"
        )
    if layout["mode"] == "compact":
        return (
            f"  V5 INIT STUDY │ {total_combos}c × {RUNS_PER_COMBO}r × "
            f"{len(MODEL_NAMES)}m × {BATTLES_PER_RUN:,} bat │ {status}"
        )
    return f"  V5 INIT │ {total_combos} combos │ {status}"


def _render_column_header(layout):
    if layout["mode"] == "full":
        model_header = " ".join(f"{_model_code(name):>{_MODEL_COL_W}}" for name in MODEL_NAMES)
        return (
            f"  {'Combo':<8} {'A':>4} {'G':>5} {'L':>3} │ "
            f"{model_header} │ {'Avg':>5} {'Run':>5} {'Battles':>11} {'Prog':<14}"
        )
    if layout["mode"] == "compact":
        return f"  {'Combo':<8} {'A':>4} {'G':>5} {'L':>3} │ {'Avg':>5} {'Run':>5} {'Battles':>11} {'Prog':<8}"
    return None


def _render_combo_line(layout, combo, row):
    combo_id = combo["combo_id"]
    avg_str = row["avg_str"]
    runs_done = row["runs_done"]
    battles_str = row["battles_str"]
    prog = row["prog"]

    if layout["mode"] == "full":
        model_str = " ".join(f"{wr:>{_MODEL_COL_W}}" for wr in row["wr_strs"])
        return (
            f"  {combo_id:<8} {combo['alpha']:>4.2f} {combo['gamma']:>5.3f} {combo['lam']:>3.1f}"
            f" │ {model_str} │ {avg_str:>5} {runs_done:>2}/{RUNS_PER_COMBO:<2} {battles_str:>11} {prog:<14}"
        )
    if layout["mode"] == "compact":
        return (
            f"  {combo_id:<8} {combo['alpha']:>4.2f} {combo['gamma']:>5.3f} {combo['lam']:>3.1f}"
            f" │ {avg_str:>5} {runs_done:>2}/{RUNS_PER_COMBO:<2} {battles_str:>11} {row['status']:<8}"
        )
    return f"  {combo_id:<8} │ {battles_str:>11} │ {row['status']:<4}"


def _render_hidden_line(hidden_count):
    return f"  ... {hidden_count} more combos hidden; enlarge the terminal to view all rows"


def _draw_table_frame(combos, layout=None):
    """Draw the static table frame once. Called at dashboard init."""
    global _LAST_DASHBOARD_SIGNATURE
    if layout is None:
        layout = _compute_dashboard_layout(combos)
    _apply_dashboard_layout(layout)
    _LAST_DASHBOARD_SIGNATURE = layout["signature"]
    sys.stdout.write("\033[2J\033[H")  # Clear screen and reset cursor
    print("═" * _W)
    print(_fit_line(_render_dashboard_header(layout, len(combos), initializing=True)))
    if layout["mode"] != "mini":
        print("═" * _W)
        print(_fit_line(_render_column_header(layout)))
        print(_fit_line(f"  {'─' * (_W - 4)}"))
    else:
        print(_fit_line(f"  {'─' * (_W - 4)}"))
    for _ in range(layout["data_rows"]):
        print(_fit_line(""))  # Padded empty rows for data (Starts at Line 6)
    print(_fit_line(f"  {'─' * (_W - 4)}"))
    print(_fit_line(""))  # Summary Line
    if layout["mode"] != "mini":
        print("═" * _W)
    print(_fit_line("  Ctrl+C to stop."))
    sys.stdout.flush()

def _draw_grid_dashboard(combos, start_time, initial_battles=0):
    """Update current state of all combos in-place."""
    global _LAST_DASHBOARD_SIGNATURE
    layout = _compute_dashboard_layout(combos)
    _apply_dashboard_layout(layout)
    if layout["signature"] != _LAST_DASHBOARD_SIGNATURE:
        _draw_table_frame(combos, layout=layout)

    elapsed_s = time.time() - start_time
    elapsed = _fmt_elapsed(elapsed_s)
    total_combos = len(combos)

    completed_combos = 0
    total_battles = 0
    grand_total = BATTLES_PER_RUN * len(MODEL_NAMES) * RUNS_PER_COMBO * total_combos

    # Update Header
    _write_line(2, _render_dashboard_header(layout, total_combos, elapsed=elapsed))

    combo_rows = []
    for combo in combos:
        combo_id = combo["combo_id"]
        combo_dir = os.path.join(GRID_RESULTS_DIR, combo_id)

        model_wrs = []
        combo_battles = 0
        runs_done = 0

        for r in range(1, RUNS_PER_COMBO + 1):
            run_dir = os.path.join(combo_dir, f"run_{r}")
            run_complete = True
            for model_name in MODEL_NAMES:
                b, _ = _get_actual_progress(run_dir, model_name)
                combo_battles += b
                total_battles += b
                if b < BATTLES_PER_RUN:
                    run_complete = False
            if run_complete:
                runs_done += 1

        for model_name in MODEL_NAMES:
            wr = _get_avg_rolling_wr(combo_dir, model_name)
            model_wrs.append(wr)

        total_for_combo = BATTLES_PER_RUN * len(MODEL_NAMES) * RUNS_PER_COMBO
        all_done = runs_done == RUNS_PER_COMBO
        any_started = combo_battles > 0

        # Create status and progress bar
        if all_done:
            status = "DONE"
            completed_combos += 1
            bar = _mini_bar(1.0)
        elif any_started:
            pct = combo_battles / total_for_combo
            status = f"{pct:.0%}"
            bar = _mini_bar(pct, width=8 if layout["mode"] == "full" else 6)
        else:
            status = "WAIT"
            bar = _mini_bar(0.0, width=8 if layout["mode"] == "full" else 6)

        wr_strs = []
        for wr in model_wrs:
            wr_strs.append(f"{wr:.1%}" if wr is not None else "—")

        valid_wrs = [w for w in model_wrs if w is not None]
        avg_wr = sum(valid_wrs) / len(valid_wrs) if valid_wrs else None
        avg_str = f"{avg_wr:.1%}" if avg_wr is not None else "—"

        battles_str = f"{_fmt_count_short(combo_battles)}/{_fmt_count_short(total_for_combo)}"
        prog = f"{bar} {status}" if layout["mode"] == "full" else status

        combo_rows.append({
            "avg_str": avg_str,
            "runs_done": runs_done,
            "battles_str": battles_str,
            "status": status,
            "prog": prog,
            "wr_strs": wr_strs,
        })

    visible_combos = combos[:layout["visible_combo_count"]]
    for i, combo in enumerate(visible_combos):
        line = _render_combo_line(layout, combo, combo_rows[i])
        _write_line(layout["data_start"] + i, line)
    if layout["hidden_count"]:
        _write_line(layout["data_start"] + layout["visible_combo_count"], _render_hidden_line(layout["hidden_count"]))

    # Update Summary
    grand_pct = total_battles / grand_total * 100 if grand_total > 0 else 0
    
    # Calculate overall battles per second for THIS SESSION only
    session_battles = max(0, total_battles - initial_battles)
    rate = session_battles / elapsed_s if elapsed_s > 0 else 0
    speed_str = f"{rate:,.0f} bat/s"

    if elapsed_s > 60 and total_battles > 0:
        # ETA is still based on the total remaining / current session rate
        remaining = (grand_total - total_battles) / rate if rate > 0 else 0
        eta_str = _fmt_elapsed(remaining)
    else:
        eta_str = "..."

    summary_line = f"  Combos: {completed_combos}/{total_combos} done   │   Total: {grand_pct:.1f}%   │   Speed: {speed_str}   │   ETA: {eta_str}"
    _write_line(layout["summary_line"], summary_line)

    # Park cursor safely below the table to prevent text overlapping
    _move_to(layout["cursor_line"])
    sys.stdout.flush()

# ── Main runners ──

def run_single_combo(combo_id, combos, run_filter=None):
    """Run all runs × models for a single combo (or one specific run)."""
    combo = next((c for c in combos if c["combo_id"] == combo_id), None)
    if not combo:
        print(f"Unknown combo: {combo_id}. Available: {[c['combo_id'] for c in combos]}")
        sys.exit(1)

    runs = [run_filter] if run_filter else list(range(1, RUNS_PER_COMBO + 1))
    print(f"Running {combo_id}: α={combo['alpha']} γ={combo['gamma']} λ={combo['lam']}")
    print(f"Runs: {runs}")

    # Start one server per concurrent model in this run.
    servers = []
    for i in range(len(MODEL_NAMES)):
        port = _get_port(i)
        log_dir = os.path.join(GRID_RESULTS_DIR, combo_id, "stdout_logs")
        servers.append(_start_showdown_server(port, log_dir))

    ports_started = [p for proc, _, p in servers if proc is not None]
    if ports_started:
        print(f"  Started servers on ports: {ports_started}")
        time.sleep(5)

    try:
        for run_id in runs:
            pool = get_pool_for_run(run_id)
            print(f"\n  Run {run_id} — pool: {pool}")
            for m_idx, model_name in enumerate(MODEL_NAMES):
                port = _get_port(m_idx)
                print(f"    {MODEL_LABELS[model_name]}...")
                _run_one_experiment(combo, run_id, model_name, port)
    finally:
        _stop_servers(servers)

    print(f"\n{combo_id} complete!")


def run_full_grid(max_parallel=8, reverse_order=False):
    """Run the entire grid search with parallel execution."""
    combos = _order_combos(build_grid(), reverse_order=reverse_order)
    os.makedirs(GRID_RESULTS_DIR, exist_ok=True)

    # Build queue: each item is (combo, run_id, model_name)
    # Interleave so we process combo1/run1/all_models, then combo1/run2/all_models, etc.
    experiment_queue = []
    for combo in combos:
        for run_id in range(1, RUNS_PER_COMBO + 1):
            for model_name in MODEL_NAMES:
                experiment_queue.append((combo, run_id, model_name))

    total_experiments = len(experiment_queue)

    # Skip already-completed experiments
    pending_queue = []
    initial_total_battles = 0
    for combo, run_id, model_name in experiment_queue:
        combo_dir = os.path.join(GRID_RESULTS_DIR, combo["combo_id"])
        run_dir = os.path.join(combo_dir, f"run_{run_id}")
        b, _ = _get_actual_progress(run_dir, model_name)
        
        initial_total_battles += b  # Track baseline for accurate speed
        
        if b < BATTLES_PER_RUN:
            pending_queue.append((combo, run_id, model_name))

    if not pending_queue:
        print("All experiments already complete!")
        _print_summary(combos)
        return

    print(f"V5 Initialization Study: {len(combos)} combos × {RUNS_PER_COMBO} runs × {len(MODEL_NAMES)} models = {total_experiments} experiments")
    print(f"Pending: {len(pending_queue)} ({total_experiments - len(pending_queue)} already done)")
    print(f"Max parallel: {max_parallel}")
    print(f"Order: {'reverse' if reverse_order else 'forward'}")

    # Start servers for max_parallel slots
    print(f"Starting {max_parallel} Showdown servers...")
    servers = []
    log_dir = os.path.join(GRID_RESULTS_DIR, "stdout_logs")
    for i in range(max_parallel):
        port = _get_port(i)
        servers.append(_start_showdown_server(port, log_dir))

    ports_started = [p for proc, _, p in servers if proc is not None]
    if ports_started:
        print(f"  Started {len(ports_started)} servers (ports {min(ports_started)}-{max(ports_started)})")
        time.sleep(8)
    print("Servers ready. Launching experiments...")
    
    # ADD THIS LINE HERE:
    _draw_table_frame(combos)

    total_start = time.time()
    active = []  # (combo, run_id, model_name, slot_idx, proc, log_f)
    # ... (rest of the function remains the same)

    all_log_handles = []
    queue_idx = 0
    free_slots = list(range(max_parallel))

    def _launch_next():
        nonlocal queue_idx
        if queue_idx >= len(pending_queue) or not free_slots:
            return False
        combo, run_id, model_name = pending_queue[queue_idx]
        queue_idx += 1
        slot = free_slots.pop(0)
        port = _get_port(slot)

        combo_id = combo["combo_id"]
        run_dir = os.path.join(GRID_RESULTS_DIR, combo_id, f"run_{run_id}")
        output_dir = os.path.join(run_dir, model_name)
        os.makedirs(output_dir, exist_ok=True)

        # Save params + pool
        params_file = os.path.join(GRID_RESULTS_DIR, combo_id, "params.json")
        if not os.path.exists(params_file):
            os.makedirs(os.path.dirname(params_file), exist_ok=True)
            with open(params_file, 'w') as f:
                json.dump(combo, f, indent=2)
        pool = get_pool_for_run(run_id)
        pool_file = os.path.join(run_dir, "pool.json")
        if not os.path.exists(pool_file):
            os.makedirs(run_dir, exist_ok=True)
            with open(pool_file, 'w') as f:
                json.dump(pool, f, indent=2)

        # Launch subprocess
        exp_log_dir = os.path.join(GRID_RESULTS_DIR, combo_id, "stdout_logs")
        os.makedirs(exp_log_dir, exist_ok=True)
        log_path = os.path.join(exp_log_dir, f"run_{run_id}_{model_name}.log")

        cmd = [
            sys.executable, __file__,
            "--run-experiment",
            "--combo-json", json.dumps(combo),
            "--exp-run-id", str(run_id),
            "--model", model_name,
            "--port", str(port),
        ]

        log_f = open(log_path, "w")
        proc = subprocess.Popen(
            cmd, cwd=EXPERIMENT_DIR,
            stdout=log_f, stderr=subprocess.STDOUT,
        )
        entry = (combo, run_id, model_name, slot, proc, log_f)
        active.append(entry)
        all_log_handles.append(log_f)
        return True

    # Launch initial batch
    while free_slots and queue_idx < len(pending_queue):
        _launch_next()

    try:
        last_dashboard = 0
        while True:
            now = time.time()
            if now - last_dashboard >= 1:
                # Add initial_total_battles here!
                _draw_grid_dashboard(combos, total_start, initial_total_battles)
                last_dashboard = now

            still_active = []
            for entry in active:
                combo, run_id, model_name, slot, proc, log_f = entry
                if proc.poll() is not None:
                    free_slots.append(slot)
                    # Launch replacements
                    while free_slots and queue_idx < len(pending_queue):
                        _launch_next()
                else:
                    still_active.append(entry)
            active[:] = still_active

            if not active and queue_idx >= len(pending_queue):
                break
            time.sleep(1)

    except KeyboardInterrupt:
        print(f"\nInterrupted! Killing {len(active)} processes...")
        for _, _, _, _, proc, _ in active:
            if proc.poll() is None:
                proc.kill()
        for f in all_log_handles:
            f.close()
        _stop_servers(servers)
        sys.exit(1)

    for f in all_log_handles:
        f.close()
    _stop_servers(servers)

    _draw_grid_dashboard(combos, total_start)
    total_elapsed = time.time() - total_start
    print(f"\nV5 STUDY COMPLETE — {total_elapsed/60:.1f} minutes total")
    _print_summary(combos)


# ── Summary ──

def _print_summary(combos=None):
    """Print a ranked summary of all combos, averaged across runs."""
    if combos is None:
        combos = build_grid()

    results = []
    for combo in combos:
        combo_dir = os.path.join(GRID_RESULTS_DIR, combo["combo_id"])
        if not os.path.isdir(combo_dir):
            continue

        row = {
            "combo_id": combo["combo_id"],
            "alpha": combo["alpha"],
            "gamma": combo["gamma"],
            "lam": combo["lam"],
        }

        wrs = []
        best_wrs = []
        completed_runs = 0
        for r in range(1, RUNS_PER_COMBO + 1):
            run_dir = os.path.join(combo_dir, f"run_{r}")
            all_done = all(
                _get_actual_progress(run_dir, m)[0] >= BATTLES_PER_RUN
                for m in MODEL_NAMES
            )
            if all_done:
                completed_runs += 1

        for model_name in MODEL_NAMES:
            short = MODEL_LABELS[model_name].replace(" + ", "_").replace(" ", "")
            avg_wr = _get_avg_rolling_wr(combo_dir, model_name)
            avg_best = _get_avg_best_rolling_wr(combo_dir, model_name)
            row[f"final_{short}"] = avg_wr
            row[f"best_{short}"] = avg_best
            if avg_wr is not None:
                wrs.append(avg_wr)
            if avg_best is not None:
                best_wrs.append(avg_best)

        row["runs_done"] = completed_runs
        row["avg_final"] = sum(wrs) / len(wrs) if wrs else 0
        row["avg_best"] = sum(best_wrs) / len(best_wrs) if best_wrs else 0
        results.append(row)

    if not results:
        print("No results found. Run the grid search first.")
        return

    results.sort(key=lambda r: r["avg_best"], reverse=True)

    # Save CSV
    summary_file = os.path.join(GRID_RESULTS_DIR, "summary.csv")
    if results:
        keys = results[0].keys()
        with open(summary_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSummary saved to: {summary_file}")

    # Print top 15
    summary_width = max(120, 44 + 9 * len(MODEL_NAMES))
    model_header = " ".join(f"{_model_code(name):>8}" for name in MODEL_NAMES)
    print(f"\n{'═' * summary_width}")
    print(f"  TOP HYPERPARAMETER COMBOS (ranked by avg best rolling WR, averaged over {RUNS_PER_COMBO} runs)")
    print(f"{'═' * summary_width}")
    print(f"  {'Rank':<5} {'Combo':<8} {'Alpha':>6} {'Gamma':>6} {'Lam':>5} {'Runs':>5} │ {model_header} │ {'AvgBest':>8}")
    print(f"  {'─' * (summary_width - 4)}")

    for rank, row in enumerate(results[:15], 1):
        model_strs = []
        for model_name in MODEL_NAMES:
            short = MODEL_LABELS[model_name].replace(" + ", "_").replace(" ", "")
            best = row.get(f"best_{short}")
            model_strs.append(f"{best:.1%}" if best is not None else "  —  ")

        model_block = " ".join(f"{val:>8}" for val in model_strs)
        print(
            f"  {rank:<5} {row['combo_id']:<8} {row['alpha']:>6.2f} {row['gamma']:>6.3f} {row['lam']:>5.1f} "
            f"{row['runs_done']:>3}/{RUNS_PER_COMBO} │ {model_block} │ {row['avg_best']:>7.1%}"
        )

    print(f"  {'─' * (summary_width - 4)}")

    # Best per model
    print(f"\n  BEST COMBO PER MODEL:")
    for model_name in MODEL_NAMES:
        short = MODEL_LABELS[model_name].replace(" + ", "_").replace(" ", "")
        best_row = max(results, key=lambda r: r.get(f"best_{short}") or 0)
        best_val = best_row.get(f"best_{short}")
        if best_val:
            print(f"    {MODEL_LABELS[model_name]:<32} → {best_row['combo_id']} "
                  f"(α={best_row['alpha']}, γ={best_row['gamma']}, λ={best_row['lam']}) "
                  f"= {best_val:.1%}")
    print()


# ── Reset ──

def _reset_grid():
    plots_dir = os.path.join(EXPERIMENT_DIR, "plots")
    has_results = os.path.isdir(GRID_RESULTS_DIR)
    has_plots = os.path.isdir(plots_dir)
    if not has_results and not has_plots:
        print("Nothing to reset.")
        return
    count = 0
    if has_results:
        count = len([d for d in os.listdir(GRID_RESULTS_DIR)
                     if os.path.isdir(os.path.join(GRID_RESULTS_DIR, d)) and d.startswith("hp_")])
    print(f"This will DELETE all grid results ({count} combo directories) and all plots.")
    confirm = input("Type RESET to confirm: ")
    if confirm.strip() != "RESET":
        print("Aborted.")
        sys.exit(1)
    if has_results:
        shutil.rmtree(GRID_RESULTS_DIR)
        print("Grid results reset.")
    if has_plots:
        shutil.rmtree(plots_dir)
        print("Plots reset.")


# ── Entry point ──

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="V5 Initialization Study")
    parser.add_argument("--combo", type=str, default=None,
                        help="Run a single combo (e.g. hp_001)")
    parser.add_argument("--run", type=int, default=None,
                        help="Run a specific run ID (1-10), requires --combo")
    parser.add_argument("--max-parallel", type=int, default=8,
                        help="Max concurrent experiments (default: 8)")
    parser.add_argument("--summary", action="store_true",
                        help="Print results summary without running")
    parser.add_argument("--reset", action="store_true",
                        help="Delete all grid results")
    parser.add_argument("--list-grid", action="store_true",
                        help="List all HP combos without running")
    parser.add_argument("--reverse-order", action="store_true",
                        help="Traverse combos from the end of the grid (e.g. hp_027 -> hp_001)")
    parser.add_argument("--tests", action="store_true",
                        help="Run test suite")
    parser.add_argument("--skip-integration", action="store_true",
                        help="Skip integration tests")
    # Internal: subprocess for parallel execution
    parser.add_argument("--run-experiment", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--combo-json", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--exp-run-id", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--model", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=9000, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.tests:
        from shared.tests import run_all_tests
        success = run_all_tests(skip_integration=args.skip_integration)
        sys.exit(0 if success else 1)

    if args.reset:
        _reset_grid()
        sys.exit(0)

    combos = _order_combos(build_grid(), reverse_order=args.reverse_order)

    if args.list_grid:
        print(
            f"Grid: {len(combos)} combos × {RUNS_PER_COMBO} runs × {len(MODEL_NAMES)} models = "
            f"{len(combos) * RUNS_PER_COMBO * len(MODEL_NAMES)} experiments"
        )
        print(f"Order: {'reverse' if args.reverse_order else 'forward'}")
        print(f"  {'ID':<8} {'Alpha':>6} {'Gamma':>6} {'Lambda':>7}")
        print(f"  {'─' * 30}")
        for c in combos:
            print(f"  {c['combo_id']:<8} {c['alpha']:>6.2f} {c['gamma']:>6.3f} {c['lam']:>7.1f}")
        print(f"\nPools per run (9 pokemon each, seeded by run_id):")
        for r in range(1, min(4, RUNS_PER_COMBO + 1)):
            pool = get_pool_for_run(r)
            print(f"  Run {r}: {pool}")
        if RUNS_PER_COMBO > 3:
            print(f"  ... ({RUNS_PER_COMBO - 3} more)")
        sys.exit(0)

    if args.summary:
        _print_summary(combos)
        sys.exit(0)

    if args.run_experiment:
        # Internal: run one combo+run+model as a subprocess
        combo = json.loads(args.combo_json)
        _run_one_experiment(combo, args.exp_run_id, args.model, args.port)
        sys.exit(0)

    if args.combo:
        run_single_combo(args.combo, combos, run_filter=args.run)
    else:
        run_full_grid(max_parallel=args.max_parallel, reverse_order=args.reverse_order)
