#!/usr/bin/env python3
"""
Run all DiffSynth inference commands in parallel across a pool of GPUs.
Commands are extracted from inference_diffsynth.sh; the CUDA_VISIBLE_DEVICES
prefix in each line is stripped and replaced with the assigned GPU.

Usage (from repo root):
    python scripts/run_diffsynth_parallel.py
    python scripts/run_diffsynth_parallel.py --gpus 4,5,6,7
    python scripts/run_diffsynth_parallel.py --sh scripts/inference_diffsynth.sh --log-dir logs/diffsynth
"""
import argparse
import os
import queue
import re
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path


# ── command extraction ────────────────────────────────────────────────────────

def parse_sh(sh_path: Path):
    """
    Extract (model, method, cmd) tuples and the PROMPT env var from the sh file.
    cmd has CUDA_VISIBLE_DEVICES stripped; GPU assignment is done at dispatch time.
    """
    commands = []
    prompt = None

    for line in sh_path.read_text().splitlines():
        line = line.strip()

        # capture PROMPT=... definition
        m = re.match(r"^PROMPT='(.+)'$", line)
        if m:
            prompt = m.group(1)
            continue

        # capture inference commands
        m = re.match(r"CUDA_VISIBLE_DEVICES=\d+\s+(python\s+\S+infer_diffsynth\.py\s+.+)", line)
        if not m:
            continue
        cmd = m.group(1)

        model_m = re.search(r"--model\s+(\S+)", cmd)
        method_m = re.search(r"--method\s+(\S+)", cmd)
        model = model_m.group(1) if model_m else "unknown"
        method = method_m.group(1) if method_m else "unknown"
        commands.append((model, method, cmd))

    return commands, prompt


# ── worker ────────────────────────────────────────────────────────────────────

def worker(gpu_id: int, job_q: queue.Queue, log_dir: Path,
           error_log: Path, print_lock: threading.Lock,
           total: int, base_env: dict,
           timings_file: Path, timings_lock: threading.Lock):
    env = base_env.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    while True:
        try:
            idx, model, method, cmd = job_q.get_nowait()
        except queue.Empty:
            break

        tag = f"[GPU{gpu_id}][{idx:>3}/{total}][{model}/{method}]"
        log_path = log_dir / f"{model}__{method}.log"

        _print(print_lock, f"\n{'='*70}\n{tag} START\n  CMD: {cmd}\n{'='*70}")

        rc = -1
        try:
            with open(log_path, "w") as lf:
                lf.write(f"CMD:   {cmd}\nGPU:   {gpu_id}\nSTART: {datetime.now()}\n\n")
                lf.flush()

                proc = subprocess.Popen(
                    cmd,
                    shell=True,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                for line in proc.stdout:
                    lf.write(line)
                    lf.flush()
                    _print(print_lock, f"{tag} {line}", end="")

                proc.wait()
                rc = proc.returncode
                lf.write(f"\nEND: {datetime.now()}  returncode={rc}\n")

            if rc == 0:
                _print(print_lock, f"{tag} ✓ OK")
            else:
                _print(print_lock, f"{tag} ✗ ERROR (rc={rc})")
                _append_error(error_log, print_lock,
                              f"rc={rc:<4}  {tag}\n  cmd: {cmd}\n  log: {log_path}\n")

        except Exception as exc:
            _print(print_lock, f"{tag} ✗ EXCEPTION: {exc}")
            _append_error(error_log, print_lock,
                          f"EXC        {tag}\n  exc: {exc}\n  cmd: {cmd}\n")
        finally:
            _append_timing_row(timings_file, timings_lock, log_path, model, method, rc)
            job_q.task_done()


# ── helpers ───────────────────────────────────────────────────────────────────

def _print(lock: threading.Lock, msg: str, end: str = "\n"):
    with lock:
        print(msg, end=end, flush=True)


def _append_error(error_log: Path, lock: threading.Lock, msg: str):
    with lock:
        with open(error_log, "a") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")


def _append_timing_row(timings_file: Path, lock: threading.Lock,
                       log_path: Path, model: str, method: str, rc: int):
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        text = ""
    status_m = re.search(r"status=(\S+)", text)
    denoise_m = re.search(r"denoise_sec=([\d.]+)", text)
    generate_m = re.search(r"generate_sec=([\d.]+)", text)

    status = status_m.group(1) if status_m else ("ok" if rc == 0 else "failed")
    denoise = f"{float(denoise_m.group(1)):.1f}" if denoise_m else ""
    generate = f"{float(generate_m.group(1)):.1f}" if generate_m else ""

    with lock:
        with open(timings_file, "a") as f:
            f.write(f"{model}\t{method}\t{status}\t{denoise}\t{generate}\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Parallel DiffSynth inference runner")
    ap.add_argument("--gpus", default="4,5,6,7",
                    help="Comma-separated GPU IDs to use (default: 4,5,6,7)")
    ap.add_argument("--sh", default="scripts/inference_diffsynth.sh",
                    help="Path to inference_diffsynth.sh")
    ap.add_argument("--log-dir", default="logs/diffsynth",
                    help="Directory for per-job logs and error summary")
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    sh_path = Path(args.sh)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    error_log = log_dir / "errors.txt"
    error_log.write_text(f"Error log — started {datetime.now()}\n\n")

    timings_file = log_dir / "timings.tsv"
    timings_file.write_text("model\tmethod\tstatus\tdenoise_sec\tgenerate_sec\n")
    timings_lock = threading.Lock()

    commands, prompt = parse_sh(sh_path)
    if not commands:
        print(f"No commands found in {sh_path}", file=sys.stderr)
        sys.exit(1)

    total = len(commands)
    print(f"Loaded {total} commands from {sh_path}")
    print(f"GPUs:  {gpus}  ({len(gpus)} workers)")
    print(f"Logs:  {log_dir}/")
    if prompt:
        print(f"PROMPT set from sh file")

    # build base env: inherit everything + PROMPT from sh
    base_env = os.environ.copy()
    if prompt:
        base_env["PROMPT"] = prompt

    # fill queue
    job_q: queue.Queue = queue.Queue()
    for i, (model, method, cmd) in enumerate(commands, 1):
        job_q.put((i, model, method, cmd))

    print_lock = threading.Lock()

    threads = [
        threading.Thread(
            target=worker,
            args=(gpu_id, job_q, log_dir, error_log, print_lock, total, base_env,
                  timings_file, timings_lock),
            daemon=True,
            name=f"gpu{gpu_id}",
        )
        for gpu_id in gpus
    ]

    start = datetime.now()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    elapsed = datetime.now() - start
    print(f"\n{'='*70}")
    print(f"All {total} jobs finished in {elapsed}")

    errors = error_log.read_text().strip()
    # count non-header lines
    error_lines = [l for l in errors.splitlines() if l and not l.startswith("Error log")]
    if error_lines:
        print(f"\n{len(error_lines)//3 + 1} error(s) recorded in {error_log}:")
        print(errors)
    else:
        print("No errors.")

    _print_timing_table(timings_file, print_lock)


def _print_timing_table(timings_file: Path, print_lock: threading.Lock):
    """Print a console table from the already-written timings.tsv."""
    try:
        lines = timings_file.read_text().splitlines()
    except OSError:
        return
    rows = []
    for line in lines[1:]:  # skip header
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        rows.append({"model": parts[0], "method": parts[1], "status": parts[2],
                     "denoise_sec": parts[3], "generate_sec": parts[4]})
    if not rows:
        return

    col_w = [
        max(len(r["model"]) for r in rows),
        max(len(r["method"]) for r in rows),
        8, 12, 13,
    ]
    header = (f"{'model':<{col_w[0]}}  {'method':<{col_w[1]}}  "
              f"{'status':<{col_w[2]}}  {'denoise_sec':>{col_w[3]}}  {'generate_sec':>{col_w[4]}}")
    lines_out = [f"\nTiming summary → {timings_file}", header, "-" * len(header)]
    for r in rows:
        denoise = r["denoise_sec"] or "-"
        generate = r["generate_sec"] or "-"
        lines_out.append(
            f"{r['model']:<{col_w[0]}}  {r['method']:<{col_w[1]}}  "
            f"{r['status']:<{col_w[2]}}  {denoise:>{col_w[3]}}  {generate:>{col_w[4]}}"
        )
    _print(print_lock, "\n".join(lines_out))


if __name__ == "__main__":
    main()
