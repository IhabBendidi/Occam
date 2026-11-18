#!/usr/bin/env python3
"""Score the official Norman all-source model on 5 seeds x 5 splits.

Run on a node with three visible CUDA GPUs.  The first task for each split
primes the shared real-side DE cache; the remaining checkpoint/split pairs can
then run in parallel without writing the same cache entry concurrently.

The runner is resumable: a task is complete only when its uncalibrated model
metrics contain all four headline numbers.  Every task has its own log.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from queue import Empty, Queue
import shutil
import subprocess
import sys
from threading import Event


REPO = Path(__file__).resolve().parent.parent
VARIANTS = ("NF32ALLA",)
SEEDS = range(42, 47)
SPLITS = range(5)
HEADLINES = ("pearson_delta", "mae", "overlap_at_N", "discrimination_score_l1")
OUT_BASE = REPO / "runs/norman_combo/eval"
LOG_DIR = REPO / "runs/norman_combo/eval_jobs"
CACHE = REPO / "runs/norman_combo/_de_real_shared"
PANEL = REPO / "data/benchmarks/norman_combo/splithalf"
CHECKPOINTS = REPO / "checkpoints/vcell/norman"


def output_root(task):
    variant, seed, _ = task
    return OUT_BASE / f"S{seed}_{variant}__decode_seq_mean"


def complete(task):
    _, _, split = task
    path = output_root(task) / f"split_{split:02d}/metrics.json"
    try:
        with path.open() as handle:
            metrics = json.load(handle)["arms"]["model_nocal"]
        return all(isinstance(metrics.get(key), (float, int)) for key in HEADLINES)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_task(task, gpu, threads):
    variant, seed, split = task
    if complete(task):
        return True, f"already complete {variant} seed {seed} split {split}"
    checkpoint = CHECKPOINTS / f"seed_{seed}/best_model.pth"
    if not checkpoint.is_file():
        return False, f"missing checkpoint: {checkpoint}"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{variant}_s{seed}_split{split}.log"
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        OMP_NUM_THREADS=str(threads),
        MKL_NUM_THREADS=str(threads),
        MPLCONFIGDIR="/tmp/occam-mplconfig",
        PYTHONUNBUFFERED="1",
        OCCAM_CELLEVAL_ATTEMPTS="3",
    )
    env["PATH"] = "/tmp/occam-celleval-bin:" + env.get("PATH", "")
    cmd = [
        str(REPO / ".venv/bin/python"), str(REPO / "scripts/eval_splithalf.py"),
        "--mode", "model", "--panel-dir", str(PANEL),
        "--checkpoint", str(checkpoint), "--compose", "decode_seq_mean",
        "--split", str(split), "--out-root", str(output_root(task)),
        "--de-real-cache-dir", str(CACHE), "--calibrate", "none",
        "--num-threads", str(threads),
    ]
    with log_path.open("a") as handle:
        handle.write(f"\n=== GPU {gpu}: {variant} seed {seed} split {split} ===\n")
        handle.flush()
        status = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=handle,
                                stderr=subprocess.STDOUT).returncode
    ok = status == 0 and complete(task)
    return ok, f"{'done' if ok else 'FAILED'} {variant} seed {seed} split {split} " \
               f"(exit {status}; log {log_path.relative_to(REPO)})"


def run_phase(tasks, gpus, threads):
    queue = Queue()
    for task in tasks:
        if not complete(task):
            queue.put(task)
    if queue.empty():
        return True
    print(f"Starting {queue.qsize()} missing evaluations on GPUs {gpus}", flush=True)
    failed = Event()

    def worker(gpu):
        while not failed.is_set():
            try:
                task = queue.get_nowait()
            except Empty:
                return
            ok, message = run_task(task, gpu, threads)
            print(message, flush=True)
            if not ok:
                failed.set()
            queue.task_done()

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(worker, gpus))
    return not failed.is_set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2", help="comma-separated CUDA GPU indexes")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",")]
    if not gpus or len(set(gpus)) != len(gpus):
        parser.error("--gpus must contain distinct GPU indexes")
    if not (PANEL / "panel.npz").is_file():
        parser.error(f"frozen Norman panel is missing: {PANEL / 'panel.npz'}")

    # Executable scripts cannot run directly from this repository's noexec
    # filesystem, so put the cell-eval entry point on /tmp.  Its shebang still
    # points to the separate Python 3.10 cell-eval environment.
    entry = REPO / ".venv-celleval/bin/cell-eval"
    if not entry.is_file():
        parser.error(f"cell-eval 0.6.0 entry point is missing: {entry}")
    wrapper = Path("/tmp/occam-celleval-bin/cell-eval")
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(entry, wrapper)
    wrapper.chmod(0o755)

    prime = [("NF32ALLA", 42, split) for split in SPLITS]
    all_tasks = [(variant, seed, split) for seed in SEEDS
                 for split in SPLITS for variant in VARIANTS]
    if args.dry_run:
        print(f"{sum(not complete(task) for task in all_tasks)}/{len(all_tasks)} tasks missing")
        print("phase 1: prime the measured-reference cache for each split")
        print("phase 2: remaining tasks, with completed tasks skipped")
        return 0
    if not run_phase(prime, gpus, args.threads):
        return 1
    if not run_phase(all_tasks, gpus, args.threads):
        return 1
    done = sum(complete(task) for task in all_tasks)
    print(f"Complete: {done}/{len(all_tasks)} checkpoint/split evaluations", flush=True)
    if done != len(all_tasks):
        return 1

    # Keep the existing frozen paper table intact.  Export these new results
    # into their own release directory with the standard repository scripts.
    result_dir = REPO / "runs/norman_combo/reproduced_scores"
    python = str(REPO / ".venv/bin/python")
    subprocess.run(
        [python, str(REPO / "scripts/export_split_metrics.py"),
         "--scope", "published", "--benchmark", "norman_combo", "--out", str(result_dir)],
        cwd=str(REPO), check=True,
    )
    subprocess.run(
        [python, str(REPO / "scripts/final_table.py"),
         "--long", str(result_dir / "metrics_long.csv"),
         "--out-dir", str(result_dir)],
        cwd=str(REPO), check=True,
    )
    print(f"New scores: {result_dir / 'scores_by_seed_split.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
