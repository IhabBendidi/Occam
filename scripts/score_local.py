#!/usr/bin/env python
"""Score Occam ledger units on THIS node's GPU instead of waiting for the cluster queue.

WHY (user, 2026-09-24). When both QOS are full, score splits can wait hours for a node while the
GPU of an interactive allocation (e.g. the code-server job) sits idle. This runs the SAME scoring
script the ledger submits (scripts/slurm/rhj_splithalf_models.sbatch, run with bash, same env:
one-row grid, PANEL_DIR / OUT_BASE / --ood-context for OOD panels), so the output is identical
in location and content to a cluster-scored split.

NO DOUBLE SCORING. The ledger's pending array tasks for the selected units are put on HOLD first
(held jobs stay in squeue, so the ledger keeps treating them as in flight and does not resubmit).
A split is scored locally only if its metrics.json is missing and its task is not RUNNING; after it
finishes, that held task is cancelled. Splits already running on the cluster are left alone.
On exit (or Ctrl-C) any task still held is RELEASED back to the queue.

Only `occam` units (rhj_fewshot + OOD panels) are supported.

USAGE
  python scripts/score_local.py --arm-regex 'AA03$' [--parallel 2] [--dry-run]
  python scripts/score_local.py --arm-regex '^S4[2-6]_ABL_L2$' --bench h1_seen
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_ledger as L  # noqa: E402

SCRIPT = "scripts/slurm/rhj_splithalf_models.sbatch"


def squeue_states() -> dict:
    """'<job>_<task>' -> state, for this user's jobs (pending arrays expanded)."""
    out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-r",
                          "-o", "%i %T"], capture_output=True, text=True).stdout
    return dict(ln.split() for ln in out.splitlines() if ln.strip())


def unit_env(u) -> dict:
    b, a, t = u["bench"], u["arm"], u["tag"]
    gdir = os.path.join(L.REPO, "configs", "_ledger")
    os.makedirs(gdir, exist_ok=True)
    grid = os.path.join(gdir, f"{b}__{a}.tsv")
    with open(grid, "w") as fh:
        fh.write("arm_id\tkind\ttag\n" + f"{a}\tloss\t{t}\n")
    env = dict(os.environ, GRID=os.path.relpath(grid, L.REPO), N_SPLITS="5", CALIBRATE="none",
               OCCAM_CELLEVAL_ATTEMPTS="4", OCCAM_REPO=L.REPO)
    if b in L.OOD:
        env.update(OUT_BASE=L.score_root(b), PANEL_DIR=f"data/benchmarks/{b}/splithalf",
                   EXTRA_ARGS="--ood-context", DE_CACHE=f"{L.score_root(b)}/_de_real_shared")
    return env


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm-regex", required=True)
    ap.add_argument("--bench", default=None, help="restrict to one benchmark")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rx = re.compile(args.arm_regex)
    units = [u for u in L.load_registry() if u["kind"] == "occam" and rx.search(u["arm"])
             and (args.bench is None or u["bench"] == args.bench)
             and os.path.isfile(L.train_marker(u))]
    st, q = L.load_state(), squeue_states()
    todo, held = [], []
    for u in units:
        rec = st.get(L.unit_key(u), {})
        root = L.unit_root(u)
        for k in L.SPLITS:
            if L.split_done(root, k):
                continue
            jobs = rec.get(str(k), {}).get("jobs", [])
            task = f"{jobs[-1]}_{k}" if jobs else None
            if task and q.get(task) == "RUNNING":
                continue
            if task and q.get(task) == "PENDING":
                held.append(task)
            todo.append((u, k, task))
    print(f"{len(todo)} splits to score locally ({len(held)} pending cluster tasks to hold)")
    if args.dry_run:
        for u, k, task in todo:
            print(f"  {u['bench']:16s} {u['arm']:12s} split {k}  cluster task {task}")
        return 0

    for t in held:
        subprocess.run(["scontrol", "hold", t], capture_output=True)
    log_dir = os.path.join(L.REPO, "runs", "slurm")

    def run(item):
        u, k, task = item
        if L.split_done(L.unit_root(u), k) or (task and squeue_states().get(task) == "RUNNING"):
            return f"skip {u['arm']} {u['bench']} split {k}"
        env = unit_env(u)
        env["SLURM_ARRAY_TASK_ID"] = str(k)
        log = os.path.join(log_dir, f"local_score_{u['bench']}__{u['arm']}_{k}.out")
        t0 = time.time()
        with open(log, "w") as fh:
            rc = subprocess.run(["bash", SCRIPT], cwd=L.REPO, env=env, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        ok = L.split_done(L.unit_root(u), k)
        if ok and task:
            subprocess.run(["scancel", task], capture_output=True)
        return (f"{'done' if ok else 'FAILED rc=' + str(rc)} {u['bench']} {u['arm']} split {k} "
                f"({(time.time() - t0) / 60:.1f} min) log {os.path.relpath(log, L.REPO)}")

    try:
        with ThreadPoolExecutor(args.parallel) as ex:
            for msg in ex.map(run, todo):
                print(msg, flush=True)
    finally:
        q = squeue_states()
        left = [t for t in held if q.get(t) == "PENDING"]
        for t in left:
            subprocess.run(["scontrol", "release", t], capture_output=True)
        if left:
            print(f"released {len(left)} held cluster tasks back to the queue")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
