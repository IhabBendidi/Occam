#!/usr/bin/env python
"""List every (arm, split, calibration-mode) whose cell-eval score is MISSING, as a task grid.

WHY THIS EXISTS
    F-infra1: a pdex shared-memory race made cell-eval return nothing for some triples while
    the SLURM task still exited 0. The directory exists, `agg_results.csv` does not, and the
    arm's mean silently becomes a mean-over-fewer-splits. This rebuilds the work list from what
    is ACTUALLY on disk rather than from what a job claimed to do -- the same discipline as
    collect_competitors.py, and for the same reason.

    ⚠️ It emits one task per (arm, split), with `calibrate` set to exactly the modes that are
    missing: `none` -> model_nocal only, `quantile` -> model_cal only, `both` -> both. Re-running
    a mode that already succeeded would waste ~8 min and risk overwriting a good score with a
    fresh race.

    ⚠️ Pass --exclude-running with the array job ids still in flight, so a triple that is
    already being computed is not queued twice.

    python scripts/build_backfill_grid.py --out configs/rhj_backfill.tsv
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess

ROOT = "runs/rhj_splithalf"
N_SPLITS = 5
CAL_FOR = {("nocal",): "none", ("cal",): "quantile", ("nocal", "cal"): "both"}


def running_triples(job_ids):
    """(arm, split) pairs whose task is still RUNNING/PENDING, read from the job's own marker."""
    if not job_ids:
        return set()
    live = set()
    out = subprocess.run(["sacct", "-j", ",".join(job_ids), "--format=JobID%20,State", "-X", "-n"],
                         capture_output=True, text=True).stdout
    live_tasks = {ln.split()[0] for ln in out.splitlines()
                  if len(ln.split()) > 1 and ln.split()[1] in ("RUNNING", "PENDING")}
    for f in glob.glob(f"{ROOT}/*/slurm_job_*.json"):
        import json
        try:
            j = json.load(open(f))
        except Exception:
            continue
        tid = f"{j.get('slurm_array_job_id')}_{j.get('slurm_task_id')}"
        if tid in live_tasks:
            live.add((j.get("arm_id"), int(j.get("split", -1))))
    return live


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="configs/rhj_backfill.tsv")
    ap.add_argument("--patterns", default="L*__*,D0*__*")
    ap.add_argument("--exclude-running", default="", help="comma-separated array job ids")
    a = ap.parse_args(argv)

    skip = running_triples([x for x in a.exclude_running.split(",") if x])
    rows, skipped = [], []
    for pat in a.patterns.split(","):
        for d in sorted(glob.glob(os.path.join(ROOT, pat.strip()))):
            arm, tag = os.path.basename(d).split("__", 1)
            for k in range(N_SPLITS):
                need = tuple(m for m in ("nocal", "cal")
                             if not os.path.isfile(
                                 f"{d}/split_{k:02d}/celleval_model_{m}/agg_results.csv"))
                if not need:
                    continue
                if (arm, k) in skip:
                    skipped.append(f"{arm} split {k}")
                    continue
                rows.append((arm, tag, k, CAL_FOR[need], ",".join(need)))

    with open(a.out, "w") as fh:
        fh.write("# F-infra1 backfill: (arm, split) triples whose cell-eval score is MISSING.\n")
        fh.write("# `calibrate` is EXACTLY the missing modes -- never re-run a mode that worked.\n")
        fh.write("# Regenerate with scripts/build_backfill_grid.py; do not hand-edit.\n")
        fh.write("arm_id\ttag\tsplit\tcalibrate\tmissing_modes\n")
        for r in rows:
            fh.write("\t".join(map(str, r)) + "\n")
    print(f"wrote {a.out}: {len(rows)} task(s)")
    if skipped:
        print(f"  skipped {len(skipped)} already in flight: {', '.join(skipped)}")
    by = {}
    for arm, _, _, cal, _ in rows:
        by[arm] = by.get(arm, 0) + 1
    print("  " + "  ".join(f"{k}:{v}" for k, v in sorted(by.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
