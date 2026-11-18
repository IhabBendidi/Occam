#!/usr/bin/env python
"""The score ledger: every expected score, whether it exists, and a resubmit of what is missing.

WHY (user, 2026-09-23: "organise better to get scores in a clean way"). Scores were arriving with
holes -- a split lost to the /dev/shm wipe, an array killed by a launcher bug -- and each hole was
found and resubmitted by hand. This makes completeness a property of the system:

  configs/score_registry.tsv   one row per SCORE UNIT = (kind, benchmark, arm, seed, tag):
                               5 expected splits, each a `split_NN/metrics.json` under one root.
  runs/_ledger/state.json      the job ids the ledger submitted per (unit, split) and the attempt
                               count, so a split with a live job is never submitted twice and a
                               split that keeps failing stops after MAX_ATTEMPTS (reported, not
                               looped).

A split is DONE when its metrics.json exists and parses; IN FLIGHT when its last recorded job is
still in squeue; otherwise MISSING and resubmittable. Units launched before the ledger existed are
seeded with their live array ids (`seed_jobs` column), so nothing already queued is duplicated.

KINDS and how each is resubmitted (all go through the existing launchers):
  baseline      scripts/repro/baselines.sh           BENCHMARK, SEED, OUT, CACHE
  occam         scripts/slurm/rhj_splithalf_models.sbatch  one-row grid, PANEL_DIR/OUT_BASE for
                an OOD panel, --ood-context
  occam_norman  scripts/slurm/norman_models.sbatch   one-row grid (arm, tag, compose)
  competitor    scripts/slurm/competitor_score.sbatch  needs the export's pred.npy; a split with
                no pred.npy is reported NEEDS-EXPORT and never auto-submitted

USAGE
  python scripts/score_ledger.py                 # status table of every unit
  python scripts/score_ledger.py --submit        # resubmit missing splits (no live job, < max tries)
  python scripts/score_ledger.py --daemon        # --submit every POLL seconds (sbatch it on cpu)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY = os.path.join(REPO, "configs", "score_registry.tsv")
STATE = os.path.join(REPO, "runs", "_ledger", "state.json")
MAX_ATTEMPTS = int(os.environ.get("LEDGER_MAX_ATTEMPTS", "3"))
SPLITS = range(5)
OOD = {"h1_seen", "k562_seen", "k562_seen_h1set"}
SCORE_ROOT = {"rhj_fewshot": "runs/rhj_splithalf", "norman_combo": "runs/norman_combo"}


def score_root(bench: str) -> str:
    return SCORE_ROOT.get(bench, f"runs/{bench}")


def unit_key(u) -> str:
    return "|".join((u["kind"], u["bench"], u["arm"], u["seed"], u["tag"]))


def unit_root(u) -> str:
    b, a, s, t = u["bench"], u["arm"], u["seed"], u["tag"]
    if u["kind"] == "baseline":
        return f"{score_root(b)}/reference" + ("" if s == "42" else f"_s{s}")
    if u["kind"] == "occam":
        return f"{score_root(b)}/{a}__{t}"
    if u["kind"] == "occam_norman":
        return f"runs/norman_combo/eval/{a}__{t.split(':')[1]}"
    if u["kind"] == "competitor":   # arm = <arm><suffix>, tag = panel tag ("" or __<panel>)
        base = "" if b in ("rhj_fewshot",) else ("__norman_combo" if b == "norman_combo" else "")
        return f"{score_root(b)}/compet_{a}{base}{t}_s{s}"
    raise ValueError(u["kind"])


def export_dir(u) -> str:
    b = u["bench"]
    bt = "__norman_combo" if b == "norman_combo" else ""
    return f"runs/competitors/{u['arm']}{bt}{u['tag']}/s{u['seed']}"


def train_marker(u) -> str:
    """`final_model.pth` of the training run this Occam unit scores (written only at the end)."""
    if u["kind"] == "occam":
        return os.path.join(REPO, f"runs/rhj_fewshot_{u['arm']}__{u['tag']}/vcell/checkpoints/"
                                  "final_model.pth")
    return os.path.join(REPO, f"runs/norman_combo_{u['arm']}__{u['tag'].split(':')[0]}/vcell/"
                              "checkpoints/final_model.pth")


def split_done(root: str, k: int) -> bool:
    p = os.path.join(REPO, root, f"split_{k:02d}", "metrics.json")
    try:
        with open(p) as fh:
            m = json.load(fh)
        return bool(m.get("arms"))
    except (OSError, ValueError):
        return False


def live_jobs() -> set:
    out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i"],
                         capture_output=True, text=True).stdout.split()
    ids = set()
    for j in out:           # "123_4", "123_[0-4%6]", "123"
        base = j.split("_")[0]
        ids.add(base)
        ids.add(j)
    return ids


def load_registry():
    with open(REGISTRY) as fh:
        rows = [r for r in csv.DictReader((ln for ln in fh if not ln.startswith("#")),
                                          delimiter="\t")]
    for r in rows:
        for k in ("seed", "tag", "seed_jobs"):
            r[k] = (r.get(k) or "").strip()
    return rows


def load_state():
    try:
        with open(STATE) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def save_state(st):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    tmp = STATE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(st, fh, indent=1, sort_keys=True)
    os.replace(tmp, STATE)


def status(u, st, live):
    root = unit_root(u)
    rec = st.get(unit_key(u), {})
    seeded = [j for j in u["seed_jobs"].split(",") if j]
    out = {}
    for k in SPLITS:
        if split_done(root, k):
            out[k] = "done"
            continue
        if u["kind"] in ("occam", "occam_norman") and not os.path.isfile(train_marker(u)):
            out[k] = "training"      # final_model.pth is written only at the end
            continue
        jobs = rec.get(str(k), {}).get("jobs", [])
        if (jobs and jobs[-1] in live) or (not jobs and any(j in live for j in seeded)):
            out[k] = "inflight"
        elif u["kind"] == "competitor" and not os.path.isfile(
                os.path.join(REPO, export_dir(u), f"split_{k}", "pred.npy")):
            out[k] = "needs-export"
        elif rec.get(str(k), {}).get("attempts", 0) >= MAX_ATTEMPTS:
            out[k] = "gave-up"
        else:
            out[k] = "missing"
    return root, out


def submit(u, splits, dry=False):
    """One sbatch array over `splits` for this unit; returns the job id (str) or None."""
    b, a, s, t = u["bench"], u["arm"], u["seed"], u["tag"]
    arr = ",".join(str(k) for k in splits) + "%3"
    q = ["--qos=hooke-predict", "--wckey=hooke-predict"]
    env = {"OCCAM_CELLEVAL_ATTEMPTS": "4"}
    if u["kind"] == "baseline":
        env.update(BENCHMARK=b, SEED=s, OUT=unit_root(u), CACHE=f"{score_root(b)}/_de_real_shared")
        script = "scripts/repro/baselines.sh"
    elif u["kind"] in ("occam", "occam_norman"):
        gdir = os.path.join(REPO, "configs", "_ledger")
        os.makedirs(gdir, exist_ok=True)
        grid = os.path.join(gdir, f"{b}__{a}.tsv")
        with open(grid, "w") as fh:
            if u["kind"] == "occam":
                fh.write("arm_id\tkind\ttag\n" + f"{a}\tloss\t{t}\n")
            else:
                tag, comp = t.split(":")
                fh.write("arm\ttag\tcompose\n" + f"{a}\t{tag}\t{comp}\n")
        env.update(GRID=os.path.relpath(grid, REPO), N_SPLITS="5", CALIBRATE="none")
        if u["kind"] == "occam" and b in OOD:
            env.update(OUT_BASE=score_root(b), PANEL_DIR=f"data/benchmarks/{b}/splithalf",
                       EXTRA_ARGS="--ood-context", DE_CACHE=f"{score_root(b)}/_de_real_shared")
        script = ("scripts/slurm/rhj_splithalf_models.sbatch" if u["kind"] == "occam"
                  else "scripts/slurm/norman_models.sbatch")
    elif u["kind"] == "competitor":
        arm, suffix = (a.split("_", 1) + [""])[:2] if a.startswith(("txpert_", "perturbdiff_")) \
            else (a, "")
        suffix = ("_" + suffix) if suffix else ""
        env.update(ARM=arm, ARM_SUFFIX=suffix, SEED=s)
        if b in OOD:
            env.update(BENCH="rhj_fewshot",
                       PANEL_OVERRIDE=os.path.join(REPO, f"data/benchmarks/{b}/splithalf"),
                       PANEL_TAG=t, SCORE_ROOT=score_root(b))
        else:
            env.update(BENCH=b)
        script = "scripts/slurm/competitor_score.sbatch"
    else:
        raise ValueError(u["kind"])
    cmd = ["sbatch", "--parsable", *q, f"--array={arr}",
           "--export=ALL," + ",".join(f"{k}={v}" for k, v in env.items()), script]
    if dry:
        print("  DRY", " ".join(cmd))
        return None
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    jid = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    return jid if jid.isdigit() else None


def run(do_submit: bool, dry: bool):
    rows, st, live = load_registry(), load_state(), live_jobs()
    tot = {"done": 0, "inflight": 0, "missing": 0, "needs-export": 0, "gave-up": 0,
           "training": 0}
    print(f"{'unit':58s} {'splits 0-4':12s} root")
    for u in rows:
        root, s = status(u, st, live)
        for v in s.values():
            tot[v] += 1
        sym = "".join({"done": "#", "inflight": "~", "missing": ".", "needs-export": "e",
                       "gave-up": "X", "training": "t"}[s[k]] for k in SPLITS)
        print(f"{unit_key(u)[:58]:58s} {sym:12s} {root}")
        miss = [k for k in SPLITS if s[k] == "missing"]
        if do_submit and miss:
            jid = submit(u, miss, dry=dry)
            if jid:
                rec = st.setdefault(unit_key(u), {})
                for k in miss:
                    e = rec.setdefault(str(k), {"jobs": [], "attempts": 0})
                    e["jobs"].append(jid)
                    e["attempts"] += 1
                print(f"   -> resubmitted splits {miss}: job {jid}")
    if do_submit and not dry:
        save_state(st)
    print("\nlegend: # done  ~ in flight  . missing  e needs export  t still training  X gave up after "
          f"{MAX_ATTEMPTS} attempts")
    print("totals:", " · ".join(f"{k} {v}" for k, v in tot.items()))
    return tot


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="with --submit: print, do not sbatch")
    ap.add_argument("--daemon", action="store_true")
    ap.add_argument("--poll", type=int, default=600)
    args = ap.parse_args()
    if not args.daemon:
        run(args.submit, args.dry_run)
        return 0
    while True:
        print(f"\n=== ledger round {time.strftime('%F %T')} ===", flush=True)
        t = run(True, False)
        if not any(t[k] for k in ("inflight", "missing", "needs-export", "training")):
            print("all units done or given up; exiting")
            return 0
        sys.stdout.flush()
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
