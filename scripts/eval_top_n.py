#!/usr/bin/env python
"""Score the top N arms by validation, submitting only the splits that are missing.

WHAT THIS ANSWERS
-----------------
Overall goal: the best model on SEEN perturbations in an UNSEEN cell line.

This script's question: which arm is actually best on the 611 held-out test perturbations? Not
"which arm does validation like most" -- that question has been answered and the answer is
wrong.

WHY THIS REPLACED THE ARGMAX-ONLY POLICY (decision 2026-09-18, F-adj5)
-----------------------------------------------------------------------
`scripts/eval_val_winner.py` scored only the validation argmax, on the evidence that the val
argmax was the cell-eval argmax (F-adj2, 1-for-1; F-adj3, 2-for-2). **It failed on the third
test.** `s469` set a new validation record (0.3805 vs `s313`'s 0.3769) and scored WORSE on
`pearson_delta` on 5 of 5 splits -- 0.3392 against 0.3415. The three record holders come out
mis-ordered: val says `s469 > s313 > s129`, cell-eval says `s313 > s469 > s129`.

That was predictable from F-adj2's own numbers and I under-reported it: **top-10-by-val rho is
+0.491**, against +0.941 over the full range. The `s469`-`s313` val gap is 0.0036 -- a near-tie,
exactly the regime rho +0.49 says validation cannot resolve. Two passes could not distinguish a
working policy from a coin flip there.

So: validation reliably identifies a BAND of good arms (all three record holders land in
0.3348-0.3415, well clear of the `transfer` bar 0.2323) and cannot order them. Score the band.

⚠️ WHAT N BUYS. rho +0.491 over the top 10 means the cell-eval winner is very likely inside the
top 10 by val, and is NOT identifiable from val within it. N=10 is the band the rho was measured
on, which is why it is the default -- a smaller N is a bet that the winner sits higher than the
measurement supports.

⚠️ RANKING IS PER METRIC. `s469` LOSES `pearson_delta` and WINS `overlap_at_N` 5/5 (+0.0027,
diff sd 0.0005). Validation's rho is +0.941 on `pearson_delta` but only +0.444 on
`overlap_at_N` and +0.198 on `discrimination_score_l1`, so this set must be read metric by
metric. `scripts/collect_adjmap_eval.py` prints all four.

Idempotent and resumable: it submits only (arm, split) pairs with no `agg_results.csv`, so
re-running after a crash or a partial batch costs nothing.

    python scripts/eval_top_n.py --dry-run
    python scripts/eval_top_n.py --n 10 --qos hooke-predict --throttle 8
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from typing import Dict, List, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

N_SPLITS = 5
LEDGER = os.path.join(REPO, "runs", "rhj_splithalf", "top_n_scored.json")


def _grid(path: str) -> Tuple[List[str], Dict[str, dict]]:
    """(arm order, arm -> row). The ORDER is load-bearing: the launcher indexes by row."""
    rows, cols = {}, None
    order = []
    for line in open(os.path.join(REPO, path)):
        if line.startswith("#"):
            continue
        p = line.rstrip("\n").split("\t")
        if cols is None:
            cols = p
            continue
        if not p or not p[0]:
            continue
        rows[p[0]] = dict(zip(cols, p))
        order.append(p[0])
    return order, rows


def _completed(rows: Dict[str, dict]) -> List[Tuple[str, float]]:
    out = []
    for arm in rows:
        run = glob.glob(os.path.join(REPO, "runs", f"rhj_fewshot_{arm}__*"))
        if not run:
            continue
        led = os.path.join(run[0], "ledger.json")
        if not os.path.isfile(led):
            continue
        st = (json.load(open(led)).get("stages") or {}).get("vcell") or {}
        if st.get("status") != "completed":
            continue
        v = (st.get("metrics") or {}).get("best_metric")
        if isinstance(v, (int, float)):
            out.append((arm, float(v)))
    out.sort(key=lambda r: -r[1])
    return out


def _missing(arm: str, eval_root: str, prefix: str) -> List[int]:
    """Splits with no scored `model_nocal`. A half-written split reads as missing."""
    d = glob.glob(os.path.join(REPO, eval_root, f"{prefix}{arm}__*"))
    if not d:
        return list(range(N_SPLITS))
    return [s for s in range(N_SPLITS)
            if not os.path.isfile(os.path.join(
                d[0], f"split_{s:02d}", "celleval_model_nocal", "agg_results.csv"))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10,
                    help="how many top-by-validation arms to score. Default 10 = the band "
                         "F-adj2's +0.491 rho was measured over.")
    ap.add_argument("--grid", default="configs/rhj_adjmap_sweep.tsv")
    ap.add_argument("--eval-root", default="runs/rhj_splithalf")
    ap.add_argument("--prefix", default="adjmap_")
    ap.add_argument("--sbatch", default="scripts/slurm/rhj_splithalf_models.sbatch")
    ap.add_argument("--qos", default="hooke-predict")
    ap.add_argument("--job-name", default="rhj_top_eval")
    ap.add_argument("--throttle", type=int, default=8,
                    help="max concurrent array tasks (Slurm %% suffix)")
    ap.add_argument("--calibrate", default="both",
                    help="keep 'both': the comparison column is model_nocal, and 'quantile' "
                         "emits only model_cal")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    order, rows = _grid(args.grid)
    vals = _completed(rows)
    if not vals:
        print("[top-n] no completed arms yet")
        return 0
    top = vals[:args.n]

    print(f"[top-n] {len(vals)} completed arm(s); scoring the top {len(top)} by validation")
    print(f"{'#':>2s} {'arm':6s} {'val':>7s} {'to do':>16s}  "
          f"{'lr':>5s} {'dlr':>5s} {'cos':>4s} {'film_mode':<20s}")
    tasks: List[int] = []
    plan: Dict[str, List[int]] = {}
    for i, (arm, v) in enumerate(top, 1):
        miss = _missing(arm, args.eval_root, args.prefix)
        r = rows[arm]
        print(f"{i:>2d} {arm:6s} {v:7.4f} {str(miss) if miss else 'all 5 done':>16s}  "
              f"{r.get('lr',''):>5s} {r.get('decoder_lr',''):>5s} "
              f"{r.get('cosine_loss_weight',''):>4s} {r.get('film_mode',''):<20s}")
        if miss:
            plan[arm] = miss
            # ⚠️ The launcher computes ARM_I = IDX/5 and SPLIT = IDX%5 over grid ROW
            # positions, so the task id is row*5 + split -- NOT the arm's number. Passing
            # `--array=313` for s313 would score arm row 62 and file it under s313.
            tasks += [order.index(arm) * N_SPLITS + s for s in miss]

    if not tasks:
        print(f"[top-n] all {len(top)} arms already scored on all {N_SPLITS} splits — "
              f"nothing to do")
        return 0

    tasks = sorted(tasks)
    arr = ",".join(map(str, tasks)) + (f"%{args.throttle}" if args.throttle else "")
    cmd = ["sbatch", "--parsable", f"--qos={args.qos}", f"--wckey={args.qos}",
           f"--job-name={args.job_name}", f"--array={arr}", args.sbatch]

    print(f"\n[top-n] {len(plan)} arm(s) need scoring: {sorted(plan)}")
    print(f"[top-n] {len(tasks)} task(s), throttled to {args.throttle}")
    print(f"[top-n] anchors: meanpert 0.1324 · transfer 0.2323 · experimental 0.3722")
    print(f"[top-n] the bar to beat is s313's pearson_delta 0.3415 (F-adj5)")
    print(f"[top-n] $ {' '.join(cmd)}")
    if args.dry_run:
        print("[top-n] --dry-run: nothing submitted")
        return 0

    env = dict(os.environ)
    env["GRID"] = args.grid
    env["OUT_PREFIX"] = args.prefix
    env["CALIBRATE"] = args.calibrate
    job = subprocess.check_output(cmd, text=True, cwd=REPO, env=env).strip()

    hist = json.load(open(LEDGER)) if os.path.isfile(LEDGER) else {"batches": []}
    hist.setdefault("batches", []).append({
        "job": job, "n": args.n, "qos": args.qos, "throttle": args.throttle,
        "calibrate": args.calibrate, "tasks": tasks,
        "arms": {a: {"val": dict(vals)[a], "splits": m} for a, m in plan.items()},
    })
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "w") as fh:
        json.dump(hist, fh, indent=2, sort_keys=True)
    print(f"[top-n] submitted {job}; recorded in {os.path.relpath(LEDGER, REPO)}")
    print(f"[top-n] next: PYENV_VERSION=scmae python scripts/collect_adjmap_eval.py "
          f"--out docs/rhj_adjmap_tables.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
