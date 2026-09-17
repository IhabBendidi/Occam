#!/usr/bin/env python
"""Score ONLY the arm that holds the validation record, and only once per record.

WHAT THIS ANSWERS
-----------------
Overall goal: the best model on SEEN perturbations in an UNSEEN cell line.

This script's question: what does the sweep's CURRENT BEST arm score on the 611 held-out test
perturbations? Not the whole sweep -- one arm, the validation argmax, re-checked as the sweep
produces new records.

WHY THIS REPLACED THE BLANKET EVAL (decision 2026-09-17, F-adj2)
----------------------------------------------------------------
Measured on 95 uncalibrated arms: Spearman(val, cell-eval `pearson_delta`) = **+0.941**, and
the best VAL arm in the scored set (`s129`, 0.3676) IS the best CELL-EVAL arm (0.3348). The
argmax test passed 1 for 1, so scoring every arm buys ranking information we do not use.

⚠️ TWO THINGS THIS POLICY DOES NOT GIVE YOU, both measured:

1. **It cannot rank near-ties.** Top-10-by-val rho is only **+0.491** (vs +0.941 over the full
   range) -- the restricted-range effect. Validation picks the single best arm; it does not
   order the ten behind it. So this script scores the ARGMAX, never "the top N".
2. **It optimises two of the four comparison metrics.** rho is +0.941 on `pearson_delta` and
   −0.704 on `mae`, but only **+0.198** on `discrimination_score_l1` and +0.444 on
   `overlap_at_N`. A val-selected arm is the best arm on `pearson_delta` and `mae`; that is
   what the record must say, and if `disc_l1` matters for a decision the arm must be scored
   directly.

The val → cell-eval gap is one-sided and tight: mean **−0.0140**, range −0.0327 … −0.0019. So
a val score `v` predicts roughly `v − 0.014` and never more than `v`.

HOW IT BEHAVES
--------------
Idempotent by design. It records every record-holder it has scored in
`runs/rhj_splithalf/val_winners.json`, and:

  * does nothing if the current argmax is already scored or already submitted
  * does nothing if the argmax has not beaten the last scored record by `--min-gain`
  * otherwise submits the 5-split eval for that one arm

⛔ IT SUBMITS A JOB. That is the point, but it means it must not be run on a timer without
someone watching: RULE 0 forbids launching work nobody asked for. Use `--dry-run` to see what
it would do.

    python scripts/eval_val_winner.py --dry-run
    python scripts/eval_val_winner.py --qos vco
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

LEDGER = os.path.join(REPO, "runs", "rhj_splithalf", "val_winners.json")
N_SPLITS = 5
# Measured over 95 arms, F-adj2. Used only to print an expectation beside the launch.
VAL_TO_CE_GAP = -0.0140


def _completed_arms(grid: str):
    """(arm, val, row) for every arm whose vcell stage reports completed."""
    tsv, cols = {}, None
    for line in open(os.path.join(REPO, grid)):
        if line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        if cols is None:
            cols = parts
            continue
        tsv[parts[0]] = dict(zip(cols, parts))

    out = []
    for arm in sorted(tsv):
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
            out.append((arm, float(v), tsv[arm]))
    return out


def _scored_splits(arm: str, eval_root: str, prefix: str, tag: str = "model_nocal") -> int:
    d = glob.glob(os.path.join(REPO, eval_root, f"{prefix}{arm}__*"))
    if not d:
        return 0
    return sum(
        1 for s in range(N_SPLITS)
        if os.path.isfile(os.path.join(d[0], f"split_{s:02d}", f"celleval_{tag}",
                                       "agg_results.csv")))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default="configs/rhj_adjmap_sweep.tsv")
    ap.add_argument("--eval-root", default="runs/rhj_splithalf")
    ap.add_argument("--prefix", default="adjmap_")
    ap.add_argument("--sbatch", default="scripts/slurm/rhj_splithalf_models.sbatch")
    ap.add_argument("--qos", default="vco")
    ap.add_argument("--job-name", default="rhj_adj_eval")
    ap.add_argument("--calibrate", default="",
                    help="passed through as CALIBRATE; empty = the launcher's own default")
    ap.add_argument("--min-gain", type=float, default=0.0,
                    help="only score a new record if it beats the last SCORED record by this "
                         "much on validation. 0.0 = any new record.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    arms = _completed_arms(args.grid)
    if not arms:
        print("[winner] no completed arms yet")
        return 0
    arms.sort(key=lambda r: -r[1])
    arm, val, row = arms[0]

    hist = json.load(open(LEDGER)) if os.path.isfile(LEDGER) else {"scored": {}}
    hist.setdefault("scored", {})

    print(f"[winner] {len(arms)} completed arm(s) · validation argmax **{arm}** = {val:.4f}")
    print(f"[winner]   lr {row.get('lr','?')} · decoder_lr {row.get('decoder_lr','?')} · "
          f"cos {row.get('cosine_loss_weight','?')} · film {row.get('film_mode','?')}")

    n = _scored_splits(arm, args.eval_root, args.prefix)
    if n >= N_SPLITS:
        print(f"[winner] already scored on all {N_SPLITS} splits — nothing to do")
        return 0
    if arm in hist["scored"] and hist["scored"][arm].get("submitted"):
        print(f"[winner] already submitted as job "
              f"{hist['scored'][arm].get('job')} ({n}/{N_SPLITS} splits landed) — "
              f"nothing to do")
        return 0

    prev = max((r["val"] for r in hist["scored"].values()), default=None)
    if prev is not None and val - prev < args.min_gain:
        print(f"[winner] {arm} val {val:.4f} does not beat the last scored record "
              f"{prev:.4f} by --min-gain {args.min_gain} — nothing to do")
        return 0

    # ⚠️ THE ARRAY INDEX IS NOT THE ARM NUMBER. `rhj_splithalf_models.sbatch` computes
    #     ARM_I = IDX / N_SPLITS ; SPLIT = IDX % N_SPLITS
    # and ARM_I is the arm's ROW POSITION in the grid, 0-based. So one arm is FIVE tasks,
    # `arm_row*5 .. arm_row*5+4`. Deriving the index from the arm's name (`s313` -> 313) and
    # passing it as `--array=313` scores arm ROW 62, split 3 -- a different arm entirely, and
    # it would have produced a perfectly plausible number.
    rows = [ln.split("\t")[0] for ln in
            open(os.path.join(REPO, args.grid)).read().split("\n")
            if ln and not ln.startswith("#")][1:]
    if arm not in rows:
        raise SystemExit(f"{arm} is not a row of {args.grid}")
    arm_row = rows.index(arm)
    lo, hi = arm_row * N_SPLITS, arm_row * N_SPLITS + N_SPLITS - 1
    cmd = ["sbatch", "--parsable", f"--qos={args.qos}", f"--wckey={args.qos}",
           f"--job-name={args.job_name}", f"--array={lo}-{hi}", args.sbatch]
    print(f"[winner] {arm} is grid row {arm_row} -> array tasks {lo}-{hi} "
          f"(5 splits of ONE arm)")
    print(f"[winner] expectation, from the measured gap ({VAL_TO_CE_GAP:+.4f}): "
          f"cell-eval pearson_delta ~ {val + VAL_TO_CE_GAP:.4f}, and NOT above {val:.4f}")
    print(f"[winner] anchors: meanpert 0.1324 · transfer 0.2323 · experimental 0.3722")
    print(f"[winner] $ {' '.join(cmd)}")
    if args.dry_run:
        print("[winner] --dry-run: nothing submitted")
        return 0

    env = dict(os.environ)
    env["GRID"] = args.grid
    env["OUT_PREFIX"] = args.prefix
    if args.calibrate:
        env["CALIBRATE"] = args.calibrate
    print(f"[winner]   GRID={args.grid} OUT_PREFIX={args.prefix}"
          + (f" CALIBRATE={args.calibrate}" if args.calibrate else ""))
    job = subprocess.check_output(cmd, text=True, cwd=REPO, env=env).strip()
    hist["scored"][arm] = {"val": val, "job": job, "submitted": True,
                           "grid_row": arm_row, "array": f"{lo}-{hi}",
                           "config": {k: row.get(k) for k in
                                      ("lr", "decoder_lr", "cosine_loss_weight",
                                       "film_mode", "film_hidden_dim", "film_mean_free")}}
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "w") as fh:
        json.dump(hist, fh, indent=2, sort_keys=True)
    print(f"[winner] submitted {job} for {arm}; recorded in "
          f"{os.path.relpath(LEDGER, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
