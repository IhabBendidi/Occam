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
from typing import List

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

LEDGER = os.path.join(REPO, "runs", "rhj_splithalf", "val_winners.json")
N_SPLITS = 5
# The val -> cell-eval gap, used ONLY to print an expectation beside a launch.
#
# ⚠️ IT IS NOT A CONSTANT, and the top-of-range value is the one this script needs. The mean
# over 95 arms is -0.0140, but the sweep's two highest-val arms both came out near -0.035
# (F-adj3: s129 val 0.3676 -> 0.3348, gap -0.0328; s313 val 0.3769 -> 0.3415, gap -0.0354),
# outside that whole fitted range. That is mild validation overfitting: the better an arm
# scores on the 103 validation perturbations, the more of the score is fit to those 103.
#
# This script only ever looks at the ARGMAX, i.e. always the top of the range, so it uses the
# top-of-range figure. Using the -0.0140 mean over-predicted s313 by +0.021.
VAL_TO_CE_GAP = -0.0350
VAL_TO_CE_GAP_MEAN = -0.0140      # the all-arm mean, kept for reference only


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


def _missing_splits(arm: str, eval_root: str, prefix: str,
                    tag: str = "model_nocal") -> List[int]:
    """Which splits still have no scored `tag` arm.

    ⚠️ Returns IDENTITIES, not a count, because resuming a crashed run needs to know WHICH
    splits to redo. An earlier version returned a count and the caller could only ask "are all
    5 done?", which is why a crash left the policy refusing to continue.

    A split counts as done only if its `agg_results.csv` exists. That is the right test and it
    is what makes a half-written split read as NOT done: when hop01 crashed mid-split-2 it left
    the two h5ads and an EMPTY `celleval_model_nocal/` directory behind, which this correctly
    ignores.
    """
    d = glob.glob(os.path.join(REPO, eval_root, f"{prefix}{arm}__*"))
    if not d:
        return list(range(N_SPLITS))
    return [s for s in range(N_SPLITS)
            if not os.path.isfile(os.path.join(d[0], f"split_{s:02d}", f"celleval_{tag}",
                                               "agg_results.csv"))]


def _arm_paths(arm: str, grid: str, eval_root: str, prefix: str):
    """(checkpoint, out_dir) for an arm, from its grid row. Raises if either is absent."""
    rows = [ln.split("\t") for ln in open(os.path.join(REPO, grid)).read().split("\n")
            if ln and not ln.startswith("#")][1:]
    for r in rows:
        if r[0] == arm:
            tag = r[2]
            ckpt = os.path.join(REPO, "runs", f"rhj_fewshot_{arm}__{tag}", "vcell",
                                "checkpoints", "best_model.pth")
            if not os.path.isfile(ckpt):
                raise SystemExit(f"no checkpoint for {arm} at {ckpt}")
            return ckpt, os.path.join(REPO, eval_root, f"{prefix}{arm}__{tag}")
    raise SystemExit(f"{arm} is not a row of {grid}")


def _run_local(arm: str, splits: List[int], ckpt: str, out: str, *, calibrate: str,
               threads: int, cache: str) -> int:
    """Run the missing splits sequentially on THIS node, re-checking before each one.

    ⚠️ RE-RUNNABLE ON PURPOSE. It re-tests for `agg_results.csv` immediately before each
    split, so a second crash costs only the split that was in flight -- which is exactly the
    failure this whole code path exists to absorb.
    """
    import subprocess

    os.makedirs(out, exist_ok=True)
    os.makedirs(cache, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(REPO, "src") + (
        ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYENV_VERSION"] = env.get("PYENV_VERSION", "scmae")

    done, failed = [], []
    for sp in splits:
        agg = os.path.join(out, f"split_{sp:02d}", "celleval_model_nocal", "agg_results.csv")
        if os.path.isfile(agg):
            print(f"[winner] split {sp} landed since we started — skipping")
            continue
        # A half-written split dir is harmless (eval_splithalf overwrites), but an empty
        # celleval_*/ is confusing to read later, so clear it first.
        sd = os.path.join(out, f"split_{sp:02d}")
        if os.path.isdir(sd):
            import shutil
            shutil.rmtree(sd)
            print(f"[winner] cleared partial {os.path.relpath(sd, REPO)}")
        print(f"\n########## {arm} split {sp} ##########", flush=True)
        rc = subprocess.call(
            [sys.executable, "scripts/eval_splithalf.py",
             "--mode", "model", "--checkpoint", ckpt, "--split", str(sp),
             "--out-root", out, "--de-real-cache-dir", cache,
             "--calibrate", calibrate, "--num-threads", str(threads)],
            cwd=REPO, env=env)
        (done if rc == 0 else failed).append(sp)
        if rc:
            print(f"[winner] ⚠️ split {sp} exited {rc}; continuing with the rest so one bad "
                  f"split does not cost the others", file=sys.stderr)
    print(f"\n[winner] local run: {len(done)} split(s) scored {done}, "
          f"{len(failed)} failed {failed}")
    return 1 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default="configs/rhj_adjmap_sweep.tsv")
    ap.add_argument("--eval-root", default="runs/rhj_splithalf")
    ap.add_argument("--prefix", default="adjmap_")
    ap.add_argument("--sbatch", default="scripts/slurm/rhj_splithalf_models.sbatch")
    ap.add_argument("--qos", default="vco")
    ap.add_argument("--job-name", default="rhj_adj_eval")
    # ⚠️ NOT the launcher's `quantile` default. `quantile` emits ONLY `model_cal`, and every
    # number this policy rests on (F-adj2: Spearman +0.941, the val argmax winning cell-eval,
    # the -0.0140 gap) was measured on `model_nocal`. MEASURED 2026-09-17: a local run left at
    # the default produced `model_cal 0.2740` for s313 and no uncalibrated number at all --
    # a value that is not comparable to s129's 0.3348 and would have been read as one.
    ap.add_argument("--calibrate", default="both",
                    help="CALIBRATE for the launcher. Keep 'both': 'quantile' emits only "
                         "model_cal, which is NOT the arm this policy's evidence is on.")
    ap.add_argument("--min-gain", type=float, default=0.0,
                    help="only score a new record if it beats the last SCORED record by this "
                         "much on validation. 0.0 = any new record.")
    ap.add_argument("--local", action="store_true",
                    help="run the missing splits sequentially on THIS node instead of "
                         "submitting to Slurm. Re-runnable: a crash costs only the split in "
                         "flight.")
    ap.add_argument("--no-resume", action="store_true",
                    help="refuse when the ledger says this arm was already submitted, even if "
                         "splits are missing. The old (pre-crash) behaviour.")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--de-cache", default="runs/rhj_splithalf/_de_real_shared")
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

    missing = _missing_splits(arm, args.eval_root, args.prefix)
    n_done = N_SPLITS - len(missing)
    if not missing:
        print(f"[winner] already scored on all {N_SPLITS} splits — nothing to do")
        return 0

    already = bool(hist["scored"].get(arm, {}).get("submitted"))
    resuming = already and n_done > 0
    if already and args.no_resume:
        print(f"[winner] already submitted as job {hist['scored'][arm].get('job')} "
              f"({n_done}/{N_SPLITS} landed) and --no-resume was passed — nothing to do")
        return 0
    if resuming:
        # ⚠️ THIS IS THE CASE THE CRASH EXPOSED. The old check saw `submitted: true` and
        # refused, so a run that died at split 2 could never be finished by the same script.
        print(f"[winner] RESUMING {arm}: {n_done}/{N_SPLITS} splits landed under job "
              f"{hist['scored'][arm].get('job')}, redoing only {missing}")
    elif already:
        print(f"[winner] {arm} was submitted as {hist['scored'][arm].get('job')} but NO split "
              f"landed; treating it as a fresh run of {missing}")

    # The record gate is skipped on a resume: the arm already won its record, and
    # re-litigating it here would block the arm from finishing its own evaluation.
    if not already:
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
    tasks = ",".join(str(arm_row * N_SPLITS + sp) for sp in missing)
    cmd = ["sbatch", "--parsable", f"--qos={args.qos}", f"--wckey={args.qos}",
           f"--job-name={args.job_name}", f"--array={tasks}", args.sbatch]
    print(f"[winner] {arm} is grid row {arm_row} -> array tasks {tasks} "
          f"(split(s) {missing} of ONE arm)")
    print(f"[winner] expectation, from the TOP-OF-RANGE gap ({VAL_TO_CE_GAP:+.4f}, F-adj3): "
          f"cell-eval pearson_delta ~ {val + VAL_TO_CE_GAP:.4f}, and NOT above {val:.4f}")
    print(f"[winner]   (the all-arm mean gap {VAL_TO_CE_GAP_MEAN:+.4f} would say "
          f"{val + VAL_TO_CE_GAP_MEAN:.4f} — it over-predicts at the top of the range)")
    print(f"[winner] anchors: meanpert 0.1324 · transfer 0.2323 · experimental 0.3722")
    print(f"[winner] $ {' '.join(cmd)}")
    ckpt, out = _arm_paths(arm, args.grid, args.eval_root, args.prefix)
    if args.local:
        print(f"[winner] --local: running split(s) {missing} on {os.uname().nodename} "
              f"sequentially, {args.threads} threads, calibrate={args.calibrate}")
        print(f"[winner]   ckpt {os.path.relpath(ckpt, REPO)}")
        print(f"[winner]   out  {os.path.relpath(out, REPO)}")
    if args.dry_run:
        print("[winner] --dry-run: nothing run or submitted")
        return 0

    if args.local:
        rc = _run_local(arm, missing, ckpt, out, calibrate=args.calibrate,
                        threads=args.threads,
                        cache=os.path.join(REPO, args.de_cache))
        rec = hist["scored"].setdefault(arm, {"val": val})
        rec.update({"submitted": True, "calibrate": args.calibrate,
                    "grid_row": arm_row, "local_node": os.uname().nodename})
        rec.setdefault("job", f"local:{os.uname().nodename}")
        runs = rec.setdefault("local_runs", [])
        runs.append({"splits_requested": missing,
                     "splits_landed": [s for s in missing
                                       if s not in _missing_splits(arm, args.eval_root,
                                                                   args.prefix)],
                     "resumed": resuming})
        os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
        with open(LEDGER, "w") as fh:
            json.dump(hist, fh, indent=2, sort_keys=True)
        still = _missing_splits(arm, args.eval_root, args.prefix)
        print(f"[winner] {N_SPLITS - len(still)}/{N_SPLITS} splits now scored"
              + (f"; still missing {still}" if still else " — COMPLETE"))
        if not still:
            print(f"[winner] next: PYENV_VERSION=scmae python scripts/paired_report.py "
                  f"--model-arm model_nocal {os.path.relpath(out, REPO)}")
        return rc

    env = dict(os.environ)
    env["GRID"] = args.grid
    env["OUT_PREFIX"] = args.prefix
    if args.calibrate:
        env["CALIBRATE"] = args.calibrate
    if args.calibrate != "both":
        print(f"[winner] ⚠️ --calibrate {args.calibrate!r}: this policy's evidence is on "
              f"`model_nocal`. Anything but 'both' may produce no uncalibrated number.")
    print(f"[winner]   GRID={args.grid} OUT_PREFIX={args.prefix}"
          + (f" CALIBRATE={args.calibrate}" if args.calibrate else ""))
    job = subprocess.check_output(cmd, text=True, cwd=REPO, env=env).strip()
    hist["scored"][arm] = {"val": val, "job": job, "submitted": True,
                           "grid_row": arm_row, "array": tasks,
                           "splits_requested": missing, "resumed": resuming,
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
