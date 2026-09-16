#!/usr/bin/env python
"""Summarise the repeated split-half paired evaluation. The ONLY valid summary of it.

WHAT THIS ANSWERS
-----------------
Overall goal: predict a perturbation's effect in a context where that perturbation was
never measured, from a single control cell.

This report's question: across 5 independent splits of the same cells, how large is the gap
between the model and experimental replicate agreement, and is it larger than the
split-to-split noise?

What is varied: the split. Held fixed: the 611-perturbation cohort, the gene axis, the
checkpoint, the arm set, the per-perturbation row counts.

What each outcome means, stated before the numbers:
  * gap clearly > 0 with a small std  -> the model is measurably short of replicate
    agreement, and the gap size is the headline.
  * gap ~ 0                           -> the model is at replicate agreement ON THIS METRIC
    at these cell counts. Not "solved": see the strata.
  * gap < 0                           -> a RED FLAG, not a triumph. At median m = 22 the
    reference and the experimental prediction are both noisy pseudobulks while a
    low-variance predictor is attenuated on one side only, so a model can beat the
    "ceiling" from noise alone.
  * the strata table is the tie-breaker: a real gap SHRINKS as m grows. One that does not
    is noise-dominated whichever sign it has.

THREE PLACES A SUBTLE ERROR WOULD ENTER THE HEADLINE
----------------------------------------------------
1. **`std` uses `ddof=1`.** numpy's default understates a 5-sample std by 11.8%.
2. **The paired gap is the mean of the 5 per-split differences.** `mean(gap)` happens to
   equal the difference of means, but `std(gap)` is NOT `sqrt(std_a^2 + std_b^2)` -- the
   splits are shared, which is the entire point of pairing.
3. **cell-eval's own `std` column means something else** (spread across perturbations
   within one run). Everything here is named `std_across_perturbations` or
   `std_across_splits`, and both are carried.

Sign convention: **positive gap = the reference arm is better.**

    python scripts/paired_report.py --run-dir runs/rhj_splithalf/a03__best_model
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Dict, List, Optional

PROTOCOL = "splithalf/v1"

# The reference the model is compared against, and the two arms that bracket it.
REFERENCE_ARM = "experimental"
FLOOR_ARM = "ctrlcell"
BASELINE_ARM = "meanpert"

HEADLINE = ("pearson_delta", "overlap_at_N", "discrimination_score_l2",
            "de_direction_match", "mae_delta", "mse_delta")
LOWER_IS_BETTER = ("mae_delta", "mse_delta", "mae", "mse")
STRATA = ((5, "m>=5"), (10, "m>=10"), (25, "m>=25"))


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def _std1(xs: List[float]) -> float:
    """Sample std, `ddof=1`. numpy's default `ddof=0` understates a 5-sample std by 11.8%."""
    if len(xs) < 2:
        return float("nan")
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _fmt(v: Optional[float], nd: int = 4) -> str:
    if v is None:
        return "undefined"
    if isinstance(v, float) and v != v:
        return "n/a"
    return f"{v:.{nd}f}"


def load_records(run_dir: str) -> List[Dict]:
    """Every split's `metrics.json`. Probes are refused, not quietly averaged in."""
    out = []
    for path in sorted(glob.glob(os.path.join(run_dir, "split_*", "metrics.json"))):
        with open(path) as fh:
            rec = json.load(fh)
        if rec.get("probe"):
            raise SystemExit(f"{path} is a PROBE record. A probe's prediction is the "
                             f"identity on the control input; it is not a result.")
        if rec.get("protocol") != PROTOCOL:
            raise SystemExit(f"{path}: protocol {rec.get('protocol')!r} != {PROTOCOL!r}")
        rec["_path"] = path
        out.append(rec)
    if not out:
        probes = glob.glob(os.path.join(run_dir, "split_*", "metrics_probe.json"))
        raise SystemExit(
            f"no metrics.json under {run_dir}/split_*/"
            + (f" ({len(probes)} probe record(s) found -- run without --probe)"
               if probes else ""))
    cohorts = {r["cohort_sha256"] for r in out}
    if len(cohorts) > 1:
        raise SystemExit(
            f"the splits disagree on the cohort ({sorted(c[:16] for c in cohorts)}). "
            f"discrimination_score_l2 is a rank over the cohort and BH-FDR is computed "
            f"over the whole frame, so these runs are not comparable. Rebuild the panel.")
    return out


def read_per_pert(split_dir: str, tag: str) -> Dict[str, Dict[str, float]]:
    """cell-eval's per-perturbation `results.csv` for one arm."""
    path = os.path.join(split_dir, f"celleval_{tag}", "results.csv")
    if not os.path.isfile(path):
        return {}
    out: Dict[str, Dict[str, float]] = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            key = row.get("target_gene") or row.get("perturbation") or row.get("")
            if not key:
                continue
            vals = {}
            for k, v in row.items():
                if k in (None, "", "target_gene", "perturbation"):
                    continue
                try:
                    vals[k] = float(v)
                except (TypeError, ValueError):
                    continue
            out[str(key)] = vals
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default="", help="default: <run-dir>/report")
    ap.add_argument("--reference", default=REFERENCE_ARM)
    ap.add_argument("--expect-splits", type=int, default=5)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.run_dir, "report")
    os.makedirs(out_dir, exist_ok=True)

    recs = load_records(args.run_dir)
    splits = [int(r["split"]) for r in recs]
    print(f"[report] {len(recs)} split(s): {splits}")
    if len(recs) != args.expect_splits:
        # Reported, not fatal: a partial report is legitimate as long as it SAYS it is
        # partial. Silently averaging 3 splits into a "5-split" claim is not.
        print(f"[report] ⚠️ PARTIAL: {len(recs)} of {args.expect_splits} splits. Every "
              f"number below is labelled `n_splits` -- do not quote it as the 5-split "
              f"result.")

    arms: List[str] = []
    for r in recs:
        for a in (r.get("anchor_arms", []) + r.get("reference_arms", [])
                  + r.get("model_arms", [])):
            if a not in arms:
                arms.append(a)
    metrics = [m for m in HEADLINE
               if any(m in r["arms"].get(a, {}) for r in recs for a in arms)]
    print(f"[report] arms {arms}")
    print(f"[report] metrics {metrics}")

    # ---- per_split.csv ----------------------------------------------------
    per_split_rows = []
    for r in recs:
        for a in arms:
            m = r["arms"].get(a, {})
            row = {"protocol": PROTOCOL, "split": r["split"], "arm": a,
                   "status": m.get("status", "scored"),
                   "n_perts": r["n_perts"], "n_rows": r["n_rows_per_arm"]}
            for k in metrics:
                row[k] = m.get(k, "")
            per_split_rows.append(row)
    _write_csv(os.path.join(out_dir, "per_split.csv"), per_split_rows)

    # ---- summary.csv: mean and ddof=1 std ACROSS SPLITS -------------------
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    summary_rows = []
    for a in arms:
        summary[a] = {}
        for k in metrics:
            vals = [r["arms"][a][k] for r in recs
                    if k in r["arms"].get(a, {})
                    and isinstance(r["arms"][a][k], (int, float))
                    and r["arms"][a][k] == r["arms"][a][k]]
            if not vals:
                summary[a][k] = {"mean": None, "std_across_splits": None, "n_splits": 0}
                continue
            summary[a][k] = {"mean": _mean(vals), "std_across_splits": _std1(vals),
                             "n_splits": len(vals)}
            summary_rows.append({"protocol": PROTOCOL, "arm": a, "metric": k,
                                 "mean": summary[a][k]["mean"],
                                 "std_across_splits": summary[a][k]["std_across_splits"],
                                 "n_splits": len(vals),
                                 "lower_is_better": int(k in LOWER_IS_BETTER)})
    _write_csv(os.path.join(out_dir, "summary.csv"), summary_rows)

    # ---- gaps.csv: the PER-SPLIT paired differences -----------------------
    ref = args.reference
    if ref not in arms:
        raise SystemExit(f"reference arm {ref!r} was not scored; arms are {arms}")
    gap_rows, gap_summary_rows = [], []
    gaps: Dict[str, Dict[str, List[float]]] = {}
    for a in arms:
        if a == ref:
            continue
        gaps[a] = {}
        for k in metrics:
            per = []
            for r in recs:
                x, y = r["arms"].get(ref, {}).get(k), r["arms"].get(a, {}).get(k)
                if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
                    continue
                if x != x or y != y:
                    continue
                # positive = the REFERENCE arm is better, in both directions
                d = (y - x) if k in LOWER_IS_BETTER else (x - y)
                per.append(d)
                gap_rows.append({"protocol": PROTOCOL, "split": r["split"], "arm": a,
                                 "metric": k, "reference": ref, "gap": d,
                                 "reference_value": x, "arm_value": y})
            gaps[a][k] = per
            if per:
                gap_summary_rows.append({
                    "protocol": PROTOCOL, "arm": a, "metric": k, "reference": ref,
                    # the mean of the per-split DIFFERENCES, and its ddof=1 std. std is NOT
                    # sqrt(std_a^2 + std_b^2): the splits are shared, which is the point.
                    "gap_mean": _mean(per), "gap_std_across_splits": _std1(per),
                    "n_splits": len(per),
                    "n_splits_arm_better": sum(1 for d in per if d < 0),
                })
    _write_csv(os.path.join(out_dir, "gaps.csv"), gap_rows)
    _write_csv(os.path.join(out_dir, "gap_summary.csv"), gap_summary_rows)

    # ---- strata.csv: the same gap by half size, by re-aggregation ---------
    strata_rows = _strata(args.run_dir, recs, arms, ref, metrics)
    _write_csv(os.path.join(out_dir, "strata.csv"), strata_rows)

    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump({"protocol": PROTOCOL, "run_dir": os.path.abspath(args.run_dir),
                   "n_splits": len(recs), "splits": splits,
                   "expect_splits": args.expect_splits,
                   "partial": len(recs) != args.expect_splits,
                   "cohort_sha256": recs[0]["cohort_sha256"],
                   "n_perts": recs[0]["n_perts"], "reference_arm": ref,
                   "arms": arms, "metrics": metrics,
                   "summary": summary,
                   "gap_summary": gap_summary_rows,
                   "notes": {k: v for r in recs for k, v in r.get("notes", {}).items()},
                   }, fh, indent=2, sort_keys=True, default=str)

    _write_report(os.path.join(out_dir, "report.md"), recs, arms, ref, metrics,
                  summary, gaps, strata_rows, args.expect_splits)
    print(f"\n[report] wrote {out_dir}/"
          f"{{per_split,summary,gaps,gap_summary,strata}}.csv + metrics.json + report.md")
    print(f"[report] read report.md -- every number belongs next to its anchor")
    return 0


def _strata(run_dir, recs, arms, ref, metrics):
    """The gap by half size `m`, re-aggregated from results.csv. No extra cell-eval runs."""
    rows = []
    half = {p: int(v) for p, v in recs[0]["half_size"].items()}
    for thresh, name in STRATA:
        keep = {p for p, m in half.items() if m >= thresh}
        if not keep:
            continue
        # per (split, arm, metric): the mean over the perturbations in this stratum
        vals: Dict[tuple, float] = {}
        for r in recs:
            sd = os.path.dirname(r["_path"])
            for a in arms:
                pp = read_per_pert(sd, a)
                if not pp:
                    continue
                for k in metrics:
                    xs = [v[k] for p, v in pp.items()
                          if p in keep and k in v and v[k] == v[k]]
                    if xs:
                        vals[(r["split"], a, k)] = _mean(xs)
        for a in arms:
            if a == ref:
                continue
            for k in metrics:
                per = []
                for r in recs:
                    x = vals.get((r["split"], ref, k))
                    y = vals.get((r["split"], a, k))
                    if x is None or y is None:
                        continue
                    per.append((y - x) if k in LOWER_IS_BETTER else (x - y))
                if per:
                    rows.append({"protocol": PROTOCOL, "stratum": name,
                                 "min_half_size": thresh, "n_perts": len(keep),
                                 "arm": a, "metric": k, "reference": ref,
                                 "gap_mean": _mean(per),
                                 "gap_std_across_splits": _std1(per),
                                 "n_splits": len(per)})
    if not rows:
        print("[report] no per-perturbation results.csv found; strata.csv is empty. "
              "The strata table is the tie-breaker for a small gap -- without it, a "
              "near-zero headline cannot be told from noise.")
    return rows


def _write_csv(path, rows):
    if not rows:
        open(path, "w").close()
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_report(path, recs, arms, ref, metrics, summary, gaps, strata, expect):
    n = len(recs)
    partial = n != expect
    with open(path, "w") as fh:
        W = fh.write
        W(f"# Split-half paired evaluation — {n} split(s)\n\n")
        if partial:
            W(f"> ⚠️ **PARTIAL: {n} of {expect} splits.** Do not quote any number here as "
              f"the {expect}-split result.\n\n")
        W(f"**Objective.** Predict a perturbation's effect in a context where that "
          f"perturbation was never measured, from a single control cell.\n\n")
        W(f"**This table's question.** Scored against the same half-A reference, how far "
          f"is the model from experimental replicate agreement, and is the gap larger than "
          f"the split-to-split noise?\n\n")
        W(f"**Varied:** the split ({n} of them). **Held fixed:** the "
          f"{recs[0]['n_perts']}-perturbation cohort (`{recs[0]['cohort_sha256'][:16]}`), "
          f"the gene axis, the checkpoint, the arm set, and the per-perturbation row "
          f"counts — {recs[0]['n_rows_per_arm']:,} rows per arm and "
          f"{recs[0]['n_ctrl_B']:,} pred-side controls in every split.\n\n")
        W("**Sign convention: positive gap = the reference arm "
          f"(`{ref}`) is better.** A negative gap is a red flag, not a triumph — at median "
          "`m` = 22 both the reference and the experimental prediction are noisy "
          "pseudobulks while a low-variance predictor is attenuated on one side only, so a "
          "model can beat the *ceiling* from noise alone. The strata table below is the "
          "tie-breaker: a real gap **shrinks** as `m` grows.\n\n")

        W("## Anchors — read every number against these\n\n")
        W(f"| arm | role |\n|---|---|\n")
        W(f"| `{FLOOR_ARM}` | control-sampling noise at `m` cells — this panel's "
          f"zero-signal floor |\n")
        W(f"| `{BASELINE_ARM}` | the mean training delta — the "
          f"\"no per-perturbation information\" baseline |\n")
        W(f"| `{ref}` | **replicate agreement at `m` cells**, not an upper bound |\n\n")

        W(f"## Per-arm, mean ± std across the {n} split(s) (`ddof=1`)\n\n")
        W("| arm | " + " | ".join(
            f"{m}{' ↓' if m in LOWER_IS_BETTER else ''}" for m in metrics) + " |\n")
        W("|---|" + "---|" * len(metrics) + "\n")
        for a in arms:
            cells = []
            for m in metrics:
                s = summary[a].get(m) or {}
                if s.get("mean") is None:
                    cells.append("undefined")
                else:
                    cells.append(f"{_fmt(s['mean'])} ± {_fmt(s['std_across_splits'])}")
            W(f"| `{a}` | " + " | ".join(cells) + " |\n")
        W("\n`↓` = lower is better. `undefined` is a real outcome here, not a missing "
          "number: see the notes.\n\n")

        W(f"## The paired gap — mean of the {n} per-split differences\n\n")
        W("| arm | " + " | ".join(metrics) + " |\n")
        W("|---|" + "---|" * len(metrics) + "\n")
        for a in arms:
            if a == ref:
                continue
            cells = []
            for m in metrics:
                per = gaps.get(a, {}).get(m) or []
                if not per:
                    cells.append("—")
                else:
                    better = sum(1 for d in per if d < 0)
                    cells.append(f"**{_fmt(_mean(per))}** ± {_fmt(_std1(per))}"
                                 + (f" ({better}/{len(per)} arm better)" if better else ""))
            W(f"| `{a}` | " + " | ".join(cells) + " |\n")
        W("\n`std` here is the std of the **differences**, not "
          "`sqrt(std_a² + std_b²)` — the splits are shared, which is the entire point of "
          "pairing. `n/N arm better` is printed so \"beat the ceiling on 3 of 5 splits\" is "
          "visible rather than buried in a mean.\n\n")

        if strata:
            W("## The same gap by half size — the tie-breaker\n\n")
            ms = sorted({r["metric"] for r in strata})
            W("| arm | stratum | n perts | " + " | ".join(ms) + " |\n")
            W("|---|---|---|" + "---|" * len(ms) + "\n")
            for a in arms:
                if a == ref:
                    continue
                for _, name in STRATA:
                    sel = [r for r in strata if r["arm"] == a and r["stratum"] == name]
                    if not sel:
                        continue
                    npe = sel[0]["n_perts"]
                    by = {r["metric"]: r for r in sel}
                    W(f"| `{a}` | {name} | {npe} | " + " | ".join(
                        _fmt(by[m]["gap_mean"]) if m in by else "—" for m in ms) + " |\n")
            W("\nA real gap **shrinks** as `m` grows. One that does not is "
              "noise-dominated whichever sign it has.\n\n")
        else:
            W("## Strata\n\nNot computed — no per-perturbation `results.csv` was found. "
              "Without it a near-zero headline cannot be told from noise.\n\n")

        notes = {k: v for r in recs for k, v in r.get("notes", {}).items()}
        if notes:
            W("## Notes carried from the runs\n\n")
            for k, v in sorted(notes.items()):
                W(f"* `{k}` — {v}\n")
            W("\n")

        W("## What is NOT comparable to this table\n\n")
        W("The descriptive full-test-set numbers (sibling directory) are **not** "
          "comparable, for three reasons:\n\n")
        W("1. `overlap_at_N` is evaluated at a different `N` — half the cells means fewer "
          "FDR-significant genes.\n")
        W("2. `discrimination_score_l2` is a rank within a different cohort "
          f"({recs[0]['n_perts']} here vs 621 there).\n")
        W("3. The descriptive run shares one control pseudobulk between the real and the "
          "predicted side, so its control noise cancels and its `pearson_delta` is "
          "structurally higher.\n\n")
        W("Only within-protocol, same-split comparisons are valid.\n")


if __name__ == "__main__":
    raise SystemExit(main())
