#!/usr/bin/env python
"""The `norman_combo` results table: every arm, on the two scored strata, all four metrics.

WHAT WE ARE TRYING TO ACHIEVE OVERALL
    Predict a perturbation's effect in a context where that exact perturbation was never
    measured. Here the form is COMPOSITION: train on single-gene perturbations, predict PAIRS.

WHAT THIS REPORT ANSWERS
    On each stratum, how does each arm compare with `sum_of_singles` — the additivity baseline,
    which is ALSO our architecture's own composition operator (`model.forward` sums the
    perturbations' graph embeddings)? A model at or below it has learned nothing about
    interaction.

WHAT IS VARIED / HELD FIXED
    Varied: the arm, and the stratum. Held fixed: the cohort, the 8,000-gene axis, the 5
    seeded split-halves, and the space (library-norm 2,000 → log1p over the full 33,694-gene
    row sum) — so `mae` is comparable between every baseline, our model and every competitor.

HOW IT AGGREGATES
    cell-eval scores every cohort pair once per arm and writes `results.csv` (one row per
    perturbation). The strata are applied HERE, by averaging those per-pair rows over each
    stratum's membership from `splits_manifest.json`. Nothing is re-scored per stratum, so the
    strata cost nothing and cannot drift from the scored numbers.
    ⚠️ Panel-level metrics (`discrimination_score_*` ranks over the whole cohort) are NOT
    re-averaged over a subset — `PANEL_LEVEL_METRICS` excludes them, because a rank computed
    over 111 pairs does not become a rank over 36 by taking a mean.

    python scripts/norman/norman_report.py [--out docs/norman_results.md]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

M = ("pearson_delta", "mae", "overlap_at_N", "discrimination_score_l1")
SCORED = ("additive", "nonadditive")
ORDER = ("sum_of_singles", "meanpert", "ctrlmean", "ctrlcell", "experimental")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs/norman_combo/reference")
    ap.add_argument("--panel-dir", default="data/benchmarks/norman_combo/splithalf")
    ap.add_argument("--also-all", action="store_true",
                    help="include the 111-pair `all` panel as context (not the headline)")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    from occam.vcell.panel.run import _aggregate_subset, _read_results_csv

    with open(os.path.join(a.panel_dir, "splits_manifest.json")) as fh:
        man = json.load(fh)
    panels = {k: set(v["pairs"]) for k, v in man["panels"].items()}
    wanted = list(SCORED) + (["all"] if a.also_all else [])

    # {arm: {panel: {metric: [per-split values]}}}
    acc: dict = {}
    n_splits = 0
    for sd in sorted(glob.glob(os.path.join(a.runs, "split_*"))):
        found = False
        for arm_dir in sorted(glob.glob(os.path.join(sd, "celleval_*"))):
            arm = os.path.basename(arm_dir)[len("celleval_"):]
            per = _read_results_csv(os.path.join(arm_dir, "results.csv"))
            if not per:
                continue
            found = True
            for pname in wanted:
                agg = _aggregate_subset(per, panels[pname])
                if not agg:
                    continue
                for m in M:
                    if m in agg:
                        acc.setdefault(arm, {}).setdefault(pname, {}).setdefault(
                            m, []).append(agg[m])
                acc[arm][pname].setdefault("n_pairs", []).append(agg["n_perturbations"])
        n_splits += int(found)

    if not acc:
        raise SystemExit(f"no scored arms under {a.runs} — has the reference pass run?")

    lines = []
    lines.append("# `norman_combo` — results\n")
    lines.append("**Objective.** Predict a perturbation's effect in a context where that exact "
                 "perturbation was never measured. Here: **train on single-gene perturbations, "
                 "predict PAIRS.**\n")
    lines.append("**Varied:** the arm. **Held fixed:** the 111-pair cohort, the 8,000-gene "
                 "axis, the 5 seeded split-halves, and the space (library-norm 2,000 → log1p, "
                 "row sum over all 33,694 source genes) — so `mae` is comparable between every "
                 "baseline, our model and every competitor.\n")
    lines.append("**Directions:** `pearson_delta` ↑ · `mae` **↓ (it is an ERROR)** · "
                 "`overlap_at_N` ↑ · `discrimination_score_l1` ↑ (chance 0.5).\n")
    lines.append("⚠️ **`sum_of_singles` is the baseline that matters**: `ctrl + Δ_A + Δ_B` from "
                 "the measured single arms. It is our architecture's own composition operator "
                 "in closed form, so **an arm at or below it has learned nothing about "
                 "interaction.**\n")
    lines.append("⚠️ **The strata are drawn from `sum_of_singles`'s own residual**, so its "
                 "score on `nonadditive` is low BY CONSTRUCTION. Model-vs-model across strata "
                 "is unaffected; the defensible claim is a margin on `additive`, where the "
                 "baseline is near its best.\n")
    lines.append(f"Splits aggregated: **{n_splits}**. `mean ± sd` over splits.\n")

    for pname in wanted:
        n = man["panels"][pname]["n_pairs"]
        lines.append(f"## `{pname}` — {n} pairs\n")
        lines.append("| arm | splits | " + " | ".join(
            f"`{m}`" + (" ↓" if m == "mae" else " ↑") for m in M) + " |")
        lines.append("|---|---|" + "---|" * len(M))
        arms = [x for x in ORDER if x in acc] + sorted(set(acc) - set(ORDER))
        for arm in arms:
            d = acc[arm].get(pname)
            if not d:
                continue
            cells = []
            for m in M:
                v = d.get(m, [])
                if not v:
                    cells.append("—")
                elif len(v) > 1:
                    cells.append(f"**{np.mean(v):.4f}** ± {np.std(v, ddof=1):.4f}")
                else:
                    cells.append(f"**{np.mean(v):.4f}**")
            star = " ⭐" if arm == "sum_of_singles" else ""
            lines.append(f"| `{arm}`{star} | {len(d.get(M[0], []))} | " + " | ".join(cells) + " |")
        lines.append("")

    text = "\n".join(lines) + "\n"
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
        print(f"wrote {a.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
