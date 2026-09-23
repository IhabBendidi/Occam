#!/usr/bin/env python
"""The final score table: ONE ROW per (benchmark, panel, model, seed, split), every cell-eval metric.

WHY (user, 2026-09-23): the paper tables are built from every model at 5 seeds x 5 splits and every
baseline at 5 control-draw seeds, with each row saying exactly which seed and split it is.

INPUT  the long export of scripts/export_split_metrics.py (--scope all), which already resolves
       benchmarks (rhj_fewshot, norman_combo, h1_seen, k562_seen, k562_seen_h1set) and the scored
       sub-panels (norman additive / nonadditive, ...). This script only reshapes and labels it.

OUTPUT data/release/final/scores_by_seed_split.csv
         benchmark, panel, model, model_kind (occam | competitor | baseline), variant (e.g. the
         composition mode), seed, split, n_perts, eval_root, then one column per metric
       data/release/final/coverage.csv
         per (benchmark, panel, model): which seeds have all 5 splits -- the table is complete when
         every expected model shows seeds 42-46 x 5 splits

SEED, from the eval root (the only place it is recorded consistently):
  baselines   reference -> 42, reference_sNN -> NN      (the half-B control draw)
  Occam       arm S<NN>_<base> -> NN, model <base>; else the run tag's trailing _s<NN>
  competitor  compet_<arm>[__<bench>]_s<NN> -> NN
Only nocal scores are kept. Withdrawn / archived roots are not in runs/, so they cannot appear.

USAGE  python scripts/export_split_metrics.py --scope all --out data/release/metrics_all
       python scripts/final_table.py [--long data/release/metrics_all/metrics_long.csv]
"""
from __future__ import annotations

import argparse
import os
import re

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINES = {"ctrlcell", "ctrlmean", "experimental", "meanpert", "transfer", "sum_of_singles"}
PANEL_TAGS = ("__norman_combo", "__h1_seen", "__k562_seen_h1set", "__k562_seen")


def label(row) -> pd.Series:
    root = row["eval_root"]
    base = os.path.basename(root)
    scored = row["scored_arm"]
    if scored in BASELINES:
        m = re.match(r"reference(?:_s(\d+))?$", base)
        seed = int(m.group(1)) if (m and m.group(1)) else 42
        return pd.Series(dict(model=scored, model_kind="baseline", variant="", seed=seed))
    if base.startswith("compet_"):
        m = re.match(r"compet_(.+)_s(\d+)$", base)
        arm, seed = (m.group(1), int(m.group(2))) if m else (base[7:], 42)
        for t in PANEL_TAGS:
            arm = arm.replace(t, "")
        return pd.Series(dict(model=arm, model_kind="competitor", variant="", seed=seed))
    arm, _, tag = base.partition("__")
    variant = ""
    if "/eval/" in root:          # norman eval roots: <ARM>__<compose>
        variant, tag = tag, ""
    m = re.match(r"S(\d+)_(.+)$", arm)
    if m:
        seed, model = int(m.group(1)), m.group(2)
    else:
        mt = re.search(r"_s(\d+)$", tag)
        seed, model = (int(mt.group(1)) if mt else 42), arm
    return pd.Series(dict(model=model, model_kind="occam", variant=variant, seed=seed))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--long", default=os.path.join(REPO, "data/release/metrics_all/metrics_long.csv"))
    ap.add_argument("--out-dir", default=os.path.join(REPO, "data/release/final"))
    args = ap.parse_args()

    # keep_default_na=False: the baselines' calibration is the literal "n/a", which pandas would
    # otherwise read as NaN and a nocal filter would silently drop every baseline row
    m = pd.read_csv(args.long, keep_default_na=False, na_values=[""])
    m = m[m["calibration"].isin(["nocal", "n/a"])]
    keys = m[["eval_root", "scored_arm"]].drop_duplicates()
    lab = keys.join(keys.apply(label, axis=1))
    m = m.merge(lab, on=["eval_root", "scored_arm"])
    idx = ["benchmark", "panel", "model", "model_kind", "variant", "seed", "split", "n_perts",
           "eval_root"]
    wide = (m.pivot_table(index=idx, columns="metric", values="value", aggfunc="first")
             .reset_index().sort_values(["benchmark", "panel", "model_kind", "model", "variant",
                                          "seed", "split"]))
    wide.columns.name = None
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "scores_by_seed_split.csv")
    wide.to_csv(out, index=False)

    cov = (wide.groupby(["benchmark", "panel", "model_kind", "model", "variant", "seed"])["split"]
               .nunique().unstack("seed").fillna(0).astype(int))
    cov["complete_seeds"] = (cov == 5).sum(axis=1)
    cov.to_csv(os.path.join(args.out_dir, "coverage.csv"))
    print(f"wrote {out}: {len(wide):,} rows x {wide.shape[1]} columns "
          f"({wide['model'].nunique()} models, {wide['benchmark'].nunique()} benchmarks)")
    print(f"coverage: {os.path.join(args.out_dir, 'coverage.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
