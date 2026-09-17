#!/usr/bin/env python
"""Phase A for the bridge GATE: a fake competitor that returns its own input.

WHAT THIS ANSWERS
-----------------
Overall goal: predict a perturbation's effect in a context where that perturbation was never
measured, from a single control cell.

This script's question: does the two-phase competitor bridge add signal or bias of its own?
A competitor's score is only interpretable if the pipe it travels through is inert.

Varied: nothing. Held fixed: everything. The "model" is the IDENTITY function on the drawn
half-B control cells, so the exported matrix IS the `ctrlcell` baseline -- routed the long way
round, through `load_panel_split` -> the arm's training space -> `to_eval_space` ->
`assert_same_space` -> `pred.npy`, then `scripts/eval_competitor.py` -> `protocol.run_split`.

WHAT EACH OUTCOME MEANS
  * scoring `pred.npy` reproduces the directly-measured `ctrlcell` row -> the bridge is inert
  * it does not -> the bridge moves the number, and every competitor measured through it
    would be uninterpretable. Fix before any training runs.

⚠️ The round trip is deliberately NOT short-circuited. The control cells are converted
eval -> 10,000 on the way in and 10,000 -> eval on the way out. That double conversion is
the thing under test: if the analytic map were wrong, this arm would not land on `ctrlcell`.

TARGET, from docs/rhj_baseline_results.md (`ctrlcell`, uncalibrated, all 5 splits):
    pearson_delta -0.0018 · overlap_at_N 0.0000 · discrimination_score_l1 0.5038

    PYTHONPATH=src:competitors/shared python scripts/export_gate_ctrlcell.py --split 0
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("src", os.path.join("competitors", "shared")):
    _p = os.path.join(REPO, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

ARM = "_gate_ctrlcell"
PANEL_DIR = os.path.join(REPO, "data", "benchmarks", "rhj_fewshot", "splithalf")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--panel-dir", default=PANEL_DIR)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args(argv)

    from export_predictions import TRAIN_TARGET_SUM, export_split, load_panel_split

    split = load_panel_split(args.panel_dir, args.split, target_sum=TRAIN_TARGET_SUM)
    out_dir = args.out_dir or os.path.join(
        REPO, "runs", "competitors", ARM, f"s{args.seed}", f"split_{args.split}")

    def predict(ctrl, pert_names):
        """The identity. `pert_names` is accepted and ignored -- that is the point."""
        del pert_names
        return np.asarray(ctrl, dtype=np.float32)

    export_split(predict, split, out_dir, arm=ARM,
                 from_target_sum=TRAIN_TARGET_SUM, from_log1p=True,
                 extra_manifest={
                     "gate": True,
                     "model": "identity on the drawn half-B control cells",
                     "expected": {"pearson_delta": -0.0018, "overlap_at_N": 0.0,
                                  "discrimination_score_l1": 0.5038,
                                  "source": "docs/rhj_baseline_results.md, arm ctrlcell"},
                 })
    print(f"\n[gate] exported. Now score it:\n"
          f"  sbatch --qos=hooke-predict --wckey=hooke-predict "
          f"scripts/slurm/rhj_competitor_gate.sbatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
