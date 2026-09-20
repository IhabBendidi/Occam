#!/usr/bin/env python
"""Pick the best checkpoint of a run on ALL FOUR cell-eval metrics, and compare to `best_metric`.

WHAT THIS ANSWERS
    Overall goal: predict a perturbation's effect in a context where that perturbation was
    never measured, from a single control cell.

    This tool's question: does selecting a checkpoint on CELL-EVAL beat selecting it on
    `val_pearson_delta`, the per-cell validation metric the trainer actually uses -- and is it
    even a different checkpoint?

    `pipeline.py` already records that the current criterion "was MEASURED to rank checkpoints
    wrongly on the real panel ... it picked one arm and put the eventual bake-off winner last",
    and says the only reason it is still used is that cell-eval was too slow to run in-loop.
    Per-epoch checkpoints (`--set stages.vcell.optim.save_every_epochs=1`) make the comparison
    possible offline.

SELECTION OVER FOUR METRICS
    `pearson_delta` UP, `mae` DOWN (it is an ERROR), `overlap_at_N` UP,
    `discrimination_score_l1` UP.

    Each metric is min-max normalised ACROSS THE CHECKPOINTS OF ONE RUN, `mae` inverted, then
    averaged. Normalising within the run is deliberate: the question is which epoch is best for
    THIS arm, so the spread that matters is the spread across its own epochs.

    ⚠️ A composite hides trades. The per-metric argmax is printed alongside, so an epoch that
    wins the mean while losing three metrics cannot pass unnoticed.

    python scripts/ckpt_sweep_select.py --run-dir runs/rhj_splithalf/CKPTSWEEP_M03
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

M = ("pearson_delta", "mae", "overlap_at_N", "discrimination_score_l1")
LOWER_BETTER = ("mae",)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="directory holding one subdir per scored checkpoint")
    ap.add_argument("--sub", default="celleval_model_nocal")
    a = ap.parse_args(argv)
    from occam.vcell.panel.celleval_v1 import _read_agg

    rows = []
    for d in sorted(glob.glob(os.path.join(a.run_dir, "*"))):
        f = os.path.join(d, a.sub, "agg_results.csv")
        if not os.path.isfile(f):
            continue
        m = _read_agg(f)
        if not m or any(k not in m for k in M):
            continue
        ep = re.search(r"(\d+)", os.path.basename(d))
        rows.append((int(ep.group(1)) if ep else -1, os.path.basename(d), m))
    if not rows:
        print(f"no scored checkpoints under {a.run_dir}")
        return 1
    rows.sort()

    lo = {k: min(r[2][k] for r in rows) for k in M}
    hi = {k: max(r[2][k] for r in rows) for k in M}
    for _, _, m in rows:
        s = {}
        for k in M:
            rng = hi[k] - lo[k]
            s[k] = 0.5 if rng == 0 else (
                (hi[k] - m[k]) / rng if k in LOWER_BETTER else (m[k] - lo[k]) / rng)
        m["_score"] = sum(s.values()) / len(s)
        m["_norm"] = s

    print(f"{'epoch':>6}  {'pearson_delta':>14}{'mae↓':>9}{'ovl@N':>9}{'disc_l1':>10}"
          f"{'4-metric':>10}")
    print("-" * 60)
    for ep, name, m in rows:
        print(f"{ep:>6}  {m['pearson_delta']:>14.4f}{m['mae']:>9.4f}"
              f"{m['overlap_at_N']:>9.4f}{m['discrimination_score_l1']:>10.4f}"
              f"{m['_score']:>10.3f}")

    best = max(rows, key=lambda r: r[2]["_score"])
    print(f"\n4-METRIC PICK: epoch {best[0]} (composite {best[2]['_score']:.3f})")
    print("per-metric argmax -- a composite hides trades, so these are printed too:")
    for k in M:
        b = (min if k in LOWER_BETTER else max)(rows, key=lambda r: r[2][k])
        print(f"  {k:<26} epoch {b[0]:>3}  {b[2][k]:.4f}"
              + ("   <- same as composite" if b[0] == best[0] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
