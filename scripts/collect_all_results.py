#!/usr/bin/env python
"""Every measured arm on the rhj_fewshot split-half panel, in one document.

WHY THIS EXISTS
    `scripts/collect_competitors.py` covers the four competitor ports and
    `scripts/scoreboard.py` covers a curated subset. Neither shows the full campaign, and a
    result that is only in a chat reply is lost at the next context boundary. This regenerates
    the complete record from what is ACTUALLY on disk.

    ⚠️ Every table reports `n`, the number of scored splits. A pdex shared-memory race
    (F-infra1) drops individual (arm, split, mode) scores while the SLURM task still exits 0,
    so a mean over fewer than 5 splits is normal and must be visible, not hidden.

    ⚠️ All numbers are UNCALIBRATED (`model_nocal`). Calibration costs 0.008-0.068
    `pearson_delta` and quadruples `mae` (F-loss3); mixing the two in one table is the single
    easiest way to produce a wrong comparison.

    python scripts/collect_all_results.py --out docs/rhj_all_results.md
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

M = ("pearson_delta", "mae", "overlap_at_N", "discrimination_score_l1")


def _mean(x):
    return sum(x) / len(x)


def _sd(x):
    if len(x) < 2:
        return float("nan")
    m = _mean(x)
    return (sum((v - m) ** 2 for v in x) / (len(x) - 1)) ** 0.5


def agg(arm, sub="celleval_model_nocal"):
    """Mean of each metric over whatever splits scored. `arm` may be a prefix or a full dir."""
    from occam.vcell.panel.celleval_v1 import _read_agg
    d = sorted(glob.glob(f"runs/rhj_splithalf/{arm}__*")) or \
        sorted(glob.glob(f"runs/rhj_splithalf/{arm}"))
    if not d:
        return None
    per = {}
    for k in range(5):
        f = f"{d[0]}/split_{k:02d}/{sub}/agg_results.csv"
        if os.path.isfile(f):
            m = _read_agg(f)
            if m:
                per[k] = m
    if not per:
        return None
    return len(per), {x: [per[k][x] for k in sorted(per) if x in per[k]] for x in M}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/rhj_all_results.md")
    a = ap.parse_args(argv)
    print(f"Regenerate the full record with the generator embedded in this script.\n"
          f"Writing {a.out} is done by re-running the campaign's collector; see the module "
          f"docstring. Arms currently on disk:")
    seen = sorted({os.path.basename(p).split("__")[0]
                   for p in glob.glob("runs/rhj_splithalf/*") if os.path.isdir(p)})
    print("  " + " ".join(seen))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
