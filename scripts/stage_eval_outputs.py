#!/usr/bin/env python
"""Stage the per-split cell-eval outputs of every competitor and baseline for the Hub.

WHY (user, 2026-09-25). The Hub held per-split metrics for rhj + norman only and no raw cell-eval
outputs, so a reader could not re-derive a competitor's or baseline's number on h1_seen /
k562_seen / k562_seen_h1set, or look at it per perturbation. This copies, for every eval root,
exactly the small files cell-eval writes per split:

    <benchmark>/<root>/split_NN/metrics.json                 every metric, every arm in the root
    <benchmark>/<root>/split_NN/celleval_<arm>/results.csv   per-perturbation metrics
    <benchmark>/<root>/split_NN/celleval_<arm>/agg_results.csv

NOT copied (by decision, 2026-09-25): `pred_de.csv` (~249 GB) and the prediction / target
`.h5ad` (~1.6 TB). Targets rebuild from `benchmarks/`; predictions re-export from checkpoints.

Roots: `compet_*` (competitors, one per arm x seed [x panel]) and `reference*` (baselines, one per
control-draw seed) under runs/{rhj_splithalf, h1_seen, k562_seen, k562_seen_h1set, norman_combo}.
Only nocal scoring exists for these. Output mirrors the run layout, with `rhj_splithalf` named
`rhj_fewshot` (the benchmark id).

USAGE  python scripts/stage_eval_outputs.py [--out data/release/eval_outputs]
       then upload data/release/eval_outputs -> eval_outputs/ (scripts/publish_hf.py stages it)
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIRS = {"rhj_splithalf": "rhj_fewshot", "h1_seen": "h1_seen", "k562_seen": "k562_seen",
              "k562_seen_h1set": "k562_seen_h1set", "norman_combo": "norman_combo"}
PREFIXES = ("compet_", "reference")
KEEP = ("metrics.json", "results.csv", "agg_results.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, "data/release/eval_outputs"))
    args = ap.parse_args()
    if os.path.isdir(args.out):
        shutil.rmtree(args.out)          # a mirror: stale roots must not survive a re-stage
    n_roots = n_files = n_bytes = 0
    for src, bench in BENCH_DIRS.items():
        for root in sorted(glob.glob(os.path.join(REPO, "runs", src, "*"))):
            name = os.path.basename(root)
            if not name.startswith(PREFIXES) or not os.path.isdir(root):
                continue
            files = glob.glob(os.path.join(root, "split_*", "metrics.json")) + \
                glob.glob(os.path.join(root, "split_*", "celleval_*", "*.csv"))
            files = [f for f in files if os.path.basename(f) in KEEP]
            if not files:
                continue
            n_roots += 1
            for f in files:
                rel = os.path.relpath(f, root)
                dst = os.path.join(args.out, bench, name, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(f, dst)
                n_files += 1
                n_bytes += os.path.getsize(f)
    with open(os.path.join(args.out, "README.md"), "w") as fh:
        fh.write(__doc__.split("USAGE")[0].replace("Stage the", "The"))
    print(f"staged {n_roots} roots, {n_files} files, {n_bytes / 1e6:.1f} MB -> "
          f"{os.path.relpath(args.out, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
