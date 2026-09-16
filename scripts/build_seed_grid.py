#!/usr/bin/env python
"""Emit the seed-replication grid for `rhj_fewshot`. Step 15 of the plan.

WHAT THIS ANSWERS
-----------------
Overall goal: predict a perturbation's effect in a context where that perturbation was never
measured, from a single control cell.

The 26-arm grid runs one seed per arm. That resolves `film_mode`-sized effects and nothing
smaller: the BatchNorm-contamination noise floor alone was ±0.007 in the research tree, and
the top-2 arms of a 24-cell factorial will differ by less than that more often than not.

**This grid's question: is the winning arm's margin larger than seed noise?**

Varied: the seed, and nothing else. Held fixed: everything — the benchmark artifact, the
gene axis, the FiLM configuration of each selected arm, 30 epochs, `num_workers` 8.

What each outcome means, pre-registered:
  * margin >> seed std  -> the arm ordering is real and can be reported as a ranking.
  * margin ~ seed std   -> the ablation resolved the AXIS (e.g. `film_mode`) and not the
    winner. Report the axis effect; do not name a best arm.
  * one seed diverges   -> the recipe is unstable at this estimand, which is itself the
    finding. `grad_norm_max` and `n_nonfinite_batches` are logged per arm for this.

THE DESIGN, AND WHAT "+9" MEANT
-------------------------------
3 arms (the top 2 by validation, plus the no-FiLM additive anchor) x 3 seeds {42, 43, 44} =
**9 data points**. The seed-42 runs already exist in the 26-arm grid, so this file declares
the **6 new jobs** and the analysis joins them against those three. Re-running seed 42 would
hit the same deterministic run directory and the ledger would refuse it, by design.

⚠️ **The selection criterion is not the benchmark result.** Arms are ranked here on
`best_metric` — the group-mean validation `pearson_delta` on the 103 HepG2 validation
perturbations, which is what checkpoint selection uses. The benchmark result is the
split-half protocol, which costs hours per checkpoint and cannot rank 26 arms. So this is
"replicate the arms that looked best on val", and the replication is then MEASURED on the
protocol. Selecting on val and reporting on the protocol is the honest order; selecting on
the protocol and reporting the same numbers would be selection on the test set.

    python scripts/build_seed_grid.py                       # writes configs/rhj_seeds_grid.tsv
    GRID=configs/rhj_seeds_grid.tsv sbatch --qos=vco --wckey=vco \
        --array=0-5%3 scripts/slurm/rhj_fewshot_grid.sbatch
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Dict, List

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SEEDS = (43, 44)          # 42 already exists in the 26-arm grid
HEADER = ("arm_id", "kind", "tag", "interaction_type", "film_mode", "film_hidden_dim",
          "film_mean_free", "loss_type", "seed")


def _collect():
    """`collect_rhj_results.read_grid`, imported rather than re-implemented."""
    path = os.path.join(REPO, "scripts", "collect_rhj_results.py")
    spec = importlib.util.spec_from_file_location("collect_rhj_results", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_status(grid: List[Dict[str, str]], runs_root: str) -> List[Dict]:
    """Each declared arm with its ledger status and validation metric."""
    out = []
    for arm in grid:
        d = os.path.join(runs_root, f"rhj_fewshot_{arm['arm_id']}__{arm['tag']}")
        rec = {**arm, "status": "MISSING", "best_metric": None, "run_dir": d}
        led = os.path.join(d, "ledger.json")
        if os.path.isfile(led):
            st = (json.load(open(led)).get("stages") or {}).get("vcell") or {}
            rec["status"] = st.get("status", "unknown")
            rec["best_metric"] = (st.get("metrics") or {}).get("best_metric")
        out.append(rec)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default="configs/rhj_fewshot_grid.tsv")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="configs/rhj_seeds_grid.tsv")
    ap.add_argument("--n-top", type=int, default=2,
                    help="how many grid arms to replicate, ranked on VALIDATION")
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                    help="the ADDITIONAL seeds; 42 already exists in the 26-arm grid")
    ap.add_argument("--arms", default="",
                    help="comma-separated arm ids, overriding the ranking entirely")
    ap.add_argument("--allow-partial", action="store_true",
                    help="rank on an unfinished grid. 'top-2 of 13' is not 'top-2'.")
    args = ap.parse_args(argv)

    mod = _collect()
    grid = mod.read_grid(os.path.join(REPO, args.grid))
    recs = read_status(grid, os.path.join(REPO, args.runs))
    done = [r for r in recs if r["status"] == "completed" and r["best_metric"] is not None]
    print(f"[seeds] {len(done)}/{len(recs)} arm(s) completed with a validation metric")

    if len(done) != len(recs) and not args.allow_partial:
        missing = [r["arm_id"] for r in recs if r["status"] != "completed"]
        raise SystemExit(
            f"the grid is not finished: {len(missing)} arm(s) are not completed "
            f"({missing[:8]}{' ...' if len(missing) > 8 else ''}).\n"
            f"Ranking now would pick the top-2 of whatever happened to finish first, "
            f"which is a different selection than the one this replication is for. Wait, "
            f"or pass --arms explicitly, or --allow-partial and say so in the writeup.")

    anchors = [r for r in recs if r["kind"] == "anchor"]
    if args.arms:
        want = [a.strip() for a in args.arms.split(",") if a.strip()]
        by_id = {r["arm_id"]: r for r in recs}
        unknown = [a for a in want if a not in by_id]
        if unknown:
            raise SystemExit(f"unknown arm id(s) {unknown}")
        chosen = [by_id[a] for a in want]
        print(f"[seeds] arms given explicitly: {[r['arm_id'] for r in chosen]}")
    else:
        ranked = sorted([r for r in done if r["kind"] == "grid"],
                        key=lambda r: -float(r["best_metric"]))
        if len(ranked) < args.n_top:
            raise SystemExit(f"only {len(ranked)} completed grid arm(s); need "
                             f"{args.n_top}")
        chosen = ranked[:args.n_top]
        print("[seeds] ranked on val group-mean pearson_delta "
              "(CHECKPOINT SELECTION metric, NOT the benchmark result):")
        for i, r in enumerate(ranked[:max(args.n_top + 3, 5)]):
            mark = "  <- replicating" if i < args.n_top else ""
            print(f"  {i + 1:2d}. {r['arm_id']}  {r['best_metric']:.4f}  "
                  f"{r['tag']}{mark}")
        if len(ranked) > args.n_top:
            margin = float(ranked[args.n_top - 1]["best_metric"]) - \
                float(ranked[args.n_top]["best_metric"])
            print(f"[seeds] the margin at the cut is {margin:.4f}. If that is below the "
                  f"seed std this replication measures, the ranking itself is not "
                  f"resolved — which is the point of running it.")

        # The no-FiLM anchor always joins: without a replicated anchor the replication can
        # say "these two arms are stable" but not "FiLM helps by more than seed noise".
        done_anchor = [r for r in anchors if r["status"] == "completed"]
        if not done_anchor and not args.allow_partial:
            raise SystemExit("no completed no-FiLM anchor. Replicating the winners without "
                             "it cannot answer whether FiLM helps by more than seed noise.")
        if done_anchor:
            best_anchor = max(done_anchor, key=lambda r: float(r["best_metric"] or -1e9))
            chosen = chosen + [best_anchor]
            print(f"[seeds] + the no-FiLM anchor {best_anchor['arm_id']} "
                  f"({best_anchor['best_metric']:.4f})")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if 42 in seeds:
        raise SystemExit("seed 42 already exists in the 26-arm grid and its run directory "
                         "is deterministic, so the ledger would refuse it. Pass only the "
                         "ADDITIONAL seeds; the analysis joins against the existing runs.")

    rows = []
    for i, (r, seed) in enumerate((r, s) for r in chosen for s in seeds):
        base = r["tag"].rsplit("_s", 1)[0]
        rows.append({
            "arm_id": f"s{i:02d}", "kind": f"seed-{r['arm_id']}",
            "tag": f"{base}_s{seed}",
            "interaction_type": r["interaction_type"], "film_mode": r["film_mode"],
            "film_hidden_dim": r["film_hidden_dim"], "film_mean_free": r["film_mean_free"],
            "loss_type": r["loss_type"], "seed": str(seed),
        })

    tags = [r["tag"] for r in rows]
    if len(set(tags)) != len(tags):
        raise SystemExit(f"duplicate tags: {sorted(t for t in tags if tags.count(t) > 1)}")

    out = os.path.join(REPO, args.out)
    with open(out, "w") as fh:
        fh.write("# rhj_fewshot SEED REPLICATION. Generated by "
                 "scripts/build_seed_grid.py -- do not hand-edit.\n")
        fh.write(f"# Replicating {[r['arm_id'] for r in chosen]} at seeds {seeds}; the "
                 f"seed-42 runs already exist in configs/rhj_fewshot_grid.tsv, so the\n")
        fh.write(f"# analysis is {len(chosen)} arms x {len(seeds) + 1} seeds = "
                 f"{len(chosen) * (len(seeds) + 1)} points from {len(rows)} new jobs.\n")
        fh.write("# Arms were ranked on the VALIDATION group-mean pearson_delta, which is "
                 "the checkpoint-selection metric and NOT the benchmark result.\n")
        fh.write("# Launch:\n")
        fh.write(f"#   GRID={args.out} sbatch --qos=vco --wckey=vco "
                 f"--array=0-{len(rows) - 1}%3 scripts/slurm/rhj_fewshot_grid.sbatch\n")
        fh.write("\t".join(HEADER) + "\n")
        for r in rows:
            fh.write("\t".join(r[k] for k in HEADER) + "\n")

    print(f"\n[seeds] wrote {args.out} — {len(rows)} new job(s), "
          f"{len(chosen)} arm(s) x {len(seeds)} additional seed(s)")
    print(f"[seeds] launch with:")
    print(f"    GRID={args.out} sbatch --qos=vco --wckey=vco "
          f"--array=0-{len(rows) - 1}%3 scripts/slurm/rhj_fewshot_grid.sbatch")
    print(f"[seeds] then collect with:")
    print(f"    python scripts/collect_rhj_results.py --grid {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
