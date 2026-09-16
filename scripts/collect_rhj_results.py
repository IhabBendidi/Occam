#!/usr/bin/env python
"""Collect every declared rhj_fewshot arm -- including the ones that did not run.

    python scripts/collect_rhj_results.py
    python scripts/collect_rhj_results.py --out docs/rhj_fewshot_results.md

Walks `runs/rhj_fewshot_*/ledger.json` and joins against `configs/rhj_fewshot_grid.tsv`,
emitting ONE ROW PER DECLARED ARM. An arm with no run directory prints `MISSING`, which is
the whole point: a table built from whatever happens to be on disk cannot show you the arm
that never started.

Also cross-checks two things that would otherwise pass unnoticed:

* **config-hash uniqueness.** Two arms sharing a hash means two array tasks trained the same
  configuration, and the grid is quietly smaller than it claims.
* **benchmark identity.** Every arm must record the same `benchmark_sha256` and
  `gene_axis_sha256`, or the arms were not trained on the same benchmark and are not
  comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_grid(path: str) -> List[Dict[str, str]]:
    rows, cols = [], None
    for line in open(path):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if cols is None:
            cols = parts
            continue
        rows.append(dict(zip(cols, parts)))
    return rows


def _fmt(v, width=9, nd=4):
    if v is None:
        return " " * (width - 1) + "-"
    if isinstance(v, float):
        return f"{v:{width}.{nd}f}"
    return f"{str(v):>{width}}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default="configs/rhj_fewshot_grid.tsv")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default=None, help="also write a markdown table here")
    args = ap.parse_args(argv)

    grid = read_grid(os.path.join(REPO, args.grid))
    runs_root = os.path.join(REPO, args.runs)

    records = []
    for arm in grid:
        d = os.path.join(runs_root, f"rhj_fewshot_{arm['arm_id']}__{arm['tag']}")
        rec = {**arm, "run_dir": os.path.relpath(d, REPO) if os.path.isdir(d) else None,
               "status": "MISSING", "best_metric": None, "epochs": None,
               "config_hash": None, "benchmark_sha256": None, "axis_sha256": None,
               "error": None, "qos": None}
        led = os.path.join(d, "ledger.json")
        if os.path.isfile(led):
            with open(led) as fh:
                data = json.load(fh)
            st = (data.get("stages") or {}).get("vcell") or {}
            rec["status"] = st.get("status", "unknown")
            rec["config_hash"] = st.get("config_hash")
            metrics = st.get("metrics") or {}
            rec["best_metric"] = metrics.get("best_metric")
            rec["epochs"] = metrics.get("epochs")
            extra = st.get("extra") or {}
            rec["benchmark_sha256"] = extra.get("benchmark_sha256")
            rec["axis_sha256"] = extra.get("gene_axis_sha256")
            if st.get("status") == "failed":
                rec["error"] = (st.get("error") or "").strip().splitlines()[-1:] or None
        sj = os.path.join(d, "slurm_job.json")
        if os.path.isfile(sj):
            try:
                rec["qos"] = json.load(open(sj)).get("qos")
            except ValueError:
                pass
        records.append(rec)

    # ---- the table -------------------------------------------------------
    hdr = (f"{'arm':5s} {'kind':7s} {'film_mode':20s} {'fh':>4s} {'fmf':>4s} "
           f"{'loss':10s} {'status':10s} {'best_metric':>12s} {'ep':>4s} {'cfg':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for r in records:
        print(f"{r['arm_id']:5s} {r['kind']:7s} {r['film_mode']:20s} "
              f"{r['film_hidden_dim']:>4s} {r['film_mean_free']:>4s} {r['loss_type']:10s} "
              f"{r['status']:10s} {_fmt(r['best_metric'], 12)} "
              f"{_fmt(r['epochs'], 4, 0)} {(r['config_hash'] or '-')[:8]:>10s}")

    n = len(records)
    done = [r for r in records if r["status"] == "completed"]
    missing = [r for r in records if r["status"] == "MISSING"]
    failed = [r for r in records if r["status"] == "failed"]
    print(f"\n{len(done)}/{n} completed · {len(failed)} failed · {len(missing)} missing")

    problems = []
    if missing:
        problems.append(f"{len(missing)} arm(s) never ran: "
                        f"{[r['arm_id'] for r in missing]}")
    for r in failed:
        problems.append(f"{r['arm_id']} failed: {r['error']}")

    hashes = {}
    for r in done:
        hashes.setdefault(r["config_hash"], []).append(r["arm_id"])
    dupes = {h: a for h, a in hashes.items() if len(a) > 1}
    if dupes:
        problems.append(f"config-hash collisions -- these arms trained the SAME config: "
                        f"{dupes}")

    benches = {r["benchmark_sha256"] for r in done if r["benchmark_sha256"]}
    axes = {r["axis_sha256"] for r in done if r["axis_sha256"]}
    if len(benches) > 1:
        problems.append(f"arms trained on {len(benches)} DIFFERENT benchmarks: {benches}")
    if len(axes) > 1:
        problems.append(f"arms trained on {len(axes)} DIFFERENT gene axes: {axes}")

    if problems:
        print("\nPROBLEMS")
        for p in problems:
            print(f"  - {p}")
    elif done:
        print(f"  all completed arms share benchmark {list(benches)[0][:16] if benches else '?'}"
              f" and axis {list(axes)[0][:16] if axes else '?'}")

    if args.out:
        out = os.path.join(REPO, args.out) if not os.path.isabs(args.out) else args.out
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            fh.write("# rhj_fewshot ablation grid — results\n\n")
            fh.write(f"{len(done)}/{n} completed · {len(failed)} failed · "
                     f"{len(missing)} missing.\n\n")
            fh.write("⚠️ Read `pearson_delta` and `discrimination_score_l2` from the "
                     "split-half protocol, not `best_metric` below: `best_metric` is the "
                     "group-mean validation `pearson_delta` used for CHECKPOINT SELECTION "
                     "on the 103 HepG2 validation perturbations. It is not the benchmark "
                     "result and is not comparable to any published number.\n\n")
            fh.write("| arm | kind | film_mode | fh | fmf | loss | status | "
                     "val pearson_delta (group-mean) | epochs |\n")
            fh.write("|---|---|---|---|---|---|---|---|---|\n")
            for r in records:
                bm = "—" if r["best_metric"] is None else f"{r['best_metric']:.4f}"
                fh.write(f"| `{r['arm_id']}` | {r['kind']} | `{r['film_mode']}` | "
                         f"{r['film_hidden_dim']} | {r['film_mean_free']} | "
                         f"`{r['loss_type']}` | {r['status']} | {bm} | "
                         f"{r['epochs'] or '—'} |\n")
            if problems:
                fh.write("\n## Problems\n\n")
                for p in problems:
                    fh.write(f"- {p}\n")
        print(f"\nwrote {out}")

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
