#!/usr/bin/env python
"""Write `files/norman_2019_k562.json` -- the per-source record the competitors read.

WHY THIS EXISTS
    `competitors/shared/occam_data.py::load_occam_benchmark` is the ONE reader all four
    ported competitors use, and it needs a per-source manifest under `<bench>/files/`:
    the path, the sha256 it was frozen against, the perturbation column, the batch
    column, the cell counts, and the per-perturbation cell counts.

    `rhj_fewshot`'s builder writes those. `norman_combo`'s never did, so the reader
    refused the whole benchmark ("no per-file manifest for cell line(s) ... found []")
    and no competitor could be trained on it. Found 2026-09-22 while wiring the
    competitors onto norman_combo.

    Everything here is DERIVED -- from `benchmark.json`'s own `sources` block and from
    the source `.h5ad` -- so the manifest cannot drift from the benchmark it belongs to.
    The sha256 is re-read from the file and asserted against the frozen one rather than
    copied, because the point of the field is to catch the file changing underneath.

USAGE
    python scripts/norman/norman_write_file_manifest.py
    python scripts/norman/norman_write_file_manifest.py --bench-dir /tmp/check_norman
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _sha256(path: str, block: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench-dir", default="data/benchmarks/norman_combo")
    ap.add_argument("--source", default=None,
                    help="the .h5ad; default = the one benchmark.json names, under "
                         "data/eval/pert/")
    ap.add_argument("--pert-col", default="gene_name")
    ap.add_argument("--batch-col", default="batch")
    ap.add_argument("--control-label", default="non-targeting")
    ap.add_argument("--skip-sha256", action="store_true",
                    help="trust benchmark.json's digest instead of re-reading 647 MB")
    args = ap.parse_args()

    import anndata as ad
    import numpy as np

    bench_path = os.path.join(args.bench_dir, "benchmark.json")
    if not os.path.isfile(bench_path):
        print(f"{bench_path} not found; build the benchmark first", file=sys.stderr)
        return 1
    bench = json.load(open(bench_path))
    sources = bench.get("sources") or {}
    if len(sources) != 1:
        print(f"expected exactly one source, got {sorted(sources)}", file=sys.stderr)
        return 1
    name, rec = next(iter(sources.items()))
    src = args.source or os.path.join(REPO, "data", "eval", "pert", name)
    if not os.path.isfile(src):
        print(f"source not found: {src}\ndownload eval/pert/{name} from the Hub "
              f"(docs/hf_release.md)", file=sys.stderr)
        return 1

    if args.skip_sha256:
        sha = str(rec["sha256"])
        print(f"[manifest] sha256 taken from benchmark.json (--skip-sha256): {sha[:16]}…")
    else:
        sha = _sha256(src)
        if sha != rec["sha256"]:
            print(f"REFUSING: {name} has changed since the benchmark was frozen\n"
                  f"  on disk  {sha}\n  frozen   {rec['sha256']}\n"
                  f"The splits refer to rows in the old file; a manifest written now "
                  f"would point the competitors at different data.", file=sys.stderr)
            return 3
        print(f"[manifest] sha256 verified against benchmark.json: {sha[:16]}…")

    a = ad.read_h5ad(src, backed="r")
    if args.pert_col not in a.obs.columns:
        print(f"--pert-col {args.pert_col!r} not in .obs "
              f"({list(a.obs.columns)[:8]})", file=sys.stderr)
        return 1
    labels = a.obs[args.pert_col].astype(str).to_numpy()
    batches = (a.obs[args.batch_col].astype(str).to_numpy()
               if args.batch_col in a.obs.columns else np.array(["1"] * len(labels)))
    is_ctrl = labels == args.control_label
    uniq, counts = np.unique(labels[~is_ctrl], return_counts=True)

    axis_path = os.path.join(args.bench_dir, "gene_axis.csv")
    n_axis = sum(1 for _ in open(axis_path)) - 1 if os.path.isfile(axis_path) else None

    out = {
        "batch_ids": sorted(set(batches.tolist())),
        "bytes": os.path.getsize(src),
        "cell_line": str(rec.get("cell_line", bench.get("cell_line", ""))),
        "mtime": os.path.getmtime(src),
        "n_batches": len(set(batches.tolist())),
        "n_controls": int(is_ctrl.sum()),
        "n_genes_on_axis": n_axis,
        "n_obs": int(a.n_obs),
        "n_perturbed": int((~is_ctrl).sum()),
        "n_vars": int(a.n_vars),
        "name": name,
        "path": os.path.abspath(src),
        "per_pert_cells": {str(k): int(v) for k, v in zip(uniq, counts)},
        # ⚠️ NOT in rhj_fewshot's records, because there it is the default. Norman's
        # perturbation column holds `A+B` for pairs and its gene ids are Ensembl in
        # `gene_id`, so the reader must be told rather than left to assume.
        "pert_col": args.pert_col,
        "batch_col": args.batch_col,
        "control_label": args.control_label,
        "gene_col": "gene_id",
        "sha256": sha,
    }

    fdir = os.path.join(args.bench_dir, "files")
    os.makedirs(fdir, exist_ok=True)
    dst = os.path.join(fdir, os.path.splitext(name)[0] + ".json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"[manifest] wrote {dst}")
    print(f"[manifest]   {out['cell_line']} · {out['n_obs']:,} cells "
          f"({out['n_controls']:,} control, {out['n_perturbed']:,} perturbed) · "
          f"{len(out['per_pert_cells']):,} conditions · {out['n_batches']} batches")
    print(f"[manifest]   pert_col={args.pert_col} batch_col={args.batch_col} "
          f"gene_col=gene_id control_label={args.control_label!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
