#!/usr/bin/env python
"""Build a tiny stand-in for the rhj_fewshot corpus, for smoke tests.

    python scripts/make_rhj_smoke.py --out /tmp/rhj_smoke

Three h5ads with the same obs/var schema and cell-line structure as the real files, a real
subset of the frozen gene axis (so every gene is in the encoder vocabulary), and real
perturbation symbols (so they resolve in the perturbation graph -- an unresolvable symbol is
dropped silently on the streaming path, which would make a smoke test pass for the wrong
reason). The real builder then runs over them, so the artifact format is exercised too.

Cheap enough to run on CPU in minutes, which is the point: the failures it is meant to catch
-- an inert `film_mean_free`, a gene mask that differs per context, an exclusion that does not
bite, `source_weight` + `nb` raising -- all appear in the first epoch.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/tmp/rhj_smoke")
    ap.add_argument("--n-genes", type=int, default=96)
    ap.add_argument("--n-perts", type=int, default=45, help="perturbations per file")
    ap.add_argument("--cells-per-pert", type=int, default=6)
    ap.add_argument("--n-controls", type=int, default=60)
    ap.add_argument("--n-val", type=int, default=5)
    ap.add_argument("--n-test", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import anndata as ad
    import numpy as np
    import pandas as pd

    bench = os.path.join(REPO, "data", "benchmarks", "rhj_fewshot")
    if not os.path.isfile(os.path.join(bench, "gene_axis.csv")):
        print("build the real benchmark first (scripts/build_rhj_fewshot.py)", file=sys.stderr)
        return 2

    # real genes off the frozen axis -> guaranteed in the encoder vocabulary
    rows = open(os.path.join(bench, "gene_axis.csv")).read().splitlines()[1:]
    axis = [r.split(",") for r in rows][: args.n_genes]
    ens = [a[0] for a in axis]
    syms = [a[1] for a in axis]

    # real perturbation symbols -> guaranteed to resolve in the graph
    with open(os.path.join(bench, "splits.json")) as fh:
        split = json.load(fh)
    pool = sorted(split["train"])
    if len(pool) < args.n_perts:
        print(f"only {len(pool)} eligible perturbations available", file=sys.stderr)
        return 2
    rng = np.random.default_rng(args.seed)
    perts = sorted(rng.choice(pool, size=args.n_perts, replace=False).tolist())

    os.makedirs(args.out, exist_ok=True)
    lines = {"nadig_2024_hepg2.h5ad": "HEPG2",
             "nadig_2024_jurkat.h5ad": "jurkat",
             "replogle_2022_rpe1_e.h5ad": "RPE1"}
    written = []
    for i, (name, line) in enumerate(sorted(lines.items())):
        labels = ([p for p in perts for _ in range(args.cells_per_pert)]
                  + ["non-targeting"] * args.n_controls)
        n = len(labels)
        # a per-line offset so the contexts are not identical, and a per-perturbation shift
        # so there is something to learn
        base = rng.negative_binomial(5, 0.35, size=(n, len(ens))).astype("float32")
        shift = {p: rng.normal(0, 0.6, size=len(ens)) for p in perts}
        for r, lab in enumerate(labels):
            if lab != "non-targeting":
                base[r] = np.clip(base[r] + shift[lab] + 0.4 * i, 0, None)
        obs = pd.DataFrame({
            "gene_name": pd.Categorical(labels),
            "is_control": np.array([l == "non-targeting" for l in labels]),
            "cell_line": pd.Categorical([line] * n),
            "dataset_batch_num": pd.Categorical(
                [f"{line}-{1 + (r % 4)}" for r in range(n)]),
        }, index=[f"{line}_{r}" for r in range(n)])
        var = pd.DataFrame({"ensembl_gene_id": ens, "gene_name": syms},
                           index=[str(j) for j in range(len(ens))])
        path = os.path.join(args.out, name)
        ad.AnnData(X=base, obs=obs, var=var).write_h5ad(path)
        written.append(path)
        print(f"  {name:32s} {line:7s} {n:5d} cells x {len(ens)} genes")

    # the held-out lists, in the shape the real builder consumes
    req = {"_": "smoke corpus", "n_val": args.n_val, "n_test": args.n_test,
           "val": perts[: args.n_val],
           "test": perts[args.n_val: args.n_val + args.n_test]}
    bench_out = os.path.join(args.out, "benchmark")
    os.makedirs(bench_out, exist_ok=True)
    with open(os.path.join(bench_out, "REQUESTED.json"), "w") as fh:
        json.dump(req, fh, indent=1)

    cmd = [sys.executable, os.path.join(REPO, "scripts", "build_rhj_fewshot.py"),
           "--sources", *written,
           "--splits-source", os.path.join(bench_out, "REQUESTED.json"),
           "--out-dir", bench_out, "--benchmark-id", "rhj_smoke",
           "--expect-val", str(args.n_val), "--expect-test", str(args.n_test),
           "--expect-train", str(args.n_perts - args.n_val - args.n_test)]
    print("\n" + " ".join(cmd) + "\n")
    rc = subprocess.call(cmd)
    if rc:
        return rc
    print(f"\nsmoke corpus: {args.out}\nsmoke benchmark: {bench_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
