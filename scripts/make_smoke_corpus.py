#!/usr/bin/env python
"""Generate a tiny synthetic corpus for the end-to-end smoke test.

Matches the real pretraining corpus's on-disk schema exactly -- dense float32
`X`, `obs/original_dataset_idx`, `obs/original_sum`, `var/gene_symbol` -- so
the smoke run exercises the same reader code paths, just at 1/300th the width
and 1/350th the height.

    python scripts/make_smoke_corpus.py --out /tmp/occam_smoke.h5ad

Counts are drawn from a negative binomial with a log-normal per-gene mean, and
a handful of genes are forced to zero everywhere, so both the NB loss and the
structural-zero handling see something realistic.
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", type=int, default=4096)
    ap.add_argument("--genes", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dead-genes", type=int, default=24,
                    help="genes held at exactly zero, mimicking structural zeros")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n, g = args.cells, args.genes

    # Log-normal gene means, negative-binomial counts.
    gene_mean = rng.lognormal(mean=0.2, sigma=1.3, size=g).astype(np.float64)
    depth = rng.lognormal(mean=0.0, sigma=0.35, size=n)
    theta = 4.0

    X = np.empty((n, g), dtype=np.float32)
    block = 512
    for start in range(0, n, block):
        stop = min(start + block, n)
        mu = np.outer(depth[start:stop], gene_mean)
        p = theta / (theta + mu)
        X[start:stop] = rng.negative_binomial(theta, p).astype(np.float32)

    if args.dead_genes > 0:
        X[:, rng.choice(g, size=min(args.dead_genes, g), replace=False)] = 0.0

    original_sum = X.sum(axis=1).astype(np.float32)
    dataset_idx = rng.integers(0, 3, size=n).astype(np.int8)
    genes = np.array([f"SMOKE{i:05d}" for i in range(g)], dtype=object)
    obs_index = np.array([str(i) for i in range(n)], dtype=object)

    str_dt = h5py.special_dtype(vlen=str)
    with h5py.File(args.out, "w") as f:
        f.attrs["encoding-type"] = "anndata"
        f.attrs["encoding-version"] = "0.1.0"
        f.create_dataset("X", data=X, chunks=(min(512, n), g), compression=None)

        obs = f.create_group("obs")
        obs.attrs["_index"] = "_index"
        obs.attrs["column-order"] = np.array(
            ["original_dataset_idx", "original_sum"], dtype=object).astype(str_dt)
        obs.attrs["encoding-type"] = "dataframe"
        obs.attrs["encoding-version"] = "0.2.0"
        obs.create_dataset("_index", data=obs_index.astype(str_dt), dtype=str_dt)
        obs.create_dataset("original_dataset_idx", data=dataset_idx)
        obs.create_dataset("original_sum", data=original_sum)

        var = f.create_group("var")
        var.attrs["_index"] = "_index"
        var.attrs["column-order"] = np.array(["gene_symbol"], dtype=object).astype(str_dt)
        var.attrs["encoding-type"] = "dataframe"
        var.attrs["encoding-version"] = "0.2.0"
        var.create_dataset("_index", data=np.array(
            [str(i) for i in range(g)], dtype=object).astype(str_dt), dtype=str_dt)
        var.create_dataset("gene_symbol", data=genes.astype(str_dt), dtype=str_dt)

        for name in ("layers", "obsm", "varm", "obsp", "varp", "uns"):
            f.create_group(name)

    nnz = float((X > 0).mean())
    print(f"wrote {args.out}")
    print(f"  shape        : {n} x {g}")
    print(f"  nonzero      : {nnz:.1%}")
    print(f"  median depth : {np.median(original_sum):.0f}")
    print(f"  max count    : {X.max():.0f}")
    print(f"  dead genes   : {int((X.max(axis=0) == 0).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
