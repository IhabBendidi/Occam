#!/usr/bin/env python
"""Rewrite an .h5ad with a DENSE X, as the `optim` data loader requires.

    python scripts/finetune/densify_h5ad.py --src corpus.h5ad --dst corpus_dense.h5ad

`occam.data.chunked.OccamChunkedDataset` (`data_mode: optim`) reads `f["X"]` with
`h5py.read_direct`, i.e. a dense dataset; a CSR/CSC `X` fails at the first read with
`'Group' object has no attribute 'shape'`. The training corpus was dense, so this is what
the recipe expects. obs / var / uns are copied unchanged; X is written row block by row
block (never fully in memory), then sampled rows are compared with the source.

Size: n_obs x n_vars x 4 bytes (1.3 M rows x 38 k genes = 200 GB).
"""

from __future__ import annotations

import argparse
import sys


def densify(src: str, dst: str, block: int = 16_384, check_rows: int = 200) -> bool:
    import anndata as ad
    import h5py
    import numpy as np
    import scipy.sparse as sp

    a = ad.read_h5ad(src, backed="r")
    n, g = a.shape
    meta = ad.AnnData(X=sp.csr_matrix((n, g), dtype=np.float32), obs=a.obs.copy(),
                      var=a.var.copy(), uns=dict(a.uns))
    a.file.close()
    meta.write_h5ad(dst)

    with h5py.File(src, "r") as fs, h5py.File(dst, "r+") as fd:
        enc = fs["X"].attrs.get("encoding-type", "array")
        if isinstance(enc, bytes):
            enc = enc.decode()
        del fd["X"]
        X = fd.create_dataset("X", shape=(n, g), dtype="float32")
        X.attrs["encoding-type"] = "array"
        X.attrs["encoding-version"] = "0.2.0"
        if enc == "array":
            for s in range(0, n, block):
                X[s:s + block] = fs["X"][s:s + block].astype(np.float32)
        elif enc == "csr_matrix":
            indptr = fs["X/indptr"]
            for s in range(0, n, block):
                e = min(s + block, n)
                p0, p1 = int(indptr[s]), int(indptr[e])
                blk = sp.csr_matrix((fs["X/data"][p0:p1], fs["X/indices"][p0:p1],
                                     indptr[s:e + 1] - p0), shape=(e - s, g))
                X[s:e] = blk.toarray().astype(np.float32)
        else:
            raise ValueError(f"unsupported X encoding {enc!r}; convert to CSR first")

    b, d = ad.read_h5ad(src, backed="r"), h5py.File(dst, "r")
    rows = np.sort(np.random.default_rng(1).choice(n, min(n, check_rows), replace=False))

    def row(r):
        x = b.X[int(r)]
        return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float32).ravel()

    ok = all(np.array_equal(row(r), d["X"][int(r)]) for r in rows)
    d.close()
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--block", type=int, default=16_384)
    args = ap.parse_args(argv)
    ok = densify(args.src, args.dst, args.block)
    print(f"wrote {args.dst}; sampled rows identical to source: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
