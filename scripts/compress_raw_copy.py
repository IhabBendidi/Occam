#!/usr/bin/env python
"""Rewrite a raw-count local copy as gzip-4 CSR with int32 counts -- the convention the other
published h5ad files follow (eval/pert/*: gzip 4; norman_2019_k562: int32 + gzip 4).

Content is unchanged (asserted: same obs, var, and X bit-for-bit after the cast back to float),
so every panel built from it is identical; only the file bytes and sha256 change.

USAGE  python scripts/compress_raw_copy.py <in.h5ad> <out.h5ad>
"""
import sys

import anndata as ad
import numpy as np
import scipy.sparse as sp

src, dst = sys.argv[1], sys.argv[2]
a = ad.read_h5ad(src)
X = sp.csr_matrix(a.X)
if not np.array_equal(X.data, np.round(X.data)):
    raise SystemExit(f"{src}: X is not integer counts; refusing to cast to int32")
if X.data.max() > np.iinfo(np.int32).max:
    raise SystemExit("count exceeds int32")
out = ad.AnnData(X=sp.csr_matrix((X.data.astype(np.int32), X.indices, X.indptr), shape=X.shape),
                 obs=a.obs, var=a.var)
out.write_h5ad(dst, compression="gzip", compression_opts=4)
b = ad.read_h5ad(dst)
Y = sp.csr_matrix(b.X)
assert b.obs.equals(a.obs) and b.var.equals(a.var), "obs/var changed"
assert (Y.astype(np.float32) != X.astype(np.float32)).nnz == 0, "X changed"
print(f"{dst}: {X.shape}, nnz {X.nnz:,}, int32 gzip-4 -- content identical to {src}")
