#!/usr/bin/env python
"""Pass 1 of the Norman-2019 rebuild: per-gene quality statistics from the raw GEO matrix.

WHY THIS EXISTS
    The benchmark's scoring axis has to be chosen on measured gene quality, not inherited from
    someone else's HVG list. The preprocessed file we were handed carries 5,045 genes chosen
    upstream by unknown criteria (and mislabelled as A549/2016; it is K562/2019 -- see
    docs/norman_benchmark.md). Rebuilding from GSE133344 gives 33,694 genes of raw integer
    counts, of which 29,187 are in Occam's vocabulary, and the axis is then OUR choice with
    OUR reasons recorded.

WHAT IT COMPUTES, in ONE streaming pass over 362,199,631 nonzeros
    per gene:  nnz (detection count), total counts, and the mean AND variance of
               log1p(count / cell_library_size * target_sum) -- i.e. the statistics in the
               EXACT space the model trains and is scored in, not in raw count space
    per cell:  library size and detected-gene count (needed for the above, and for QC)

    Single-pass is possible because the MatrixMarket file is CELL-MAJOR with ascending gene
    index inside each cell (verified on 3,000,000 entries: 0 out-of-order). A cell's library
    size is therefore known by the time its last entry is read, so the normalised statistics
    can be accumulated without a second pass. Chunks are cut at CELL boundaries and the
    partial tail is carried, so no cell is ever normalised by a partial library.

⚠️ Genes are indexed 1..33694 in the file and 0-based here. `var` order is the order of
GSE133344_filtered_genes.tsv.gz and is never re-sorted.

    python scripts/norman/norman_gene_stats.py --out data/external/GSE133344/gene_stats.npz
"""
from __future__ import annotations

import argparse
import gzip
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

MTX = "data/external/GSE133344/GSE133344_filtered_matrix.mtx.gz"
GENES = "data/external/GSE133344/GSE133344_filtered_genes.tsv.gz"
TARGET_SUM = 2000.0          # Occam's space, pinned by the encoder checkpoint


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mtx", default=MTX)
    ap.add_argument("--genes", default=GENES)
    ap.add_argument("--out", default="data/external/GSE133344/gene_stats.npz")
    ap.add_argument("--target-sum", type=float, default=TARGET_SUM)
    ap.add_argument("--chunk", type=int, default=20_000_000, help="nonzeros per read")
    # ⚠️ The axis must be chosen on the cells the BENCHMARK uses, not on all 111,668 barcodes.
    # 20,500 of them fail Norman's own QC (good_coverage & number_of_cells == 1) and are never
    # scored, so letting them set a gene's detection rate would pick the axis on data the
    # benchmark discards. `qc_cells.npz` carries the surviving matrix columns.
    ap.add_argument("--keep-cells", default="data/external/GSE133344/qc_cells.npz",
                    help="npz with a `col` array of 0-based matrix columns to keep; '' = all")
    a = ap.parse_args(argv)

    rows = [l.rstrip("\n").split("\t") for l in gzip.open(a.genes, "rt")]
    ens = np.array([r[0] for r in rows])
    sym = np.array([r[1] for r in rows])
    n_genes = len(ens)

    with gzip.open(a.mtx, "rt") as fh:
        fh.readline(); fh.readline()
        hdr = fh.readline().split()
    G, C, NNZ = int(hdr[0]), int(hdr[1]), int(hdr[2])
    assert G == n_genes, f"header says {G} genes, the genes file has {n_genes}"
    print(f"[stats] {G:,} genes x {C:,} cells, {NNZ:,} nonzeros, target_sum {a.target_sum:g}")

    keep_col = None
    if a.keep_cells:
        kz = np.load(a.keep_cells, allow_pickle=True)
        keep_col = np.zeros(C + 1, dtype=bool)
        keep_col[np.asarray(kz["col"], dtype=np.int64) + 1] = True    # file is 1-based
        n_keep = int(keep_col.sum())
        print(f"[stats] restricting to {n_keep:,} QC cells from {a.keep_cells} "
              f"({100*n_keep/C:.1f}% of barcodes)")
    n_eff = C if keep_col is None else int(keep_col.sum())

    g_nnz = np.zeros(n_genes, dtype=np.int64)
    g_cnt = np.zeros(n_genes, dtype=np.float64)
    g_sum = np.zeros(n_genes, dtype=np.float64)     # sum of log1p-normalised
    g_sq = np.zeros(n_genes, dtype=np.float64)      # sum of squares
    c_lib = np.zeros(C + 1, dtype=np.float64)
    c_nnz = np.zeros(C + 1, dtype=np.int64)

    reader = pd.read_csv(a.mtx, sep=r"\s+", skiprows=3, header=None,
                         names=["g", "c", "v"], dtype={"g": np.int32, "c": np.int32,
                                                       "v": np.float32},
                         chunksize=a.chunk, engine="c", compression="gzip")
    carry = None
    seen = read = 0
    t0 = time.time()
    for ci, chunk in enumerate(reader):
        if carry is not None and len(carry):
            chunk = pd.concat([carry, chunk], ignore_index=True)
        g = chunk["g"].to_numpy(); c = chunk["c"].to_numpy(); v = chunk["v"].to_numpy()
        # cut at the last COMPLETE cell: the final cell in the chunk may continue next chunk
        last = c[-1]
        keep = c < last
        carry = chunk.iloc[np.flatnonzero(~keep)]
        g, c, v = g[keep], c[keep], v[keep]
        if not len(g):
            continue
        lib = np.bincount(c, weights=v, minlength=C + 1)
        nz = np.bincount(c, minlength=C + 1)
        c_lib += lib; c_nnz += nz
        x = np.log1p(v / lib[c] * a.target_sum)
        if keep_col is not None:
            m = keep_col[c]
            g, v, x, c = g[m], v[m], x[m], c[m]
            if not len(g):
                seen += int(m.size - m.sum()) if False else 0
                continue
        g0 = g - 1
        g_nnz += np.bincount(g0, minlength=n_genes).astype(np.int64)
        g_cnt += np.bincount(g0, weights=v, minlength=n_genes)
        g_sum += np.bincount(g0, weights=x, minlength=n_genes)
        g_sq += np.bincount(g0, weights=x * x, minlength=n_genes)
        read += int(keep.sum())
        seen += len(g)
        print(f"  chunk {ci}: {read:,}/{NNZ:,} nonzeros read ({100*read/NNZ:.1f}%), "
              f"{seen:,} kept, {time.time()-t0:.0f}s", flush=True)
    if carry is not None and len(carry):
        g = carry["g"].to_numpy(); c = carry["c"].to_numpy(); v = carry["v"].to_numpy()
        lib = np.bincount(c, weights=v, minlength=C + 1)
        c_lib += lib; c_nnz += np.bincount(c, minlength=C + 1)
        x = np.log1p(v / lib[c] * a.target_sum)
        if keep_col is not None:
            m = keep_col[c]
            g, v, x = g[m], v[m], x[m]
        g0 = g - 1
        g_nnz += np.bincount(g0, minlength=n_genes).astype(np.int64)
        g_cnt += np.bincount(g0, weights=v, minlength=n_genes)
        g_sum += np.bincount(g0, weights=x, minlength=n_genes)
        g_sq += np.bincount(g0, weights=x * x, minlength=n_genes)
        read += len(carry)
        seen += len(g)
    assert read == NNZ, f"read {read:,} nonzeros, header promised {NNZ:,}"

    # zeros count as log1p(0) = 0 in both mean and variance, so the moments are over ALL C cells
    mean = g_sum / n_eff
    var = np.maximum(g_sq / n_eff - mean ** 2, 0.0)
    det = g_nnz / n_eff
    np.savez(a.out, ensembl=ens, symbol=sym, nnz=g_nnz, total_counts=g_cnt,
             mean_log1p=mean, var_log1p=var, detection_rate=det,
             cell_lib=c_lib[1:], cell_nnz=c_nnz[1:], n_cells=np.int64(n_eff),
             n_barcodes=np.int64(C),
             target_sum=np.float64(a.target_sum))
    print(f"\n[stats] wrote {a.out} in {time.time()-t0:.0f}s")
    print(f"  cells: library size median {np.median(c_lib[1:]):.0f} "
          f"(p1 {np.percentile(c_lib[1:],1):.0f}, p99 {np.percentile(c_lib[1:],99):.0f}); "
          f"genes/cell median {int(np.median(c_nnz[1:]))}")
    for thr in (0.0, 0.01, 0.05, 0.10, 0.25, 0.50):
        print(f"  genes with detection rate > {thr:.0%}: {int((det > thr).sum()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
