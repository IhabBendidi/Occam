#!/usr/bin/env python
"""Pass 3: write the Norman-2019 benchmark `.h5ad` — raw counts, CSR, QC cells, our labels.

WHY THIS EXISTS
    The preprocessed file we were handed has `X` ALREADY normalised and log1p'd, and its raw
    counts only in a DENSE `layers/counts`. Our loader reads CSR from `X` and takes the row sum
    BEFORE gene alignment (`stream_v2.load_chunk_sparse`), so feeding it that file would
    double-transform every cell — silently, because log1p of a log1p'd value is still a
    plausible-looking number. This writes the file the loader expects: **raw integer counts,
    CSR, in `X`**, and nothing else.

WHAT IT WRITES
    data/eval/pert/norman_2019_k562.h5ad
      X                CSR int32 counts, 91,168 cells x 33,694 genes (ALL source genes)
      obs.gene_name    "non-targeting" | "GENE" | "GENE_A+GENE_B"   <- the model/eval label
      obs.n_perts      0 | 1 | 2
      obs.batch        gemgroup 1..8
      obs.is_control   bool
      obs.cell_line    "K562"    ⚠️ CORRECTED -- see below
      obs.cell_barcode, obs.guide_identity  (provenance, verbatim from GEO)
      var.gene_id      Ensembl, verbatim and in GEO order
      var.gene_name    symbol, verbatim
      var.in_scoring_axis  bool, the 8,000-gene axis from norman_pick_axis.py

⚠️ ALL 33,694 GENES ARE KEPT, not just the 8,000 scored. The library size must be the sum over
everything the assay measured, exactly as `rhj_fewshot`'s sources keep 8,749-9,624 genes while
its panel scores 6,983. Writing only the axis would shrink every library size and change the
normalisation of every cell.

⚠️ `cell_line = "K562"`, not the `A549` the upstream file claims. Norman et al. 2019 (Science
365:786, GSE133344) is K562, and the copied file's own counts agree: XIST detected in 75.9% of
cells (female; A549 is male) with HBZ mean 28.0, HBG2 17.2, GYPA in 87.9% (erythroid; A549 is
lung). `docs/norman_benchmark.md` records the evidence.

    python scripts/norman/norman_build_h5ad.py
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

D = "data/external/GSE133344/"


def _dec(x):
    return x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mtx", default=D + "GSE133344_filtered_matrix.mtx.gz")
    ap.add_argument("--genes", default=D + "GSE133344_filtered_genes.tsv.gz")
    ap.add_argument("--qc-cells", default=D + "qc_cells.npz")
    ap.add_argument("--axis", default="data/benchmarks/norman_combo/axis.npz")
    ap.add_argument("--out", default="data/eval/pert/norman_2019_k562.h5ad")
    ap.add_argument("--chunk", type=int, default=20_000_000)
    a = ap.parse_args(argv)

    import anndata as ad
    from scipy.sparse import csr_matrix

    rows = [l.rstrip("\n").split("\t") for l in gzip.open(a.genes, "rt")]
    ens = np.array([r[0] for r in rows]); sym = np.array([r[1] for r in rows])
    n_genes = len(ens)

    q = np.load(a.qc_cells, allow_pickle=True)
    keep_col = np.asarray(q["col"], dtype=np.int64)          # 0-based matrix columns, sorted
    labels = np.array([_dec(x) for x in q["label"]])
    kinds = np.array([_dec(x) for x in q["kind"]])
    gems = np.asarray(q["gemgroup"], dtype=np.int16)
    barcodes = np.array([_dec(x) for x in q["barcode"]])
    n_cells = len(keep_col)
    # matrix column (1-based) -> output row
    col2row = np.full(keep_col.max() + 2, -1, dtype=np.int64)
    col2row[keep_col + 1] = np.arange(n_cells)

    with gzip.open(a.mtx, "rt") as fh:
        fh.readline(); fh.readline()
        hdr = fh.readline().split()
    G, C, NNZ = int(hdr[0]), int(hdr[1]), int(hdr[2])
    assert G == n_genes, f"{G} != {n_genes}"
    print(f"[build] source {G:,} x {C:,}, {NNZ:,} nnz -> keeping {n_cells:,} QC cells")

    ii, jj, vv = [], [], []
    reader = pd.read_csv(a.mtx, sep=r"\s+", skiprows=3, header=None, names=["g", "c", "v"],
                         dtype={"g": np.int32, "c": np.int32, "v": np.int32},
                         chunksize=a.chunk, engine="c", compression="gzip")
    read = 0; t0 = time.time()
    for ci, ch in enumerate(reader):
        g = ch["g"].to_numpy(); c = ch["c"].to_numpy(); v = ch["v"].to_numpy()
        r = col2row[c]
        m = r >= 0
        ii.append(r[m].astype(np.int32)); jj.append((g[m] - 1).astype(np.int32))
        vv.append(v[m].astype(np.int32))
        read += len(g)
        print(f"  chunk {ci}: {read:,}/{NNZ:,} ({100*read/NNZ:.1f}%), "
              f"{sum(len(x) for x in ii):,} kept, {time.time()-t0:.0f}s", flush=True)
    assert read == NNZ, f"read {read:,} of {NNZ:,}"
    ii = np.concatenate(ii); jj = np.concatenate(jj); vv = np.concatenate(vv)
    X = csr_matrix((vv, (ii, jj)), shape=(n_cells, n_genes), dtype=np.int32)
    X.sort_indices()
    print(f"[build] CSR {X.shape}, {X.nnz:,} nnz, dtype {X.dtype}, "
          f"integral={bool((X.data == np.round(X.data)).all())}")

    az = np.load(a.axis, allow_pickle=True)
    axis_ens = set(_dec(x) for x in az["axis_ensembl"])
    in_axis = np.array([e in axis_ens for e in ens])
    assert int(in_axis.sum()) == len(axis_ens) == 8000, \
        f"axis has {len(axis_ens)} ids, {int(in_axis.sum())} matched in var"

    obs = pd.DataFrame({
        "cell_barcode": barcodes,
        "gene_name": pd.Categorical(labels),
        "n_perts": np.where(kinds == "control", 0, np.where(kinds == "single", 1, 2)).astype(np.int8),
        "batch": gems.astype(np.int16),
        "is_control": (kinds == "control"),
        "cell_line": pd.Categorical(["K562"] * n_cells),
        "cell_type": pd.Categorical(["K562"] * n_cells),
        "perturbation_class": pd.Categorical(kinds),
    })
    obs.index = pd.Index(barcodes, name=None)
    var = pd.DataFrame({"gene_id": ens, "gene_name": sym, "in_scoring_axis": in_axis})
    var.index = pd.Index(ens, name=None)
    A = ad.AnnData(X=X, obs=obs, var=var)
    A.uns["source"] = "GSE133344 (Norman et al. 2019, Science 365:786) filtered matrix"
    A.uns["qc"] = "good_coverage == True AND number_of_cells == 1 (Norman's own filter)"
    A.uns["cell_line_note"] = (
        "K562. The upstream norman_2016_a549_preprocessed.h5ad labels these cells A549, which "
        "is wrong: XIST is detected in 75.9% of cells (female; A549 is male) and HBZ/HBG2/GYPA "
        "are high (erythroid; A549 is lung adenocarcinoma).")
    A.uns["scoring_axis_sha256"] = str(_dec(az["axis_sha256"]))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    A.write_h5ad(a.out, compression="gzip")
    sha = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    meta = {"path": a.out, "sha256": sha, "bytes": os.path.getsize(a.out),
            "shape": list(A.shape), "nnz": int(X.nnz),
            "cells_by_class": {k: int((kinds == k).sum()) for k in ("control", "single", "double")},
            "conditions": int(len(set(labels))), "batches": sorted(set(int(x) for x in gems)),
            "scoring_axis_genes": int(in_axis.sum()),
            "library_size": {"median": float(np.median(np.asarray(X.sum(1)).ravel())),
                             "min": int(np.asarray(X.sum(1)).ravel().min()),
                             "max": int(np.asarray(X.sum(1)).ravel().max())},
            "seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(a.out + ".json", "w"), indent=1)
    print(f"\n[build] wrote {a.out}  ({meta['bytes']/1e9:.2f} GB)  sha256 {sha[:16]}…")
    print(f"  {A.shape[0]:,} cells x {A.shape[1]:,} genes, {X.nnz:,} nnz, "
          f"{meta['scoring_axis_genes']:,} on the scoring axis")
    print(f"  classes {meta['cells_by_class']}  ·  library median "
          f"{meta['library_size']['median']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
