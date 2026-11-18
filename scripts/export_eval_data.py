#!/usr/bin/env python
"""Export the evaluation datasets as a compact, self-contained bundle.

    python scripts/export_eval_data.py --dry-run
    python scripts/export_eval_data.py                 # writes data/eval/
    python scripts/export_eval_data.py --only pbmc_12k

Occam's benchmarks read eight `.h5ad` files that live in three different
absolute locations on one cluster. This rewrites them into `data/eval/` with
relative paths, so the released bundle can be downloaded into the repo and the
benchmarks run with no configuration.

Two reasons the copies are far smaller than the originals:

**They are stored dense and heavily fragmented.** `nadig_2024_hepg2` holds
9.79 GB of logical data in a 247 GB file -- HDF5 free space never reclaimed.
Repacking to sparse recovers ~25x on the three perturbation files.

**`obsm` is dropped.** Each perturbation file carries 28 `obsm` arrays of
precomputed baseline embeddings (`X_ica`, `X_nmf`, `X_ridge_pca`, ...) totalling
~2.3 GB. Those are other models' outputs, not evaluation inputs, and shipping
them would invite comparison against numbers whose provenance is not recorded
anywhere.

Everything else is kept in full: all of `X`, all of `obs`, all of `var`,
including Ensembl IDs. Only `obsm` is removed. `manifest.json` records the
columns carried and the arrays dropped.

Expect ~1,069 GB of source to become ~15-20 GB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np


def _ensure_src_on_path() -> None:
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


_ensure_src_on_path()

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Everything in `obs` and `var` is carried through. The evaluation code keys on
# `gene_name` / `is_control` / `dataset_batch_num` (perturbation benchmark) and
# on per-dataset label/batch columns (scIB), and matches genes against a list of
# candidate `var` columns -- but that list can grow, and a column dropped here
# cannot be recovered without the source. So nothing is selected away.
#
# `obsm` is the one exception, and is dropped: see the module docstring.
DROP_SLOTS = ("obsm",)

# Source locations are arguments, not constants: they are wherever these
# datasets happen to live on the machine doing the export, and no absolute path
# belongs in this repository.
#   --eval-root  holds nadig_2024_*.h5ad and replogle_2022_rpe1_e.h5ad
#   --scib-root  holds pancreas_scib.ensembl.h5ad, pbmc_{12k,68k}.ensembl.h5ad
# (name, group, source filename, why it is here). The filename is resolved
# against --eval-root or --scib-root according to the group.
DATASETS = [
    ("nadig_2024_hepg2", "pert", "nadig_2024_hepg2.h5ad",
     "perturbation benchmark, 1 of 3"),
    ("nadig_2024_jurkat", "pert", "nadig_2024_jurkat.h5ad",
     "perturbation benchmark, 2 of 3"),
    ("replogle_2022_rpe1_e", "pert", "replogle_2022_rpe1_e.h5ad",
     "perturbation benchmark, 3 of 3"),
    ("pancreas_scib", "scib", "pancreas_scib.ensembl.h5ad",
     "scIB, label=celltype batch=batch"),
    ("pbmc_12k", "scib", "pbmc_12k.ensembl.h5ad",
     "scIB, label=labels batch=batch"),
    ("pbmc_68k", "scib", "pbmc_68k.ensembl.h5ad",
     "scIB, label=celltype batch=batch"),
]

_ROOT_FOR_GROUP = {"pert": "eval_root", "scib": "scib_root"}

def _decode(vals):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in vals]


def _read_df_col(grp, name):
    """Read an h5ad dataframe column, resolving anndata's categorical encodings."""
    import h5py

    if name not in grp:
        return None
    node = grp[name]
    if isinstance(node, h5py.Group):
        if "categories" in node and "codes" in node:
            cats = _decode(node["categories"][:])
            codes = node["codes"][:]
            return np.array([cats[c] if 0 <= c < len(cats) else "" for c in codes],
                            dtype=object)
        return None
    if "__categories" in grp and name in grp["__categories"]:
        cats = _decode(grp["__categories"][name][:])
        return np.array([cats[c] if 0 <= c < len(cats) else "" for c in node[:]],
                        dtype=object)
    arr = node[:]
    if arr.dtype.kind in ("S", "O", "U"):
        return np.array(_decode(arr), dtype=object)
    return arr


def _index_of(grp):
    idx = grp.attrs.get("_index", "_index")
    idx = idx.decode() if isinstance(idx, bytes) else idx
    return idx


def _read_obs_var(src: str, take: Optional[int] = None):
    """Read `obs` and `var` exactly as anndata would, without loading `X`.

    Backed mode materialises the two frames and leaves the matrix on disk, so
    this costs a few hundred MB even for the largest file here. Using anndata
    rather than reimplementing its decoding is the point: these files span
    several anndata versions and three different categorical encodings, and a
    hand-rolled reader silently produced all-NaN columns on the first attempt.
    """
    import anndata as ad

    adata = ad.read_h5ad(src, backed="r")
    try:
        obs = adata.obs.copy()
        var = adata.var.copy()
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()
    if take is not None and take < len(obs):
        obs = obs.iloc[:take].copy()
    return obs, var


def export_one(name: str, group: str, src: str, note: str,
               out_root: str, limit_rows: int = 0, chunk: int = 4096) -> Dict:
    """Rewrite one dataset as sparse, carrying obs and var in full."""
    import anndata as ad
    import h5py
    import scipy.sparse as sp

    out_dir = os.path.join(out_root, group)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.h5ad")

    t0 = time.time()
    with h5py.File(src, "r") as f:
        X = f["X"]
        sparse_in = isinstance(X, h5py.Group)
        shape = tuple(X.attrs["shape"]) if sparse_in else X.shape
        n_obs, n_var = int(shape[0]), int(shape[1])
        take = min(limit_rows, n_obs) if limit_rows else n_obs

        dropped = {}
        for slot in DROP_SLOTS:
            if slot in f and hasattr(f[slot], "keys"):
                dropped[slot] = sorted(f[slot].keys())
        # Anything else present and non-empty is carried as a note, so a slot
        # that starts being used does not vanish silently.
        other_slots = {
            k: sorted(f[k].keys())
            for k in ("layers", "obsp", "varm", "varp")
            if k in f and hasattr(f[k], "keys") and len(f[k].keys()) > 0
        }

        blocks, nnz, total = [], 0, 0.0
        if sparse_in:
            indptr = X["indptr"][:]
            data, indices = X["data"], X["indices"]
            for start in range(0, take, chunk):
                stop = min(start + chunk, take)
                lo, hi = int(indptr[start]), int(indptr[stop])
                blk = sp.csr_matrix(
                    (data[lo:hi], indices[lo:hi],
                     indptr[start:stop + 1] - indptr[start]),
                    shape=(stop - start, n_var)).astype(np.float32)
                blocks.append(blk)
                nnz += blk.nnz
                total += float(blk.data.sum())
        else:
            for start in range(0, take, chunk):
                stop = min(start + chunk, take)
                blk = sp.csr_matrix(np.asarray(X[start:stop, :], dtype=np.float32))
                blocks.append(blk)
                nnz += blk.nnz
                total += float(blk.data.sum())

    mat = sp.vstack(blocks).tocsr() if blocks else sp.csr_matrix((0, n_var), np.float32)

    obs, var = _read_obs_var(src, take)

    if len(obs) != mat.shape[0]:
        raise ValueError(f"{name}: obs has {len(obs)} rows but X has {mat.shape[0]}")
    if len(var) != mat.shape[1]:
        raise ValueError(f"{name}: var has {len(var)} rows but X has {mat.shape[1]} columns")

    ad.AnnData(X=mat, obs=obs, var=var).write_h5ad(out_path, compression="gzip")

    return {
        "name": name, "group": group,
        "file": os.path.relpath(out_path, REPO),
        # Only the basename: where the file happened to live on the exporting
        # machine is not provenance anyone else can use, and an absolute path
        # does not belong in a released manifest.
        "source_filename": os.path.basename(src),
        "source_bytes": os.path.getsize(src),
        "bytes": os.path.getsize(out_path),
        "shape": [int(mat.shape[0]), int(mat.shape[1])],
        "nnz": int(nnz),
        "density_pct": round(100.0 * nnz / max(mat.shape[0] * mat.shape[1], 1), 3),
        "sum": total,
        "obs_columns": list(obs.columns),
        "var_columns": list(var.columns),
        "dropped": dropped,
        "other_slots_present": other_slots,
        "sha256": _sha256(out_path),
        "seconds": round(time.time() - t0, 1),
        "note": note,
        "partial": bool(limit_rows and take < n_obs),
    }


def _sha256(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-root", required=True,
                    help="directory holding the perturbation benchmark .h5ad files")
    ap.add_argument("--scib-root", required=True,
                    help="directory holding the scIB .h5ad files")
    ap.add_argument("--out", default=os.path.join(REPO, "data", "eval"))
    ap.add_argument("--only", action="append", default=[],
                    help="dataset name; repeatable")
    ap.add_argument("--limit-rows", type=int, default=0,
                    help="first N cells only, to validate the pipeline")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    roots = {"eval_root": args.eval_root, "scib_root": args.scib_root}
    wanted = [(name, group, os.path.join(roots[_ROOT_FOR_GROUP[group]], filename), note)
              for name, group, filename, note in DATASETS
              if not args.only or name in args.only]
    if args.only:
        unknown = set(args.only) - {d[0] for d in DATASETS}
        if unknown:
            ap.error(f"unknown dataset(s): {sorted(unknown)}\n"
                     f"available: {[d[0] for d in DATASETS]}")

    print(f"[eval-export] {len(wanted)} dataset(s) -> {args.out}\n")
    missing = []
    src_total = 0
    for name, group, src, note in wanted:
        ok = os.path.isfile(src)
        size = os.path.getsize(src) / 1e9 if ok else 0.0
        src_total += size
        print(f"  {name:26s} {group:14s} {size:8.1f} GB  "
              f"{'' if ok else 'MISSING  '}{note}")
        if not ok:
            missing.append((name, src))
    print(f"\n  source total: {src_total:,.0f} GB")
    if missing:
        print(f"\n  {len(missing)} source file(s) missing:", file=sys.stderr)
        for name, src in missing:
            print(f"    {name}: {src}", file=sys.stderr)
        if len(missing) == len(wanted):
            return 2
    if args.dry_run:
        print("\n[eval-export] --dry-run: nothing written")
        return 0

    manifest_path = os.path.join(args.out, "manifest.json")
    entries = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            entries = {e["name"]: e for e in json.load(fh).get("datasets", [])}
        # Older manifests recorded an absolute source path. Drop it: no absolute
        # path belongs in a tracked file, and a reused entry must not smuggle
        # one back in.
        for e in entries.values():
            src = e.pop("source", None)
            if src and "source_filename" not in e:
                e["source_filename"] = os.path.basename(src)

    print()
    for name, group, src, note in wanted:
        if not os.path.isfile(src):
            continue
        out_path = os.path.join(args.out, group, f"{name}.h5ad")
        prior = entries.get(name)
        if prior and os.path.isfile(out_path) and not args.limit_rows \
                and prior.get("sha256") == _sha256(out_path):
            print(f"  {name:26s} unchanged, skipping")
            continue
        print(f"  {name:26s} converting...", flush=True)
        meta = export_one(name, group, src, note, args.out, args.limit_rows)
        entries[name] = meta
        print(f"  {name:26s} {meta['shape'][0]:>8,} x {meta['shape'][1]:>6,}  "
              f"{meta['density_pct']:>5.1f}% nz  "
              f"{meta['source_bytes'] / 1e9:>7.1f} GB -> {meta['bytes'] / 1e9:5.2f} GB  "
              f"({meta['seconds']:.0f}s, {len(meta['obs_columns'])} obs / "
              f"{len(meta['var_columns'])} var cols, "
              f"dropped {sum(len(v) for v in meta['dropped'].values())} obsm)")

    ordered = [entries[d[0]] for d in DATASETS if d[0] in entries]
    os.makedirs(args.out, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump({
            "schema": 1,
            "partial": any(e.get("partial") for e in ordered),
            "datasets": ordered,
            "totals": {
                "n_datasets": len(ordered),
                "source_bytes": sum(e["source_bytes"] for e in ordered),
                "bytes": sum(e["bytes"] for e in ordered),
                "nnz": sum(e["nnz"] for e in ordered),
            },
        }, fh, indent=2, sort_keys=True)

    t = {"src": sum(e["source_bytes"] for e in ordered) / 1e9,
         "out": sum(e["bytes"] for e in ordered) / 1e9}
    print(f"\n[eval-export] {len(ordered)} datasets, "
          f"{t['src']:,.0f} GB -> {t['out']:.2f} GB "
          f"({t['src'] / max(t['out'], 1e-9):.0f}x smaller)")
    print(f"[eval-export] manifest: {manifest_path}")
    if args.limit_rows:
        print("[eval-export] ⚠️  PARTIAL: manifest records partial=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
