#!/usr/bin/env python
"""Build an evaluation panel from any perturbation corpus.

    python scripts/build_panel.py --source data.h5ad --out-dir data/panels/mypanel \
        --context K562 --context-col cell_line

The panel is the scoring artifact for `occam.vcell.panel`. It emits two files
whose relationship is the whole point:

    panel_real.h5ad            perturbed cells + a SCORING control pool
    panel_controls_model.h5ad  a DISJOINT control pool the model may see

Controls are drawn once at 2x the requested size and then split in half. That
ordering matters. If the two halves were drawn separately, or thinned with
different RNG streams, a leaked cell could not be detected downstream -- and a
model scored against control cells it was built from looks good for the wrong
reason.

⚠️ Do not construct the model-visible pool as a SUBSET of the scoring panel.
That is the specific mistake this script exists to make impossible; the loader
asserts disjointness and will refuse such a panel.

Depth thinning, when requested, happens ONCE before the split, so both halves
come from one depth distribution and a duplicated cell is byte-identical on
both sides (which is what makes the loader's row-hash check sound).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

BUILDER_VERSION = "1.0.0"
CONTROL = "non-targeting"


# ---------------------------------------------------------------------------
# h5ad reading (h5py directly, to avoid anndata version skew)
# ---------------------------------------------------------------------------
def _decode(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def _read_obs_column(f, name: str) -> Optional[List[str]]:
    """Read an obs column, resolving anndata's categorical encoding."""
    grp = f["obs"]
    if name not in grp:
        return None
    node = grp[name]
    if isinstance(node, type(grp)):          # categorical: codes + categories
        codes = node["codes"][:]
        cats = _decode(node["categories"][:])
        return [cats[c] if 0 <= c < len(cats) else "" for c in codes]
    if "__categories" in grp and name in grp["__categories"]:
        cats = _decode(grp["__categories"][name][:])
        return [cats[c] if 0 <= c < len(cats) else "" for c in node[:]]
    return _decode(node[:])


def _read_var_names(f, gene_col: str = "") -> List[str]:
    var = f["var"]
    if gene_col and gene_col in var:
        return _decode(var[gene_col][:])
    idx = var.attrs.get("_index", "_index")
    idx = idx.decode() if isinstance(idx, bytes) else idx
    return _decode(var[idx][:])


def _n_obs(f) -> int:
    x = f["X"]
    if isinstance(x, type(f["var"])):        # sparse group
        return int(x.attrs["shape"][0])
    return int(x.shape[0])


def _read_rows(f, rows: np.ndarray, n_genes: int) -> "scipy.sparse.csr_matrix":  # noqa: F821
    """Read the given rows as CSR, handling dense and CSR-on-disk layouts."""
    import scipy.sparse as sp

    rows = np.sort(np.asarray(rows))
    x = f["X"]
    if not isinstance(x, type(f["var"])):    # dense
        out = []
        for start in range(0, len(rows), 512):
            chunk = rows[start:start + 512]
            out.append(sp.csr_matrix(x[chunk, :]))
        return sp.vstack(out).tocsr() if out else sp.csr_matrix((0, n_genes))

    indptr = x["indptr"][:]
    data, indices = x["data"], x["indices"]
    rows_data, rows_idx, new_indptr = [], [], [0]
    for r in rows:
        lo, hi = int(indptr[r]), int(indptr[r + 1])
        rows_data.append(data[lo:hi])
        rows_idx.append(indices[lo:hi])
        new_indptr.append(new_indptr[-1] + (hi - lo))
    return sp.csr_matrix(
        (np.concatenate(rows_data) if rows_data else np.array([]),
         np.concatenate(rows_idx) if rows_idx else np.array([], dtype=np.int32),
         np.array(new_indptr)),
        shape=(len(rows), n_genes),
    )


# ---------------------------------------------------------------------------
# Depth thinning
# ---------------------------------------------------------------------------
def thin_to_targets(X, targets: np.ndarray, rng) -> "scipy.sparse.csr_matrix":  # noqa: F821
    """Binomially thin each cell toward its target library size.

    Thinning, not rescaling: the generative operation for shallower sequencing.
    It matters because dropout, not magnitude, is what a per-gene detection
    filter reacts to -- rescaling a deep cell leaves it detected everywhere.
    Never up-samples.
    """
    lib = np.asarray(X.sum(axis=1)).ravel()
    p = np.clip(np.where(lib > 0, targets / np.maximum(lib, 1), 1.0), 0.0, 1.0)
    per_entry = np.repeat(p, np.diff(X.indptr))
    X = X.copy()
    X.data = rng.binomial(X.data.astype(np.int64), per_entry).astype(np.float32)
    X.eliminate_zeros()
    return X


def _pool_library_sizes(paths: Sequence[str]) -> np.ndarray:
    import h5py

    sizes = []
    for path in paths:
        with h5py.File(path, "r") as f:
            x = f["X"]
            if isinstance(x, type(f["var"])):
                indptr = x["indptr"][:]
                data = x["data"]
                for i in range(len(indptr) - 1):
                    sizes.append(float(data[int(indptr[i]):int(indptr[i + 1])].sum()))
            else:
                for start in range(0, x.shape[0], 2048):
                    sizes.extend(x[start:start + 2048, :].sum(axis=1).tolist())
    return np.asarray(sizes, dtype=np.float64)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", required=True,
                    help="input .h5ad; repeatable")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pert-col", default="target_gene",
                    help="obs column holding the perturbation (default: target_gene)")
    ap.add_argument("--control-label", default=CONTROL)
    ap.add_argument("--context-col", default="",
                    help="obs column holding the cell line / context; blank to skip")
    ap.add_argument("--context", default="",
                    help="context value to select, or the label to stamp when "
                         "--context-col is blank")
    ap.add_argument("--gene-col", default="",
                    help="var column with gene names; blank uses the var index")
    ap.add_argument("--cells-per-pert", type=int, default=400)
    ap.add_argument("--n-controls", type=int, default=18400,
                    help="controls PER POOL; twice this many are drawn")
    ap.add_argument("--n-perts", type=int, default=0, help="0 = all that pass the floor")
    ap.add_argument("--min-cells-per-pert", type=int, default=0,
                    help="0 = same as --cells-per-pert")
    ap.add_argument("--gene-axis", choices=["identical", "intersect"], default="intersect")
    ap.add_argument("--controls-from", choices=["all", "first"], default="all",
                    help="'first' takes controls from the first source only, for corpora "
                         "that replicate the same control cells across split files")
    ap.add_argument("--thin-to-h5ad", action="append", default=[],
                    help="pool per-cell library sizes from these files and thin to them")
    ap.add_argument("--thin-to-median", type=float, default=0.0)
    ap.add_argument("--dedupe-genes", choices=["drop", "error"], default="drop")
    ap.add_argument("--seed", type=int, default=0, help="perturbation + control selection")
    ap.add_argument("--split-seed", type=int, default=1234,
                    help="which control half becomes model-visible")
    ap.add_argument("--seen-from", action="append", default=[], metavar="TAG=PATH",
                    help="checkpoint whose split_info['train'] defines 'seen'; repeatable")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the plan and feasibility, write nothing")
    args = ap.parse_args()

    import h5py
    import scipy.sparse as sp

    min_cells = args.min_cells_per_pert or args.cells_per_pert
    rng = np.random.default_rng(args.seed)

    # ---- pass 1: obs only ------------------------------------------------
    print(f"[build] pass 1: scanning {len(args.source)} source(s), obs only")
    metas = []
    gene_sets = []
    for path in args.source:
        if not os.path.isfile(path):
            print(f"ERROR: source not found: {path}", file=sys.stderr)
            return 2
        with h5py.File(path, "r") as f:
            perts = _read_obs_column(f, args.pert_col)
            if perts is None:
                print(f"ERROR: {path} has no obs column {args.pert_col!r}.\n"
                      f"  available: {sorted(f['obs'].keys())[:20]}", file=sys.stderr)
                return 2
            contexts = _read_obs_column(f, args.context_col) if args.context_col else None
            genes = _read_var_names(f, args.gene_col)
            n = _n_obs(f)
        keep = np.ones(n, dtype=bool)
        if contexts is not None and args.context:
            keep = np.array([c == args.context for c in contexts], dtype=bool)
        perts = np.array(perts, dtype=object)
        metas.append({"path": path, "perts": perts, "keep": keep, "genes": genes, "n": n})
        gene_sets.append(set(genes))
        n_ctrl = int((keep & (perts == args.control_label)).sum())
        print(f"  {os.path.basename(path)}: {n} cells, {len(genes)} genes, "
              f"{int(keep.sum())} in context, {n_ctrl} controls")

    # ---- gene axis -------------------------------------------------------
    if args.gene_axis == "identical":
        first = metas[0]["genes"]
        for m in metas[1:]:
            if m["genes"] != first:
                print(f"ERROR: --gene-axis identical, but {m['path']} has a different "
                      f"gene axis ({len(m['genes'])} vs {len(first)} genes).\n"
                      f"  Use --gene-axis intersect.", file=sys.stderr)
                return 2
        shared = list(first)
    else:
        shared = sorted(set.intersection(*gene_sets))
        if not shared:
            print("ERROR: the sources share no genes", file=sys.stderr)
            return 2
    print(f"[build] gene axis: {len(shared)} genes ({args.gene_axis})")

    # Duplicate gene names break the symbol->id alignment downstream, and the
    # failure there is a generic 'map is not 1:1' rather than a name.
    seen_g, dupes, keep_gene = set(), [], []
    for g in shared:
        if g in seen_g:
            dupes.append(g)
            keep_gene.append(False)
        else:
            seen_g.add(g)
            keep_gene.append(True)
    if dupes:
        if args.dedupe_genes == "error":
            print(f"ERROR: {len(dupes)} duplicate gene name(s): {dupes[:10]}\n"
                  f"  Fix upstream, or use --dedupe-genes drop.", file=sys.stderr)
            return 2
        print(f"[build] dropping {len(dupes)} duplicate gene name(s): {dupes[:5]}"
              f"{'...' if len(dupes) > 5 else ''}")
    keep_gene = np.asarray(keep_gene)
    final_genes = [g for g, k in zip(shared, keep_gene) if k]

    # ---- perturbation and control candidates -----------------------------
    pert_cells: Dict[str, List[Tuple[int, int]]] = {}
    ctrl_cells: List[Tuple[int, int]] = []
    for fi, m in enumerate(metas):
        take_ctrl = args.controls_from == "all" or fi == 0
        for i in np.flatnonzero(m["keep"]):
            p = str(m["perts"][i])
            if p == args.control_label:
                if take_ctrl:
                    ctrl_cells.append((fi, int(i)))
            else:
                pert_cells.setdefault(p, []).append((fi, int(i)))

    eligible = sorted(p for p, c in pert_cells.items() if len(c) >= min_cells)
    print(f"[build] perturbations: {len(pert_cells)} present, {len(eligible)} with "
          f">= {min_cells} cells")
    print(f"[build] controls available: {len(ctrl_cells)}"
          f"{' (first source only)' if args.controls_from == 'first' else ''}")

    if args.n_perts:
        eligible = list(rng.permutation(eligible))[:args.n_perts]

    # ---- feasibility, before touching X ----------------------------------
    need = 2 * args.n_controls
    if len(ctrl_cells) < need:
        print(f"\nERROR: need {need} control cells ({args.n_controls} per pool x 2) "
              f"but only {len(ctrl_cells)} are available -- short by "
              f"{need - len(ctrl_cells)}.\n"
              f"  The two pools MUST be disjoint, so half the requested size is the "
              f"hard ceiling.\n"
              f"  Lower --n-controls to at most {len(ctrl_cells) // 2}.", file=sys.stderr)
        return 2
    if not eligible:
        print(f"\nERROR: no perturbation has >= {min_cells} cells.", file=sys.stderr)
        return 2

    print(f"[build] plan: {len(eligible)} perturbations x {args.cells_per_pert} cells "
          f"+ {args.n_controls} scoring controls = "
          f"{len(eligible) * args.cells_per_pert + args.n_controls} rows in panel_real")
    print(f"[build]       {args.n_controls} model-visible controls (disjoint)")
    if args.dry_run:
        print("[build] --dry-run: nothing written")
        return 0

    # ---- pass 2: read the selected rows ----------------------------------
    selected: List[Tuple[int, int, str]] = []
    for p in eligible:
        cells = pert_cells[p]
        idx = rng.choice(len(cells), size=args.cells_per_pert, replace=False)
        selected += [(cells[i][0], cells[i][1], p) for i in idx]
    ctrl_idx = rng.choice(len(ctrl_cells), size=need, replace=False)
    selected += [(ctrl_cells[i][0], ctrl_cells[i][1], "__CTRL__") for i in ctrl_idx]

    print(f"[build] pass 2: reading {len(selected)} rows")
    blocks, labels, src_file, src_row = [], [], [], []
    for fi, m in enumerate(metas):
        rows = [(r, lab) for f2, r, lab in selected if f2 == fi]
        if not rows:
            continue
        order = np.argsort([r for r, _ in rows])
        rows = [rows[i] for i in order]
        row_ids = np.array([r for r, _ in rows])
        with h5py.File(m["path"], "r") as f:
            X = _read_rows(f, row_ids, len(m["genes"]))
        # Reindex onto the shared axis, then drop duplicate columns.
        loc = {g: i for i, g in enumerate(m["genes"])}
        cols = np.array([loc[g] for g in shared])
        X = X[:, cols][:, keep_gene]
        blocks.append(X)
        labels += [lab for _, lab in rows]
        src_file += [os.path.basename(m["path"])] * len(rows)
        src_row += row_ids.tolist()
        print(f"  {os.path.basename(m['path'])}: {X.shape[0]} rows")

    Xall = sp.vstack(blocks).tocsr()
    labels = np.array(labels, dtype=object)
    src_file = np.array(src_file, dtype=object)
    src_row = np.array(src_row, dtype=np.int64)

    # ---- thin ONCE, before the split -------------------------------------
    thin_info = {"mode": "none"}
    median_before = float(np.median(np.asarray(Xall.sum(axis=1)).ravel()))
    if args.thin_to_h5ad or args.thin_to_median:
        if args.thin_to_h5ad:
            pool = _pool_library_sizes(args.thin_to_h5ad)
            targets = rng.choice(pool, size=Xall.shape[0], replace=True)
            thin_info = {"mode": "h5ad_pool", "pool_files": list(args.thin_to_h5ad),
                         "pool_median": float(np.median(pool))}
        else:
            targets = np.full(Xall.shape[0], args.thin_to_median)
            thin_info = {"mode": "fixed_median", "target": args.thin_to_median}
        print(f"[build] thinning (once, before the control split)")
        Xall = thin_to_targets(Xall, targets, rng)
    median_after = float(np.median(np.asarray(Xall.sum(axis=1)).ravel()))
    thin_info.update(median_umi_before=median_before, median_umi_after=median_after)
    print(f"[build] median library size: {median_before:.0f} -> {median_after:.0f}")

    # ---- split the controls 2x-then-half ---------------------------------
    split_rng = np.random.default_rng(args.split_seed)
    ci = split_rng.permutation(np.flatnonzero(labels == "__CTRL__"))
    scoring_c, model_c = ci[:args.n_controls], ci[args.n_controls:need]
    assert not (set(scoring_c.tolist()) & set(model_c.tolist()))
    pi = np.flatnonzero(labels != "__CTRL__")
    real_rows = np.concatenate([pi, scoring_c])

    os.makedirs(args.out_dir, exist_ok=True)
    real_path = os.path.join(args.out_dir, "panel_real.h5ad")
    ctrl_path = os.path.join(args.out_dir, "panel_controls_model.h5ad")
    context = args.context or "panel"

    _write(real_path, Xall[real_rows], final_genes,
           np.where(labels[real_rows] == "__CTRL__", args.control_label, labels[real_rows]),
           context, src_file[real_rows], src_row[real_rows], args.control_label)
    _write(ctrl_path, Xall[model_c], final_genes,
           np.array([args.control_label] * len(model_c), dtype=object),
           context, src_file[model_c], src_row[model_c], args.control_label)

    with open(os.path.join(args.out_dir, "genes.csv"), "w", encoding="utf-8") as fh:
        fh.write("gene_name\n" + "\n".join(final_genes) + "\n")
    with open(os.path.join(args.out_dir, "perts.csv"), "w", encoding="utf-8") as fh:
        fh.write("target_gene\n" + "\n".join(eligible) + "\n")
    with open(os.path.join(args.out_dir, "control_split.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "scoring": [[src_file[i], int(src_row[i])] for i in scoring_c],
            "model": [[src_file[i], int(src_row[i])] for i in model_c],
        }, fh)

    manifest = {
        "builder_version": BUILDER_VERSION,
        "command": " ".join(sys.argv),
        "seed": args.seed,
        "split_seed": args.split_seed,
        "context": context,
        "sources": [{"path": m["path"], "n_obs": m["n"], "n_var": len(m["genes"]),
                     "sha256": _sha256(m["path"])} for m in metas],
        "gene_axis": {"n": len(final_genes), "mode": args.gene_axis,
                      "duplicates_dropped": len(dupes),
                      "sha256": hashlib.sha256("\n".join(final_genes).encode()).hexdigest()},
        "perts": {"n": len(eligible), "cells_per_pert": args.cells_per_pert,
                  "floor": min_cells},
        "controls": {"per_pool": args.n_controls, "available": len(ctrl_cells),
                     "disjoint": True, "disjoint_proof": "2x-draw-then-split + provenance"},
        "thinning": thin_info,
        "files": {"panel_real.h5ad": list(Xall[real_rows].shape),
                  "panel_controls_model.h5ad": list(Xall[model_c].shape)},
    }
    with open(os.path.join(args.out_dir, "panel_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, default=str)

    _write_seen_split(args.seen_from, eligible, args.out_dir)

    # ---- verify our own output before exiting ----------------------------
    print("[build] verifying the panel we just wrote")
    try:
        _ensure_src_on_path()
        from occam.vcell.panel.panel_io import assert_disjoint_controls

        real_obs = {"_index": [str(i) for i in range(len(real_rows))],
                    "target_gene": labels[real_rows].tolist(),
                    "source_file": src_file[real_rows].tolist(),
                    "source_row": src_row[real_rows].tolist()}
        ctrl_obs = {"_index": [str(i) for i in range(len(model_c))],
                    "target_gene": [args.control_label] * len(model_c),
                    "source_file": src_file[model_c].tolist(),
                    "source_row": src_row[model_c].tolist()}
        method = assert_disjoint_controls(real_obs, ctrl_obs,
                                          Xall[real_rows], Xall[model_c])
        print(f"[build] disjointness verified ({method})")
    except Exception as exc:  # noqa: BLE001
        print(f"[build] WARNING: could not self-verify: {exc}")

    print(f"\n[build] wrote {args.out_dir}")
    print(f"  panel_real.h5ad            {Xall[real_rows].shape}")
    print(f"  panel_controls_model.h5ad  {Xall[model_c].shape}")
    return 0


def _ensure_src_on_path() -> None:
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


def _sha256(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _write(path, X, genes, labels, context, src_file, src_row, control_label) -> None:
    import anndata as ad
    import pandas as pd

    # anndata 0.9 refuses object-dtype columns, and a numpy object array stays
    # object even when a dtype is declared -- coerce through a plain list.
    obs = pd.DataFrame({
        "target_gene": [str(v) for v in labels],
        "context": [str(context)] * X.shape[0],
        # Provenance. This is what makes the disjointness assertion able to
        # fail: comparing regenerated obs indices cannot.
        "source_file": [str(v) for v in src_file],
        "source_row": np.asarray(src_row, dtype=np.int64),
    }, index=[str(i) for i in range(X.shape[0])])
    for col in ("target_gene", "context", "source_file"):
        obs[col] = obs[col].astype("category")
    var = pd.DataFrame(index=pd.Index(genes, name=None))
    ad.AnnData(X=X.tocsr(), obs=obs, var=var).write_h5ad(path, compression="gzip")


def _write_seen_split(specs, perts, out_dir) -> None:
    """seen_split.json from checkpoints' split_info['train'], if torch is around."""
    split = {}
    if specs:
        try:
            import torch
        except ImportError:
            print("[build] torch unavailable; writing an empty seen_split.json")
            torch = None
        if torch is not None:
            for spec in specs:
                tag, _, path = spec.partition("=")
                if not path or not os.path.isfile(path):
                    print(f"[build] --seen-from {spec}: no such checkpoint, skipping")
                    continue
                try:
                    info = torch.load(path, map_location="cpu",
                                      weights_only=False).get("split_info", {})
                    train = set(info.get("train", ()))
                    split[tag] = [p for p in perts if p in train]
                    print(f"[build] seen({tag}): {len(split[tag])}/{len(perts)}")
                except Exception as exc:  # noqa: BLE001
                    print(f"[build] --seen-from {spec}: {exc}")
    with open(os.path.join(out_dir, "seen_split.json"), "w", encoding="utf-8") as fh:
        json.dump(split, fh, indent=1)


if __name__ == "__main__":
    raise SystemExit(main())
