#!/usr/bin/env python
"""Freeze the H1 "seen perturbation, UNSEEN cell line" split-half panel (`h1_seen`).

WHAT THIS IS FOR
----------------
Objective: measure how far models trained on `rhj_fewshot` (HepG2 + Jurkat + RPE1) carry a
perturbation they HAVE seen into a cell line they have NEVER seen -- not its perturbations,
not even its controls: H1 hESC from the VCC 2025 dataset. No model is trained here; existing
checkpoints are scored with the same split-half protocol as `rhj_fewshot`.

Question: how much of the RHJ score survives the cell-line change when the perturbation is
seen? Anchors on this panel: `ctrlcell` (floor), `transfer` (copy the RHJ delta of the same
perturbation -- the non-learned "memorise and carry over" reference), `experimental`.

WHAT GOES IN
------------
* perturbations: every H1 perturbation (any of the 3 VCC 2025 files, all have ground truth
  locally) that is ELIGIBLE in `rhj_fewshot` (the 2,381 trained in at least one RHJ line)
  -- 99. Stratum `rhj_split` per perturbation: `train` (trained in HepG2, Jurkat and RPE1)
  or `val`/`test` (withheld from HepG2, trained only in Jurkat/RPE1).
* controls: the 38,176 non-targeting cells, read ONCE (identical in all three files).
* the A/B halves are drawn from these H1 cells only, exactly as RHJ draws them from its
  held-out HepG2 cohort. Model input = half-B controls; reference = half A.

DESIGN CHOICES (stated, recorded in the manifest)
-------------------------------------------------
* gene axis = the 6,535 `rhj_fewshot` axis genes present in H1 (448 are absent). The model
  input carries zeros there, as it already does for the 37k non-axis encoder columns.
* normalisation: log1p(count / rowsum x 2000), rowsum over the H1 genes that are ALSO in the
  RHJ HepG2 source file (Ensembl match) -- the same gene universe the checkpoints' inputs
  were normalised over. No depth thinning: H1's depth is part of the shift.
* `meanpert`: RHJ HepG2 training grand-mean DELTA (what the models could know), placed on
  the H1 control mean so `meanpert_arm`'s absolute-profile convention holds.
* `transfer`: cell-weighted mean delta of the same perturbation over the RHJ lines where it
  was TRAINED (all 3 for `train`; Jurkat + RPE1 for `val`/`test`).

OUTPUT  data/benchmarks/h1_seen/
    h1_seen_raw.h5ad                 local filtered copy: raw counts, all 18,080 genes
    benchmark.json, gene_axis.csv, perturbations.csv
    splithalf/{panel.npz, splits.npz, cohort.json, splits_manifest.json,
               train_grand_mean.npy, transfer_deltas.npz}

USAGE
    # from the three VCC 2025 H1 files (how the published benchmark was built)
    python scripts/build_h1_seen_panel.py --h1-files \
        adata_Training.h5ad adata_Validation.h5ad adata_Test.h5ad
    # from the PUBLISHED local copy (what everyone else does -- see
    # scripts/repro/build_benchmark_h1.sh), verified against the published panel
    python scripts/build_h1_seen_panel.py \
        --from-raw data/benchmarks/h1_seen/h1_seen_raw.h5ad \
        --out-dir /tmp/h1_check --verify-against data/benchmarks/h1_seen

    Refuses to overwrite an existing benchmark.json unless --force.

Both modes give the same panel: the local copy holds exactly the selected rows, in build
order, with raw counts on all 18,080 H1 genes, so every later step is identical.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import date

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import numpy as np  # noqa: E402

from occam.h5ad_columns import read_column  # noqa: E402

BUILDER_VERSION = "1.0.0"
BENCH_ID = "h1_seen"
CELL_LINE = "H1"
CONTROL = "non-targeting"
PERT_COL = "target_gene"
BATCH_COL = "batch"
# The three VCC 2025 H1 files (Training / Validation / Test), passed with --h1-files in that
# order. Their paths and sha256 are recorded in benchmark.json `sources`.
H1_SPLITS = ("Training", "Validation", "Test")
RHJ_HEPG2 = "nadig_2024_hepg2.h5ad"
RHJ_OTHERS = ("nadig_2024_jurkat.h5ad", "replogle_2022_rpe1_e.h5ad")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def _read_rows(path: str, rows: np.ndarray, n_vars: int):
    """Raw counts of `rows` (sorted) as a scipy CSR, from a dense OR CSR h5ad X."""
    import h5py
    import scipy.sparse as sp

    # Iterate over FILE blocks, not over selected rows: a block of selected rows can span
    # most of the file, and a dense 18,080-gene read of that span would not fit in memory.
    out = []
    rows = np.asarray(rows, dtype=np.int64)
    with h5py.File(path, "r") as f:
        X = f["X"]
        dense = not isinstance(X, h5py.Group)
        n_obs = X.shape[0] if dense else len(X["indptr"]) - 1
        if not dense:
            indptr = np.asarray(X["indptr"][:], dtype=np.int64)
            data, indices = X["data"], X["indices"]
        CH = 2048 if dense else 16384
        for s in range(0, n_obs, CH):
            e = min(s + CH, n_obs)
            r = rows[(rows >= s) & (rows < e)]
            if r.size == 0:
                continue
            if dense:
                block = sp.csr_matrix(np.asarray(X[s:e], dtype=np.float32))
            else:
                lo, hi = int(indptr[s]), int(indptr[e])
                block = sp.csr_matrix(
                    (np.asarray(data[lo:hi], dtype=np.float32),
                     np.asarray(indices[lo:hi], dtype=np.int64), indptr[s:e + 1] - lo),
                    shape=(e - s, n_vars))
            out.append(block[r - s])
            if (s // CH) % 20 == 0:
                print(f"    {os.path.basename(path)}: {e:,}/{n_obs:,} rows", flush=True)
    return sp.vstack(out, format="csr")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rhj-bench-dir", default=os.path.join(REPO, "data", "benchmarks",
                                                            "rhj_fewshot"))
    ap.add_argument("--out-dir", default=os.path.join(REPO, "data", "benchmarks", BENCH_ID))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-cells", type=int, default=10)
    ap.add_argument("--h1-files", nargs=3, metavar=("TRAINING", "VALIDATION", "TEST"),
                    help="the VCC 2025 adata_{Training,Validation,Test}.h5ad; required "
                         "unless --from-raw")
    ap.add_argument("--from-raw", default="",
                    help="start from the published local copy h1_seen_raw.h5ad instead of "
                         "the three VCC 2025 files")
    ap.add_argument("--verify-against", default="",
                    help="a published h1_seen directory; exit 3 unless the rebuild matches "
                         "it in every scientific quantity")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not args.from_raw and not args.h1_files:
        ap.error("pass --h1-files TRAINING VALIDATION TEST, or --from-raw <h1_seen_raw.h5ad>")
    H1_FILES = dict(zip(H1_SPLITS, args.h1_files or ()))

    import anndata as ad
    import h5py
    import pandas as pd

    from build_splithalf_panel import _load_axis_restricted
    from occam.vcell.benchmark import _sha256 as file_sha256
    from occam.vcell.benchmark import load_benchmark
    from occam.vcell.panel.splithalf import (assert_disjoint_halves, draw_splits,
                                             freeze_cohort, train_pert_grand_mean)

    out, sh = args.out_dir, os.path.join(args.out_dir, "splithalf")
    if os.path.isfile(os.path.join(out, "benchmark.json")) and not args.force:
        raise SystemExit(f"{out}/benchmark.json exists; pass --force to rebuild")
    os.makedirs(sh, exist_ok=True)

    rhj = load_benchmark(args.rhj_bench_dir, expect_id="rhj_fewshot", verify_sources=False)
    target_sum = float(rhj.raw["encoder_target_sum"])
    with open(os.path.join(args.rhj_bench_dir, "perturbations.csv")) as fh:
        prow = {r["gene_name"]: r for r in csv.DictReader(fh) if r["eligible"] == "1"}
    print(f"[h1] rhj_fewshot: {len(prow):,} eligible perturbations, axis "
          f"{len(rhj.axis_ensembl):,}, target_sum {target_sum:g}")

    def rhj_src(name):
        """The RHJ source file, found where it is on THIS machine and sha256-checked.

        `files/*.json` records the absolute path of the machine that froze rhj_fewshot, so
        on anyone else's machine it points nowhere; fall back to data/eval/pert/<name>,
        where the HF download puts it. Either way the content must hash to the frozen digest.
        """
        rec = dict(rhj.raw["sources"][name])
        with open(os.path.join(args.rhj_bench_dir, "files",
                               name.replace(".h5ad", ".json"))) as fh:
            rec.update(json.load(fh))
        if not os.path.isfile(rec["path"]):
            rec["path"] = os.path.join(REPO, "data", "eval", "pert", name)
        if not os.path.isfile(rec["path"]):
            raise SystemExit(f"{name} not found; download eval/pert/{name} from the HF repo "
                             f"into data/eval/pert/ (docs/hf_release.md)")
        if name not in _checked:
            have = file_sha256(rec["path"])
            if have != rec["sha256"]:
                raise SystemExit(f"{rec['path']}: sha256 {have[:16]} != frozen "
                                 f"{rec['sha256'][:16]}; not the rhj_fewshot source")
            _checked.add(name)
        return rec

    _checked = set()

    if args.from_raw:
        # ---- the published local copy: genes, rows and obs are already selected ------
        with h5py.File(args.from_raw, "r") as f:
            var_sym = np.asarray(read_column(f["var"], f["var"].attrs["_index"]), dtype=str)
            var_ens = np.asarray(read_column(f["var"], "ensembl_gene_id"), dtype=str)
    else:
        # ---- H1 genes: symbols from every file (must agree), Ensembl from Training ------
        var_sym = None
        for tag, p in H1_FILES.items():
            with h5py.File(p, "r") as f:
                s = np.asarray(read_column(f["var"], f["var"].attrs["_index"]), dtype=str)
            if var_sym is None:
                var_sym = s
            elif not np.array_equal(var_sym, s):
                raise SystemExit(f"{tag}: var differs from Training; cannot share gene order")
        with h5py.File(H1_FILES["Training"], "r") as f:
            var_ens = np.asarray(read_column(f["var"], "ensembl_gene_id"), dtype=str)
    n_vars = len(var_sym)
    h1_pos = {g: i for i, g in enumerate(var_ens)}
    axis_pairs = [(e, s) for e, s in zip(rhj.axis_ensembl, rhj.axis_symbols) if e in h1_pos]
    axis_ens = [e for e, _ in axis_pairs]
    axis_sym = [s for _, s in axis_pairs]
    axis_cols_h1 = np.array([h1_pos[e] for e in axis_ens], dtype=np.int64)
    rhj_axis_idx = np.array([rhj.axis_ensembl.index(e) for e in axis_ens], dtype=np.int64)
    print(f"[h1] axis: {len(axis_ens):,} of {len(rhj.axis_ensembl):,} rhj axis genes in H1")

    with h5py.File(rhj_src(RHJ_HEPG2)["path"], "r") as f:
        hep_ens = set(np.asarray(read_column(f["var"], "ensembl_gene_id"), dtype=str))
    norm_cols = np.array([i for i, e in enumerate(var_ens) if e in hep_ens], dtype=np.int64)
    print(f"[h1] normalisation universe: {len(norm_cols):,} H1 genes also in {RHJ_HEPG2}")

    seen = sorted(set(prow))
    if args.from_raw:
        a = ad.read_h5ad(args.from_raw)
        import scipy.sparse as sp
        raw = sp.csr_matrix(a.X, dtype=np.float32)
        obs = a.obs.copy()
        del a
        bad = sorted(set(obs[PERT_COL]) - set(seen) - {CONTROL})
        if bad:
            raise SystemExit(f"{args.from_raw} holds perturbations that are not eligible in "
                             f"rhj_fewshot: {bad[:5]}; wrong file or wrong rhj benchmark")
        raw_path = args.from_raw
        print(f"[h1] local copy {args.from_raw}: {raw.shape[0]:,} cells x "
              f"{raw.shape[1]:,} genes (raw counts)")
    else:
        # ---- which rows ----------------------------------------------------------
        pieces, meta = [], []
        for tag, p in H1_FILES.items():
            with h5py.File(p, "r") as f:
                lab = np.asarray(read_column(f["obs"], PERT_COL), dtype=str)
                bat = np.asarray(read_column(f["obs"], BATCH_COL), dtype=str)
                idx = np.asarray(read_column(f["obs"], f["obs"].attrs["_index"]), dtype=str)
            want = np.isin(lab, seen)
            if tag == "Training":          # controls are identical in all 3 files: take once
                want |= lab == CONTROL
            rows = np.flatnonzero(want)
            n_ctrl = int((lab[rows] == CONTROL).sum())
            print(f"[h1] {tag}: {len(set(lab[rows]) - {CONTROL})} seen perts, "
                  f"{len(rows) - n_ctrl:,} perturbed + {n_ctrl:,} control cells")
            pieces.append((tag, p, rows))
            meta.append(pd.DataFrame({PERT_COL: lab[rows], BATCH_COL: bat[rows],
                                      "source_file": tag, "source_obs": idx[rows]}))

        import scipy.sparse as sp
        raw = sp.vstack([_read_rows(p, rows, n_vars) for _, p, rows in pieces], format="csr")
        obs = pd.concat(meta, ignore_index=True)
        obs["rhj_split"] = [prow[g]["split"] if g in prow else "control" for g in obs[PERT_COL]]
        obs.index = [f"H1s_{i}" for i in range(len(obs))]
        print(f"[h1] {raw.shape[0]:,} cells x {raw.shape[1]:,} genes (raw counts)")

        adata = ad.AnnData(X=raw, obs=obs,
                           var=pd.DataFrame({"ensembl_gene_id": var_ens}, index=var_sym))
        raw_path = os.path.join(out, "h1_seen_raw.h5ad")
        adata.write_h5ad(raw_path)
        print(f"[h1] wrote {raw_path}")

    # ---- normalise onto the axis -------------------------------------------
    rs = np.asarray(raw[:, norm_cols].sum(axis=1)).ravel().astype(np.float64)
    rs = np.where(rs > 0, rs, 1.0)
    Xa = raw[:, axis_cols_h1].toarray().astype(np.float32)
    X = np.log1p(Xa / rs[:, None].astype(np.float32) * target_sum).astype(np.float32)
    del Xa
    labels = obs[PERT_COL].to_numpy(dtype=str)
    batches = obs[BATCH_COL].to_numpy(dtype=str)
    lib = np.asarray(raw.sum(axis=1)).ravel()
    print(f"[h1] library size median {np.median(lib):,.0f} UMI; normalisation rowsum "
          f"median {np.median(rs):,.0f}")

    strata = {g: prow[g]["split"] for g in seen if g in set(labels)}
    cohort = freeze_cohort(
        labels, eligible=list(strata), min_cells=args.min_cells, context=CELL_LINE,
        extra={"benchmark_id": BENCH_ID, "seen_from": "rhj_fewshot",
               "rhj_split": {p: strata[p] for p in sorted(strata)},
               "n_axis_genes": len(axis_ens), "builder_version": BUILDER_VERSION,
               "built": str(date.today()), "pert_col_in_source": PERT_COL,
               "batch_col": BATCH_COL})
    print(f"[h1] cohort {cohort['n_perts']} perturbations; strata "
          f"{ {k: list(cohort['rhj_split'].values()).count(k) for k in ('train', 'val', 'test')} }")

    # ---- meanpert (RHJ HepG2 training) and transfer (RHJ lines where trained) ----
    rhj_X = {}
    for name in (RHJ_HEPG2,) + RHJ_OTHERS:
        rec = rhj_src(name)
        print(f"[h1] reading {name} on the rhj axis")
        Xs, ls, _ = _load_axis_restricted(rec["path"], rhj.axis_ensembl, target_sum)
        rhj_X[name] = (Xs[:, rhj_axis_idx], ls, rec)
        del Xs
    hep_train = sorted(g for g, r in prow.items() if r["split"] == "train")
    hep_grand, grand_info = train_pert_grand_mean(rhj_X[RHJ_HEPG2][0], rhj_X[RHJ_HEPG2][1],
                                                  hep_train)
    # `meanpert_arm` emits ctrl_used + (grand_mean - B control mean): it expects an ABSOLUTE
    # profile in the EVALUATED line. A HepG2 profile minus an H1 control mean would be the
    # HepG2-vs-H1 line difference, not a perturbation effect. So store the HepG2 grand
    # DELTA placed on the H1 control mean: the arm then emits ctrl_used + hepg2_delta +
    # (H1 all-ctrl mean - H1 B-ctrl mean). The last term is half the A/B control-mean
    # difference over 38k controls -- recorded, not hidden.
    hep_X, hep_l = rhj_X[RHJ_HEPG2][0], rhj_X[RHJ_HEPG2][1]
    hep_delta = hep_grand - hep_X[hep_l == CONTROL].mean(axis=0)
    h1_ctrl_mean = X[labels == CONTROL].mean(axis=0)
    grand = (hep_delta + h1_ctrl_mean).astype(np.float32)
    grand_info = {k: v for k, v in grand_info.items() if k != "cells_per_pert"}
    grand_info["source"] = (f"rhj_fewshot {RHJ_HEPG2} training perturbations: grand DELTA "
                            f"(equal weight per pert) + H1 all-control mean")
    np.save(os.path.join(sh, "train_grand_mean.npy"), grand)

    perts = cohort["perts"]
    deltas = np.zeros((len(perts), len(axis_ens)), dtype=np.float64)
    w = np.zeros(len(perts))
    used = {p: [] for p in perts}
    for name, (Xs, ls, rec) in rhj_X.items():
        cmean = Xs[ls == CONTROL].mean(axis=0)
        for i, p in enumerate(perts):
            if name == RHJ_HEPG2 and cohort["rhj_split"][p] != "train":
                continue          # withheld from HepG2 in training: not the model's data
            r = np.flatnonzero(ls == p)
            if r.size:
                deltas[i] += (Xs[r].mean(axis=0) - cmean) * r.size
                w[i] += r.size
                used[p].append(rec.get("cell_line") or name)
    if (w == 0).any():
        raise SystemExit(f"no transfer source for {[p for p, x in zip(perts, w) if x == 0]}")
    deltas /= w[:, None]
    np.savez_compressed(os.path.join(sh, "transfer_deltas.npz"),
                        deltas=deltas.astype(np.float32),
                        perts=np.asarray(perts, dtype=object))
    transfer_info = {"weighting": "cells per (file, pert)",
                     "rule": "RHJ lines where the perturbation was TRAINED: all 3 for "
                             "rhj_split=train, Jurkat+RPE1 for val/test",
                     "lines_per_pert": used, "cells_per_pert_median": float(np.median(w))}
    del rhj_X

    # ---- panel + splits ------------------------------------------------------
    keep = np.flatnonzero(np.isin(labels, list(perts) + [CONTROL]))
    Xp, lp, bp = X[keep], labels[keep], batches[keep]
    splits = draw_splits(lp, bp, cohort, n_splits=args.n_splits, base_seed=args.seed)
    for s in splits:
        assert_disjoint_halves(s)
        print(f"[h1] {s.summary()}")

    axis_sha = _sha("\n".join(axis_ens).encode())
    with open(os.path.join(out, "gene_axis.csv"), "w") as fh:
        fh.write("ensembl_gene_id,gene_name\n")
        fh.writelines(f"{e},{s}\n" for e, s in axis_pairs)
    with open(os.path.join(out, "perturbations.csv"), "w") as fh:
        fh.write("gene_name,rhj_split,n_cells_H1\n")
        fh.writelines(f"{p},{cohort['rhj_split'][p]},{cohort['cells_per_pert'][p]}\n"
                      for p in perts)
    bench = {
        "benchmark_id": BENCH_ID, "cell_line": CELL_LINE, "builder_version": BUILDER_VERSION,
        "built": str(date.today()), "git_sha": _git_sha(), "command": " ".join(sys.argv),
        "purpose": "seen perturbation, unseen cell line: score rhj_fewshot checkpoints on H1",
        "seen_from": {"benchmark_id": "rhj_fewshot", "benchmark_sha256": rhj.sha256,
                      "gene_axis_sha256": rhj.axis_sha256,
                      "n_axis_genes": len(rhj.axis_ensembl)},
        "gene_axis_sha256": axis_sha, "n_axis_genes": len(axis_ens),
        "n_axis_genes_missing_in_h1": len(rhj.axis_ensembl) - len(axis_ens),
        "encoder_target_sum": target_sum,
        "normalisation": f"rowsum over {len(norm_cols)} H1 genes also in {RHJ_HEPG2}, "
                         f"then axis restriction, then log1p(x/rowsum*{target_sum:g})",
        "sources": ({"raw_copy": {"path": raw_path}} if args.from_raw else
                    {t: {"path": p, "sha256": file_sha256(p)} for t, p in H1_FILES.items()}),
        "raw_copy": {"path": os.path.relpath(os.path.abspath(raw_path), REPO),
                     "sha256": file_sha256(raw_path), "n_cells": int(raw.shape[0])},
        "library_size_median": float(np.median(lib)),
    }
    bench_bytes = json.dumps(bench, indent=2, sort_keys=True).encode()
    with open(os.path.join(out, "benchmark.json"), "wb") as fh:
        fh.write(bench_bytes)

    np.savez_compressed(os.path.join(sh, "panel.npz"), X=Xp, labels=lp, batches=bp,
                        axis_ensembl=np.asarray(axis_ens, dtype=object),
                        axis_symbols=np.asarray(axis_sym, dtype=object), source_rows=keep)
    np.savez_compressed(os.path.join(sh, "splits.npz"),
                        **{f"{k}_{s.index}": getattr(s, k) for s in splits
                           for k in ("pert_A", "pert_B", "ctrl_A", "ctrl_B", "dropped")})
    with open(os.path.join(sh, "cohort.json"), "w") as fh:
        json.dump(cohort, fh, indent=2, sort_keys=True)
    m = np.array(sorted(cohort["half_size"].values()))
    manifest = {
        "protocol": "splithalf/v1", "builder_version": BUILDER_VERSION,
        "built": str(date.today()), "git_sha": _git_sha(), "command": " ".join(sys.argv),
        "benchmark_id": BENCH_ID, "benchmark_sha256": _sha(bench_bytes),
        "gene_axis_sha256": axis_sha, "n_axis_genes": len(axis_ens),
        "ood_context": True, "seen_from": bench["seen_from"],
        "cell_line": CELL_LINE, "pert_col_in_source": PERT_COL, "batch_col": BATCH_COL,
        "control_label": CONTROL, "target_sum": target_sum,
        "normalisation": bench["normalisation"],
        "n_panel_rows": int(Xp.shape[0]), "n_panel_controls": int((lp == CONTROL).sum()),
        "cohort_sha256": cohort["cohort_sha256"], "n_cohort_perts": int(cohort["n_perts"]),
        "min_cells": int(args.min_cells), "base_seed": int(args.seed),
        "n_splits": int(args.n_splits),
        "half_size_strata": {"m>=5": int((m >= 5).sum()), "m>=25": int((m >= 25).sum()),
                             "m_median": int(np.median(m))},
        "train_grand_mean": grand_info, "transfer": transfer_info,
        "splits": [{"split": int(s.index), "n_pert_per_half": int(len(s.pert_A)),
                    "n_ctrl_per_half": int(len(s.ctrl_A)),
                    "n_dropped_odd": int(s.n_dropped_odd)} for s in splits],
    }
    with open(os.path.join(sh, "splits_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    # ---- self-verification ---------------------------------------------------
    z = np.load(os.path.join(sh, "panel.npz"), allow_pickle=True)
    zs = np.load(os.path.join(sh, "splits.npz"))
    ctrl_ids = obs.loc[obs[PERT_COL] == CONTROL, "source_obs"]
    checks = [
        ("panel round-trip", z["X"].shape == Xp.shape),
        ("axis length", len(z["axis_ensembl"]) == len(axis_ens)),
        ("controls unique", ctrl_ids.is_unique),
        ("every label is cohort or control", set(lp.tolist()) == set(perts) | {CONTROL}),
        ("halves disjoint", all(np.intersect1d(zs[f"pert_A_{i}"], zs[f"pert_B_{i}"]).size
                                == 0 and np.intersect1d(zs[f"ctrl_A_{i}"],
                                                        zs[f"ctrl_B_{i}"]).size == 0
                                for i in range(args.n_splits))),
        ("halves equal", all(len(zs[f"pert_A_{i}"]) == len(zs[f"pert_B_{i}"])
                             for i in range(args.n_splits))),
        ("grand mean on axis", grand.shape == (len(axis_ens),)),
    ]
    rng = np.random.default_rng(0)
    for r in rng.choice(len(keep), 3, replace=False):
        row = raw[keep[r]].toarray().ravel()
        ref = np.log1p(row[axis_cols_h1] / max(row[norm_cols].sum(), 1) * target_sum)
        checks.append((f"row {keep[r]} re-normalised by hand", np.allclose(ref, Xp[r],
                                                                            atol=1e-5)))
    bad = [n for n, ok in checks if not ok]
    for n, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    if bad:
        raise SystemExit(f"self-verification failed: {bad}")
    print(f"\n[h1] done -> {out}")
    if args.verify_against:
        return verify(out, args.verify_against)
    return 0


def verify(rebuilt: str, published: str) -> int:
    """Exit 0 if `rebuilt` equals `published` in every scientific quantity, else 3.

    Not a byte hash: benchmark.json embeds git_sha / date / paths and .npz files are zips
    with timestamps. What is compared is what a score depends on.
    """
    def J(d, f):
        with open(os.path.join(d, f)) as fh:
            return json.load(fh)

    bad = []
    br, bp = J(rebuilt, "benchmark.json"), J(published, "benchmark.json")
    for k in ("gene_axis_sha256", "n_axis_genes", "encoder_target_sum", "normalisation"):
        if br.get(k) != bp.get(k):
            bad.append(f"benchmark.{k}: {br.get(k)!r} != {bp.get(k)!r}")
    if br["seen_from"]["gene_axis_sha256"] != bp["seen_from"]["gene_axis_sha256"]:
        bad.append("seen_from.gene_axis_sha256 differs: a different rhj_fewshot axis")
    cr, cp = J(rebuilt, "splithalf/cohort.json"), J(published, "splithalf/cohort.json")
    for k in ("cohort_sha256", "cells_per_pert", "rhj_split", "n_controls"):
        if cr.get(k) != cp.get(k):
            bad.append(f"cohort.{k} differs")
    zr = np.load(os.path.join(rebuilt, "splithalf", "panel.npz"), allow_pickle=True)
    zp = np.load(os.path.join(published, "splithalf", "panel.npz"), allow_pickle=True)
    for k in ("labels", "batches", "axis_ensembl"):
        if not np.array_equal(zr[k], zp[k]):
            bad.append(f"panel.{k} differs")
    if zr["X"].shape != zp["X"].shape or not np.allclose(zr["X"], zp["X"], atol=1e-6):
        bad.append("panel.X differs")
    sr = np.load(os.path.join(rebuilt, "splithalf", "splits.npz"))
    sp_ = np.load(os.path.join(published, "splithalf", "splits.npz"))
    if set(sr.files) != set(sp_.files) or any(not np.array_equal(sr[k], sp_[k])
                                               for k in sr.files):
        bad.append("splits.npz differs (A/B halves)")
    for f, key in (("train_grand_mean.npy", None), ("transfer_deltas.npz", "deltas")):
        a = np.load(os.path.join(rebuilt, "splithalf", f), allow_pickle=True)
        b = np.load(os.path.join(published, "splithalf", f), allow_pickle=True)
        a, b = (a[key], b[key]) if key else (a, b)
        if a.shape != b.shape or not np.allclose(a, b, atol=1e-5):
            bad.append(f"{f} differs")
    print("\n[h1] verification against", published)
    if bad:
        for b in bad:
            print("  MISMATCH ", b)
        print("  -> NOT the published benchmark; scores are not comparable")
        return 3
    print("  MATCH: axis, cohort, cells, labels, batches, X, all 5 A/B splits, meanpert and "
          "transfer baselines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
