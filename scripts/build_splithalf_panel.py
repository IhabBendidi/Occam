#!/usr/bin/env python
"""Freeze the split-half evaluation panel for `rhj_fewshot`: cells, cohort, splits.

WHAT THIS IS FOR
----------------
Objective: report how far the model is from **experimental reproducibility** on the 621
held-out HepG2 perturbations, not from an arbitrary floor. The reviewer asked for a repeated
paired split-half protocol, and its whole purpose is to stop one split -- or one
power-dependent DE gene set -- from determining the result.

This script freezes everything the protocol must not be free to vary:

    panel.npz               the scored cells, preprocessed exactly as model input
    cohort.json             the perturbation set, frozen once, identical in all 25 files
    splits.npz              per-split row indices for A and B, perturbed and control
    splits_manifest.json    the seed recipe, the hashes, the batch composition, the drops
    train_grand_mean.npy    the mean-training-perturbation baseline's profile
    CARD.md                 what it is, and the two things it cannot express

Question it answers: nothing on its own. It is the fixed input that makes
`occam.vcell.panel.protocol` reproducible -- run twice, byte-identical.

WHY THE COHORT IS FROZEN ONCE AND NEVER INTERSECTED AGAIN
---------------------------------------------------------
`discrimination_score_l2` is a rank over the cohort and BH-FDR is computed over the whole
flattened DE frame, so changing the membership silently rescales both. The cohort is
therefore written once, `min_cells=10` (m >= 5 per half), and every file the protocol emits
is checked against it. It is deliberately NOT intersected with the model's `pert2id`: that
would make the benchmark checkpoint-dependent. The protocol refuses up front instead.

WHAT THIS PANEL CANNOT EXPRESS
------------------------------
* **The experimental arm is not an upper bound at these cell counts.** Reference and
  experimental prediction are both ~m-cell pseudobulks (median m 22), so that arm is
  attenuated on both sides while a low-variance predictor is attenuated on one. Read the
  count-stratified gap: it must shrink as m grows.
* **`ctrlmean`'s `pearson_delta` is undefined by construction** -- see `protocol.py`.

USAGE
-----
    python scripts/build_splithalf_panel.py                  # the frozen default
    python scripts/build_splithalf_panel.py --n-splits 5 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import date
from typing import Dict, List


def _ensure_src_on_path() -> None:
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


_ensure_src_on_path()

import numpy as np  # noqa: E402

from occam.h5ad_columns import read_column  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILDER_VERSION = "1.0.0"

# The evaluated context. HepG2 is the only one with held-out perturbations; RPE1 and Jurkat
# train on all of them, which is the entire point of the benchmark's per-(file, pert) split.
CELL_LINE = "HEPG2"
SOURCE = "nadig_2024_hepg2.h5ad"
PERT_COL = "gene_name"
BATCH_COL = "dataset_batch_num"
CONTROL = "non-targeting"

# The two contexts the transfer arm reads its per-perturbation deltas from. HepG2 is
# excluded on purpose: it is the held-out context, and including it would leak the answer.
TRANSFER_SOURCES = ("nadig_2024_jurkat.h5ad", "replogle_2022_rpe1_e.h5ad")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _load_axis_restricted(path: str, axis_ensembl: List[str], target_sum: float):
    """(X on the axis, log1p-normalised), perturbation labels, batch labels.

    ⚠️ Library-size normalisation uses the cell's **full-panel** row sum over all 9,624 file
    genes, not the sum over the 6,983 axis genes -- which is what `stream_v2.load_chunk_sparse`
    does (`rowsum` is taken BEFORE alignment). Normalising by the axis sum instead would put
    the panel in a different space from the model's input and quietly break every metric.
    """
    import h5py

    with h5py.File(path, "r") as f:
        ens = read_column(f["var"], "ensembl_gene_id")
        if ens is None:
            raise KeyError(f"{path}: var has no ensembl_gene_id column")
        pos = {g: i for i, g in enumerate(ens)}
        missing = [g for g in axis_ensembl if g not in pos]
        if missing:
            raise ValueError(
                f"{len(missing)} axis gene(s) are absent from {os.path.basename(path)} "
                f"(e.g. {missing[:5]}). The axis is the 3-way intersection, so this means "
                f"the file changed under the benchmark.")
        keep = np.array([pos[g] for g in axis_ensembl], dtype=np.int64)

        labels = np.asarray(read_column(f["obs"], PERT_COL), dtype=str)
        batches = np.asarray(read_column(f["obs"], BATCH_COL), dtype=str)

        X = f["X"]
        if not (isinstance(X, h5py.Group) and "indptr" in X):
            raise ValueError(f"{path}: X is not CSR; this builder reads sparse rows only")
        n_obs = len(labels)
        n_vars = len(ens)
        indptr = np.asarray(X["indptr"][:], dtype=np.int64)
        data, indices = X["data"], X["indices"]

        # local file column -> position on the axis, -1 for the 2,641 dropped genes
        col_of = np.full(n_vars, -1, dtype=np.int64)
        col_of[keep] = np.arange(len(keep))

        out = np.zeros((n_obs, len(keep)), dtype=np.float32)
        CH = 8192
        for s in range(0, n_obs, CH):
            e = min(s + CH, n_obs)
            lo, hi = int(indptr[s]), int(indptr[e])
            v = np.asarray(data[lo:hi], dtype=np.float32)
            c = np.asarray(indices[lo:hi], dtype=np.int64)
            lens = np.diff(indptr[s:e + 1])
            row = np.repeat(np.arange(e - s, dtype=np.int64), lens)
            # the row sum over the FULL panel, before the axis restriction
            rs = np.bincount(row, weights=v.astype(np.float64), minlength=e - s)
            rs = np.where(rs > 0, rs, 1.0)
            tgt = col_of[c]
            m = tgt >= 0
            out[s + row[m], tgt[m]] = np.log1p(
                v[m] / rs[row[m]].astype(np.float32) * float(target_sum))
            if (s // CH) % 4 == 0:
                print(f"    read {e:,}/{n_obs:,} rows", flush=True)
    return out, labels, batches


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench-dir", default=os.path.join(REPO, "data", "benchmarks",
                                                        "rhj_fewshot"))
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-cells", type=int, default=10)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.bench_dir, "splithalf")
    os.makedirs(out_dir, exist_ok=True)

    from occam.vcell.benchmark import load_benchmark
    from occam.vcell.panel.splithalf import (assert_disjoint_halves, draw_splits,
                                             freeze_cohort, train_pert_grand_mean)

    bench = load_benchmark(args.bench_dir, expect_id="rhj_fewshot", verify_sources=False)
    print(f"[panel] {bench.summary()}")

    src = dict(bench.raw["sources"][SOURCE])
    with open(os.path.join(args.bench_dir, "files",
                           SOURCE.replace(".h5ad", ".json"))) as fh:
        src.update(json.load(fh))
    target_sum = float(bench.raw["encoder_target_sum"])
    test_perts = sorted(bench.split["test_unseen_pert"])

    # The HepG2 TRAINING perturbations, read from the artifact rather than inferred from the
    # labels: `split == train` is exactly the 1,657 the model trained on in this context.
    # Inferring it as "every label that is not val or test" would silently fold in the 10
    # ineligible perturbations the model never saw.
    import csv
    with open(os.path.join(args.bench_dir, "perturbations.csv")) as fh:
        prows = list(csv.DictReader(fh))
    hepg2_train = sorted(r["gene_name"] for r in prows if r["split"] == "train")
    print(f"[panel] {len(hepg2_train):,} HepG2 training perturbations (artifact `split`)")

    # The split is row indices into THIS file. If it moved or changed, the indices point at
    # different cells and every number downstream is quietly wrong.
    from occam.vcell.benchmark import _sha256 as _file_sha256
    have = _file_sha256(src["path"])
    if have != src["sha256"]:
        raise SystemExit(
            f"{SOURCE} has changed since the benchmark was built "
            f"({have[:16]} != {src['sha256'][:16]}). The frozen split refers to rows in the "
            f"old file; refusing to build a panel against the new one.")

    print(f"[panel] reading {SOURCE} on the {len(bench.axis_ensembl):,}-gene axis "
          f"(target_sum {target_sum:g}, full-panel row sums)")
    X, labels, batches = _load_axis_restricted(src["path"], bench.axis_ensembl, target_sum)
    print(f"[panel] {X.shape[0]:,} cells x {X.shape[1]:,} genes")

    # ---- the frozen cohort ------------------------------------------------
    cohort = freeze_cohort(
        labels, eligible=test_perts, min_cells=args.min_cells, context=CELL_LINE,
        extra={"benchmark_id": bench.benchmark_id, "benchmark_sha256": bench.sha256,
               "gene_axis_sha256": bench.axis_sha256, "n_axis_genes": len(bench.axis_ensembl),
               "source": SOURCE, "source_sha256": src["sha256"],
               "builder_version": BUILDER_VERSION, "built": str(date.today()),
               "pert_col_in_source": PERT_COL, "batch_col": BATCH_COL})
    short = [p for p in test_perts if p not in set(cohort["perts"])]
    print(f"[panel] cohort {cohort['n_perts']}/{len(test_perts)} perturbations "
          f"(min_cells={args.min_cells}); {len(short)} too shallow")
    m = np.array(sorted(cohort["half_size"].values()))
    print(f"[panel] half size m: median {int(np.median(m))}, min {int(m.min())}, "
          f"max {int(m.max())} · m>=5 {int((m >= 5).sum())} · m>=10 "
          f"{int((m >= 10).sum())} · m>=25 {int((m >= 25).sum())}")

    # ---- only the scored cells go in the panel ----------------------------
    keep_rows = np.flatnonzero(np.isin(labels, list(cohort["perts"]) + [CONTROL]))
    Xp, lp, bp = X[keep_rows], labels[keep_rows], batches[keep_rows]
    print(f"[panel] panel rows {len(keep_rows):,} "
          f"({int((lp == CONTROL).sum()):,} controls)")

    # ---- the mean-training-perturbation baseline, from the FULL file ------
    # Precomputed here so the panel need not carry 1,657 training perturbations' cells.
    grand_info: Dict[str, object] = {}
    grand = None
    if hepg2_train:
        grand, grand_info = train_pert_grand_mean(X, labels, hepg2_train)
        np.save(os.path.join(out_dir, "train_grand_mean.npy"), grand)
        print(f"[panel] train grand mean over {grand_info['n_perts']:,} HepG2 training "
              f"perturbations ({grand_info['total_cells']:,} cells, equal weight per "
              f"perturbation)")

    # ---- the CROSS-CELL-LINE TRANSFER deltas ------------------------------
    # For each cohort perturbation: the count-weighted mean of its (perturbed mean - that
    # file's control mean) over Jurkat and RPE1, on this same axis. Adding this to a HepG2
    # control cell is the non-learned version of exactly what the model is asked to do, and
    # on the validation side it beat every one of the 24 FiLM arms by 0.20 (docs
    # rhj_fewshot.md F5). It must therefore appear in the benchmark table, or the report
    # omits the thing that wins.
    #
    # ⚠️ HepG2 is deliberately NOT one of the source files. Including it would read the
    # held-out context's own answer.
    deltas = np.zeros((cohort["n_perts"], Xp.shape[1]), dtype=np.float64)
    weights = np.zeros(cohort["n_perts"], dtype=np.float64)
    pert_index = {p: i for i, p in enumerate(cohort["perts"])}
    transfer_info: Dict[str, object] = {"sources": [], "weighting": "cells per (file, pert)"}
    for name in TRANSFER_SOURCES:
        rec = dict(bench.raw["sources"][name])
        with open(os.path.join(args.bench_dir, "files",
                               name.replace(".h5ad", ".json"))) as fh:
            rec.update(json.load(fh))
        have = _file_sha256(rec["path"])
        if have != rec["sha256"]:
            raise SystemExit(f"{name} has changed since the benchmark was built "
                             f"({have[:16]} != {rec['sha256'][:16]})")
        print(f"[panel] transfer source {name} on the axis")
        Xs, ls, _bs = _load_axis_restricted(rec["path"], bench.axis_ensembl, target_sum)
        cmean = Xs[ls == CONTROL].mean(axis=0)
        n_hit = 0
        for pert, i in pert_index.items():
            rows = np.flatnonzero(ls == pert)
            if rows.size == 0:
                continue
            deltas[i] += (Xs[rows].mean(axis=0) - cmean) * rows.size
            weights[i] += rows.size
            n_hit += 1
        transfer_info["sources"].append(
            {"file": name, "sha256": rec["sha256"], "cell_line": rec.get("cell_line"),
             "n_controls": int((ls == CONTROL).sum()),
             "n_cohort_perts_present": n_hit})
        print(f"[panel]   {n_hit}/{cohort['n_perts']} cohort perturbations present, "
              f"{int((ls == CONTROL).sum()):,} controls")
        del Xs, ls, _bs

    missing_transfer = [p for p, i in pert_index.items() if weights[i] == 0]
    if missing_transfer:
        raise SystemExit(
            f"{len(missing_transfer)} cohort perturbation(s) appear in NEITHER transfer "
            f"source (e.g. {missing_transfer[:5]}). Every cohort perturbation is supposed "
            f"to be present in all three files; refusing to write a transfer arm with "
            f"holes in it.")
    deltas /= weights[:, None]
    np.savez_compressed(os.path.join(out_dir, "transfer_deltas.npz"),
                        deltas=deltas.astype(np.float32),
                        perts=np.asarray(cohort["perts"], dtype=object))
    transfer_info["n_perts"] = int(len(pert_index))
    transfer_info["cells_per_pert_median"] = float(np.median(weights))
    print(f"[panel] transfer deltas: {deltas.shape}, median "
          f"{int(np.median(weights))} source cells per perturbation")

    # ---- the splits -------------------------------------------------------
    splits = draw_splits(lp, bp, cohort, n_splits=args.n_splits, base_seed=args.seed)
    for sp in splits:
        assert_disjoint_halves(sp)
        print(f"[panel] {sp.summary()}")

    np.savez_compressed(
        os.path.join(out_dir, "panel.npz"),
        X=Xp, labels=lp, batches=bp,
        axis_ensembl=np.asarray(bench.axis_ensembl, dtype=object),
        axis_symbols=np.asarray(bench.axis_symbols, dtype=object),
        source_rows=keep_rows)
    np.savez_compressed(
        os.path.join(out_dir, "splits.npz"),
        **{f"{k}_{sp.index}": getattr(sp, k)
           for sp in splits
           for k in ("pert_A", "pert_B", "ctrl_A", "ctrl_B", "dropped")})
    with open(os.path.join(out_dir, "cohort.json"), "w") as fh:
        json.dump(cohort, fh, indent=2, sort_keys=True)

    manifest = {
        "protocol": "splithalf/v1",
        "builder_version": BUILDER_VERSION,
        "built": str(date.today()),
        "git_sha": _git_sha(),
        "command": " ".join(sys.argv),
        "benchmark_id": bench.benchmark_id,
        "benchmark_sha256": bench.sha256,
        "gene_axis_sha256": bench.axis_sha256,
        "n_axis_genes": len(bench.axis_ensembl),
        "source": SOURCE, "source_sha256": src["sha256"],
        "cell_line": CELL_LINE, "pert_col_in_source": PERT_COL, "batch_col": BATCH_COL,
        "control_label": CONTROL,
        "target_sum": target_sum,
        "normalisation": "full-panel row sum over all file genes, then axis restriction, "
                         "then log1p -- matches stream_v2.load_chunk_sparse",
        "n_panel_rows": int(Xp.shape[0]),
        "n_panel_controls": int((lp == CONTROL).sum()),
        "cohort_sha256": cohort["cohort_sha256"],
        "n_cohort_perts": int(cohort["n_perts"]),
        "min_cells": int(args.min_cells),
        "too_shallow": short,
        "seed_recipe": "blake2b(SPLIT_SALT | base_seed | split_index | group), one stream "
                       "PER (split, perturbation) -- so adding a perturbation to the cohort "
                       "does not reshuffle any other perturbation's halves",
        "base_seed": int(args.seed),
        "n_splits": int(args.n_splits),
        "half_size_strata": {"m>=5": int((m >= 5).sum()), "m>=10": int((m >= 10).sum()),
                             "m>=25": int((m >= 25).sum()), "m_median": int(np.median(m))},
        "train_grand_mean": {k: v for k, v in grand_info.items() if k != "cells_per_pert"},
        "transfer": transfer_info,
        "splits": [],
    }
    for sp in splits:
        cb = np.unique(bp[sp.ctrl_A], return_counts=True)
        cbb = np.unique(bp[sp.ctrl_B], return_counts=True)
        manifest["splits"].append({
            "split": int(sp.index),
            "n_pert_per_half": int(len(sp.pert_A)),
            "n_ctrl_per_half": int(len(sp.ctrl_A)),
            "n_dropped_odd": int(sp.n_dropped_odd),
            "pert_A_sha256": _sha256_bytes(np.ascontiguousarray(sp.pert_A).tobytes()),
            "pert_B_sha256": _sha256_bytes(np.ascontiguousarray(sp.pert_B).tobytes()),
            "ctrl_A_sha256": _sha256_bytes(np.ascontiguousarray(sp.ctrl_A).tobytes()),
            "ctrl_B_sha256": _sha256_bytes(np.ascontiguousarray(sp.ctrl_B).tobytes()),
            # the control halves are where stratification genuinely works and is mandatory:
            # an unstratified split leaves an (A_ctrl_mean - B_ctrl_mean) batch term that
            # enters the real and the predicted delta with OPPOSITE signs
            "ctrl_batch_max_abs_diff": int(max(
                abs(int(dict(zip(*cb)).get(b, 0)) - int(dict(zip(*cbb)).get(b, 0)))
                for b in set(cb[0]) | set(cbb[0]))),
        })
    with open(os.path.join(out_dir, "splits_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    # ---- self-verification: re-read what was just written -----------------
    print("\n[panel] self-verification")
    z = np.load(os.path.join(out_dir, "panel.npz"), allow_pickle=True)
    zs = np.load(os.path.join(out_dir, "splits.npz"))
    checks = [
        ("panel rows round-trip", z["X"].shape == Xp.shape),
        ("panel labels round-trip", np.array_equal(z["labels"], lp)),
        ("axis length", len(z["axis_ensembl"]) == len(bench.axis_ensembl)),
        ("cohort is the eligible test set", set(cohort["perts"]) <= set(test_perts)),
        ("every panel label is cohort or control",
         set(lp.tolist()) == set(cohort["perts"]) | {CONTROL}),
        ("halves are equal per split",
         all(len(zs[f"pert_A_{i}"]) == len(zs[f"pert_B_{i}"])
             for i in range(args.n_splits))),
        ("halves are disjoint per split",
         all(np.intersect1d(zs[f"pert_A_{i}"], zs[f"pert_B_{i}"]).size == 0
             for i in range(args.n_splits))),
        ("control halves are equal",
         all(len(zs[f"ctrl_A_{i}"]) == len(zs[f"ctrl_B_{i}"])
             for i in range(args.n_splits))),
        ("grand mean is on the axis",
         grand is None or grand.shape == (len(bench.axis_ensembl),)),
        ("transfer deltas cover every cohort perturbation",
         np.load(os.path.join(out_dir, "transfer_deltas.npz"),
                 allow_pickle=True)["deltas"].shape
         == (cohort["n_perts"], len(bench.axis_ensembl))),
        ("transfer deltas are finite",
         bool(np.isfinite(np.load(os.path.join(out_dir, "transfer_deltas.npz"),
                                  allow_pickle=True)["deltas"]).all())),
        ("transfer perts are the cohort, in order",
         list(np.load(os.path.join(out_dir, "transfer_deltas.npz"),
                      allow_pickle=True)["perts"]) == list(cohort["perts"])),
        ("no non-finite value in the panel", bool(np.isfinite(z["X"]).all())),
    ]
    bad = [n for n, ok in checks if not ok]
    for n, ok in checks:
        print(f"  {'OK  ' if ok else 'FAIL'} {n}")
    if bad:
        print(f"\n[panel] {len(bad)} check(s) FAILED: {bad}")
        return 1
    print(f"\n[panel] {len(checks)}/{len(checks)} checks passed -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
