#!/usr/bin/env python
"""Pass 4: freeze the Norman-2019 double-perturbation panel — cohort, val holdout, 5 splits,
and the `sum_of_singles` baseline.

WHAT WE ARE TRYING TO ACHIEVE OVERALL
    Predict a perturbation's effect in a context where that exact perturbation was never
    measured. `rhj_fewshot` tests an unseen perturbation in a seen cell line; THIS benchmark
    tests COMPOSITION — train on single-gene perturbations, predict PAIRS, where both genes
    were seen individually and the pair never was.

WHAT THIS ARTIFACT ENABLES
    Scored against the same half-A reference, how does a model fed half-B control cells compare
    with (a) the half-B perturbed cells themselves (`experimental`) and (b) the sum of the two
    measured single-gene deltas (`sum_of_singles`)?

WHAT IS VARIED / HELD FIXED
    Varied: which cells land in half A and which in half B — 5 independent batch-stratified
    splits. Held fixed: the cohort, the 8,000-gene axis, the val holdout, and the per-pair cell
    counts (`m(p) = floor(n_p/2)` depends only on `n_p`).

WHAT EACH OUTCOME MEANS
    A model below `sum_of_singles` has learned nothing about interaction: our architecture
    composes by SUMMING perturbation embeddings (`model.py`, the `max_perts > 1` branch), so
    the additive baseline is that architecture's own hypothesis in closed form. A model above
    it has learned something non-additive. `experimental` is the empirical reference, NOT a
    ceiling — at these counts both sides are noisy pseudobulks (see the rhj CARD).

⚠️ THE VAL HOLDOUT IS 20 DOUBLE PAIRS, NOT SINGLES. F-ckpt4 measured that our validation
metric selects epochs well and misranks arms; the mitigation is that the validation quantity
must be the same KIND as the test one. A val set of singles would select checkpoints on
single-gene prediction and be blind to composition failure — the whole question. Cost: 20 of
131 pairs.

⚠️ CONTROLS ARE SHARED between the training corpus and the panel, exactly as in `rhj_fewshot`
(whose panel controls also came from the training file). The held-out quantity is the
PERTURBATION, not the control distribution.

    python scripts/norman/norman_build_panel.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import date

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

CONTROL = "non-targeting"
BUILDER_VERSION = "norman_combo/v1.0.0"


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:                                                    # noqa: BLE001
        return "unknown"


def _load_axis(path, axis_ensembl, target_sum):
    """(X on the axis, log1p-normalised), labels, batches — the rhj convention exactly.

    ⚠️ Library-size normalisation uses the cell's row sum over ALL 33,694 file genes, not over
    the 8,000 axis genes, because `stream_v2.load_chunk_sparse` takes `rowsum` BEFORE gene
    alignment. Normalising by the axis sum would put the panel in a different space from the
    model's own input.
    """
    import h5py

    with h5py.File(path, "r") as f:
        def col(g, k):
            o = f[g][k]
            if isinstance(o, h5py.Group) and "codes" in o:
                cats = [c.decode() if isinstance(c, bytes) else str(c)
                        for c in o["categories"][:]]
                return np.asarray(cats)[o["codes"][:]]
            d = o[:]
            return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in d]) \
                if d.dtype.kind in "SO" else d
        ens = list(col("var", "gene_id"))
        pos = {g: i for i, g in enumerate(ens)}
        missing = [g for g in axis_ensembl if g not in pos]
        if missing:
            raise SystemExit(f"{len(missing)} axis genes absent from the source "
                             f"(e.g. {missing[:5]}); the axis and the file disagree")
        keep = np.array([pos[g] for g in axis_ensembl], dtype=np.int64)
        labels = np.asarray(col("obs", "gene_name"), dtype=str)
        batches = np.asarray(col("obs", "batch"), dtype=str)
        X = f["X"]
        indptr = np.asarray(X["indptr"][:], dtype=np.int64)
        data, indices = X["data"], X["indices"]
        n_obs, n_vars = len(labels), len(ens)
        col_of = np.full(n_vars, -1, dtype=np.int64)
        col_of[keep] = np.arange(len(keep))
        out = np.zeros((n_obs, len(keep)), dtype=np.float32)
        CH = 8192
        for s in range(0, n_obs, CH):
            e = min(s + CH, n_obs)
            lo, hi = int(indptr[s]), int(indptr[e])
            v = np.asarray(data[lo:hi], dtype=np.float32)
            c = np.asarray(indices[lo:hi], dtype=np.int64)
            row = np.repeat(np.arange(e - s, dtype=np.int64), np.diff(indptr[s:e + 1]))
            rs = np.bincount(row, weights=v.astype(np.float64), minlength=e - s)
            rs = np.where(rs > 0, rs, 1.0)
            tgt = col_of[c]
            m = tgt >= 0
            out[s + row[m], tgt[m]] = np.log1p(
                v[m] / rs[row[m]].astype(np.float32) * float(target_sum))
    return out, labels, batches


def _val_holdout(pairs, counts, degree, n_val, seed):
    """Stratified val pairs: cross of (cell-count tercile x min-degree tercile).

    Both axes plausibly predict difficulty, and an unstratified draw of 20 from 131 can easily
    take the 20 smallest pairs or 20 hub-gene pairs. Deterministic given the seed.
    """
    rng = np.random.default_rng(seed)
    cnt = np.array([counts[p] for p in pairs])
    deg = np.array([min(degree[g] for g in p.split("+")) for p in pairs])
    ct = np.searchsorted(np.quantile(cnt, [1 / 3, 2 / 3]), cnt, side="right")
    dt = np.searchsorted(np.quantile(deg, [1 / 3, 2 / 3]), deg, side="right")
    cells = {}
    for i, p in enumerate(pairs):
        cells.setdefault((int(ct[i]), int(dt[i])), []).append(p)
    keys = sorted(cells)
    per = {k: len(cells[k]) for k in keys}
    total = sum(per.values())
    take = {k: int(round(n_val * per[k] / total)) for k in keys}
    # fix rounding drift deterministically, largest strata first
    while sum(take.values()) != n_val:
        order = sorted(keys, key=lambda k: (-per[k], k))
        for k in order:
            if sum(take.values()) == n_val:
                break
            if sum(take.values()) < n_val and take[k] < per[k]:
                take[k] += 1
            elif sum(take.values()) > n_val and take[k] > 0:
                take[k] -= 1
    val = []
    for k in keys:
        pool = sorted(cells[k])
        val += list(rng.choice(pool, size=take[k], replace=False))
    return sorted(val), {"strata": {str(k): {"pairs": per[k], "val": take[k]} for k in keys}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="data/eval/pert/norman_2019_k562.h5ad")
    ap.add_argument("--axis", default="data/benchmarks/norman_combo/axis.npz")
    ap.add_argument("--out-dir", default="data/benchmarks/norman_combo/splithalf")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-cells", type=int, default=40)
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--target-sum", type=float, default=2000.0)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)

    from occam.vcell.panel.splithalf import (assert_disjoint_halves, draw_splits,
                                             freeze_cohort, train_pert_grand_mean)

    az = np.load(a.axis, allow_pickle=True)
    axis_ens = [str(x) for x in az["axis_ensembl"]]
    axis_sym = [str(x) for x in az["axis_symbols"]]
    axis_sha = str(az["axis_sha256"])
    src_sha = hashlib.sha256(open(a.source, "rb").read()).hexdigest()
    print(f"[panel] source {a.source} sha256 {src_sha[:16]}…")
    print(f"[panel] axis {len(axis_ens):,} genes sha256 {axis_sha[:16]}…")

    X, labels, batches = _load_axis(a.source, axis_ens, a.target_sum)
    print(f"[panel] {X.shape[0]:,} cells x {X.shape[1]:,} axis genes "
          f"(target_sum {a.target_sum:g}, FULL-file row sums)")

    is_double = np.array(["+" in l for l in labels])
    is_ctrl = labels == CONTROL
    is_single = ~is_double & ~is_ctrl
    pairs = sorted(set(labels[is_double]))
    singles = sorted(set(labels[is_single]))
    counts = {p: int((labels == p).sum()) for p in pairs}
    degree = {}
    for p in pairs:
        for g in p.split("+"):
            degree[g] = degree.get(g, 0) + 1
    print(f"[panel] {len(pairs)} double pairs · {len(singles)} single conditions · "
          f"{int(is_ctrl.sum()):,} controls")

    val_pairs, val_info = _val_holdout(pairs, counts, degree, a.n_val, a.seed)
    test_pairs = [p for p in pairs if p not in set(val_pairs)]
    print(f"[panel] val holdout {len(val_pairs)} pairs (stratified), test {len(test_pairs)}")
    print(f"        val: {', '.join(val_pairs)}")

    cohort = freeze_cohort(
        labels, eligible=test_pairs, min_cells=a.min_cells, context="K562",
        extra={"benchmark_id": "norman_combo", "task": "train on singles, predict doubles",
               "gene_axis_sha256": axis_sha, "n_axis_genes": len(axis_ens),
               "source": os.path.basename(a.source), "source_sha256": src_sha,
               "builder_version": BUILDER_VERSION, "built": str(date.today()),
               "git_sha": _git_sha(), "val_pairs": val_pairs, "val_strata": val_info,
               "n_single_conditions": len(singles), "single_genes": sorted(
                   set(g for s in singles for g in s.split("+")))})
    m = np.array(sorted(cohort["half_size"].values()))
    short = [p for p in test_pairs if p not in set(cohort["perts"])]
    print(f"[panel] cohort {cohort['n_perts']}/{len(test_pairs)} pairs "
          f"(min_cells={a.min_cells}); {len(short)} too shallow: {short}")
    print(f"[panel] half size m: median {int(np.median(m))} min {int(m.min())} "
          f"max {int(m.max())} · m>=20 {int((m>=20).sum())} · m>=40 {int((m>=40).sum())}")

    keep_rows = np.flatnonzero(np.isin(labels, list(cohort["perts"]) + [CONTROL]))
    Xp, lp, bp = X[keep_rows], labels[keep_rows], batches[keep_rows]
    print(f"[panel] panel rows {len(keep_rows):,} ({int((lp == CONTROL).sum()):,} controls)")

    # ---- meanpert: grand mean over the TRAINING (single) perturbations ----
    grand, grand_info = train_pert_grand_mean(X, labels, singles)
    np.save(os.path.join(a.out_dir, "train_grand_mean.npy"), grand)
    print(f"[panel] meanpert grand mean over {grand_info['n_perts']} single conditions "
          f"({grand_info['total_cells']:,} cells, equal weight per condition)")

    # ---- sum_of_singles: THE baseline this benchmark exists to beat -------
    # For pair (A, B): delta_A + delta_B, each measured on the TRAINING single arms of the
    # same file, against the same control mean. Both orientations of a single gene
    # (`X+ctrl` and `ctrl+X` upstream) are pooled here because they are the same
    # perturbation -- our labels already collapse them to the bare gene symbol.
    cmean = X[is_ctrl].mean(axis=0)
    sdelta = {}
    for s in singles:
        rows = np.flatnonzero(labels == s)
        sdelta[s] = X[rows].mean(axis=0) - cmean
    sos = np.zeros((cohort["n_perts"], X.shape[1]), dtype=np.float32)
    missing = []
    for i, p in enumerate(cohort["perts"]):
        gs = p.split("+")
        if not all(g in sdelta for g in gs):
            missing.append(p)
            continue
        sos[i] = sum(sdelta[g] for g in gs)
    if missing:
        raise SystemExit(
            f"{len(missing)} cohort pair(s) have a gene with NO single arm (e.g. "
            f"{missing[:5]}). Every pair is supposed to have both; refusing to write a "
            f"sum_of_singles baseline with holes in it.")
    np.savez_compressed(os.path.join(a.out_dir, "sum_of_singles.npz"),
                        deltas=sos, perts=np.asarray(cohort["perts"], dtype=object))
    n_single_cells = {s: int((labels == s).sum()) for s in singles}
    print(f"[panel] sum_of_singles: {sos.shape}, from {len(singles)} single arms "
          f"(median {int(np.median(list(n_single_cells.values())))} cells each)")

    # ---- the splits -------------------------------------------------------
    splits = draw_splits(lp, bp, cohort, n_splits=a.n_splits, base_seed=a.seed)
    for sp in splits:
        assert_disjoint_halves(sp)
    np.savez_compressed(
        os.path.join(a.out_dir, "splits.npz"),
        # ⚠️ KEY CONVENTION: `<field>_<split>`, e.g. `pert_A_0`. `eval_splithalf.load_frozen`
        # reads exactly this and raises "split N is not in splits.npz" on anything else --
        # which is what `s0_pert_A` produced on the first build.
        **{f"{k}_{sp.index}": getattr(sp, k)
           for sp in splits for k in ("pert_A", "pert_B", "ctrl_A", "ctrl_B", "dropped")})
    np.savez_compressed(
        os.path.join(a.out_dir, "panel.npz"),
        X=Xp, labels=np.asarray(lp, dtype=object), batches=np.asarray(bp, dtype=object),
        axis_ensembl=np.asarray(axis_ens, dtype=object),
        axis_symbols=np.asarray(axis_sym, dtype=object),
        source_rows=keep_rows.astype(np.int64))
    with open(os.path.join(a.out_dir, "cohort.json"), "w") as fh:
        json.dump(cohort, fh, indent=2, sort_keys=True, default=str)

    ctrl_bal = []
    for sp in splits:
        ba = np.asarray(bp)[sp.ctrl_A]; bb = np.asarray(bp)[sp.ctrl_B]
        d = {b: abs(int((ba == b).sum()) - int((bb == b).sum())) for b in set(bp)}
        ctrl_bal.append(max(d.values()))
    _ident = f"{axis_sha}|{cohort['cohort_sha256']}|{a.n_splits}|{a.seed}"
    manifest = {
        "benchmark_id": "norman_combo", "builder_version": BUILDER_VERSION,
        # A stable identity for the benchmark DEFINITION (axis + cohort + split recipe), not
        # for the file bytes. `eval_splithalf` prints it so a result can be traced to the
        # artifact that produced it; omitting it made the first launch die with
        # KeyError: 'benchmark_sha256'.
        "benchmark_sha256": hashlib.sha256(_ident.encode()).hexdigest(),
        "benchmark_sha256_recipe": "sha256(gene_axis_sha256|cohort_sha256|n_splits|base_seed)",
        "cohort_sha256": cohort["cohort_sha256"],
        "built": str(date.today()), "git_sha": _git_sha(),
        "source": os.path.basename(a.source), "source_sha256": src_sha,
        "gene_axis_sha256": axis_sha, "n_axis_genes": len(axis_ens),
        "task": "train on single-gene perturbations, predict double-gene pairs",
        "cell_line": "K562 (upstream file mislabels it A549; see docs/norman_benchmark.md)",
        "n_splits": a.n_splits, "base_seed": a.seed, "min_cells": a.min_cells,
        "seed_recipe": "blake2b(SPLIT_SALT | base_seed | split_index | group), one stream per "
                       "(split, group) so adding a pair reshuffles no other pair",
        "cohort_pairs": cohort["n_perts"], "val_pairs": val_pairs,
        "val_stratification": val_info,
        "single_conditions": len(singles), "controls": int(is_ctrl.sum()),
        "half_size_m": {"median": int(np.median(m)), "min": int(m.min()), "max": int(m.max()),
                        "ge_20": int((m >= 20).sum()), "ge_40": int((m >= 40).sum()),
                        "ge_60": int((m >= 60).sum())},
        "per_split": [{"index": sp.index, "pert_per_half": int(len(sp.pert_A)),
                       "ctrl_per_half": int(len(sp.ctrl_A)),
                       "dropped_odd": int(sp.n_dropped_odd)} for sp in splits],
        "control_half_batch_max_imbalance": ctrl_bal,
        "baselines": ["ctrlcell", "ctrlmean", "meanpert", "sum_of_singles", "experimental"],
        "meanpert_source": grand_info,
    }
    with open(os.path.join(a.out_dir, "splits_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, default=str)
    print(f"\n[panel] wrote {a.out_dir}")
    print(f"  per split: {manifest['per_split'][0]['pert_per_half']:,} perturbed per half, "
          f"{manifest['per_split'][0]['ctrl_per_half']:,} controls per half")
    print(f"  control-half batch imbalance (max per batch, per split): {ctrl_bal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
