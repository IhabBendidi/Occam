#!/usr/bin/env python
"""The exclusion gate from the METRIC side: held-out vs seen perturbations, same cell line.

    python scripts/verify_rhj_exclusion_by_metric.py \
        --checkpoint runs/rhj_fewshot_a03__.../vcell/checkpoints/best_model.pth

WHAT THIS ANSWERS
-----------------
Overall goal: predict a perturbation's effect in a context where that perturbation was never
measured, from a single control cell. The whole benchmark rests on the 724 held-out
perturbations being genuinely unseen **in HepG2** while remaining trained in Jurkat and RPE1.

`scripts/verify_rhj_exclusion.py` proves that from the DATA side: it reads the eligibility
gate and shows zero held-out HepG2 rows and nonzero Jurkat/RPE1 rows. That is conclusive
about which rows the loader admits — and it is the only check that runs before training.

**This script asks the same question of the trained model instead**, which is a different
and weaker-but-independent line of evidence: score the 103 **held-out** HepG2 validation
perturbations and an equally sized random sample of HepG2 **training** perturbations, on the
same cells, the same axis, the same controls, the same code path.

Varied: whether the perturbation was in training for this cell line. Held fixed: everything
else — cell line, gene axis, control pool, batch composition, eval seed, sample size.

WHAT EACH OUTCOME MEANS, pre-registered
---------------------------------------
* **seen ≫ held-out** — expected, and the size of the drop is itself a finding: it is the
  few-shot generalisation gap this benchmark exists to measure. On the H1 panel the
  equivalent gap was −0.203 `pearson_delta`, which is the whole distance from memorisation
  to chance.
* **seen ≈ held-out** — ⛔ two possibilities, and they are not equally likely. Either the
  exclusion silently failed and the "held-out" perturbations were trained on (check
  `verify_rhj_exclusion.py` and the checkpoint's `benchmark.excluded_perts`), or the model
  genuinely transfers from the other two cell lines. The data-side gate distinguishes them:
  if it passed, the collapse is a real transfer result and a strong one.
* **held-out > seen** — an implausible ordering; suspect the two sets were swapped, or that
  the seen sample is dominated by shallow perturbations. The per-set cell counts are printed
  for exactly this.

⚠️ This is not a benchmark number. It uses the VALIDATION path and its group-mean metric,
which exists for checkpoint selection. The benchmark result is the split-half protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List


def _ensure_src_on_path() -> None:
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


_ensure_src_on_path()

import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_DIR = os.path.join(REPO, "data", "benchmarks", "rhj_fewshot", "splithalf")
BENCH_DIR = os.path.join(REPO, "data", "benchmarks", "rhj_fewshot")
CONTROL = "non-targeting"


def _load_hepg2_axis(bench_dir: str):
    """The HepG2 cells on the frozen axis, reusing the split-half panel builder's reader."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_splithalf_panel", os.path.join(REPO, "scripts", "build_splithalf_panel.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from occam.vcell.benchmark import load_benchmark

    bench = load_benchmark(bench_dir, expect_id="rhj_fewshot", verify_sources=False)
    src = dict(bench.raw["sources"]["nadig_2024_hepg2.h5ad"])
    with open(os.path.join(bench_dir, "files", "nadig_2024_hepg2.json")) as fh:
        src.update(json.load(fh))
    X, labels, batches = mod._load_axis_restricted(
        src["path"], bench.axis_ensembl, float(bench.raw["encoder_target_sum"]))
    return bench, X, labels, batches


def _matched_sample(seen: List[str], target: List[str], counts: Dict[str, int],
                    rng) -> List[str]:
    """A seen sample matched to the held-out set's cell-count distribution, not uniform.

    A uniform draw over 1,657 training perturbations has a different depth profile from the
    103 held-out ones, and pearson_delta at these counts is depth-sensitive -- so an
    unmatched sample would confound "was it trained on" with "how many cells does it have".
    Matching is by nearest cell count, sampled without replacement.
    """
    pool = sorted(seen, key=lambda p: counts.get(p, 0))
    pool_counts = np.array([counts.get(p, 0) for p in pool], dtype=float)
    picked, used = [], set()
    for p in target:
        want = counts.get(p, 0)
        order = np.argsort(np.abs(pool_counts - want), kind="stable")
        for i in order:
            if pool[i] not in used:
                picked.append(pool[i])
                used.add(pool[i])
                break
    rng.shuffle(picked)
    return picked


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bench-dir", default=BENCH_DIR)
    ap.add_argument("--panel-dir", default=PANEL_DIR)
    ap.add_argument("--output-activation", default=None,
                    choices=["rectified_sigmoid", "softplus"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-controls", type=int, default=2000,
                    help="control cells shared by both sets, drawn once")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "eval_splithalf", os.path.join(REPO, "scripts", "eval_splithalf.py"))
    es = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(es)

    from occam.vcell.panel.panel_io import Panel
    from occam.vcell.panel.predict import predict_panel

    trainer = es.load_model(args.checkpoint, args.device, args.output_activation)

    bench, X, labels, batches = _load_hepg2_axis(args.bench_dir)
    counts = {p: int(n) for p, n in zip(*np.unique(labels, return_counts=True))}

    heldout = sorted(set(bench.split["val"]) & set(labels.tolist()))
    excluded = set().union(*bench.exclude_file_perts.values())
    seen_all = sorted({p for p in set(labels.tolist())
                       if p != CONTROL and p not in excluded})
    rng = np.random.default_rng(args.seed)
    seen = _matched_sample(seen_all, heldout, counts, rng)

    print(f"[check] held-out (val) perturbations : {len(heldout)}")
    print(f"[check] seen (training) sample       : {len(seen)} of {len(seen_all)}, "
          f"matched on cell count")
    hc = [counts[p] for p in heldout]
    sc = [counts[p] for p in seen]
    print(f"[check] cells/pert  held-out median {int(np.median(hc))} "
          f"(min {min(hc)}, max {max(hc)}) · seen median {int(np.median(sc))} "
          f"(min {min(sc)}, max {max(sc)})")
    if not heldout or not seen:
        raise SystemExit("one of the two sets is empty; nothing to compare")

    # the shared control pool, drawn ONCE so control noise is common to both sets
    ctrl_rows = np.flatnonzero(labels == CONTROL)
    take = rng.choice(ctrl_rows, size=min(args.n_controls, len(ctrl_rows)),
                      replace=False)
    pos = {g: i for i, g in enumerate(trainer.encoder_gene_names)}
    enc_cols = np.array([pos[g] for g in bench.axis_ensembl], dtype=np.int64)
    symbols = list(bench.axis_symbols)

    out: Dict[str, object] = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "benchmark_sha256": bench.sha256, "gene_axis_sha256": bench.axis_sha256,
        "cell_line": "HEPG2", "seed": args.seed,
        "n_controls_shared": int(len(take)),
        "sets": {}, "note": "validation path, group-mean metric; NOT a benchmark number",
    }

    for name, perts in (("heldout", heldout), ("seen", seen)):
        rows = np.flatnonzero(np.isin(labels, perts))
        panel = Panel(
            real_X_model=np.concatenate([X[rows], X[take]]),
            real_labels=np.concatenate([labels[rows],
                                        np.full(len(take), CONTROL)]),
            real_counts=None,
            ctrl_X_model=X[take], ctrl_counts=None, ctrl_batches=None,
            enc_cols=enc_cols, panel_cols=np.arange(X.shape[1]),
            panel_genes=symbols, n_enc=len(trainer.encoder_gene_names))
        missing = [p for p in perts if int(trainer.pert2id.get(p, -1)) < 0]
        if missing:
            raise SystemExit(f"{name}: {len(missing)} perturbation(s) have no pert2id "
                             f"entry ({missing[:5]}); predict_panel would drop them "
                             f"silently and the two sets would not be comparable")
        pred, plabels, ctrl_used, _ = predict_panel(
            trainer.model, panel, dict(trainer.pert2id), trainer.device,
            batch_size=512, seed=args.seed)

        # group-mean pearson_delta against the SHARED control pseudobulk
        cmean = panel.ctrl_X_model.mean(axis=0)
        pds = []
        for p in perts:
            sel = np.flatnonzero(np.asarray(plabels, dtype=str) == p)
            if sel.size == 0:
                continue
            rsel = np.flatnonzero(labels[rows] == p)
            dp = pred[sel].mean(axis=0) - cmean
            dr = X[rows][rsel].mean(axis=0) - cmean
            if dp.std() < 1e-12 or dr.std() < 1e-12:
                continue
            pds.append(float(np.corrcoef(dp, dr)[0, 1]))
        out["sets"][name] = {
            "n_perts": len(perts), "n_scored": len(pds),
            "n_cells": int(len(rows)),
            "cells_per_pert_median": int(np.median([counts[p] for p in perts])),
            "pearson_delta_groupmean_mean": float(np.mean(pds)) if pds else None,
            "pearson_delta_groupmean_median": float(np.median(pds)) if pds else None,
            "perts": perts,
        }
        print(f"[check] {name:8s} n={len(pds):4d}  "
              f"pearson_delta (group-mean) mean "
              f"{np.mean(pds):.4f}  median {np.median(pds):.4f}")

    h = out["sets"]["heldout"]["pearson_delta_groupmean_mean"]
    s = out["sets"]["seen"]["pearson_delta_groupmean_mean"]
    gap = None if (h is None or s is None) else s - h
    out["gap_seen_minus_heldout"] = gap
    print(f"\n[check] gap (seen − held-out) = {gap:+.4f}")
    if gap is None:
        verdict = "INDETERMINATE — one set scored nothing"
    elif gap > 0.02:
        verdict = ("EXPECTED — seen perturbations score higher. The size of this gap is "
                   "the few-shot generalisation gap the benchmark measures.")
    elif gap > -0.02:
        verdict = ("⚠️ COLLAPSED — seen and held-out score the same. Either the exclusion "
                   "failed (check verify_rhj_exclusion.py and the checkpoint's "
                   "benchmark.excluded_perts) or the model genuinely transfers from Jurkat "
                   "and RPE1. The data-side gate decides which; if it passed, this is a "
                   "real transfer result.")
    else:
        verdict = ("⛔ INVERTED — held-out scores HIGHER than seen. Implausible; suspect "
                   "the sets were swapped or the matched sample is degenerate.")
    out["verdict"] = verdict
    print(f"[check] {verdict}")

    path = args.out or os.path.join(REPO, "runs", "rhj_exclusion_by_metric.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print(f"[check] wrote {path}")
    print("[check] ⚠️ validation path, group-mean metric, checkpoint-selection scale. "
          "The benchmark result is scripts/eval_splithalf.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
