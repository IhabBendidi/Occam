#!/usr/bin/env python
"""Run the repeated split-half paired evaluation on the frozen `rhj_fewshot` panel.

WHAT THIS ANSWERS
-----------------
Overall goal: predict a perturbation's effect in a context where that perturbation was never
measured, from a single control cell.

This script's question, per split: scored against the same half-A reference, how does the
model fed half-B control cells compare with the half-B perturbed cells themselves?

Varied: which cells land in half A and which in half B (5 splits). Held fixed: the cohort
(611 perturbations), the 6,983-gene axis, the checkpoint, the arm set, and the
per-perturbation row counts. See `data/benchmarks/rhj_fewshot/splithalf/CARD.md`.

How to read it: **positive gap = the experimental arm is better**, which is the expected
result. A negative gap is a red flag, not a triumph — at median `m` = 22 a low-variance
predictor can beat a two-sided-noisy reference from noise alone. The count-stratified gap
must shrink as `m` grows.

THE PROBE MODE
--------------
`--probe` runs with **no checkpoint**: the prediction is the identity on the control input,
so `model_nocal` collapses onto `ctrlcell` and only the zero-signal anchors are scored. Its
purpose is the cost measurement, because cell-eval's runtime depends on the panel (42,402
cells x 6,983 genes x 612 groups) and not at all on the model. Two arms are scored so the
real-side DE, which is shared across arms via the target-file sha256 cache, is separable:

    arm 1 (cold) = real DE + pred DE + anndata metrics
    arm 2 (warm) = pred DE + anndata metrics
    real DE      = arm 1 - arm 2

Every probe record is stamped `probe: true` and written to `metrics_probe.json`, never
`metrics.json`, so it cannot be read as a result.

    # the cost measurement, before committing 25 cell-eval runs
    python scripts/eval_splithalf.py --probe --split 0

    # (step 14) the real thing, one array task per split
    python scripts/eval_splithalf.py --split $SLURM_ARRAY_TASK_ID --checkpoint ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _ensure_src_on_path() -> None:
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


_ensure_src_on_path()

import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_DIR = os.path.join(REPO, "data", "benchmarks", "rhj_fewshot", "splithalf")


def load_frozen(panel_dir: str, split_index: int):
    """(base Panel, labels, HalfSplit, cohort, grand_mean) from the frozen artifact."""
    from occam.vcell.panel.panel_io import Panel
    from occam.vcell.panel.splithalf import HalfSplit

    z = np.load(os.path.join(panel_dir, "panel.npz"), allow_pickle=True)
    zs = np.load(os.path.join(panel_dir, "splits.npz"))
    with open(os.path.join(panel_dir, "cohort.json")) as fh:
        cohort = json.load(fh)
    with open(os.path.join(panel_dir, "splits_manifest.json")) as fh:
        man = json.load(fh)
    if man["cohort_sha256"] != cohort["cohort_sha256"]:
        raise SystemExit("cohort.json and splits_manifest.json disagree on the cohort hash; "
                         "rebuild with scripts/build_splithalf_panel.py")

    X = np.asarray(z["X"], dtype=np.float32)
    labels = np.asarray(z["labels"], dtype=str)
    symbols = [str(s) for s in z["axis_symbols"]]

    key = f"_{int(split_index)}"
    if f"pert_A{key}" not in zs:
        raise SystemExit(f"split {split_index} is not in splits.npz "
                         f"(built with --n-splits {man['n_splits']})")
    split = HalfSplit(
        index=int(split_index),
        pert_A=zs[f"pert_A{key}"], pert_B=zs[f"pert_B{key}"],
        ctrl_A=zs[f"ctrl_A{key}"], ctrl_B=zs[f"ctrl_B{key}"],
        dropped=zs[f"dropped{key}"],
        n_dropped_odd=int(len(zs[f"dropped{key}"])))

    # `panel_genes` must hold SYMBOLS: `discrimination_score_l2` excludes each
    # perturbation's own gene by NAME, and with Ensembl ids the exclusion silently
    # matches nothing and the score is quietly wrong.
    n = X.shape[1]
    base = Panel(real_X_model=X, real_labels=labels, real_counts=None,
                 ctrl_X_model=X[:0], ctrl_counts=None, ctrl_batches=None,
                 enc_cols=np.arange(n), panel_cols=np.arange(n),
                 panel_genes=symbols, n_enc=n)

    gm_path = os.path.join(panel_dir, "train_grand_mean.npy")
    grand = np.load(gm_path) if os.path.isfile(gm_path) else None
    return base, labels, split, cohort, grand, man


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel-dir", default=PANEL_DIR)
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--out-root", default="")
    ap.add_argument("--probe", action="store_true",
                    help="no checkpoint: identity prediction, anchors only, cost only")
    ap.add_argument("--probe-arms", default="ctrlcell,ctrlmean",
                    help="two arms, so the shared real-side DE cost is separable")
    ap.add_argument("--num-threads", type=int, default=16)
    ap.add_argument("--de-batch-size", type=int, default=2000,
                    help="cell-eval's default of 100 makes ~43,000 pool tasks and races")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.probe:
        raise SystemExit(
            "only --probe is implemented so far. The model-loading path lands with step 14, "
            "when there is a trained rhj_fewshot checkpoint to test it against; writing it "
            "blind would be an untested branch in the scoring path.")

    out_root = args.out_root or os.path.join(
        REPO, "runs", "rhj_splithalf", "probe" if args.probe else "eval")
    out_dir = os.path.join(out_root, f"split_{args.split:02d}")
    os.makedirs(out_dir, exist_ok=True)

    from occam.vcell.panel.protocol import run_split

    base, labels, split, cohort, grand, man = load_frozen(args.panel_dir, args.split)
    print(f"[splithalf] panel {base.real_X_model.shape[0]:,} x {base.n_keep:,} · "
          f"cohort {cohort['n_perts']} · split {split.index} · "
          f"benchmark {man['benchmark_sha256'][:16]}")

    # The probe's "model": prediction = the control input, verbatim. No checkpoint, no
    # graph, and therefore no pert2id -- the dummy below is never read, because the
    # identity predict_fn ignores its perturbation index. Stamped in the record.
    def identity(ctrl_t, pert_indices):
        return ctrl_t

    pert2id = {p: 0 for p in cohort["perts"]}
    arms = [a.strip() for a in args.probe_arms.split(",") if a.strip()]
    print(f"[splithalf] PROBE: identity prediction, scoring {arms}")

    t0 = time.time()
    rec = run_split(
        base, labels, split, cohort, out_dir,
        model=None, pert2id=pert2id, device="cpu", predict_fn=identity,
        grand_mean=grand, grand_mean_info=man.get("train_grand_mean"),
        calibrate="none", batch_size=4096, seed=args.seed,
        celleval_kwargs={"num_threads": args.num_threads,
                         "de_batch_size": args.de_batch_size},
        score_only=arms)
    total = time.time() - t0

    t = rec.get("timings_s", {})
    print(f"\n[splithalf] PROBE TIMINGS (split {split.index}, "
          f"{rec['n_rows_per_arm']:,} rows/arm, {rec['n_ctrl_B']:,} pred-side controls)")
    for tag in arms:
        print(f"  {tag:14s} {t.get(tag, float('nan')):8.1f} s")
    if len(arms) >= 2 and all(a in t for a in arms[:2]):
        cold, warm = t[arms[0]], t[arms[1]]
        print(f"  {'real-side DE':14s} {cold - warm:8.1f} s  (cold - warm)")
        print(f"  {'pred DE+metrics':14s} {warm:8.1f} s  (per additional arm)")
        print(f"\n  → 5 arms x 1 split ≈ {(cold - warm + 5 * warm) / 60:.0f} min")
        print(f"  → 5 arms x 5 splits ≈ {5 * (cold - warm + 5 * warm) / 3600:.1f} h "
              f"(≈{(cold - warm + 5 * warm) / 3600:.1f} h per array task)")
    print(f"  {'wall clock':14s} {total:8.1f} s")
    print(f"\n[splithalf] this is a PROBE — {out_dir}/metrics_probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
