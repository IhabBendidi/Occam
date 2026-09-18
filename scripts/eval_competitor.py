#!/usr/bin/env python
"""Phase B of the bridge: score a competitor's prediction matrix through OUR path.

WHAT THIS ANSWERS
-----------------
Overall goal: predict a perturbation's effect in a context where that perturbation was never
measured, from a single control cell.

This script's question: where does an external model sit on `rhj_fewshot`, measured on
exactly the instrument our own arms are measured on -- same 611-perturbation cohort, same 5
splits, same 6,983-gene axis, same drawn half-B controls, same cell-eval 0.6.0, same five
baselines?

HOW IT WORKS
------------
Phase A (`competitors/shared/export_predictions.py`, in the competitor's own venv) wrote a
`(18575, 6983)` float32 matrix already mapped into our evaluation space. This wraps that
matrix in a matrix-backed `predict_fn` and hands it to the existing
`occam.vcell.panel.protocol.run_split`, with `model=None`, `device="cpu"` and
`calibrate="none"` -- the same trick the reference and probe paths already use.

The competitor therefore inherits, without being able to opt out:

  * `_assert_in_pert2id`      -- refuses before predicting if a cohort perturbation is absent
  * `assert_disjoint_halves`  -- the A/B split is genuinely disjoint
  * `assert_cohort`           -- ENFORCES the frozen 611-perturbation cohort
  * the label-order check     -- `run_split` re-verifies the row order
  * the non-finite guard      -- counted and printed, never silently patched
  * the mandatory anchors     -- `build_arms` has no way to disable them
  * `pred_ctrl = ctrl_B`      -- the half-B controls on the prediction side
  * the sha256-keyed real-DE cache, shared with every other arm

⚠️ CALIBRATION: uncalibrated only. Their models emit gene space directly and were never
calibrated upstream. On our own model, control-fit quantile calibration cost 0.047
`pearson_delta` and quadrupled `mae` (docs/rhj_splithalf.md F12), so `model_nocal` is the
like-for-like column. The arm therefore comes out named `model_nocal`, because `build_arms`
deliberately refuses unknown arm names; `--out-root` is what disambiguates competitors.

    python scripts/eval_competitor.py --arm txpert --seed 42 --split 0 \
        --pred-dir runs/competitors/txpert/s42/split_0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _ensure_src_on_path() -> None:
    for sub in ("src", os.path.join("competitors", "shared")):
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), sub)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


_ensure_src_on_path()

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_DIR = os.path.join(REPO, "data", "benchmarks", "rhj_fewshot", "splithalf")


def load_prediction(pred_dir: str, expect_labels) -> np.ndarray:
    """`pred.npy`, with its row order and provenance verified against the panel."""
    pred_path = os.path.join(pred_dir, "pred.npy")
    if not os.path.isfile(pred_path):
        raise SystemExit(
            f"no pred.npy in {pred_dir}. Run Phase A first:\n"
            f"  competitors/shared/export_predictions.py, in the arm's own venv")
    pred = np.load(pred_path)

    lab_path = os.path.join(pred_dir, "labels.json")
    if os.path.isfile(lab_path):
        with open(lab_path) as fh:
            got = [str(x) for x in json.load(fh)]
        want = [str(x) for x in expect_labels]
        if got != want:
            first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), len(want))
            raise SystemExit(
                f"{pred_dir}: the exported row order does not match this split.\n"
                f"  {len(got)} exported rows vs {len(want)} expected; first mismatch at "
                f"{first}: {got[first] if first < len(got) else '<end>'!r} vs "
                f"{want[first] if first < len(want) else '<end>'!r}\n"
                f"  Rows must be the half-A perturbed cells ordered by "
                f"argsort(labels, kind='stable'). Re-export with the matching --split.")
    else:
        print(f"[compet] ⚠️ no labels.json in {pred_dir}; the row order is UNVERIFIED. "
              f"run_split will still re-check it, but re-export to get the earlier, "
              f"clearer failure.")

    man_path = os.path.join(pred_dir, "manifest.json")
    if os.path.isfile(man_path):
        with open(man_path) as fh:
            man = json.load(fh)
        print(f"[compet] exported by {man.get('arm')} from "
              f"target_sum {man.get('emission', {}).get('from_target_sum')} "
              f"(log1p={man.get('emission', {}).get('from_log1p')})")
    return pred


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, help="txpert | state | cellflow | perturbdiff")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--pred-dir", default="",
                    help="default runs/competitors/<arm>/s<seed>/split_<k>")
    ap.add_argument("--panel-dir", default=PANEL_DIR)
    ap.add_argument("--out-root", default="",
                    help="default runs/rhj_splithalf/compet_<arm>_s<seed>")
    ap.add_argument("--de-real-cache-dir",
                    default=os.path.join(REPO, "runs", "rhj_splithalf",
                                         "_de_real_shared"),
                    help="shared with every other arm; the real-side DE depends only on the "
                         "split's target file and is keyed on its sha256")
    ap.add_argument("--num-threads", type=int, default=16)
    ap.add_argument("--de-batch-size", type=int, default=2000)
    ap.add_argument("--skip-space-check", action="store_true",
                    help="NOT recommended. Phase A already ran assert_representable; this "
                         "re-runs it here because the two phases can drift.")
    args = ap.parse_args(argv)

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "eval_splithalf", os.path.join(REPO, "scripts", "eval_splithalf.py"))
    es = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(es)

    from occam.vcell.panel.protocol import run_split

    pred_dir = args.pred_dir or os.path.join(
        REPO, "runs", "competitors", args.arm, f"s{args.seed}", f"split_{args.split}")
    out_root = args.out_root or os.path.join(
        REPO, "runs", "rhj_splithalf", f"compet_{args.arm}_s{args.seed}")
    out_dir = os.path.join(out_root, f"split_{args.split:02d}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.de_real_cache_dir, exist_ok=True)

    # No encoder vocabulary: competitors are natively 6,983 wide, no padding.
    base, labels, split, cohort, grand, man, transfer = es.load_frozen(
        args.panel_dir, args.split, encoder_gene_names=None)
    print(f"[compet] {args.arm} s{args.seed} · panel {base.real_X_model.shape[0]:,} x "
          f"{base.n_keep:,} · cohort {cohort['n_perts']} · split {split.index}")

    # the row order run_split will re-check, reproduced here so the error is early and clear
    lab = np.asarray(labels, dtype=str)
    a_rows = np.asarray(split.pert_A, dtype=np.int64)
    a_rows = a_rows[np.argsort(lab[a_rows], kind="stable")]
    pred = load_prediction(pred_dir, lab[a_rows])

    if pred.shape != (len(a_rows), base.n_keep):
        raise SystemExit(
            f"pred.npy is {pred.shape}, expected {(len(a_rows), base.n_keep)} for split "
            f"{args.split}")

    if not args.skip_space_check:
        from emission_space import assert_representable, space_diagnostics
        assert_representable(pred, n_genes=base.n_keep,
                             name=f"{args.arm} pred.npy")
        rep = space_diagnostics(pred, base.real_X_model[a_rows],
                                pred_name=f"{args.arm} pred.npy",
                                target_name="panel half-A")
        print(f"[compet] representable ✓   rowsum ratio {rep['rowsum_ratio']:.4f} · "
              f"gene-mean corr {rep['gene_mean_corr']:.4f} · "
              f"zero {rep['zero_fraction_pred']:.4f}")
        print(f"[compet] ⚠️ diagnostics only -- library size and density are confounded "
              f"with model quality; OUR OWN arm is 0.9399 / 0.9998 / 0.0000")

    cursor = {"i": 0}

    def predict_fn(ctrl_t, pert_indices):
        """Serve the precomputed rows in order. `ctrl_t` is ignored by construction."""
        import torch
        n = ctrl_t.shape[0]
        i = cursor["i"]
        out = pred[i:i + n]
        cursor["i"] = i + n
        return torch.from_numpy(np.ascontiguousarray(out))

    rec = run_split(
        base, labels, split, cohort, out_dir,
        model=None, pert2id={p: 0 for p in cohort["perts"]}, device="cpu",
        predict_fn=predict_fn,
        grand_mean=grand, grand_mean_info=man.get("train_grand_mean"),
        transfer_deltas=transfer, transfer_info=man.get("transfer"),
        calibrate="none", batch_size=4096, seed=args.seed,
        celleval_kwargs={"num_threads": args.num_threads,
                         "de_batch_size": args.de_batch_size,
                         "de_real_cache_dir": args.de_real_cache_dir},
        score_only=["model_nocal"])

    if cursor["i"] != len(pred):
        raise SystemExit(
            f"predict_fn served {cursor['i']:,} of {len(pred):,} rows; the panel and the "
            f"exported matrix disagree about how many cells this split has")

    m = rec["arms"].get("model_nocal", {})
    print(f"\n[compet] {args.arm} s{args.seed} split {args.split}")
    for k in ("pearson_delta", "mae", "overlap_at_N", "discrimination_score_l1"):
        v = m.get(k)
        print(f"  {k:26s} {'—' if v is None else f'{v:.4f}'}")
    print("\n[compet] baselines, all 5 splits: ctrlcell -0.0018 · meanpert 0.1324 · "
          "transfer 0.2323 · experimental 0.3722")
    print("[compet] ⚠️ one split is not a result; run all 5 then scripts/paired_report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
