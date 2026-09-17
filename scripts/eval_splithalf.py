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


def load_frozen(panel_dir: str, split_index: int, encoder_gene_names=None):
    """(base Panel, labels, HalfSplit, cohort, grand_mean, manifest, transfer) from disk.

    `encoder_gene_names` is the checkpoint's 44,728-entry Ensembl vocabulary. The panel is
    stored on the 6,983-gene axis, but the model's encoder takes the full vector, so
    `enc_cols` has to map each axis gene to its encoder column: `predict_panel` scatters
    with `full[:, panel.enc_cols] = ...`, leaving the other 37,745 columns at zero, which is
    exactly the input the model was trained on. Omit it (the probe) and the mapping is the
    identity on 6,983 columns.
    """
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
    ensembl = [str(g) for g in z["axis_ensembl"]]
    if encoder_gene_names is None:
        enc_cols, n_enc = np.arange(n), n
    else:
        pos = {g: i for i, g in enumerate(encoder_gene_names)}
        outside = [g for g in ensembl if g not in pos]
        if outside:
            raise SystemExit(
                f"{len(outside)} axis gene(s) are not in this checkpoint's vocabulary "
                f"(e.g. {outside[:5]}). Those columns could never be written, so the axis "
                f"and this checkpoint do not belong together.")
        enc_cols = np.array([pos[g] for g in ensembl], dtype=np.int64)
        n_enc = len(encoder_gene_names)
    base = Panel(real_X_model=X, real_labels=labels, real_counts=None,
                 ctrl_X_model=X[:0], ctrl_counts=None, ctrl_batches=None,
                 enc_cols=enc_cols, panel_cols=np.arange(n),
                 panel_genes=symbols, n_enc=n_enc)

    gm_path = os.path.join(panel_dir, "train_grand_mean.npy")
    grand = np.load(gm_path) if os.path.isfile(gm_path) else None

    td_path = os.path.join(panel_dir, "transfer_deltas.npz")
    transfer = None
    if os.path.isfile(td_path):
        tz = np.load(td_path, allow_pickle=True)
        transfer = {str(p): tz["deltas"][i] for i, p in enumerate(tz["perts"])}
    return base, labels, split, cohort, grand, man, transfer


def load_model(ckpt_path: str, device: str, output_activation=None):
    """A `VirtualCellTrainer` for eval only, with the activation trap closed."""
    import torch

    from occam.vcell.pipeline import VirtualCellTrainer

    if not os.path.isfile(ckpt_path):
        raise SystemExit(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    recorded = (ckpt.get("decoder_config") or {}).get("output_activation")
    if output_activation is None and recorded is None:
        raise SystemExit(
            "this checkpoint does not record `output_activation` and none was given. "
            "Rebuilding would default to 'rectified_sigmoid'; if the decoder was trained "
            "with softplus that is ~99% exact zeros -- a dead model that still loads "
            "cleanly. Pass --output-activation explicitly.")
    if recorded and output_activation and output_activation != recorded:
        raise SystemExit(
            f"--output-activation {output_activation!r} contradicts the {recorded!r} "
            f"recorded in the checkpoint. One of the two is wrong; refusing to guess.")
    act = output_activation or recorded
    print(f"[splithalf] output activation: {act}"
          f"{' (from the checkpoint)' if recorded else ' (from the flag)'}")
    trainer = VirtualCellTrainer.from_vcell_checkpoint(
        ckpt_path, device=device, output_activation=act)
    print(f"[splithalf] checkpoint epoch {trainer.current_epoch}, "
          f"best_metric {trainer.best_metric:.6f}, {len(trainer.pert2id):,} pert2id entries")
    return trainer


def assert_checkpoint_matches(ckpt_path: str, manifest) -> None:
    """Refuse a checkpoint trained on a different gene axis or a different benchmark.

    A model trained on 6,983 input columns and scored on 9,624 is silent and severe: every
    column it never saw arrives as a real number instead of a structural zero, and nothing
    in the scoring path notices.
    """
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    b = ckpt.get("benchmark")
    if not b:
        raise SystemExit(
            f"{os.path.basename(ckpt_path)} records no `benchmark` block, so the axis it "
            f"was trained on cannot be verified against the panel's "
            f"{manifest['n_axis_genes']:,}-gene axis. It predates the benchmark stage; "
            f"retrain, or score it on the panel it was built for.")
    bad = []
    for key, want in (("gene_axis_sha256", manifest["gene_axis_sha256"]),
                      ("benchmark_sha256", manifest["benchmark_sha256"]),
                      ("n_axis_genes", manifest["n_axis_genes"]),
                      ("id", manifest["benchmark_id"])):
        if b.get(key) != want:
            bad.append(f"{key}: checkpoint {b.get(key)!r} != panel {want!r}")
    if bad:
        raise SystemExit("this checkpoint was not trained on this benchmark:\n  "
                         + "\n  ".join(bad))
    print(f"[splithalf] checkpoint benchmark OK: {b['id']} · axis "
          f"{b['gene_axis_sha256'][:16]} · {b['n_axis_genes']:,} genes · "
          f"scope {b.get('axis_scope') or '?'}")
    excl = b.get("excluded_perts") or {}
    n_ex = sum(len(v) for v in excl.values())
    print(f"[splithalf] it held out {n_ex:,} (file, perturbation) pair(s) from training; "
          f"val cell lines {ckpt.get('val_cell_lines')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel-dir", default=PANEL_DIR)
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--out-root", default="")
    ap.add_argument("--probe", action="store_true",
                    help="no checkpoint: identity prediction, anchors only, cost only")
    ap.add_argument("--checkpoint", default="",
                    help="a vcell checkpoint, e.g. runs/rhj_fewshot_a00__.../vcell/"
                         "checkpoints/best_model.pth")
    ap.add_argument("--output-activation", default=None,
                    choices=["rectified_sigmoid", "softplus"],
                    help="required unless the checkpoint records one; the wrong value "
                         "loads strict=True and emits ~99%% exact zeros")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--calibrate", default="quantile",
                    choices=["none", "quantile", "both"],
                    help="control-fit quantile calibration is worth ~0.37 pearson_delta "
                         "against ~0.02 for the head/loss choice, so 'none' measures "
                         "very little. 'quantile' gives the protocol's 5 arms; 'both' "
                         "adds an uncalibrated 6th and a 6th cell-eval run.")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--mode", default="all", choices=["all", "reference", "model"],
                    help="`reference` scores only the 5 MODEL-INDEPENDENT arms (the two "
                         "anchors, experimental, transfer, meanpert) and needs NO "
                         "checkpoint -- they are identical for every checkpoint, so "
                         "scoring them once per split instead of once per checkpoint turns "
                         "27x5x6 = 810 cell-eval runs into 25 + 135. `model` scores only "
                         "the model arm and reuses the shared real-side DE cache.")
    ap.add_argument("--de-real-cache-dir", default="",
                    help="shared across splits AND checkpoints. The real-side DE depends "
                         "only on the split's target file, so one cache serves every run; "
                         "it is keyed on that file's own sha256, so a rebuilt panel misses "
                         "by construction rather than being scored against a stale truth.")
    ap.add_argument("--probe-arms", default="ctrlcell,ctrlmean",
                    help="two arms, so the shared real-side DE cost is separable")
    ap.add_argument("--num-threads", type=int, default=16)
    ap.add_argument("--de-batch-size", type=int, default=2000,
                    help="cell-eval's default of 100 makes ~43,000 pool tasks and races")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    needs_ckpt = args.mode in ("all", "model") and not args.probe
    if args.probe and args.checkpoint:
        raise SystemExit("pass either --probe or --checkpoint, not both")
    if needs_ckpt and not args.checkpoint:
        raise SystemExit(f"--mode {args.mode} needs --checkpoint (only --mode reference "
                         f"and --probe run without one)")
    if args.mode == "reference" and args.checkpoint:
        raise SystemExit("--mode reference scores only the model-INDEPENDENT arms; passing "
                         "a checkpoint would imply otherwise. Use --mode model or all.")

    out_root = args.out_root or os.path.join(
        REPO, "runs", "rhj_splithalf",
        "probe" if args.probe else ("reference" if args.mode == "reference" else "eval"))
    out_dir = os.path.join(out_root, f"split_{args.split:02d}")
    os.makedirs(out_dir, exist_ok=True)

    from occam.vcell.panel.protocol import run_split

    model = pert2id = predict_fn = None
    device, calibrate, batch_size = "cpu", "none", 4096
    encoder_genes = None
    arms = None

    if args.probe:
        arms = [a.strip() for a in args.probe_arms.split(",") if a.strip()]
    elif args.mode == "reference":
        # The five arms that do not depend on the checkpoint. `predict_panel` still runs,
        # with the identity, purely to draw `ctrl_used` -- and it draws it from the SAME
        # seeded RNG the model runs use, so `ctrl_used` is bit-identical across every run
        # of this split and `meanpert`/`transfer` cancel the model's control noise exactly.
        arms = ["ctrlmean", "ctrlcell", "experimental", "transfer", "meanpert"]
    else:
        trainer = load_model(args.checkpoint, args.device, args.output_activation)
        model, pert2id, device = trainer.model, trainer.pert2id, trainer.device
        encoder_genes = trainer.encoder_gene_names
        calibrate, batch_size = args.calibrate, args.batch_size

    base, labels, split, cohort, grand, man, transfer = load_frozen(
        args.panel_dir, args.split, encoder_gene_names=encoder_genes)
    print(f"[splithalf] panel {base.real_X_model.shape[0]:,} x {base.n_keep:,} "
          f"(encoder width {base.n_enc:,}) · cohort {cohort['n_perts']} · "
          f"split {split.index} · benchmark {man['benchmark_sha256'][:16]}")

    if args.probe or args.mode == "reference":
        # Prediction = the control input, verbatim. No checkpoint, no graph, and therefore
        # no pert2id -- the dummy below is never read, because the identity predict_fn
        # ignores its perturbation index.
        def predict_fn(ctrl_t, pert_indices):
            return ctrl_t

        pert2id = {p: 0 for p in cohort["perts"]}
        what = "PROBE" if args.probe else "REFERENCE"
        print(f"[splithalf] {what}: identity prediction, scoring {arms}")
    else:
        assert_checkpoint_matches(args.checkpoint, man)
        if args.mode == "model":
            arms = None      # build_arms emits them; score_only filters below

    # `model` mode scores only the model arm; the other five come from the reference run.
    score_only = arms
    if args.mode == "model":
        score_only = ["model_cal"] if calibrate == "quantile" else None

    t0 = time.time()
    rec = run_split(
        base, labels, split, cohort, out_dir,
        model=model, pert2id=pert2id, device=device, predict_fn=predict_fn,
        grand_mean=grand, grand_mean_info=man.get("train_grand_mean"),
        transfer_deltas=transfer, transfer_info=man.get("transfer"),
        calibrate=calibrate, batch_size=batch_size, seed=args.seed,
        celleval_kwargs={"num_threads": args.num_threads,
                         "de_batch_size": args.de_batch_size,
                         **({"de_real_cache_dir": args.de_real_cache_dir}
                            if args.de_real_cache_dir else {})},
        score_only=score_only)
    total = time.time() - t0

    if not args.probe:
        print(f"\n[splithalf] split {split.index} scored in {total / 60:.1f} min")
        for tag in (rec.get("anchor_arms", []) + rec.get("reference_arms", [])
                    + rec.get("model_arms", [])):
            m = rec["arms"].get(tag, {})
            pd_ = m.get("pearson_delta")
            print(f"  {tag:14s} pearson_delta "
                  f"{'undefined' if pd_ is None else f'{pd_:.4f}'}"
                  f"   overlap_at_N {m.get('overlap_at_N', float('nan')):.4f}"
                  f"   mae {m.get('mae', float('nan')):.4f}")
        print("\n[splithalf] ⚠️ one split is not a result. The claim is the mean of the 5 "
              "per-split paired differences -- run every split, then "
              "scripts/paired_report.py.")
        return 0

    t = rec.get("timings_s", {})
    # The SCORED order, not the requested one: the first arm scored is the one that pays
    # for the real-side DE, and build_arms fixes the emission order (anchors first).
    order = [a for a in rec.get("scored_order", arms) if a in t]
    print(f"\n[splithalf] PROBE TIMINGS (split {split.index}, "
          f"{rec['n_rows_per_arm']:,} rows/arm, {rec['n_ctrl_B']:,} pred-side controls)")
    for tag in order:
        print(f"  {tag:14s} {t[tag]:8.1f} s"
              f"{'   <- cold, includes the real-side DE' if tag == order[0] else ''}")
    if len(order) >= 2 and "scored_order" not in rec:
        print("  ⚠️ this record predates `scored_order`, so which arm paid for the "
              "real-side DE cannot be read from it. Check the job log for "
              "`cached real-side DE ->` and assign cold/warm by hand.")
    elif len(order) >= 2:
        cold, warm = t[order[0]], t[order[1]]
        if cold <= warm:
            # A negative real-DE time is impossible; it means the cold arm was
            # misidentified. Refuse rather than print it.
            print(f"  ⚠️ the first scored arm ({order[0]}) was not slower than the second "
                  f"({order[1]}): {cold:.1f} s vs {warm:.1f} s. The cold arm computes the "
                  f"real-side DE, so this cannot be right and no decomposition is printed.")
            print(f"  {'wall clock':14s} {total:8.1f} s")
            print(f"\n[splithalf] this is a PROBE — {out_dir}/metrics_probe.json")
            return 0
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
