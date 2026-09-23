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
    # ⚠️ The hash cross-check is the guard against a panel built from one cohort and split
    # from another. `norman_combo`'s manifest does not carry the key (its builder writes the
    # cohort hash only into cohort.json), so a MISSING key is tolerated while a MISMATCHED
    # one still refuses -- silence and disagreement are different things.
    if "cohort_sha256" in man and man["cohort_sha256"] != cohort["cohort_sha256"]:
        raise SystemExit("cohort.json and splits_manifest.json disagree on the cohort hash; "
                         "rebuild the panel")

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

    # norman_combo: the ADDITIVITY baseline, `delta_A + delta_B` per pair. Absent on
    # rhj_fewshot (single-gene perturbations have nothing to compose), present on any
    # combinatorial panel — so its absence is normal and its presence adds one arm.
    sos_path = os.path.join(panel_dir, "sum_of_singles.npz")
    sos = None
    if os.path.isfile(sos_path):
        sz = np.load(sos_path, allow_pickle=True)
        sos = {str(p): sz["deltas"][i] for i, p in enumerate(sz["perts"])}
    return base, labels, split, cohort, grand, man, transfer, sos


def load_model(ckpt_path: str, device: str, output_activation=None):
    """`(trainer, ckpt)` for eval only, with the activation trap closed.

    ⚠️ `from_vcell_checkpoint` returns a TUPLE `(trainer, ckpt)`. Binding it to one name
    gives a tuple that looks like a trainer until the first attribute access -- which is
    how 135 array tasks died at `trainer.current_epoch`. `scripts/eval_panel.py` had the
    same bug. The checkpoint dict is returned here so callers reuse it instead of loading
    898 MB twice.
    """
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
    trainer, loaded = VirtualCellTrainer.from_vcell_checkpoint(
        ckpt_path, device=device, output_activation=act)
    print(f"[splithalf] checkpoint epoch {trainer.current_epoch}, "
          f"best_metric {trainer.best_metric:.6f}, {len(trainer.pert2id):,} pert2id entries")
    return trainer, loaded


def _cohort_labels(panel_dir: str):
    """Every perturbation the panel will score, from cohort.json. None if unavailable."""
    p = os.path.join(panel_dir or "", "cohort.json")
    if not panel_dir or not os.path.isfile(p):
        return None
    with open(p) as fh:
        cohort = json.load(fh)
    per = cohort.get("cells_per_pert") or {}
    return sorted(str(k) for k in per) or None


def _leakage_check(ckpt, cohort):
    """(ok, lines) -- is every scored perturbation actually held out of this checkpoint?

    This is the property the benchmark digest is a PROXY for, checked directly: a
    perturbation the model trained on must not appear on the scoring side. Verified
    against both records the checkpoint carries -- `benchmark.excluded_perts` (what the
    training stage deliberately held out, per source file) and `split_info.train` (the
    perturbations it actually trained on).
    """
    excluded = set()
    for v in (ckpt.get("benchmark", {}).get("excluded_perts") or {}).values():
        excluded |= {str(x) for x in v}
    trained = {str(x) for x in (ckpt.get("split_info", {}).get("train") or [])}
    missing = [q for q in cohort if q not in excluded]
    leaked = sorted(q for q in cohort if q in trained)
    lines = []
    if leaked:
        lines.append(f"{len(leaked)} scored perturbation(s) are in this checkpoint's "
                     f"OWN TRAINING LIST, e.g. {leaked[:5]} -- scoring here would be "
                     f"measuring memorisation")
    if missing:
        lines.append(f"{len(missing)} scored perturbation(s) are not in the "
                     f"checkpoint's `excluded_perts`, e.g. {missing[:5]} -- the training "
                     f"run did not deliberately hold them out")
    return (not leaked and not missing), lines


def assert_checkpoint_matches(ckpt_path: str, manifest, ckpt=None, panel_dir=None,
                              require_exact: bool = False) -> None:
    """Refuse a checkpoint trained on a different gene axis or a different benchmark.

    A model trained on 6,983 input columns and scored on 9,624 is silent and severe: every
    column it never saw arrives as a real number instead of a structural zero, and nothing
    in the scoring path notices.

    ⛔ REBUILT PANELS ARE ACCEPTED, deliberately, since 2026-09-22. `benchmark_sha256` is
    `sha256(benchmark.json)`, and that file embeds `git_sha`, the build date and the
    output path -- so a panel rebuilt with `scripts/repro/build_benchmark_*.sh` can be
    identical in every scientific quantity and still carry a different digest (measured:
    f458c2d9... vs 8ac7da0d...). Requiring byte equality therefore made the published
    checkpoints unusable on a panel a reader built themselves, which is the normal thing
    for a reader to do.

    So when the digests differ, the guard does NOT give up: it checks the property the
    digest is a proxy for, which is stronger than the digest itself --

      1. the SCORING AXIS is identical (`gene_axis_sha256` is a content hash), the
         benchmark `id`, `n_axis_genes` and `axis_scope` agree; and
      2. every perturbation the panel will score is HELD OUT of this checkpoint: absent
         from `split_info.train` and present in `benchmark.excluded_perts`.

    A different split cannot pass (2), and a different axis cannot pass (1). Pass
    `require_exact=True` (`--require-exact-benchmark`) to restore byte equality.
    """
    if ckpt is None:
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
                      ("n_axis_genes", manifest["n_axis_genes"]),
                      ("id", manifest["benchmark_id"])):
        if b.get(key) != want:
            bad.append(f"{key}: checkpoint {b.get(key)!r} != panel {want!r}")
    digest_differs = b.get("benchmark_sha256") != manifest.get("benchmark_sha256")
    if bad:
        raise SystemExit("this checkpoint was not trained on this benchmark:\n  "
                         + "\n  ".join(bad))
    if digest_differs:
        if require_exact:
            raise SystemExit(
                f"benchmark_sha256: checkpoint {b.get('benchmark_sha256')!r} != panel "
                f"{manifest.get('benchmark_sha256')!r}, and --require-exact-benchmark "
                f"was given. Drop that flag to accept a verified rebuild.")
        cohort = _cohort_labels(panel_dir)
        if cohort is None:
            raise SystemExit(
                f"benchmark_sha256 differs (checkpoint {str(b.get('benchmark_sha256'))[:16]}"
                f"... != panel {str(manifest.get('benchmark_sha256'))[:16]}...) and "
                f"{panel_dir}/cohort.json is missing, so the hold-out cannot be verified "
                f"another way. Score against the panel the checkpoint was built for.")
        ok, lines = _leakage_check(ckpt, cohort)
        if not ok:
            raise SystemExit(
                "benchmark_sha256 differs AND the hold-out does not check out, so this "
                "is a different benchmark rather than a rebuild of the same one:\n  "
                + "\n  ".join(lines))
        print(f"[splithalf] benchmark_sha256 DIFFERS "
              f"(checkpoint {str(b.get('benchmark_sha256'))[:16]}... != panel "
              f"{str(manifest.get('benchmark_sha256'))[:16]}...), accepted as a REBUILD:")
        print(f"[splithalf]   same scoring axis ({b['gene_axis_sha256'][:16]}..., "
              f"{b['n_axis_genes']:,} genes), same benchmark id, same axis scope")
        print(f"[splithalf]   all {len(cohort):,} scored perturbation(s) held out of "
              f"training (absent from split_info.train, present in excluded_perts)")
        print(f"[splithalf]   that digest is sha256(benchmark.json), which embeds "
              f"git_sha / build date / output path and so cannot survive a rebuild; the "
              f"hold-out check above is the property it stands in for")
    print(f"[splithalf] checkpoint benchmark OK: {b['id']} · axis "
          f"{b['gene_axis_sha256'][:16]} · {b['n_axis_genes']:,} genes · "
          f"scope {b.get('axis_scope') or '?'}")
    excl = b.get("excluded_perts") or {}
    n_ex = sum(len(v) for v in excl.values())
    print(f"[splithalf] it held out {n_ex:,} (file, perturbation) pair(s) from training; "
          f"val cell lines {ckpt.get('val_cell_lines')}")


def assert_ood_context(ckpt_path: str, manifest, ckpt=None, panel_dir=None,
                       out_dir=None) -> dict:
    """`--ood-context`: accept a checkpoint on a panel from a DIFFERENT cell line/benchmark.

    Used for `h1_seen`: rhj_fewshot checkpoints scored on H1, a cell line none of them saw.
    The two refusals of `assert_checkpoint_matches` are correct for a rebuild of the same
    benchmark and wrong here, by design: the axis is a SUBSET of the training axis, and the
    cohort perturbations ARE in training (that is the point: seen pert, unseen line).

    What is still enforced:
      * the panel must declare itself `ood_context` in its manifest, so the flag cannot be
        used to push a checkpoint through a same-context panel it does not match;
      * the panel's `seen_from.gene_axis_sha256` must be the axis this checkpoint trained on;
      * every panel axis gene must be in the checkpoint's gene vocabulary (load_frozen).
    It writes `context_shift.json` recording the seen / not-seen overlap per cohort pert.
    """
    if ckpt is None:
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not manifest.get("ood_context"):
        raise SystemExit("--ood-context given, but this panel's splits_manifest.json does not "
                         "declare `ood_context: true`. It is a same-context panel; drop the "
                         "flag and let the benchmark check run.")
    b = ckpt.get("benchmark") or {}
    src = manifest.get("seen_from") or {}
    if b.get("gene_axis_sha256") != src.get("gene_axis_sha256"):
        raise SystemExit(
            f"--ood-context: checkpoint axis {str(b.get('gene_axis_sha256'))[:16]} is not the "
            f"axis this panel was derived from ({str(src.get('gene_axis_sha256'))[:16]}, "
            f"{src.get('benchmark_id')})")
    cohort = _cohort_labels(panel_dir) or []
    trained = {str(x) for x in (ckpt.get("split_info", {}).get("train") or [])}
    excluded = set()
    for v in (b.get("excluded_perts") or {}).values():
        excluded |= {str(x) for x in v}
    rec = {
        "checkpoint": ckpt_path, "checkpoint_benchmark": b.get("id"),
        "panel_benchmark": manifest.get("benchmark_id"),
        "panel_cell_line": manifest.get("cell_line"),
        "panel_n_axis_genes": manifest.get("n_axis_genes"),
        "checkpoint_n_axis_genes": b.get("n_axis_genes"),
        "n_cohort": len(cohort),
        "n_in_split_info_train": sum(q in trained for q in cohort),
        "n_in_excluded_perts": sum(q in excluded for q in cohort),
    }
    print(f"[splithalf] OOD CONTEXT: {b.get('id')} checkpoint on {rec['panel_benchmark']} "
          f"({rec['panel_cell_line']}), axis {rec['panel_n_axis_genes']} of "
          f"{rec['checkpoint_n_axis_genes']} genes; {rec['n_in_split_info_train']}/"
          f"{len(cohort)} cohort perts in the checkpoint's training list")
    if out_dir:
        with open(os.path.join(out_dir, "context_shift.json"), "w") as fh:
            json.dump(rec, fh, indent=2, sort_keys=True)
    return rec


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
    ap.add_argument("--center-delta", action="store_true",
                    help="also emit `model_nocal_cdel`: ctrl + (delta - panel-mean delta). "
                         "Removes the component of the predicted effect common to EVERY "
                         "perturbation -- the quantity `meanpert` scores. Expected signature: "
                         "discrimination UP, mae/mse DOWN. On H1 it SPLIT the checkpoints "
                         "(+0.1394 on one, -0.3335 on others), so a single-base result says "
                         "nothing about the others.")
    ap.add_argument("--compose", default="decode_seq_mean",
                    help="multi-perturbation composition mode (norman_combo). DEFAULT "
                         "`decode_seq_mean` since 2026-09-22: best mean rank (2.08/8) and "
                         "best single cell over the 72-cell matrix in docs/norman_results.md, "
                         "and a NO-OP on single-perturbation panels. One of "
                         "sum | latent_seq_{ab,ba,mean} | decode_seq_{ab,ba,mean} | "
                         "delta_sum | delta_sum_realctrl | resid_seq_{ab,ba,mean}. "
                         "⚠️ The modes are IDENTICAL on a single perturbation, so this is a "
                         "no-op on rhj_fewshot; it only changes doubles. ⚠️ The "
                         "`*_realctrl`/`resid_seq_*` family also changes the EMISSION "
                         "convention (deltas added to the real control, not to the model's "
                         "reconstruction), so its numbers are comparable within that family "
                         "and to `delta_sum_realctrl`, not directly to `sum`.")
    ap.add_argument("--require-exact-benchmark", action="store_true",
                    help="refuse unless the panel's `benchmark_sha256` byte-matches the "
                         "one the checkpoint recorded. Off by default: that digest is "
                         "sha256(benchmark.json), which embeds git_sha, the build date "
                         "and the output path, so a panel rebuilt by "
                         "scripts/repro/build_benchmark_*.sh cannot reproduce it even "
                         "when identical in content. Without this flag a rebuild is "
                         "accepted after verifying the same scoring axis AND that every "
                         "scored perturbation is held out of the checkpoint's training.")
    ap.add_argument("--ood-context", action="store_true",
                    help="score a checkpoint on a panel from a cell line it never saw "
                         "(e.g. h1_seen). Replaces the same-benchmark check with "
                         "assert_ood_context; the panel must declare `ood_context`.")
    ap.add_argument("--calibrate", default="none",
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
    ap.add_argument("--arms", default="",
                    help="in --mode reference, score only these arms. For retrying a "
                         "single arm after the transient pdex shared-memory race "
                         "(FileNotFoundError: /psm_...) without repaying the other four.")
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

    model = pert2id = predict_fn = loaded_ckpt = None
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
        # ⚠️ Which reference arms EXIST depends on the panel, so the list is resolved
        # against the artifact below rather than hardcoded: `transfer` needs
        # transfer_deltas.npz (rhj_fewshot has it; a single-cell-line combinatorial panel
        # cannot) and `sum_of_singles` needs sum_of_singles.npz (norman_combo has it;
        # single-gene perturbations have nothing to compose). Requesting an arm the panel
        # cannot build used to leave it silently absent from the record, which reads as
        # "scored 0" rather than "not applicable".
        arms = ["ctrlmean", "ctrlcell", "experimental", "meanpert"]
        if args.arms:
            want = [a.strip() for a in args.arms.split(",") if a.strip()]
            unknown = sorted(set(want) - set(arms))
            if unknown:
                raise SystemExit(f"--arms {unknown} are not reference arms; "
                                 f"choose from {arms}")
            arms = want
            print(f"[splithalf] --arms restricts this pass to {arms}; the record it "
                  f"writes will cover only those, so merge it rather than overwrite")
    else:
        trainer, loaded_ckpt = load_model(
            args.checkpoint, args.device, args.output_activation)
        model, pert2id, device = trainer.model, trainer.pert2id, trainer.device
        encoder_genes = trainer.encoder_gene_names
        calibrate, batch_size = args.calibrate, args.batch_size
        # `sum` is `forward()` itself, so the wrapper is only installed for the other
        # modes -- keeping the rhj_fewshot call path byte-for-byte what it was when
        # --compose defaulted to `sum`. The default is now `decode_seq_mean`, which takes
        # this branch and returns pred1 unchanged on a single perturbation.
        if args.compose != "sum":
            from occam.vcell.model import VirtualCellModel
            if args.compose not in VirtualCellModel.COMPOSE_MODES:
                raise SystemExit(f"--compose {args.compose!r} is not a mode; expected "
                                 f"{list(VirtualCellModel.COMPOSE_MODES)}")
            _m, _mode = model, args.compose

            def predict_fn(ctrl_t, pert_idx):
                return _m.forward_composed(ctrl_t, pert_idx, _mode)["prediction"]
            print(f"[splithalf] composition mode: {args.compose} "
                  f"({'REAL-control emission -- read within its own family' if args.compose in VirtualCellModel.REAL_BASE_MODES else 'reconstruction-based emission'})")

    base, labels, split, cohort, grand, man, transfer, sos = load_frozen(
        args.panel_dir, args.split, encoder_gene_names=encoder_genes)

    if args.mode == "reference" and not args.arms:
        # resolved now that the panel is loaded: only the arms it can actually build
        if transfer is not None:
            arms.insert(3, "transfer")
        if sos is not None:
            arms.append("sum_of_singles")
        print(f"[splithalf] reference arms for this panel: {arms}")
    print(f"[splithalf] panel {base.real_X_model.shape[0]:,} x {base.n_keep:,} "
          f"(encoder width {base.n_enc:,}) · cohort {cohort['n_perts']} · "
          f"split {split.index} · benchmark "
          f"{str(man.get('benchmark_sha256') or man.get('gene_axis_sha256') or 'n/a')[:16]}")

    if args.probe or args.mode == "reference":
        # Prediction = the control input, verbatim. No checkpoint, no graph, and therefore
        # no pert2id -- the dummy below is never read, because the identity predict_fn
        # ignores its perturbation index.
        def predict_fn(ctrl_t, pert_indices):
            return ctrl_t

        # PER GENE, not per label (same fix as scripts/eval_competitor.py): the identity
        # predict_fn never reads it, and `_assert_in_pert2id` rightly refuses a whole "A+B"
        # key, so a label-keyed map made every norman_combo reference run die at start
        # (seeds 43-46, 2026-09-23). Identical on single-gene panels.
        pert2id = {g: 0 for p in cohort["perts"] for g in str(p).split("+") if g}
        what = "PROBE" if args.probe else "REFERENCE"
        print(f"[splithalf] {what}: identity prediction, scoring {arms}")
    else:
        if args.ood_context:
            assert_ood_context(args.checkpoint, man, ckpt=loaded_ckpt,
                               panel_dir=args.panel_dir, out_dir=out_dir)
        else:
            assert_checkpoint_matches(args.checkpoint, man, ckpt=loaded_ckpt,
                                      panel_dir=args.panel_dir,
                                      require_exact=args.require_exact_benchmark)
        if args.mode == "model":
            arms = None      # build_arms emits them; score_only filters below

    # `model` mode scores only the model arm; the other five come from the reference run.
    score_only = arms
    if args.mode == "model":
        # Only the model arms; the other five come from the reference pass. With
        # `both` that is the calibrated AND uncalibrated prediction, which is how the
        # ~0.37 pearson_delta that calibration is worth on comparable panels gets
        # measured on THIS one rather than assumed.
        score_only = {"quantile": ["model_cal"],
                      "none": ["model_nocal"],
                      "both": ["model_cal", "model_nocal"]}[calibrate]
        if args.center_delta and calibrate in ("none", "both"):
            score_only = list(score_only) + ["model_nocal_cdel"]

    t0 = time.time()
    rec = run_split(
        base, labels, split, cohort, out_dir,
        model=model, pert2id=pert2id, device=device, predict_fn=predict_fn,
        grand_mean=grand, grand_mean_info=man.get("train_grand_mean"),
        transfer_deltas=transfer, transfer_info=man.get("transfer"),
        sos_deltas=sos, sos_info=man.get("sum_of_singles"),
        calibrate=calibrate, center_delta=bool(args.center_delta),
        batch_size=batch_size, seed=args.seed,
        probe=args.probe,
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
