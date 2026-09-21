"""The reviewer's repeated split-half paired evaluation, run end to end.

WHAT THIS IS FOR
----------------
Objective: say how far the model is from **experimental reproducibility** on held-out HepG2
perturbations, rather than how far it is from an arbitrary floor. The question this module
answers, per split: *scored against the same half-A reference, how does the model fed half-B
control cells compare with the half-B perturbed cells themselves?*

Varied: which cells land in half A and which in half B (5 independent, batch-stratified
splits). Held fixed: the frozen cohort, the gene axis, the checkpoint, the arm set, and the
per-perturbation cell counts -- `m(p) = floor(n_p/2)` depends only on `n_p`, so every arm in
every split emits exactly the same rows for the same perturbation.

What each outcome means. A **negative** gap (model better than the experimental arm) is not
a triumph: at these cell counts (median m 22) reference and experimental prediction are
*both* noisy pseudobulks while a low-variance predictor is attenuated on one side only, so
a model can beat the "ceiling" from noise alone. Read the count-stratified gap: it must
SHRINK as m grows. A gap that does not shrink is noise-dominated, whichever sign it has.

THE ARMS, AND WHAT EACH ONE IS THERE TO RULE OUT
------------------------------------------------
| arm | prediction | rules out |
|---|---|---|
| `ctrlmean` | B control mean, tiled | a metric that scores a constant |
| `ctrlcell` | the drawn B control cell | the load-bearing floor: control-sampling noise at m |
| `experimental` | the B perturbed cells | the reference: replicate agreement at m cells |
| `model_nocal` / `model_cal` | model fed the drawn B controls | the measurement |
| `transfer` | `ctrl_used + this pert's Jurkat/RPE1 delta` | ⭐ the non-learned version of the task itself |
| `meanpert` | `ctrl_used + (train grand mean - B ctrl mean)` | "it learned the average knockdown" |

`ctrlmean`'s prediction delta is **zero by construction** -- with B controls on the pred
side the ctrlmean prediction *is* the B control mean. MEASURED (job 18021925, split 0):
`pearson_delta` comes back as **0.000623**, not NaN, because the delta is zero only in exact
arithmetic and float32 rounding leaves ~1e-8 of noise to correlate. That is a readable zero
rather than a hole in the table, so the guard below is a belt-and-braces branch rather than
the expected path; it fires only if the value really is non-finite, and then records
`undefined (zero-variance prediction delta)` instead of letting a NaN drop out of a mean.
`ctrlcell` is the load-bearing floor either way: its delta is real control-sampling noise at
`m` cells. See docs/rhj_splithalf.md F1.

COST
----
One split = 6 cell-eval runs sharing one real-side DE (keyed on the target file's own
sha256, so cross-split reuse is impossible by construction). Run the timing probe
(`score_only=["ctrlcell"]`) before committing all 5 splits: every cost estimate for this
protocol extrapolates across a 5x change in panel size.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from occam.vcell.panel.arms import REFERENCE_TAGS, build_arms, split_arms
from occam.vcell.panel.panel_io import Panel
from occam.vcell.panel.run import score_arm
from occam.vcell.panel.splithalf import (
    CONTROL, HalfSplit, assert_disjoint_halves, meanpert_arm, panel_for_split,
    train_pert_grand_mean, transfer_arm, sum_of_singles_arm,
)

__all__ = ["PROTOCOL", "UNDEFINED", "ProtocolError", "run_split", "run_splithalf_protocol"]

PROTOCOL = "splithalf/v1"
UNDEFINED = "undefined (zero-variance prediction delta)"


class ProtocolError(RuntimeError):
    """The protocol's preconditions do not hold; nothing was scored."""


def _assert_in_pert2id(cohort: Mapping[str, object], pert2id: Mapping[str, int]) -> None:
    """Refuse BEFORE predicting: `predict_panel` drops unknown perturbations silently."""
    missing = [p for p in cohort["perts"] if int(pert2id.get(p, -1)) < 0]
    if missing:
        raise ProtocolError(
            f"{len(missing)} of {len(cohort['perts'])} cohort perturbations have no "
            f"pert2id entry, e.g. {missing[:5]}. predict_panel drops them silently, which "
            f"would shrink the cohort and rescale discrimination_score_l2 and BH-FDR. The "
            f"cohort is deliberately NOT intersected with pert2id at freeze time -- that "
            f"would make the benchmark checkpoint-dependent. Fix the graph vocabulary or "
            f"re-freeze the cohort explicitly.")


def run_split(
    base: Panel,
    labels: Sequence[str],
    split: HalfSplit,
    cohort: Mapping[str, object],
    out_dir: str,
    *,
    model,
    pert2id: Mapping[str, int],
    device,
    train_perts: Sequence[str] = (),
    grand_mean: Optional[np.ndarray] = None,
    grand_mean_info: Optional[Mapping[str, object]] = None,
    transfer_deltas: Optional[Mapping[str, np.ndarray]] = None,
    transfer_info: Optional[Mapping[str, object]] = None,
    # norman_combo: the additivity baseline. Structurally like `transfer` (ctrl + a per-pert
    # delta) but a different quantity, so it gets its own tag rather than reusing that one --
    # "transfer" on a single-cell-line benchmark would be a lie in every output file.
    sos_deltas: Optional[Mapping[str, np.ndarray]] = None,
    sos_info: Optional[Mapping[str, object]] = None,
    predict_fn=None,
    calibrate: str = "none",
    center_delta: bool = False,
    seed: int = 42,
    eval_control: str = "single",
    ctrl_avg_k: int = 1,
    batch_size: int = 512,
    celleval_kwargs: Optional[Mapping] = None,
    score_only: Optional[Sequence[str]] = None,
    probe: bool = False,
) -> Dict[str, object]:
    """Score every arm of ONE split. Returns the split's record (also written as JSON).

    `score_only` restricts which arms reach cell-eval. `probe` is a SEPARATE flag: it stamps
    the record `probe: true` and writes `metrics_probe.json` instead of `metrics.json`, so a
    cost measurement can never be read as a result.

    ⚠️ These two were one flag until 2026-09-17 and it was wrong. The reference pass
    legitimately uses `score_only` to skip the model arm -- the other five arms do not
    depend on the checkpoint -- and it was therefore stamped a probe and written to
    `metrics_probe.json`, where `paired_report.py` refuses it. Restricting the arm set and
    being a throwaway measurement are different things.

    The `meanpert` baseline comes either from `train_perts` (averaged out of `base` here) or
    from a precomputed `grand_mean`. The precomputed form is the normal one: the training
    perturbations' cells are needed for nothing else, and carrying 1,657 perturbations'
    worth of HepG2 cells through the panel to average them once would quadruple its size.

    `predict_fn` is forwarded to `predict_panel`. The timing probe passes the identity
    (prediction = the control input), which needs no checkpoint and makes the measured cost
    exactly the cell-eval cost the 25 real runs will pay.
    """
    from occam.vcell.panel.calibrate import apply_quantile_map, fit_control_quantile_map
    from occam.vcell.panel.predict import predict_panel

    os.makedirs(out_dir, exist_ok=True)
    labels = np.asarray(labels, dtype=str)
    _assert_in_pert2id(cohort, pert2id)
    assert_disjoint_halves(split)

    panel, exp_X, exp_labels = panel_for_split(base, labels, split, cohort)
    # The pred-side controls. `predict_panel` draws the model's input from exactly these,
    # and they are also what gets written into every prediction file.
    ctrl_B = panel.ctrl_X_model
    ctrl_mean_B = ctrl_B.mean(axis=0)

    # Seeded per split, so the control draw is reproducible and differs across splits.
    split_seed = int(seed) + 1000 * int(split.index)
    pred, arm_labels, ctrl_used, cin_idx = predict_panel(
        model, panel, dict(pert2id), device, predict_fn=predict_fn,
        eval_control=eval_control, batch_size=batch_size, seed=split_seed,
        ctrl_avg_k=ctrl_avg_k)

    if not np.array_equal(np.asarray(arm_labels, dtype=str), np.asarray(exp_labels, str)):
        raise ProtocolError(
            f"split {split.index}: the predicted label sequence does not match the B-half "
            f"label sequence ({len(arm_labels)} vs {len(exp_labels)} rows). Every arm must "
            f"emit the same rows in the same order or the arms are not comparable "
            f"cell-for-cell, which is the entire basis of the paired gap.")

    calibrated = None
    # `build_arms` accepts none|quantile|both. "fit" was never a valid value -- this gate
    # silently skipped the fit for `quantile`, so `calibrated` stayed None and build_arms
    # refused. It cost two launches of 135 array tasks, because every test used "none" or
    # "both" and only the model launcher passes "quantile".
    if calibrate in ("quantile", "both"):
        # `predict_fn` must reach the calibration fit too. Without it the quantile map is
        # fitted on `model(...)` while the prediction came from `predict_fn`, so the map
        # corrects a distribution that was never emitted -- and with model=None it is a
        # TypeError inside the fit.
        src_q, ref_q = fit_control_quantile_map(
            model, panel, dict(pert2id), device, predict_fn=predict_fn,
            batch_size=batch_size, seed=split_seed, ctrl_avg_k=ctrl_avg_k)
        calibrated = apply_quantile_map(pred, src_q, ref_q, device)

    # ---- the two reference arms -------------------------------------------
    extra: Dict[str, np.ndarray] = {"experimental": exp_X}
    meanpert_info: Dict[str, object] = {}
    if len(train_perts) and grand_mean is not None:
        raise ProtocolError(
            "pass either `train_perts` (averaged here) or a precomputed `grand_mean`, not "
            "both -- two different estimands would silently share one arm name")
    grand, info = None, dict(grand_mean_info or {})
    if grand_mean is not None:
        grand = np.asarray(grand_mean, dtype=np.float32)
        if grand.shape != (panel.n_keep,):
            raise ProtocolError(
                f"grand_mean has shape {grand.shape}, expected ({panel.n_keep},) on the "
                f"panel's kept-gene axis")
    elif len(train_perts):
        grand, info = train_pert_grand_mean(base.real_X_model, labels, train_perts)
    if grand is not None:
        mp, clipped = meanpert_arm(ctrl_used, grand, ctrl_mean_B)
        extra["meanpert"] = mp
        meanpert_info = dict(info)
        meanpert_info["clipped_fraction"] = clipped

    transfer_meta: Dict[str, object] = {}
    if transfer_deltas is not None:
        tp, tclip = transfer_arm(ctrl_used, arm_labels, transfer_deltas)
        extra["transfer"] = tp
        transfer_meta = dict(transfer_info or {})
        transfer_meta["clipped_fraction"] = tclip
    sos_meta: Dict[str, object] = {}
    if sos_deltas is not None:
        sp, sclip = sum_of_singles_arm(ctrl_used, arm_labels, sos_deltas)
        extra["sum_of_singles"] = sp
        sos_meta = dict(sos_info or {})
        sos_meta["clipped_fraction"] = sclip

    arms = build_arms(pred, arm_labels, ctrl_used, panel, center_delta=center_delta,
                      calibrate=calibrate,
                      calibrated=calibrated, extra=extra)
    anchor_tags, ref_tags, model_tags = split_arms(arms)
    print(f"[splithalf] split {split.index}: {len(arms)} arm(s) — "
          f"anchors {anchor_tags}, reference {ref_tags}, model {model_tags}")

    wanted = None
    if score_only is not None:  # noqa: SIM102 -- kept flat for the message below
        wanted = [str(t) for t in score_only]
        emitted = [t for t, _, _ in arms]
        unknown = sorted(set(wanted) - set(emitted))
        if unknown:
            raise ProtocolError(f"score_only names arm(s) {unknown} that were not emitted; "
                                f"emitted: {emitted}")
        if probe:
            print(f"[splithalf] TIMING PROBE: scoring only {wanted} — this output is not "
                  f"a result and is written to metrics_probe.json")
        else:
            print(f"[splithalf] scoring {len(wanted)} of the arms: {wanted}")

    record: Dict[str, object] = {
        "protocol": PROTOCOL,
        "split": int(split.index),
        "probe": bool(probe),
        "seed": split_seed,
        "cohort_sha256": cohort["cohort_sha256"],
        "n_perts": int(cohort["n_perts"]),
        "n_rows_per_arm": int(pred.shape[0]),
        "n_ctrl_A": int(len(split.ctrl_A)),
        "n_ctrl_B": int(len(split.ctrl_B)),
        "n_dropped_odd": int(split.n_dropped_odd),
        "half_size": {p: int(v) for p, v in dict(cohort["half_size"]).items()},
        "anchor_arms": anchor_tags, "reference_arms": ref_tags, "model_arms": model_tags,
        "meanpert": meanpert_info,
        "transfer": transfer_meta,
        "sum_of_singles": sos_meta,
        "arms": {}, "timings_s": {}, "notes": {},
        # The order arms were actually SCORED in, which is `build_arms`' fixed emission
        # order and not the caller's request order. The real-side DE is computed by the
        # first scored arm and read from the cache by the rest, so which arm was first is
        # what makes the cold/warm cost decomposition readable.
        "scored_order": [],
    }

    for tag, matrix, lbls in arms:
        if wanted is not None and tag not in wanted:
            record["arms"][tag] = {
                "status": "not scored (timing probe)" if probe
                else "not scored in this pass"}
            continue
        t0 = time.time()
        record["scored_order"].append(tag)
        # `pred_ctrl=ctrl_B` is the protocol: the prediction file carries the B controls, so
        # its delta is B-relative while the target file's is A-relative.
        metrics, arm_dir, target_path = score_arm(
            tag, matrix, lbls, panel, out_dir, celleval_kwargs, pred_ctrl=ctrl_B)
        record["timings_s"][tag] = round(time.time() - t0, 1)
        if not metrics:
            record["arms"][tag] = {"status": "cell-eval returned nothing"}
            continue
        if tag == "ctrlmean":
            pd_val = metrics.get("pearson_delta")
            if pd_val is None or not np.isfinite(pd_val):
                metrics = dict(metrics)
                metrics.pop("pearson_delta", None)
                record["notes"]["ctrlmean.pearson_delta"] = UNDEFINED
        record["arms"][tag] = metrics
        record.setdefault("target_file", target_path)

    name = "metrics_probe.json" if probe else "metrics.json"
    with open(os.path.join(out_dir, name), "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True, default=str)
    print(f"[splithalf] split {split.index}: wrote {os.path.join(out_dir, name)}")
    return record


def run_splithalf_protocol(
    base: Panel,
    labels: Sequence[str],
    splits: Sequence[HalfSplit],
    cohort: Mapping[str, object],
    out_root: str,
    **kwargs,
) -> List[Dict[str, object]]:
    """`run_split` over every split, into `out_root/split_NN/`. Returns the records.

    A split that raises is recorded and the rest continue -- 5 splits are the unit of the
    claim, and losing all 5 to one bad one wastes hours. `manifest.json` lists every
    declared split including the failures, so a missing split is impossible to overlook.
    """
    os.makedirs(out_root, exist_ok=True)
    records: List[Dict[str, object]] = []
    for split in splits:
        out_dir = os.path.join(out_root, f"split_{int(split.index):02d}")
        try:
            records.append(run_split(base, labels, split, cohort, out_dir, **kwargs))
        except Exception as exc:                      # recorded, never swallowed silently
            print(f"[splithalf] split {split.index} FAILED: {type(exc).__name__}: {exc}")
            records.append({"protocol": PROTOCOL, "split": int(split.index),
                            "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
    manifest = {
        "protocol": PROTOCOL,
        "n_splits_declared": len(splits),
        "n_splits_scored": sum(1 for r in records if r.get("status") != "FAILED"),
        "cohort_sha256": cohort["cohort_sha256"],
        "control_label": CONTROL,
        "reference_tags": list(REFERENCE_TAGS),
        "splits": [{"split": r["split"], "status": r.get("status", "ok"),
                    "probe": r.get("probe", False)} for r in records],
    }
    with open(os.path.join(out_root, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"[splithalf] {manifest['n_splits_scored']}/{len(splits)} split(s) scored — "
          f"{os.path.join(out_root, 'manifest.json')}")
    return records
