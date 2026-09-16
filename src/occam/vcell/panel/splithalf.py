"""Repeated split-half halves, the frozen cohort, and a per-split `Panel`.

THE REVIEWER'S PROTOCOL, AND THE ONE IDEA THAT MAKES IT CHEAP
-------------------------------------------------------------
Within each perturbation and context, split both perturbed and control cells into balanced
halves (A) and (B). Half A defines the reference: its mean profile and its DE genes. Two
predictions are then scored against that same reference -- the B perturbed cells
("experimental replicate agreement") and the model fed the B control cells -- with the
experimental arm's DE computed from B and the model's DE computed relative to the same B
controls.

cell-eval already does exactly this, natively, because `pearson_delta` uses **each file's
own** control pseudobulk: the real delta is (real pert bulk - real control bulk) and the
predicted delta is (pred bulk - PRED-file control bulk). So the entire asymmetry is obtained
by choosing what goes in which file:

    real file  = A perturbed + A controls    -> reference mean and reference DE, from A
    pred file  = the arm's prediction + B controls -> predicted delta and DE, relative to B

`panel_for_split` therefore returns a stock `Panel` whose `real_X_model` is the A side and
whose `ctrl_X_model` is the B controls, after which `predict_panel`,
`fit_control_quantile_map`, `apply_quantile_map` and `build_arms` all work unchanged. No
patch to cell-eval, and no second scoring path to keep in sync.

TWO THINGS TO KNOW BEFORE READING ANY NUMBER
--------------------------------------------
* **The experimental arm is not an upper bound at these cell counts.** Reference and
  experimental prediction are *both* ~22-cell pseudobulks, so that arm is attenuated on both
  sides while a low-variance predictor is attenuated on one. The model can beat it from noise
  alone on perturbations whose true effect is small relative to sigma/sqrt(m). Report it as
  "replicate agreement at m cells", stratified by m.
* **`ctrlmean`'s `pearson_delta` is undefined here by construction.** With B controls on the
  pred side, the ctrlmean prediction IS the B control mean, so its delta is exactly zero and
  pearsonr returns NaN. `ctrlcell` is the load-bearing floor instead -- its delta is the mean
  of m drawn B controls minus the full B control mean, i.e. pure control-sampling noise at m
  cells, which is the right floor and scales with m.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from occam.vcell.panel.panel_io import Panel

__all__ = [
    "CONTROL", "SPLIT_SALT", "CohortDriftError", "HalfSplit",
    "freeze_cohort", "group_seed", "balanced_halves", "draw_splits",
    "assert_cohort", "assert_disjoint_halves", "panel_for_split",
    "train_pert_grand_mean", "meanpert_arm",
]

CONTROL = "non-targeting"
SPLIT_SALT = "occam.vcell.panel.splithalf/v1"
CONTROL_GROUP = "__controls__"


class CohortDriftError(RuntimeError):
    """The perturbation set being written is not the frozen cohort."""


@dataclass(frozen=True)
class HalfSplit:
    """One balanced split: row indices into the benchmark panel."""

    index: int
    pert_A: np.ndarray
    pert_B: np.ndarray
    ctrl_A: np.ndarray
    ctrl_B: np.ndarray
    dropped: np.ndarray
    n_dropped_odd: int

    def summary(self) -> str:
        return (f"split {self.index}: pert {len(self.pert_A)}|{len(self.pert_B)} · "
                f"ctrl {len(self.ctrl_A)}|{len(self.ctrl_B)} · "
                f"{self.n_dropped_odd} odd cell(s) dropped")


# ---------------------------------------------------------------------------
# cohort
# ---------------------------------------------------------------------------

def freeze_cohort(
    labels: Sequence[str],
    eligible: Sequence[str],
    *,
    min_cells: int = 10,
    context: str = "",
    out_path: Optional[str] = None,
    extra: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Freeze the scored perturbation set, once, and record why each one was dropped.

    `min_cells` is 10, giving m >= 5 per half. Not lower: at min_cells=4 you get m=2, and
    nothing errors -- which is why it is dangerous. A 2-cell pseudobulk has enormous
    variance, and the variance is ASYMMETRIC between the arms (the experimental arm is 2 raw
    cells; a mean-based prediction has near-zero variance), so the "upper bound" stops being
    one at exactly the perturbations where it matters most.

    The cohort must then be identical in every scored file, because
    `discrimination_score_l2` is a rank over the cohort (score = 1 - rank/n_perts) and
    BH-FDR is computed over the whole flattened DE frame at once. Changing the membership
    silently rescales both.
    """
    labels = np.asarray(labels, dtype=str)
    counts: Dict[str, int] = {}
    for lab, n in zip(*np.unique(labels[labels != CONTROL], return_counts=True)):
        counts[str(lab)] = int(n)

    elig = set(eligible)
    kept, dropped = [], {}
    for p in sorted(counts):
        if p not in elig:
            dropped[p] = f"not in the eligible set"
        elif counts[p] < min_cells:
            dropped[p] = f"{counts[p]} cells < min_cells={min_cells}"
        else:
            kept.append(p)

    cohort = {
        "protocol": "splithalf/v1",
        "context": context,
        "pert_col": "target_gene",
        "control_label": CONTROL,
        "min_cells": int(min_cells),
        "n_perts": len(kept),
        "perts": kept,
        "cells_per_pert": {p: counts[p] for p in kept},
        "half_size": {p: counts[p] // 2 for p in kept},
        "n_controls": int((labels == CONTROL).sum()),
        "dropped": dropped,
        "cohort_sha256": hashlib.sha256("\n".join(kept).encode()).hexdigest(),
    }
    if extra:
        cohort.update(dict(extra))
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(cohort, fh, indent=2, sort_keys=True)
    return cohort


def assert_cohort(labels: Sequence[str], cohort: Mapping[str, object]) -> None:
    """Refuse to write a file whose perturbation set is not the frozen cohort."""
    got = sorted({str(x) for x in labels} - {CONTROL})
    want = list(cohort["perts"])
    if got != want:
        missing = sorted(set(want) - set(got))
        extra = sorted(set(got) - set(want))
        raise CohortDriftError(
            f"cohort drift: {len(missing)} missing {missing[:5]}, "
            f"{len(extra)} extra {extra[:5]}. discrimination_score_l2 is a rank over the "
            f"cohort and BH-FDR is computed over the whole frame, so a changed cohort "
            f"silently rescales both. Refusing to write.")


# ---------------------------------------------------------------------------
# the splits
# ---------------------------------------------------------------------------

def group_seed(base_seed: int, split_index: int, group: str) -> int:
    """A stable seed for one (split, group), independent of iteration order.

    Per-group, NOT one generator consumed in a loop over perturbations: with a single stream,
    adding or removing one perturbation from the cohort would shift the halves of every later
    perturbation, so a cohort change would silently change results everywhere instead of only
    where it should.
    """
    key = f"{SPLIT_SALT}|{base_seed}|{split_index}|{group}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")


def balanced_halves(
    batches: Sequence[str], rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch-stratified equal-size halves of one group -> (A, B, dropped).

    Deals alternately over a batch-blocked permutation, carrying parity ACROSS batch
    boundaries so the leftover cell of each odd batch alternates sides rather than always
    landing on A.

    For perturbed cells this cannot equalise composition -- median 1 cell per
    (perturbation, batch), 53% of strata singletons -- but it is free and it removes the
    worst case, where an unstratified permutation puts a whole batch on one side. For
    CONTROLS it genuinely works and is mandatory: an unstratified 2,488/2,488 split leaves an
    (A_ctrl_mean - B_ctrl_mean) batch term that enters the real and predicted deltas with
    OPPOSITE signs.
    """
    batches = np.asarray(batches, dtype=str)
    n = len(batches)
    order: List[int] = []
    for b in rng.permutation(np.unique(batches)):
        idx = np.flatnonzero(batches == b)
        order.extend(idx[rng.permutation(len(idx))].tolist())

    a = np.asarray(order[0::2], dtype=np.int64)
    b_ = np.asarray(order[1::2], dtype=np.int64)
    dropped = np.zeros(0, dtype=np.int64)
    if n % 2:
        # Drop from the LARGER half, chosen with this group's own RNG. Never into A: a
        # quieter reference than the prediction biases the ceiling upward.
        big = a if len(a) > len(b_) else b_
        k = int(rng.integers(0, len(big)))
        dropped = np.asarray([big[k]], dtype=np.int64)
        keep = np.delete(big, k)
        if big is a:
            a = keep
        else:
            b_ = keep
    assert len(a) == len(b_), "halves must be equal"
    return a, b_, dropped


def draw_splits(
    labels: Sequence[str],
    batches: Sequence[str],
    cohort: Mapping[str, object],
    *,
    n_splits: int = 5,
    base_seed: int = 42,
) -> List[HalfSplit]:
    """`n_splits` balanced, batch-stratified splits over the frozen cohort."""
    labels = np.asarray(labels, dtype=str)
    batches = np.asarray(batches, dtype=str)
    perts = list(cohort["perts"])
    ctrl_rows = np.flatnonzero(labels == CONTROL)
    rows_of = {p: np.flatnonzero(labels == p) for p in perts}

    splits: List[HalfSplit] = []
    for s in range(n_splits):
        pa, pb, dropped = [], [], []
        for p in perts:                      # sorted, and each seeded independently
            rows = rows_of[p]
            rng = np.random.default_rng(group_seed(base_seed, s, p))
            a, b, d = balanced_halves(batches[rows], rng)
            pa.append(rows[a]); pb.append(rows[b]); dropped.append(rows[d])
        crng = np.random.default_rng(group_seed(base_seed, s, CONTROL_GROUP))
        ca, cb, cd = balanced_halves(batches[ctrl_rows], crng)
        dropped.append(ctrl_rows[cd])
        splits.append(HalfSplit(
            index=s,
            pert_A=np.concatenate(pa), pert_B=np.concatenate(pb),
            ctrl_A=ctrl_rows[ca], ctrl_B=ctrl_rows[cb],
            dropped=np.concatenate(dropped) if dropped else np.zeros(0, np.int64),
            n_dropped_odd=int(sum(len(d) for d in dropped)),
        ))
    return splits


def assert_disjoint_halves(split: HalfSplit) -> None:
    """A and B must share no cell, on either side. The whole design rests on it."""
    for name, a, b in (("perturbed", split.pert_A, split.pert_B),
                       ("control", split.ctrl_A, split.ctrl_B)):
        overlap = np.intersect1d(a, b)
        if overlap.size:
            raise ValueError(
                f"split {split.index}: {overlap.size} {name} cell(s) are in BOTH halves "
                f"(e.g. rows {overlap[:5].tolist()}). The halves must be disjoint or the "
                f"reference and the prediction share cells and every metric is inflated.")
    if len(split.pert_A) != len(split.pert_B):
        raise ValueError(f"split {split.index}: perturbed halves differ in size")
    if len(split.ctrl_A) != len(split.ctrl_B):
        raise ValueError(f"split {split.index}: control halves differ in size")


# ---------------------------------------------------------------------------
# the per-split Panel
# ---------------------------------------------------------------------------

def panel_for_split(
    base: Panel,
    labels: Sequence[str],
    split: HalfSplit,
    cohort: Mapping[str, object],
) -> Tuple[Panel, np.ndarray, np.ndarray]:
    """(A-side Panel with ctrl_X_model = the B controls, B perturbed matrix, labels).

    The B rows are ordered to match the A label sequence, so every arm shares one labels
    vector and the row layouts are identical across arms -- which is what makes the
    experimental arm and the model arm comparable cell-for-cell.
    """
    labels = np.asarray(labels, dtype=str)

    a_rows = np.asarray(split.pert_A, dtype=np.int64)
    b_rows = np.asarray(split.pert_B, dtype=np.int64)
    # order both halves by perturbation so row i of each refers to the same perturbation
    a_order = np.argsort(labels[a_rows], kind="stable")
    b_order = np.argsort(labels[b_rows], kind="stable")
    a_rows, b_rows = a_rows[a_order], b_rows[b_order]
    if not np.array_equal(labels[a_rows], labels[b_rows]):
        raise ValueError(
            f"split {split.index}: the A and B label sequences differ after ordering; "
            f"equal halves per perturbation are required")

    real_rows = np.concatenate([a_rows, split.ctrl_A])
    real_labels = np.concatenate([labels[a_rows],
                                  np.full(len(split.ctrl_A), CONTROL)])
    assert_cohort(real_labels, cohort)

    panel = Panel(
        real_X_model=base.real_X_model[real_rows],
        real_labels=real_labels,
        real_counts=(base.real_counts[real_rows]
                     if getattr(base, "real_counts", None) is not None else None),
        # The B controls are what the model sees AND what lands on the pred side -- the one
        # substitution that produces the protocol's whole A/B asymmetry.
        ctrl_X_model=base.real_X_model[split.ctrl_B],
        ctrl_counts=(base.real_counts[split.ctrl_B]
                     if getattr(base, "real_counts", None) is not None else None),
        ctrl_batches=(np.asarray(base.ctrl_batches)[split.ctrl_B]
                      if getattr(base, "ctrl_batches", None) is not None else None),
        enc_cols=base.enc_cols, panel_cols=base.panel_cols,
        panel_genes=base.panel_genes, n_enc=base.n_enc,
    )
    return panel, base.real_X_model[b_rows], labels[b_rows]


# ---------------------------------------------------------------------------
# the mean-training-perturbation baseline
# ---------------------------------------------------------------------------

def train_pert_grand_mean(
    X: np.ndarray, labels: Sequence[str], train_perts: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Equal-weight-per-PERTURBATION mean of the training perturbations' mean profiles.

    Per perturbation, not per cell. `baselines.MeanBaseline` averages per cell, which on
    HepG2 is dominated by TFAM's 1,213 cells and GFM1's 749 -- a different estimand, and a
    materially different baseline.
    """
    labels = np.asarray(labels, dtype=str)
    want = sorted(set(train_perts) & set(labels.tolist()))
    if not want:
        raise ValueError("no training perturbations present in this panel")
    means = np.zeros((len(want), X.shape[1]), dtype=np.float64)
    counts = {}
    for i, p in enumerate(want):
        rows = np.flatnonzero(labels == p)
        means[i] = X[rows].mean(axis=0)
        counts[p] = int(len(rows))
    grand = means.mean(axis=0).astype(np.float32)
    return grand, {"n_perts": len(want), "weighting": "equal per perturbation",
                   "cells_per_pert": counts,
                   "total_cells": int(sum(counts.values()))}


def meanpert_arm(
    ctrl_used: np.ndarray, grand_mean: np.ndarray, ctrl_mean_B: np.ndarray,
    clip: bool = True,
) -> Tuple[np.ndarray, float]:
    """ctrl_used + (grand_mean - B control mean), clipped at 0. Returns (pred, clipped_frac).

    The delta is built from the **B** controls only: using all controls would hand the
    baseline a free correction of the A-control sampling noise that the model does not get.

    `ctrl_used` must be the SAME drawn control cells the model arm received, so the
    control-draw noise cancels in exactly the comparison this baseline exists to support.
    """
    delta = np.asarray(grand_mean, dtype=np.float32) - np.asarray(ctrl_mean_B, np.float32)
    pred = np.asarray(ctrl_used, dtype=np.float32) + delta[None, :]
    frac = 0.0
    if clip:
        neg = pred < 0
        frac = float(neg.mean())
        pred = np.clip(pred, 0.0, None)
    return pred, frac
