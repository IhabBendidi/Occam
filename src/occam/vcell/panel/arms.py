"""Building the arms to score: the model's predictions, and the anchors.

An "arm" is one ``(tag, prediction_matrix, labels)`` triple handed to cell-eval.
The model's own prediction is one arm; the rest are transformations of it
(calibration, delta shrink) plus the zero-signal anchors.

**The anchors are not optional.** On this evaluation the zero-signal
``ctrlmean`` arm scores HIGHER than every model arm on ``pearson_delta``: across
41 (run, arm) pairs measured in the research tree, the model won 0/41 on
``pearson_delta``, 9/41 on ``discrimination_score_l2``, and 41/41 on
``overlap_at_N``. A metrics table without the anchor row is therefore not
interpretable, and :func:`build_arms` refuses to produce one.

Note this inverts the caveat the research tree's docs carry. There the warning
was that ``overlap_at_N`` is compromised because a zero-signal predictor beat
the model on it -- but that was measured on a different panel (TREK). On this
panel the direction is reversed. Which metric the anchor wins is a property of
the panel, so establish it per panel from the anchor arms rather than
inheriting the claim.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from occam.vcell.panel.panel_io import Panel

__all__ = ["Arm", "build_arms", "ANCHOR_TAGS"]

# Arms that carry no biology by construction.
ANCHOR_TAGS = ("ctrlmean", "ctrlcell")

Arm = Tuple[str, np.ndarray, np.ndarray]


def _tag(base: str, alpha: float) -> str:
    return base if alpha == 1.0 else f"{base}_a{alpha:g}".replace("0.", "")


def build_arms(
    pred: np.ndarray,
    labels: np.ndarray,
    ctrl_used: np.ndarray,
    panel: Panel,
    *,
    calibrate: str = "both",
    shrinks: Sequence[float] = (1.0,),
    anchors: Sequence[str] = ANCHOR_TAGS,
    calibrated: Optional[np.ndarray] = None,
) -> List[Arm]:
    """Assemble the arms to score.

    Args:
        pred: model output, one row per real perturbed cell.
        labels: the perturbation label per row.
        ctrl_used: the control cell each prediction was actually generated from.
            Shrink is taken against this, per cell, not against a pooled mean.
        panel: the loaded panel, for its model-visible control pool.
        calibrate: ``none`` | ``quantile`` | ``both``. With ``quantile`` or
            ``both``, ``calibrated`` must be supplied.
        shrinks: alpha values for the delta shrink. ``pred_shrunk = ctrl +
            alpha * (pred - ctrl)``, i.e. a shrink of the log fold change.
            Worth sweeping rather than picking: the predictions are
            directionally right but far too large, and the metrics that read
            magnitude collapse under shrink while the scale-invariant ones do
            not, so the sweep separates those two failures.
        anchors: zero-signal arms to include. Must be non-empty.
        calibrated: the control-fit quantile-calibrated prediction.

    Returns:
        ``[(tag, matrix, labels)]``, anchors FIRST so that a job killed at the
        wall clock has still produced an interpretable floor.
    """
    if not anchors:
        raise ValueError(
            "anchors must not be empty. On this evaluation the zero-signal "
            "ctrlmean arm outscores every model arm on pearson_delta (0/41 in "
            "the research tree's measurements), so a table without it cannot be "
            "read. There is deliberately no way to turn the anchors off."
        )
    unknown = set(anchors) - set(ANCHOR_TAGS)
    if unknown:
        raise ValueError(f"unknown anchor(s) {sorted(unknown)}; expected {list(ANCHOR_TAGS)}")

    want = str(calibrate).lower()
    if want not in ("none", "quantile", "both"):
        raise ValueError(f"calibrate must be none|quantile|both, got {calibrate!r}")
    if want in ("quantile", "both") and calibrated is None:
        raise ValueError(
            f"calibrate={want!r} needs the calibrated prediction. Fit it with "
            f"occam.vcell.panel.calibrate.fit_control_quantile_map and pass it as "
            f"`calibrated`."
        )

    shrinks = [float(a) for a in shrinks] or [1.0]

    def shrink(arr: np.ndarray, alpha: float) -> np.ndarray:
        if alpha == 1.0:
            return arr
        return (ctrl_used + alpha * (arr - ctrl_used)).astype(np.float32)

    arms: List[Arm] = []

    # ---- anchors first ------------------------------------------------------
    if "ctrlmean" in anchors:
        # The model-visible control mean, emitted for every perturbation. The
        # only place a mean over cells is taken, and here averaging IS the
        # definition of the baseline: it carries no biology, so whatever it
        # scores is what the metric reads from the emission alone.
        cm = np.tile(panel.ctrl_X_model.mean(axis=0, keepdims=True), (pred.shape[0], 1))
        arms.append(("ctrlmean", cm.astype(np.float32), labels))

    if "ctrlcell" in anchors:
        # The drawn control cell itself, unchanged. A tighter zero-signal
        # anchor than ctrlmean and free -- the control was already drawn to
        # generate the prediction. It differs from ctrlmean in exactly one
        # respect, per-cell noise instead of a pooled mean, which separates
        # "the model has no signal" from "the metric rewards the pooled-mean
        # emission shape". Two independent anchors is what makes the result
        # readable rather than ambiguous.
        arms.append(("ctrlcell", ctrl_used.astype(np.float32), labels))

    # ---- model arms ---------------------------------------------------------
    if want in ("none", "both"):
        for alpha in shrinks:
            arms.append((_tag("model_nocal", alpha), shrink(pred, alpha), labels))

    if want in ("quantile", "both"):
        for alpha in shrinks:
            arms.append((_tag("model_cal", alpha), shrink(calibrated, alpha), labels))

    return arms


def split_anchor_and_model(arms: Sequence[Arm]) -> Tuple[List[str], List[str]]:
    """``(anchor_tags, model_tags)`` present in ``arms``."""
    anchor = [t for t, _, _ in arms if t in ANCHOR_TAGS]
    model = [t for t, _, _ in arms if t not in ANCHOR_TAGS]
    return anchor, model
