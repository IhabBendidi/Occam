"""Phase A of the bridge: a competitor emits a prediction matrix in OUR evaluation space.

WHY TWO PHASES
--------------
A competitor's checkpoint can only be loaded in that competitor's own interpreter, and our
scoring path (`occam.vcell.panel.protocol.run_split`) can only run in ours. Two interpreters
cannot share a Python callable, so the bridge is a file:

    Phase A (competitor venv, THIS module)
        read our frozen panel -> feed the model the drawn half-B control cells
        -> map the emission into our evaluation space -> write pred.npy

    Phase B (our venv, scripts/eval_competitor.py)
        wrap pred.npy in a matrix-backed predict_fn -> protocol.run_split

Everything that makes the comparison fair lives in Phase B, which the competitor never
touches: the frozen 611-perturbation cohort, the 5 splits, the mandatory zero-signal anchors,
`pred_ctrl = ctrl_B`, the sha256-keyed real-DE cache, and cell-eval 0.6.0.

WHAT AN ARM MUST PROVIDE
------------------------
One callable:

    predict(ctrl: np.ndarray, pert_names: Sequence[str]) -> np.ndarray

`ctrl` is `(n, 6983)` float32 in **the arm's own training space** (by default
`log1p(target_sum=10_000)`, which is what `occam_data` hands it), one real half-B HepG2
control cell per row. `pert_names` is the perturbation to apply to each row, same length,
same order. The return must be `(n, 6983)` in the arm's **emission** space.

This module then converts that emission into our evaluation space and refuses anything that
is not in it. It does NOT ask the arm to know anything about our space -- that is the point:
their internal preprocessing is their business, the interface is ours.

⚠️ ROW ORDER IS A CONTRACT
--------------------------
The rows are the half-A perturbed cells of a split, ordered by
`np.argsort(labels, kind="stable")`, and `run_split` re-checks that order before scoring.
`labels.json` is written alongside so Phase B can verify rather than assume.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from emission_space import (  # noqa: E402
    EVAL_TARGET_SUM, TRAIN_TARGET_SUM, EmissionSpaceError,
    assert_representable, describe_space, space_diagnostics, to_eval_space,
)

__all__ = ["PanelSplit", "load_panel_split", "export_split", "CONTROL"]

CONTROL = "non-targeting"


class PanelSplit:
    """One split of the frozen panel, in the arm's training space.

    Attributes
    ----------
    ctrl : (18575, 6983) float32
        The drawn half-B control cells, one per predicted row -- the model's input.
    labels : (18575,) str
        The perturbation for each row, in the order `run_split` will re-check.
    target : (18575, 6983) float32
        The half-A perturbed cells, in OUR evaluation space. Used only by
        `space_diagnostics`; never shown to the model.
    """

    def __init__(self, index, ctrl, labels, target, ctrl_b_rows, cohort, manifest):
        self.index = index
        self.ctrl = ctrl
        self.labels = labels
        self.target = target
        self.ctrl_b_rows = ctrl_b_rows
        self.cohort = cohort
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.labels)


def load_panel_split(panel_dir: str, split_index: int, *,
                     target_sum: float = TRAIN_TARGET_SUM) -> PanelSplit:
    """The frozen panel's split `k`, with the control input in the arm's training space.

    The panel on disk is in our **evaluation** space (`target_sum=2000`). The model expects
    its **training** space, so the control cells are converted with the same analytic map
    used in reverse on the way out. At the default 10,000 that is
    `log1p(expm1(x) * 5.0)` -- exact, because the per-cell denominator cancels.
    """
    z = np.load(os.path.join(panel_dir, "panel.npz"), allow_pickle=True)
    zs = np.load(os.path.join(panel_dir, "splits.npz"))
    with open(os.path.join(panel_dir, "cohort.json")) as fh:
        cohort = json.load(fh)
    with open(os.path.join(panel_dir, "splits_manifest.json")) as fh:
        man = json.load(fh)
    if man["cohort_sha256"] != cohort["cohort_sha256"]:
        raise RuntimeError("cohort.json and splits_manifest.json disagree; rebuild the panel")

    X = np.asarray(z["X"], dtype=np.float32)                 # our EVAL space
    labels_all = np.asarray(z["labels"], dtype=str)
    k = int(split_index)
    key = f"_{k}"
    if f"pert_A{key}" not in zs:
        raise RuntimeError(f"split {k} is not in splits.npz "
                           f"(built with n_splits={man['n_splits']})")

    a_rows = np.asarray(zs[f"pert_A{key}"], dtype=np.int64)
    ctrl_b = np.asarray(zs[f"ctrl_B{key}"], dtype=np.int64)
    # the same stable ordering run_split re-checks
    a_rows = a_rows[np.argsort(labels_all[a_rows], kind="stable")]
    labels = labels_all[a_rows]

    # The control DRAW must match predict_panel's, or `meanpert`, `transfer` and the model
    # arm stop sharing control noise and the paired comparison loses its point.
    # predict_panel: rng = default_rng(seed + 1000*split); cin_idx = rng.integers(0, n_ctrl, n)
    rng = np.random.default_rng(42 + 1000 * k)
    cin_idx = rng.integers(0, len(ctrl_b), size=len(a_rows))
    ctrl_eval = X[ctrl_b][cin_idx]

    ctrl_train = ctrl_eval
    if abs(target_sum - EVAL_TARGET_SUM) > 1e-9:
        # eval -> train: the inverse of to_eval_space, same cancellation
        ctrl_train = np.log1p(
            np.expm1(ctrl_eval.astype(np.float64)) * (target_sum / EVAL_TARGET_SUM)
        ).astype(np.float32)

    print(f"[export] split {k}: {len(labels):,} rows · {len(ctrl_b):,} half-B controls · "
          f"control input at target_sum {target_sum:g}")
    print(f"[export]   input space : {describe_space(ctrl_train, target_sum=target_sum).summary()}")
    return PanelSplit(k, ctrl_train, labels, X[a_rows], ctrl_b, cohort, man)


def export_split(predict: Callable[[np.ndarray, Sequence[str]], np.ndarray],
                 split: PanelSplit, out_dir: str, *,
                 arm: str,
                 from_target_sum: float = TRAIN_TARGET_SUM,
                 from_log1p: bool = True,
                 batch_size: int = 2048,
                 extra_manifest: Optional[Dict[str, object]] = None) -> str:
    """Run an arm over one split and write `pred.npy` + `labels.json` + `manifest.json`.

    Returns the directory written. Raises `EmissionSpaceError` rather than writing a
    prediction that is not in our evaluation space -- a wrong-space matrix that reaches
    cell-eval produces a plausible number from a meaningless comparison.
    """
    os.makedirs(out_dir, exist_ok=True)
    n = len(split)

    chunks: List[np.ndarray] = []
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        out = np.asarray(predict(split.ctrl[s:e], list(split.labels[s:e])),
                         dtype=np.float32)
        if out.shape != (e - s, split.ctrl.shape[1]):
            raise ValueError(
                f"{arm}: predict returned {out.shape} for rows [{s}:{e}), expected "
                f"{(e - s, split.ctrl.shape[1])}. Every arm must emit one row per input "
                f"cell on the 6,983-gene axis.")
        chunks.append(out)
        if (s // batch_size) % 4 == 0:
            print(f"[export]   {e:,}/{n:,} rows", flush=True)
    raw = np.concatenate(chunks, axis=0)

    print(f"[export]   raw emission : "
          f"{describe_space(raw, target_sum=from_target_sum).summary()}")
    pred = to_eval_space(raw, from_target_sum=from_target_sum, from_log1p=from_log1p)
    print(f"[export]   eval space   : {describe_space(pred).summary()}")

    # ⛔ HARD: can this be scored at all? Model-agnostic -- a terrible arm passes.
    assert_representable(pred, n_genes=split.target.shape[1],
                         name=f"{arm} prediction", target_sum=EVAL_TARGET_SUM,
                         target=split.target)

    # DIAGNOSTIC: never a refusal. Library size, density and mean correlation are all
    # confounded with model quality, and refusing on them would exclude exactly the weak
    # arms this benchmark exists to measure. Recorded in the manifest, read beside the score.
    diag = space_diagnostics(pred, split.target, pred_name=f"{arm} prediction",
                             target_name="panel target")
    print(f"[export]   representable ✓   diagnostics: rowsum ratio "
          f"{diag['rowsum_ratio']:.4f} · gene-mean corr {diag['gene_mean_corr']:.4f} · "
          f"zero {diag['zero_fraction_pred']:.4f} (target "
          f"{diag['zero_fraction_target']:.4f})")
    print(f"[export]   for reference, OUR OWN arm: rowsum ratio 0.9399 · "
          f"gene-mean corr 0.9998 · zero 0.0000")

    np.save(os.path.join(out_dir, "pred.npy"), pred)
    with open(os.path.join(out_dir, "labels.json"), "w") as fh:
        json.dump(list(map(str, split.labels)), fh)
    manifest = {
        "arm": arm,
        "split": split.index,
        "n_rows": int(n),
        "n_genes": int(pred.shape[1]),
        "cohort_sha256": split.cohort["cohort_sha256"],
        "benchmark_sha256": split.manifest["benchmark_sha256"],
        "gene_axis_sha256": split.manifest["gene_axis_sha256"],
        "emission": {
            "from_target_sum": float(from_target_sum),
            "from_log1p": bool(from_log1p),
            "to_target_sum": float(EVAL_TARGET_SUM),
            "map": ("log1p(expm1(x) * to/from)" if from_log1p
                    else "log1p(x * to/from)"),
        },
        "representable": True,
        "space_diagnostics": diag,
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, default=str)
    print(f"[export] wrote {out_dir}/pred.npy  {pred.shape} {pred.dtype}")
    return out_dir
