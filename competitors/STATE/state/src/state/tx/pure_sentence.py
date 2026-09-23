"""STATE prediction in PURE cell sentences, and its sequential composition for pairs.

ONE implementation, two callers, so they cannot drift:

  * `competitors/STATE/export_occam.py`         -- the export that is scored.
  * `state/tx/callbacks/validation_celleval.py` -- the in-training validation pass whose
                                                   `val/celleval/selection_score` picks
                                                   `best_celleval.ckpt`.

WHY THE VALIDATION PASS MUST USE THIS ON A COMBINATORIAL BENCHMARK
    On `norman_combo` the export answers `"A+B"` by sequential composition
    (`compose_sequential`, `decode_seq_mean`): predict A from the control, feed that output
    back in as the control and predict B; also B then A; average the two chains. The
    validation pass used to predict the 20 val pairs through the one-hot of the literal
    label `"A+B"` -- a `pert_encoder` column that never receives a gradient, because no pair
    cell is in training. The selection score was therefore measuring an untrained input,
    not what is exported: best fell on the first validation for 3 of 5 seeds and val
    `pearson_delta` swung 0.13-0.47. Audit 2026-09-23.

    On a single-perturbation benchmark (`rhj_fewshot`) none of this runs: the callback keeps
    its original loader-driven path byte for byte.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

# `competitors/shared` is five levels up from state/src/state/tx/.
_SHARED = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "..", "shared"))
if os.path.isdir(_SHARED) and _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

__all__ = ["PAIR_SEP", "is_multi", "plan_pure_sentences", "predict_pure_sentences",
           "make_composed_pair_predict"]

PAIR_SEP = "+"


def is_multi(label: str, sep: str = PAIR_SEP) -> bool:
    """True for a combination label (`"A+B"`), with the same rule `compose_sequential` uses."""
    return len([p for p in str(label).split(sep) if p]) > 1


def plan_pure_sentences(names: np.ndarray, sentence_len: int) -> list[np.ndarray]:
    """Row groups of ONE perturbation each, at most `sentence_len` long, first-appearance order.

    `SentenceBatchSampler._create_sentences` groups training sentences by
    (cell_type, perturbation), i.e. one perturbation per sentence, always. The backbone is a
    bidirectional transformer over the sentence, so a mixed sentence predicts each cell in
    the company of other perturbations -- a regime absent from training (C27).
    """
    S = int(sentence_len)
    plan: list[np.ndarray] = []
    # first-appearance order without pandas: np.unique sorts, so recover the original order
    # from the returned first indices. Order only affects sentence locality, never
    # correctness -- but a stable order keeps two runs byte-comparable.
    _, _first = np.unique(names, return_index=True)
    for lbl in names[np.sort(_first)]:
        idx = np.flatnonzero(names == lbl)
        for i in range(0, len(idx), S):
            plan.append(idx[i:i + S])

    n = len(names)
    covered = np.concatenate(plan) if plan else np.empty(0, dtype=np.int64)
    if len(covered) != n or len(np.unique(covered)) != n:
        raise RuntimeError(
            f"the sentence plan covers {len(np.unique(covered))} of {n} rows "
            f"({len(covered)} slots). Every row must be predicted exactly once or the "
            f"output matrix is silently part-stale.")
    for sent in plan:
        if len(set(names[sent])) != 1:
            raise RuntimeError(
                f"a sentence carries {len(set(names[sent]))} perturbations. Every "
                f"sentence must hold exactly one -- the backbone attends across the "
                f"whole sentence, so a mixed one predicts each cell in the company of "
                f"the wrong perturbations. See C27.")
    return plan


def predict_pure_sentences(model, ctrl: np.ndarray, pert_names: Sequence[str], *,
                           pert_onehot_map: Mapping[str, torch.Tensor],
                           batch_onehot: torch.Tensor,
                           batch_names: Sequence[str],
                           cell_type_onehot: torch.Tensor,
                           cell_type_names: Sequence[str],
                           sentence_len: int,
                           device,
                           dataset_name: str = "occam_panel") -> np.ndarray:
    """One predicted cell per control row, one `predict_step` per PURE sentence.

    `batch_onehot`, `batch_names`, `cell_type_onehot`, `cell_type_names` are PER ROW of
    `ctrl` (same length, same order). A combination label is refused: it would be predicted
    through the one-hot of the literal `"A+B"`, which never trains -- pairs go through
    `make_composed_pair_predict`.

    ⚠️ ONE SENTENCE PER `predict_step`, ALWAYS, at its TRUE length (no padding).
    `padded=False` means "this whole batch IS one sentence" (`state_transition.py:409-415`),
    so submitting several sentences at once flattens them into one mixed sentence with no
    error (C27/S1). STATE's own inference (`_infer.py:810`) runs a short sentence unpadded.

    History carried over from `export_occam.py`, where this loop used to live:
      * Tiling a 30-cell sentence to 64 with `np.resize` changed the prediction by 7.2%
        (non-uniform reweighting: cells 0-3 thrice, 4-29 twice). Exact 2x duplication is
        attention-invariant (0.0007%). Not padding is faithful and ~2.3x cheaper.
      * Submitting 4 pure sentences with `padded=False` flattened them into ONE 256-cell
        sentence over ~8-9 perturbations (NoRoPE backbone: no error). Measured 52% change
        per cell, per-perturbation correlation 0.12-0.25 with the faithful prediction -- the
        bulk of the 0.2049 -> 0.1306 collapse.
    """
    X = np.asarray(ctrl, dtype=np.float32)
    n, g = X.shape
    names = np.asarray([str(p) for p in pert_names], dtype=object)
    for arr, what in ((batch_onehot, "batch_onehot"), (batch_names, "batch_names"),
                      (cell_type_onehot, "cell_type_onehot"),
                      (cell_type_names, "cell_type_names")):
        if len(arr) != n:
            raise RuntimeError(
                f"predict got {n} rows but {what} has {len(arr)}: every row would be "
                f"conditioned on another row's covariate.")
    multi = [p for p in names if is_multi(p)]
    if multi:
        raise RuntimeError(
            f"{len(multi)} row(s) carry a combination label (e.g. {multi[0]!r}); their "
            f"one-hot column never trains. Predict pairs by sequential composition "
            f"(make_composed_pair_predict), as the export does.")
    onehot = torch.stack([pert_onehot_map[p] for p in names])
    batch_names = np.asarray([str(v) for v in batch_names], dtype=object)
    cell_type_names = np.asarray([str(v) for v in cell_type_names], dtype=object)

    plan = plan_pure_sentences(names, sentence_len)
    out = np.full((n, g), np.nan, dtype=np.float32)
    with torch.no_grad():
        for si, sent in enumerate(plan):
            t = len(sent)                       # true sentence length; NOT padded to S
            ctrl_b = torch.from_numpy(X[sent]).reshape(1, t, g).to(device)
            pert_b = onehot[sent].reshape(1, t, -1).to(device)
            batch = {
                # ⚠️ pert_cell_emb is the TARGET, ignored at predict time. Filled with the
                # control cells so the tensor has the right shape.
                "pert_cell_emb": ctrl_b,
                "ctrl_cell_emb": ctrl_b,
                "pert_emb": pert_b,
                "pert_name": [str(v) for v in names[sent]],
                "cell_type": [str(v) for v in cell_type_names[sent]],
                "cell_type_onehot": cell_type_onehot[sent].to(device),
                "batch": batch_onehot[sent].to(device),
                "batch_name": [str(v) for v in batch_names[sent]],
                "dataset_name": [dataset_name] * t,
            }
            preds = model.predict_step(batch, si, padded=False)
            p = preds["preds"] if isinstance(preds, dict) else preds
            p = np.asarray(p.reshape(-1, g).float().cpu(), dtype=np.float32)
            if p.shape[0] != t:
                raise RuntimeError(
                    f"predict_step returned {p.shape[0]} rows for {t} input cells; "
                    f"a positional prediction matrix would then be misaligned")
            out[sent] = p

    if np.isnan(out).any():
        raise RuntimeError(
            f"{int(np.isnan(out).any(1).sum())} row(s) were never written by any "
            f"sentence; refusing to emit a partly-NaN prediction.")
    return out


def make_composed_pair_predict(single_predict: Callable[[np.ndarray, Sequence[str]], np.ndarray],
                               n_rows: int, order: str = "mean"):
    """`compose_sequential(single_predict)` for a set of rows that are ALL pairs.

    The composition itself is the export's own `competitors/shared/export_predictions.py::
    compose_sequential` -- imported, not re-implemented. `single_predict(ctrl, names)` must
    take the FULL `n_rows` pair set every time (its per-row covariates are indexed by it),
    which holds because every row is a pair, so each chained call covers all rows.

    `compose_sequential` first calls `single_predict` on the raw `"A+B"` labels and then
    OVERWRITES every multi row with the chain; that first call's output is discarded. It is
    intercepted here and answered with NaN without touching the model, so the untrained
    `"A+B"` one-hot column is never read. Any NaN that survived would mean a pair row was
    NOT overwritten, and is refused.
    """
    from export_predictions import compose_sequential

    def _guarded(ctrl: np.ndarray, names: Sequence[str]) -> np.ndarray:
        names = [str(p) for p in names]
        if len(names) != n_rows:
            raise RuntimeError(
                f"composition called the single predictor on {len(names)} rows, expected "
                f"all {n_rows} pair rows; the per-row covariates would misalign.")
        flags = [is_multi(p) for p in names]
        if all(flags):
            # compose_sequential's pass-through call: discarded, never run on the model
            return np.full(np.asarray(ctrl).shape, np.nan, dtype=np.float32)
        if any(flags):
            raise RuntimeError("mixed single/pair labels reached the pair predictor")
        return single_predict(ctrl, names)

    composed = compose_sequential(_guarded, order=order)

    def _predict(ctrl: np.ndarray, names: Sequence[str]) -> np.ndarray:
        if not all(is_multi(p) for p in names):
            raise RuntimeError("make_composed_pair_predict takes pair rows only")
        out = composed(ctrl, names)
        if np.isnan(out).any():
            raise RuntimeError(
                f"{int(np.isnan(out).any(1).sum())} pair row(s) were not produced by the "
                f"composition; refusing to score them.")
        return out

    return _predict
