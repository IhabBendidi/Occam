"""One predicted cell per real perturbed cell, from one real control cell.

Ported from the research tree's held-out-context evaluation module. Behaviour
is preserved; what changed is that the second, count-space scorer and its
emission machinery are gone -- Occam scores with cell-eval v1 only -- and the
names are no longer dataset-specific, because the panel builder is
corpus-agnostic. See docs/evaluation.md.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from occam.vcell.panel.panel_io import Panel

CONTROL = "non-targeting"

import torch

def _avg_ctrl_embeddings(model, panel: Panel, idx: np.ndarray, device,
                         batch_size: int = 512) -> torch.Tensor:
    """Mean of the encoder embeddings of the k pool-A control cells in each row of `idx`.

    (n, k) row indices -> (n, hidden) tensor. Averaging is done in EMBEDDING space because
    that is where training averages them (`ctrl_avg_k` on the control-embedding pool); the
    encoder is nonlinear, so mean(encode(x)) != encode(mean(x)) and the two must not be mixed.
    """
    n, k = idx.shape
    flat = idx.reshape(-1)
    out = torch.empty(len(flat), model.encoder_hidden_size, device=device)
    full = np.zeros((batch_size, panel.n_enc), dtype=np.float32)
    with torch.no_grad():
        for s0 in range(0, len(flat), batch_size):
            e0 = min(s0 + batch_size, len(flat))
            full[: e0 - s0] = 0.0
            full[: e0 - s0, panel.enc_cols] = panel.ctrl_X_model[flat[s0:e0]]
            out[s0:e0] = model.encoder(torch.from_numpy(full[: e0 - s0]).to(device)).float()
    return out.view(n, k, -1).mean(dim=1)


def predict_panel(model, panel: Panel, pert2id: Dict[str, int], device,
               predict_fn=None, eval_control: str = "single",
               batch_size: int = 512, seed: int = 42,
               input_override: Optional[np.ndarray] = None,
               ctrl_avg_k: int = 1
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One predicted cell per real perturbed cell, each from ONE real control cell.

    No averaging: `eval_control='single'` draws a single model-visible control cell per
    predicted cell. Returns `(pred, labels)` restricted to the panel's kept genes, with the
    per-cell ordering matching the real panel's perturbed cells so that seen/unseen
    stratification and any per-cell pairing stay valid.

    `input_override` — (n_real, n_keep), indexed on the panel's OWN row axis — replaces the
    drawn control cell as the model input, row for row. This is a **deliberate cheat** and
    exists for exactly one purpose: the round-trip ceiling (`scripts/eval_h1_roundtrip.py`),
    which feeds the TRUE PERTURBED cell through encoder->decoder to measure the best score
    any latent-space method could reach through a given frozen decoder. `ctrl_used` is still
    the drawn control, so shrink and the gamma variants keep their usual meaning.

    ⚠️ Nothing that reports a model score may pass this. It leaks the target.
    """
    if predict_fn is None:
        def predict_fn(ctrl_t, pert_idx):
            return model(ctrl_t, pert_idx)["prediction"]

    rng = np.random.default_rng(seed)
    predict_panel.last_nonfinite_rows = 0        # read by the trainer's refuse-to-save guard
    if model is not None:
        model.eval()

    sel = np.where(panel.real_labels != CONTROL)[0]
    known = np.array([panel.real_labels[i] in pert2id for i in sel])
    if not known.all():
        missing = sorted({panel.real_labels[i] for i in sel[~known]})
        print(f"[panel] {len(missing)} perturbations absent from pert2id, dropped: "
              f"{missing[:8]}{' ...' if len(missing) > 8 else ''}")
        sel = sel[known]
    labels = panel.real_labels[sel]
    n = len(sel)
    n_ctrl = panel.ctrl_X_model.shape[0]

    # Control input per predicted cell: one real control cell, drawn uniformly.
    #
    # Only 'single' is offered. An earlier path also had a 'batch' mode matching each
    # predicted cell to a control from the same gem_group, but that needs a batch label on
    # the cell being predicted, and `panel_real` carries only `target_gene` and `context`.
    # Matching to a *random* batch is just a uniform draw with extra steps, so pretending to
    # offer the option would only invite a false claim of batch control in the writeup.
    if eval_control != "single":
        raise ValueError(
            f"panel_eval_control={eval_control!r} is not available on the H1 panel: its "
            f"perturbed cells carry no batch label to match against. Use 'single'.")
    cin_idx = rng.integers(0, n_ctrl, size=n)
    K = max(1, int(ctrl_avg_k))
    cin_extra = rng.integers(0, n_ctrl, size=(n, K - 1)) if K > 1 else None
    if K > 1:
        print(f"[panel] control input = mean of {K} pool-A control embeddings per predicted cell "
              f"(matches --ctrl_avg_k)", flush=True)

    if input_override is not None:
        if input_override.shape[0] != panel.real_X_model.shape[0]:
            raise ValueError(
                f"input_override has {input_override.shape[0]} rows but the panel has "
                f"{panel.real_X_model.shape[0]} — it must be indexed on the panel's own "
                f"row axis so that `sel` selects the matching perturbed cells")
        print(f"[panel] ⚠️ INPUT OVERRIDE ACTIVE — the model is being fed the TRUE PERTURBED "
              f"cell, not a control. This is a CEILING measurement, not a model score.")

    # The encoder needs the full 44,728-vector; only kept columns are ever scored.
    out = np.empty((n, panel.n_keep), dtype=np.float32)
    full = np.zeros((batch_size, panel.n_enc), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            full[: e - s] = 0.0
            if input_override is not None:
                full[: e - s, panel.enc_cols] = input_override[sel[s:e]]
            else:
                full[: e - s, panel.enc_cols] = panel.ctrl_X_model[cin_idx[s:e]]
            ct = torch.from_numpy(full[: e - s]).to(device)
            if K > 1 and input_override is None:
                # pass a (B, hidden) embedding: VirtualCellModel.forward detects the width and
                # skips its own encoder, so the averaged embedding is used verbatim
                idx_k = np.concatenate([cin_idx[s:e, None], cin_extra[s:e]], axis=1)
                ct = _avg_ctrl_embeddings(model, panel, idx_k, device)
            pidx = [[pert2id[p]] for p in labels[s:e]]
            pred = predict_fn(ct, pidx).float().cpu().numpy()
            out[s:e] = pred[:, panel.enc_cols]
            if (s // batch_size) % 50 == 0:
                print(f"    [panel] {e}/{n} cells predicted", flush=True)
    print(f"[panel] predicted {n} cells over {len(set(labels))} perturbations "
          f"(eval_control={eval_control})")
    ctrl_used = panel.ctrl_X_model[cin_idx]

    # ================= NON-FINITE PREDICTION GUARD (F158) =================
    # The training loop has a NaN guard; INFERENCE had none, and that asymmetry cost 7 of 11
    # E8 arms. The model can be perfectly healthy — finite weights, descending train loss —
    # and still overflow on an occasional cell. A single non-finite value anywhere in this
    # matrix propagates into the emitted h5ad and makes the scorer return its degenerate
    # floor (0.0591, pds -0.0262), so ONE bad cell out of 102,000 destroys the whole panel
    # score. Unlike training there is no tolerable rate here: inference has to be clean once.
    #
    # Fallback = that cell's OWN control row, i.e. "predict no change for this cell". This is
    # the panel's existing convention (unreachable genes are already filled from the control
    # profile a few lines below) and it is conservative: it can only pull a cell toward the
    # control, never invent signal.
    #
    # ⚠️ COUNTED AND PRINTED, never silent. A bare np.nan_to_num would hide the instability
    # this is meant to expose, and would let a badly-diverging arm score respectably.
    _bad = ~np.isfinite(out)
    _nbad = int(_bad.sum())
    if _nbad:
        _rows = np.unique(np.nonzero(_bad)[0])
        out[_bad] = ctrl_used[_bad]
        _pct_cells = 100.0 * _rows.size / max(out.shape[0], 1)
        print(f"[panel] ⚠️ NON-FINITE PREDICTIONS: {_nbad:,} values across {_rows.size:,} of "
              f"{out.shape[0]:,} cells ({_pct_cells:.3f}%) replaced by each cell's own "
              f"control row (F158). The model is unstable on those inputs — treat this "
              f"score as degraded, and if the percentage is not tiny, lower decoder_lr "
              f"rather than trusting the number.", flush=True)
        predict_panel.last_nonfinite_rows = int(_rows.size)
    # ================= end non-finite prediction guard =================

    return out, labels, ctrl_used, cin_idx


def predict_null(model, panel: Panel, device, ctrl_idx: np.ndarray,
                 predict_fn=None, batch_size: int = 512) -> np.ndarray:
    """Model output for control cells under the NULL perturbation token (`pert=[-1]`).

    This is the model's own estimate of "this control cell, unperturbed", in the model's
    own output space. It is the reference the perturbation delta should be taken against:

        delta = model(ctrl, p) - model(ctrl, null)

    Both terms live in the SAME (compressed) output space, so their difference isolates the
    perturbation effect from the model's systematic offset. Subtracting the raw control
    instead mixes two spaces.

    Only meaningful because control-identity samples are trained (`control_identity_ratio`,
    plus the H1 pool-A identity samples), which is what gives the null token a defined
    meaning at all.
    """
    if predict_fn is None:
        def predict_fn(ctrl_t, pert_idx):
            return model(ctrl_t, pert_idx)["prediction"]
    if model is not None:
        model.eval()
    n = len(ctrl_idx)
    out = np.empty((n, panel.n_keep), dtype=np.float32)
    full = np.zeros((batch_size, panel.n_enc), dtype=np.float32)
    with torch.no_grad():
        for s0 in range(0, n, batch_size):
            e = min(s0 + batch_size, n)
            full[: e - s0] = 0.0
            full[: e - s0, panel.enc_cols] = panel.ctrl_X_model[ctrl_idx[s0:e]]
            ct = torch.from_numpy(full[: e - s0]).to(device)
            out[s0:e] = predict_fn(ct, [[-1]] * (e - s0)).float().cpu().numpy()[:, panel.enc_cols]
    return out
