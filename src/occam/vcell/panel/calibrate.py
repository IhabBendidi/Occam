"""Control-fit per-gene quantile calibration, and the gamma amplification estimate.

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

from occam.vcell.panel.predict import _avg_ctrl_embeddings

CONTROL = "non-targeting"

import torch

def estimate_gamma(null_pred: np.ndarray, src_q, ref_q, device,
                   n_pairs: int = 4000, seed: int = 42) -> float:
    """Controls-only estimate of how much the quantile map amplifies a SIGNAL-FREE delta.

    Take two different control cells, run both under the null token, and measure how much
    the map stretches the difference between them:

        gamma = median over pairs of  ||Q(c_a) - Q(c_b)|| / ||c_a - c_b||

    That difference contains cell-to-cell noise and NO perturbation, so gamma isolates the
    map's noise amplification. Dividing a calibrated delta by it removes exactly the
    inflation the map introduces while leaving the map's per-gene RELATIVE reweighting
    intact.

    Why this is the right quantity. The map's slope is fitted to the control marginal,
    which is pure noise; there is no reason a perturbation effect is compressed by the same
    factor as noise. Measured on ess_paired ep30, the model's per-gene spread is 16.6x
    narrower than the real controls' (0.0134 vs 0.2220), and the map inflates the delta
    3.5x (0.0534 -> 0.2287 against a true 0.0650) — the mechanism this undoes.

    Uses no ground truth and no perturbed cells, so it is re-derivable on any new context
    from the controls alone, unlike a scalar shrink fitted to a scored panel.
    """
    rng = np.random.default_rng(seed)
    n = null_pred.shape[0]
    a = rng.integers(0, n, size=n_pairs)
    b = rng.integers(0, n, size=n_pairs)
    keep = a != b
    a, b = a[keep], b[keep]
    raw = np.linalg.norm(null_pred[a] - null_pred[b], axis=1)
    mapped_a = apply_quantile_map(null_pred[a], src_q, ref_q, device)
    mapped_b = apply_quantile_map(null_pred[b], src_q, ref_q, device)
    mapped = np.linalg.norm(mapped_a - mapped_b, axis=1)
    ok = raw > 1e-8
    ratio = mapped[ok] / raw[ok]
    g = float(np.median(ratio))
    print(f"[panel] gamma (map amplification of a signal-free delta) = {g:.3f}  "
          f"[p25={np.percentile(ratio,25):.3f} p75={np.percentile(ratio,75):.3f}, "
          f"{len(ratio)} control pairs]")
    return g


def _column_quantiles(x: np.ndarray, qs: np.ndarray, device, gene_chunk: int = 4096) -> np.ndarray:
    """np.quantile(x, qs, axis=0) ('linear'), by sorting gene columns on the GPU in chunks."""
    if str(device) == "cpu" or not torch.cuda.is_available():
        return np.quantile(x, qs, axis=0).astype(np.float32)
    n = x.shape[0]
    pos = torch.as_tensor(np.asarray(qs), dtype=torch.float64) * (n - 1)
    lo = pos.floor().long().to(device); hi = pos.ceil().long().to(device)
    w = (pos - pos.floor()).float().to(device)
    out = np.empty((len(qs), x.shape[1]), dtype=np.float32)
    for g0 in range(0, x.shape[1], gene_chunk):
        g1 = min(g0 + gene_chunk, x.shape[1])
        col = torch.from_numpy(np.ascontiguousarray(x[:, g0:g1])).to(device)
        srt = torch.sort(col, dim=0).values
        out[:, g0:g1] = (srt[lo] + (srt[hi] - srt[lo]) * w[:, None]).float().cpu().numpy()
        del col, srt
    return out


def fit_control_quantile_map(model, panel: Panel, pert2id: Dict[str, int], device,
                             predict_fn=None, max_cells: int = 20000,
                             batch_size: int = 512, seed: int = 42,
                             ctrl_avg_k: int = 1
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-gene quantile map, fitted on the MODEL-VISIBLE H1 controls only.

    Source = what the model emits when fed those control cells; reference = what those
    control cells actually are. The perturbation ids are sampled only because the virtual-
    cell model has no null perturbation to condition on — a knockdown moves a small fraction
    of genes, so the per-gene marginal of the predicted set is dominated by the model's
    output distribution, which is exactly the domain shift being corrected (§12.5).

    The map is monotone per gene, so it cannot invent a perturbation-specific effect, and it
    never touches `panel_real`. Same construction as the decoder stage's `--eval_calibrate
    quantile --eval_calib_on controls`, which is the campaign default.
    """
    if predict_fn is None:
        def predict_fn(ctrl_t, pert_idx):
            return model(ctrl_t, pert_idx)["prediction"]

    rng = np.random.default_rng(seed)
    if model is not None:
        model.eval()
    n_ctrl = panel.ctrl_X_model.shape[0]
    take = np.sort(rng.choice(n_ctrl, size=min(max_cells, n_ctrl), replace=False))
    pert_ids = list(pert2id.values())
    draw = rng.choice(len(pert_ids), size=len(take))
    _K = max(1, int(ctrl_avg_k))
    _extra = rng.integers(0, n_ctrl, size=(len(take), _K - 1)) if _K > 1 else None

    src = np.empty((len(take), panel.n_keep), dtype=np.float32)
    full = np.zeros((batch_size, panel.n_enc), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(take), batch_size):
            e = min(s + batch_size, len(take))
            full[: e - s] = 0.0
            full[: e - s, panel.enc_cols] = panel.ctrl_X_model[take[s:e]]
            ct = torch.from_numpy(full[: e - s]).to(device)
            if _K > 1:
                # the map's SOURCE must be what the model emits for the inputs it will actually
                # be given at inference, so average k controls here too
                idx_k = np.concatenate([take[s:e, None], _extra[s:e]], axis=1)
                ct = _avg_ctrl_embeddings(model, panel, idx_k, device)
            pidx = [[pert_ids[draw[j]]] for j in range(s, e)]
            src[s:e] = predict_fn(ct, pidx).float().cpu().numpy()[:, panel.enc_cols]
    ref = panel.ctrl_X_model[take]

    qs = np.linspace(0.0, 1.0, 1001)
    # column quantiles on the GPU (numpy 'linear' interpolation reproduced up to float
    # error): sorting 20k x 18k twice on the CPU was ~1 min of every periodic eval
    src_q = _column_quantiles(src, qs, device)
    ref_q = _column_quantiles(ref, qs, device)
    src_q = np.maximum.accumulate(src_q, axis=0)   # strictly non-decreasing for searchsorted
    print(f"[panel] calibration map fitted on {len(take)} model-visible control cells "
          f"({panel.n_keep} genes)")
    return (torch.from_numpy(src_q.T).contiguous().to(device),
            torch.from_numpy(ref_q.T).contiguous().to(device))


def apply_quantile_map(pred: np.ndarray, src_q: torch.Tensor, ref_q: torch.Tensor,
                       device, batch_size: int = 4096) -> np.ndarray:
    """Piecewise-linear per-gene quantile mapping, batched over genes on the GPU."""
    out = np.empty_like(pred)
    for s in range(0, pred.shape[0], batch_size):
        e = min(s + batch_size, pred.shape[0])
        v = torch.from_numpy(pred[s:e]).to(device).t().contiguous()   # (G, B)
        i = torch.searchsorted(src_q, v).clamp(1, src_q.shape[1] - 1)
        x0 = torch.gather(src_q, 1, i - 1); x1 = torch.gather(src_q, 1, i)
        y0 = torch.gather(ref_q, 1, i - 1); y1 = torch.gather(ref_q, 1, i)
        w = torch.where(x1 > x0, (v - x0) / (x1 - x0).clamp(min=1e-12),
                        torch.zeros_like(v)).clamp(0, 1)
        out[s:e] = (y0 + w * (y1 - y0)).t().contiguous().cpu().numpy()
    return out
