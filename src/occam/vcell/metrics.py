"""Evaluation metrics for Virtual Cell Mode.

Primary metric: pearson_delta — correlation between predicted and true
perturbation effects (relative to control), computed per (perturbation, cell_type).

Also computes: pearson (raw correlation), MSE/MAE (raw and delta).
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr


def _safe_pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, returning 0.0 on non-finite or constant input.

    The non-finite guard is load-bearing, not defensive boilerplate. `scipy.stats.pearsonr`
    calls `np.asarray_chkfinite`, which RAISES on inf/NaN rather than returning NaN — so a
    diverged model does not produce a bad metric, it kills the whole training run from
    inside the evaluation. That is exactly what happened to the LR-sweep cell
    `wg_exact / lr 3e-3 / decoder_lr 2e-3`, which lost its entire history at epoch 20:

        ValueError: array must not contain infs or NaNs

    Returning 0.0 instead lets a diverged configuration finish and REPORT that it diverged,
    which is a usable sweep result. Divergence is still visible as `loss=nan` in the
    training log and as a floored metric here.
    """
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return 0.0
    if a.std() == 0 or b.std() == 0:
        return 0.0
    corr, _ = pearsonr(a, b)
    return 0.0 if np.isnan(corr) else float(corr)


def _overlap_at_n(pred_delta: np.ndarray, truth_delta: np.ndarray, n: int) -> float:
    """Fraction of top-N DE genes (by |delta|) recovered in prediction."""
    if len(pred_delta) < n:
        return 0.0
    top_pred = set(np.argsort(np.abs(pred_delta))[-n:])
    top_truth = set(np.argsort(np.abs(truth_delta))[-n:])
    return len(top_pred & top_truth) / n


def _pert_metrics_sc(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    controls: np.ndarray,
) -> Dict[str, float]:
    """Per-(perturbation, cell_type) metrics computed WITHOUT averaging any cells.

    Each cell is scored against ITS OWN paired control cell:

        delta_i        = pred_i  - ctrl_i
        truth_delta_i  = truth_i - ctrl_i
        pearson_delta  = mean_i corr(delta_i, truth_delta_i)   over genes, per cell

    The only mean taken is over the resulting per-cell SCALARS, which is reporting, not
    aggregation of expression.

    This differs from `_pert_metrics`, which first collapses the cells of a group to a
    pseudobulk mean and subtracts a pooled control mean. That made the validation metric —
    and therefore checkpoint selection — a pseudobulk quantity even though training was
    single-cell, and it is systematically more optimistic: averaging cancels the per-cell
    noise the model cannot predict, so a model that only ever emits the conditional mean
    scores near-perfectly on it.

    Args:
        predictions:  (n_cells, n_genes)
        ground_truth: (n_cells, n_genes)
        controls:     (n_cells, n_genes) the PAIRED control cell for each row.
    """
    pred_delta = predictions - controls
    truth_delta = ground_truth - controls

    # vectorised per-row Pearson: centre each row, then cosine of the centred rows
    def _rowwise_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # same reasoning as _safe_pearsonr: a diverged model must yield a floored metric,
        # not propagate NaN into every downstream aggregate
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        na = np.sqrt((a * a).sum(axis=1))
        nb = np.sqrt((b * b).sum(axis=1))
        denom = na * nb
        out = np.zeros(a.shape[0], dtype=np.float64)
        ok = denom > 0
        out[ok] = (a[ok] * b[ok]).sum(axis=1) / denom[ok]
        return np.nan_to_num(out)

    pd_cells = _rowwise_corr(pred_delta, truth_delta)
    p_cells = _rowwise_corr(predictions, ground_truth)
    diff = predictions - ground_truth
    ddiff = pred_delta - truth_delta

    # overlap is inherently a ranking over genes, so it stays per cell then averages
    n_cells = predictions.shape[0]
    ov20 = np.empty(n_cells)
    ov50 = np.empty(n_cells)
    for i in range(n_cells):
        ov20[i] = _overlap_at_n(pred_delta[i], truth_delta[i], 20)
        ov50[i] = _overlap_at_n(pred_delta[i], truth_delta[i], 50)

    return {
        "pearson_delta": float(pd_cells.mean()),
        "pearson": float(p_cells.mean()),
        "mse": float((diff ** 2).mean()),
        "mae": float(np.abs(diff).mean()),
        "mse_delta": float((ddiff ** 2).mean()),
        "mae_delta": float(np.abs(ddiff).mean()),
        "overlap_20": float(ov20.mean()),
        "overlap_50": float(ov50.mean()),
    }


def _pert_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    control_mean: np.ndarray,
) -> Dict[str, float]:
    """Compute all per-(perturbation, cell_type) metrics.

    Args:
        predictions: (n_cells, n_genes) predicted expression.
        ground_truth: (n_cells, n_genes) actual expression.
        control_mean: (n_genes,) mean control expression for this cell type.

    Returns:
        Dict with pearson_delta, pearson, mse, mae, mse_delta, mae_delta,
        overlap_20, overlap_50.
    """
    pred_mean = predictions.mean(axis=0)
    truth_mean = ground_truth.mean(axis=0)

    pred_delta = pred_mean - control_mean
    truth_delta = truth_mean - control_mean

    return {
        "pearson_delta": _safe_pearsonr(pred_delta, truth_delta),
        "pearson": _safe_pearsonr(pred_mean, truth_mean),
        "mse": float(np.mean((pred_mean - truth_mean) ** 2)),
        "mae": float(np.mean(np.abs(pred_mean - truth_mean))),
        "mse_delta": float(np.mean((pred_delta - truth_delta) ** 2)),
        "mae_delta": float(np.mean(np.abs(pred_delta - truth_delta))),
        "overlap_20": _overlap_at_n(pred_delta, truth_delta, 20),
        "overlap_50": _overlap_at_n(pred_delta, truth_delta, 50),
    }


def compute_perturbation_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    pert_names: np.ndarray,
    cell_types: np.ndarray,
    control_means: Optional[Dict[str, np.ndarray]] = None,
    gene_mask: Optional[np.ndarray] = None,
    gene_masks_per_ct: Optional[Dict[str, np.ndarray]] = None,
    controls: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Compute metrics per (perturbation, cell_type) pair.

    Metrics per pair: pearson_delta, pearson, mse, mae, mse_delta, mae_delta.
    Aggregates: mean/median/std over all pairs for each metric.

    Args:
        predictions: (N, n_genes) predicted expression.
        ground_truth: (N, n_genes) actual expression.
        pert_names: (N,) perturbation gene name per sample.
        cell_types: (N,) cell type per sample.
        control_means: {cell_type: (n_genes,) mean control expression}.
        gene_mask: Optional (n_genes,) global bool mask for relevant genes.
        gene_masks_per_ct: Optional {cell_type: (n_genes,) bool} per-cell-type
            gene masks. When provided, each (pert, ct) pair is evaluated only
            on genes present in that cell type's source data file.

    Returns:
        (aggregate_metrics, per_pert_metrics)
    """
    # `controls` (per-cell paired controls) selects the SINGLE-CELL path, which averages no
    # cells anywhere. Without it the legacy pseudobulk path runs, which collapses each
    # group to a mean and subtracts a pooled control mean.
    single_cell = controls is not None
    if not single_cell and control_means is None:
        raise ValueError("compute_perturbation_metrics needs either `controls` "
                         "(single-cell) or `control_means` (pseudobulk)")

    # Apply global gene mask if provided (and no per-ct masks)
    if gene_mask is not None and gene_masks_per_ct is None:
        predictions = predictions[:, gene_mask]
        ground_truth = ground_truth[:, gene_mask]
        if single_cell:
            controls = controls[:, gene_mask]
        else:
            control_means = {
                ct: cm[gene_mask] for ct, cm in control_means.items()
            }

    # Group by (perturbation, cell_type)
    unique_perts = np.unique(pert_names)
    unique_cts = np.unique(cell_types)

    # Collect per-pair metrics
    metric_keys = [
        "pearson_delta", "pearson", "mse", "mae", "mse_delta", "mae_delta",
        "overlap_20", "overlap_50",
    ]
    all_values = {k: [] for k in metric_keys}
    per_pert_metrics = {}

    for pert in unique_perts:
        if pert == "non-targeting":
            continue
        for ct in unique_cts:
            mask = (pert_names == pert) & (cell_types == ct)
            n_cells = mask.sum()
            if n_cells == 0:
                continue
            if not single_cell and ct not in control_means:
                continue

            preds_pair = predictions[mask]
            truth_pair = ground_truth[mask]
            ctrl_pair = controls[mask] if single_cell else control_means[ct]

            # Apply per-cell-type gene mask
            if gene_masks_per_ct is not None and ct in gene_masks_per_ct:
                ct_gmask = gene_masks_per_ct[ct]
                preds_pair = preds_pair[:, ct_gmask]
                truth_pair = truth_pair[:, ct_gmask]
                ctrl_pair = ctrl_pair[:, ct_gmask] if single_cell else ctrl_pair[ct_gmask]

            pair_metrics = (_pert_metrics_sc(preds_pair, truth_pair, ctrl_pair)
                            if single_cell else
                            _pert_metrics(preds_pair, truth_pair, ctrl_pair))

            for k in metric_keys:
                all_values[k].append(pair_metrics[k])

            key = f"{pert}_{ct}"
            per_pert_metrics[key] = {
                **pair_metrics,
                "cell_type": ct,
                "perturbation": pert,
                "n_cells": int(n_cells),
            }

    # Aggregate
    aggregate = {}
    n_pairs = 0
    for k in metric_keys:
        vals = np.array(all_values[k]) if all_values[k] else np.array([0.0])
        n_pairs = len(vals)
        aggregate[f"{k}_mean"] = float(np.mean(vals))
        aggregate[f"{k}_median"] = float(np.median(vals))
        aggregate[f"{k}_std"] = float(np.std(vals))

    aggregate["n_pert_celltype_pairs"] = n_pairs
    aggregate["n_unique_perturbations"] = len(
        set(m["perturbation"] for m in per_pert_metrics.values())
    )

    return aggregate, per_pert_metrics
