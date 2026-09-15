"""Evaluation metrics for benchmark tasks.

Adapted from biomodal_codebase/biomodalities/utils/metrics.py — rewritten
without torchmetrics dependency.
"""

from typing import Dict, List, Sequence

import numpy as np
import torch
from scipy import stats


def accuracy_at_k(
    outputs: torch.Tensor, targets: torch.Tensor, top_k: Sequence[int] = (1, 5)
) -> Sequence[torch.Tensor]:
    with torch.no_grad():
        maxk = min(max(top_k), outputs.size(1))
        batch_size = targets.size(0)
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        res = []
        for k in top_k:
            k = min(k, maxk)
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate_reconstruction_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Compute MSE, MAE, Pearson, Spearman, R-squared for reconstruction."""
    predictions_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    mse = float(np.mean((predictions_np - targets_np) ** 2))
    mae = float(np.mean(np.abs(predictions_np - targets_np)))

    # Per-feature correlations, averaged (vectorized for speed)
    n_features = predictions_np.shape[1]
    p_stds = np.std(predictions_np, axis=0)
    t_stds = np.std(targets_np, axis=0)
    valid_mask = (p_stds > 1e-8) & (t_stds > 1e-8)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) > 0:
        # Vectorized Pearson: correlation of centered/scaled columns
        p_valid = predictions_np[:, valid_indices]
        t_valid = targets_np[:, valid_indices]
        p_centered = p_valid - p_valid.mean(axis=0)
        t_centered = t_valid - t_valid.mean(axis=0)
        p_norm = p_centered / (np.linalg.norm(p_centered, axis=0) + 1e-12)
        t_norm = t_centered / (np.linalg.norm(t_centered, axis=0) + 1e-12)
        pearson_corrs = np.sum(p_norm * t_norm, axis=0)

        # Vectorized Spearman: rank then Pearson
        p_ranks = np.apply_along_axis(stats.rankdata, 0, p_valid)
        t_ranks = np.apply_along_axis(stats.rankdata, 0, t_valid)
        p_rc = p_ranks - p_ranks.mean(axis=0)
        t_rc = t_ranks - t_ranks.mean(axis=0)
        p_rn = p_rc / (np.linalg.norm(p_rc, axis=0) + 1e-12)
        t_rn = t_rc / (np.linalg.norm(t_rc, axis=0) + 1e-12)
        spearman_corrs = np.sum(p_rn * t_rn, axis=0)

        average_pearson = float(np.mean(pearson_corrs))
        average_spearman = float(np.mean(spearman_corrs))
    else:
        average_pearson = 0.0
        average_spearman = 0.0

    ss_res = np.sum((targets_np - predictions_np) ** 2)
    ss_tot = np.sum((targets_np - np.mean(targets_np)) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "MSE": mse,
        "MAE": mae,
        "Average Pearson Correlation": average_pearson,
        "Average Spearman Correlation": average_spearman,
        "R-squared": r_squared,
    }


class StructuralTranscriptomeDistance:
    """Measures structural preservation in reconstruction per batch."""

    def __init__(self, distance_function='frobenius'):
        self.distance_function = distance_function
        self.total_distance = 0.0
        self.total_max_distance = 0.0
        self.num_elements = 0

    def _adjust_data(self, transcriptomes, batch_ids, is_control, control_label):
        if not isinstance(is_control, np.ndarray):
            is_control = np.array(is_control)
        control_mask = is_control == control_label
        if isinstance(batch_ids, torch.Tensor):
            batch_ids = batch_ids.numpy()
        unique_batches = np.unique(batch_ids)
        adjusted = transcriptomes.clone()
        for batch in unique_batches:
            batch_mask = batch_ids == batch
            control_batch_mask = batch_mask & control_mask
            if np.any(control_batch_mask):
                mean_control = transcriptomes[torch.from_numpy(control_batch_mask)].mean(dim=0)
                adjusted[torch.from_numpy(batch_mask)] -= mean_control
        return adjusted

    def update(self, preds, target, batch_ids, is_control, control_label):
        pred_adjusted = self._adjust_data(preds, batch_ids, is_control, control_label)
        target_adjusted = self._adjust_data(target, batch_ids, is_control, control_label)
        if isinstance(batch_ids, torch.Tensor):
            batch_ids = batch_ids.numpy()
        for batch in np.unique(batch_ids):
            batch_mask = batch_ids == batch
            pred_batch = pred_adjusted[torch.from_numpy(batch_mask)]
            target_batch = target_adjusted[torch.from_numpy(batch_mask)]
            if len(pred_batch) > 0 and self.distance_function == 'frobenius':
                distance = torch.norm(pred_batch - target_batch, p='fro') / target_batch.shape[0]
                max_distance = torch.norm(target_batch, p='fro') / target_batch.shape[0]
                self.total_distance += distance.item()
                self.total_max_distance += max_distance.item()
                self.num_elements += 1

    def compute(self):
        if self.num_elements == 0:
            return 0.0
        distance = self.total_distance / self.num_elements
        max_distance = 2 * (self.total_max_distance / self.num_elements)
        integrity = 1 - (distance / max_distance) if max_distance > 0 else 0.0
        return integrity

    def reset(self):
        self.total_distance = 0.0
        self.total_max_distance = 0.0
        self.num_elements = 0
