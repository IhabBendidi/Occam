"""Inverse Local Simpson's Index (iLISI) metric for batch integration.

Adapted from biomodal_codebase/biomodalities/eval/ilisi.py — rewritten
without torchmetrics dependency.
"""

import torch
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from typing import List


class TorchILISIMetric:
    """Computes iLISI to measure batch diversity in local neighborhoods."""

    def __init__(self, unique_labels: List[str], perplexity: int = 30,
                 use_pca: bool = False, seed: int = None):
        self.perplexity = perplexity
        self.use_pca = use_pca
        self.unique_labels = unique_labels
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self._embeddings = []
        self._labels = []

    def _extract_labels(self, labels):
        return torch.tensor([self.label_to_idx[label] for label in labels], dtype=torch.long)

    def update(self, embeddings: torch.Tensor, metadata: pd.Series):
        label_indices = self._extract_labels(metadata) if not isinstance(metadata, torch.Tensor) else metadata
        self._embeddings.append(embeddings.detach().cpu())
        self._labels.append(label_indices)

    def compute(self) -> torch.Tensor:
        all_embeddings = torch.cat(self._embeddings, dim=0).numpy()
        all_labels = torch.cat(self._labels, dim=0).numpy()

        if self.use_pca:
            pca = PCA().fit(all_embeddings)
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.where(cumulative_variance >= 0.9)[0][0] + 1
            n_components = max(32, n_components)
            pca = PCA(n_components=n_components)
            all_embeddings = pca.fit_transform(all_embeddings)

        lisi_scores = self._compute_lisi_single_label(all_embeddings, all_labels, self.perplexity)
        return torch.mean(torch.tensor(lisi_scores, dtype=torch.float32))

    def reset(self):
        self._embeddings = []
        self._labels = []

    @staticmethod
    def _compute_lisi_single_label(embeddings: np.ndarray, labels, perplexity: int = 30):
        nearest_neighbors = NearestNeighbors(n_neighbors=perplexity * 3, algorithm='kd_tree').fit(embeddings)
        distances, indices = nearest_neighbors.kneighbors(embeddings)
        indices = indices[:, 1:]
        distances = distances[:, 1:]
        labels = pd.Categorical(labels)
        num_categories = len(labels.categories)
        simpson_indices = TorchILISIMetric._compute_simpson(distances.T, indices.T, labels, num_categories, perplexity)
        ilisi_sc = np.divide(
            1.0, simpson_indices,
            out=np.zeros_like(simpson_indices, dtype=float),
            where=simpson_indices != 0,
        )
        return ilisi_sc

    @staticmethod
    def _compute_simpson(distances, indices, labels, num_categories, perplexity, tol=1e-5):
        num_points = distances.shape[1]
        simpson_index = np.zeros(num_points)
        log_target_entropy = np.log(perplexity)

        for point_idx in range(num_points):
            beta, beta_min, beta_max = 1.0, -np.inf, np.inf
            probabilities = np.exp(-distances[:, point_idx] * beta) + 1e-12
            prob_sum = np.sum(probabilities)
            if prob_sum == 0:
                entropy = 0
                probabilities = np.zeros(distances.shape[0])
            else:
                entropy = np.log(prob_sum) + beta * np.sum(distances[:, point_idx] * probabilities) / prob_sum
                probabilities /= prob_sum
            entropy_diff = entropy - log_target_entropy

            for _ in range(100):
                if abs(entropy_diff) < tol:
                    break
                beta, beta_min, beta_max = TorchILISIMetric._adjust_beta(entropy_diff, beta, beta_min, beta_max)
                probabilities = np.exp(-distances[:, point_idx] * beta)
                prob_sum = np.sum(probabilities)
                if prob_sum == 0:
                    entropy = 0
                    probabilities = np.zeros(distances.shape[0])
                else:
                    entropy = np.log(prob_sum) + beta * np.sum(distances[:, point_idx] * probabilities) / prob_sum
                    probabilities /= prob_sum
                entropy_diff = entropy - log_target_entropy

            for category in labels.categories:
                category_indices = labels[indices[:, point_idx]] == category
                category_prob_sum = np.sum(probabilities[category_indices])
                simpson_index[point_idx] += category_prob_sum * category_prob_sum

        return simpson_index

    @staticmethod
    def _adjust_beta(entropy_diff, beta, beta_min, beta_max):
        if entropy_diff > 0:
            beta_min = beta
            beta = beta * 2 if not np.isfinite(beta_max) else (beta + beta_max) / 2
        else:
            beta_max = beta
            beta = beta / 2 if not np.isfinite(beta_min) else (beta + beta_min) / 2
        return beta, beta_min, beta_max
