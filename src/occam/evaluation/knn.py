"""Weighted k-NN classifier for perturbation classification evaluation.

Adapted from biomodal_codebase/biomodalities/eval/knn.py — rewritten
without torchmetrics dependency.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.decomposition import PCA


class WeightedKNNClassifier:
    """Weighted k-NN classifier for evaluating embedding quality."""

    def __init__(
        self,
        unique_labels: List[str],
        k: int = 20,
        T: float = 0.07,
        max_distance_matrix_size: int = int(5e6),
        distance_fx: str = "cosine",
        epsilon: float = 0.00001,
        use_pca: bool = False,
    ):
        self.k = k
        self.T = T
        self.max_distance_matrix_size = max_distance_matrix_size
        self.distance_fx = distance_fx
        self.epsilon = epsilon
        self.use_pca = use_pca
        self.unique_labels = unique_labels
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

        self.train_features = []
        self.train_targets = []
        self.test_features = []
        self.test_targets = []

    def _extract_labels(self, labels):
        if isinstance(labels, pd.Series):
            labels = labels.tolist()
        return torch.tensor([self.label_to_idx[label] for label in labels], dtype=torch.long)

    def update(
        self,
        train_features: torch.Tensor = None,
        train_targets=None,
        test_features: torch.Tensor = None,
        test_targets=None,
    ):
        assert (train_features is None) == (train_targets is None)
        assert (test_features is None) == (test_targets is None)

        if train_targets is not None and not isinstance(train_targets, torch.Tensor):
            train_targets = self._extract_labels(train_targets)
        if test_targets is not None and not isinstance(test_targets, torch.Tensor):
            test_targets = self._extract_labels(test_targets)

        if train_features is not None:
            assert train_features.size(0) == train_targets.size(0)
            self.train_features.append(train_features.detach())
            self.train_targets.append(train_targets.detach())

        if test_features is not None:
            assert test_features.size(0) == test_targets.size(0)
            self.test_features.append(test_features.detach())
            self.test_targets.append(test_targets.detach())

    @torch.no_grad()
    def compute(self) -> Tuple[float, float]:
        if not self.train_features or not self.test_features:
            return -1, -1

        train_features = torch.cat(self.train_features)
        train_targets = torch.cat(self.train_targets)
        test_features = torch.cat(self.test_features)
        test_targets = torch.cat(self.test_targets)

        if self.distance_fx == "cosine":
            train_features = F.normalize(train_features)
            test_features = F.normalize(test_features)

        # Apply PCA if flagged, reducing to 32 principal components (matching biomodal)
        if self.use_pca and train_features.shape[1] > 32:
            pca = PCA(n_components=32)
            train_features = torch.tensor(pca.fit_transform(train_features.numpy()), dtype=torch.float32)
            test_features = torch.tensor(pca.transform(test_features.numpy()), dtype=torch.float32)

        num_classes = len(self.unique_labels)
        num_train_images = train_targets.size(0)
        num_test_images = test_targets.size(0)
        chunk_size = min(
            max(1, self.max_distance_matrix_size // num_train_images),
            num_test_images,
        )
        k = min(self.k, num_train_images)

        top1, top5, total = 0.0, 0.0, 0
        retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
        for idx in range(0, num_test_images, chunk_size):
            features = test_features[idx: min((idx + chunk_size), num_test_images), :]
            targets = test_targets[idx: min((idx + chunk_size), num_test_images)]
            batch_size = targets.size(0)

            if self.distance_fx == "cosine":
                similarities = torch.mm(features, train_features.t())
            elif self.distance_fx == "euclidean":
                similarities = 1 / (torch.cdist(features, train_features) + self.epsilon)
            else:
                raise NotImplementedError

            similarities, indices = similarities.topk(k, largest=True, sorted=True)
            candidates = train_targets.view(1, -1).expand(batch_size, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices)

            retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)

            if self.distance_fx == "cosine":
                similarities = similarities.clone().div_(self.T).exp_()

            probs = torch.sum(
                torch.mul(
                    retrieval_one_hot.view(batch_size, -1, num_classes),
                    similarities.view(batch_size, -1, 1),
                ),
                1,
            )
            _, predictions = probs.sort(1, True)

            correct = predictions.eq(targets.data.view(-1, 1))
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = top5 + correct.narrow(1, 0, min(5, k, correct.size(-1))).sum().item()
            total += targets.size(0)

        top1 = top1 * 100.0 / total
        top5 = top5 * 100.0 / total

        self.reset()

        return top1, top5

    def reset(self):
        self.train_features = []
        self.train_targets = []
        self.test_features = []
        self.test_targets = []
