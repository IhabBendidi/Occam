"""Composable per-sample transform classes for Occam data processing."""

from abc import ABC, abstractmethod

import anndata
import numpy as np
import torch


class OccamTransform(ABC):
    """Base class for Occam per-sample transforms."""

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the transform to a single cell's gene expression vector.

        Args:
            x: 1D tensor of gene expression values for one cell.

        Returns:
            Transformed 1D tensor.
        """
        pass


class NormalizePerCell(OccamTransform):
    """Library-size normalization: x = x / x.sum() * target_sum."""

    def __init__(self, target_sum: float = 1e4):
        self.target_sum = target_sum

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        total = x.sum()
        if total > 0:
            return x / total * self.target_sum
        return x


class Log1p(OccamTransform):
    """Log1p transform: x = log(1 + x)."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log1p(x)


class Scale(OccamTransform):
    """Z-score scaling using precomputed per-gene mean and std.

    Args:
        mean: 1D tensor of per-gene means (after NormalizePerCell + Log1p).
        std: 1D tensor of per-gene standard deviations.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std.clamp(min=1e-8)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def compute_gene_stats(
    adata_path: str,
    target_sum: float = 1e4,
    chunk_size: int = 10000,
    gene_indices: np.ndarray = None,
) -> dict:
    """Compute per-gene mean and std over an h5ad file using Welford's algorithm.

    Statistics are computed on NormalizePerCell + Log1p transformed data,
    matching the default Occam transform pipeline.

    Args:
        adata_path: Path to .h5ad file.
        target_sum: Target sum for library-size normalization.
        chunk_size: Number of cells to process at a time.
        gene_indices: Optional array of gene indices to subset. If None, uses all genes.

    Returns:
        Dict with 'mean' (np.float32 array), 'std' (np.float32 array), 'n_cells' (int).
    """
    adata = anndata.read_h5ad(adata_path, backed="r")
    n_obs = adata.n_obs
    n_genes = adata.n_vars if gene_indices is None else len(gene_indices)

    # Welford's online algorithm for numerically stable mean/variance
    count = 0
    mean = np.zeros(n_genes, dtype=np.float64)
    m2 = np.zeros(n_genes, dtype=np.float64)

    for start in range(0, n_obs, chunk_size):
        end = min(start + chunk_size, n_obs)
        chunk = adata.X[start:end]

        if hasattr(chunk, "toarray"):
            chunk = chunk.toarray()
        chunk = np.asarray(chunk, dtype=np.float64)

        if gene_indices is not None:
            chunk = chunk[:, gene_indices]

        # NormalizePerCell
        row_sums = chunk.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        chunk = chunk / row_sums * target_sum

        # Log1p
        chunk = np.log1p(chunk)

        # Welford batch update
        for i in range(chunk.shape[0]):
            count += 1
            delta = chunk[i] - mean
            mean += delta / count
            delta2 = chunk[i] - mean
            m2 += delta * delta2

    variance = m2 / max(count - 1, 1)
    std = np.sqrt(variance)

    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "n_cells": count,
    }


# Backward compatibility aliases
OccamTransform = OccamTransform
