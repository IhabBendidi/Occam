"""Weight init, gene-variance-factor computation, and the augmentation controller.

Extracted from ``decoder_training_pipeline.py:436-835``.

The GVF cache is keyed on a content-aware SHA-256 fingerprint (file paths,
mtimes, sizes, ``target_sum``, gene-name hash), so it invalidates when the
corpus changes rather than when a path string does.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import math
import os
import sys
import json
import csv
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import anndata
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from occam.model import AutoEncoder, MODEL_CONFIGS, resolve_decoder_hidden
from occam.decoder.external import HybridNormGlobalSkipDecoder
from occam.transforms import NormalizePerCell, Log1p
from occam.data.datasets import OccamChunkedDataset, OccamAdataDataset
from occam.data.datasets import optim_train_val_chunk_split, compute_global_chunk_info
from occam.ddp import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    is_rank0_env,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Numpy 2.x -> 1.x compatibility shim
if not hasattr(np, '_core'):
    np._core = np.core
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray

# ---- Constants ----

VALID_INIT_SCHEMES = (
    "default", "kaiming_normal", "kaiming_uniform",
    "xavier_normal", "xavier_uniform", "orthogonal", "trunc_normal",
)

_DEFAULT_AUG_HPARAMS = {
    'initial_gene_jitter_strength': 0.2,
    'final_gene_jitter_strength': 0.05,
    'gene_jitter_prob': 0.6,
    'initial_gene_mask_rate': 0.05,
    'final_gene_mask_rate': 0.0,
    'gene_mask_prob': 0.5,
    'max_effective_mask_rate': 0.25,
    'initial_mask_activation_threshold': 0.5,
    'final_mask_activation_threshold': 0.0,
    'initial_sensitivity_p': 2.0,
    'final_sensitivity_p': 0.25,
    'sensitivity_p_phase1_epochs_ratio': 0.25,
    'initial_sparsity_modulator_strength': 0.75,
    'final_sparsity_modulator_strength': 0.1,
    'initial_cov_influence_weight': 1.5,
    'final_cov_influence_weight': 0.5,
    'gvf_lower_p': 0.1,
    'gvf_upper_p': 0.95,
}

# BMDB benchmark paths
_BMDB_DIR = None  # resolved lazily
from occam.decoder_train.config import _DEFAULT_AUG_HPARAMS

__all__ = ["compute_gvf_chunked", "AugmentationController", "init_weights"]


def init_weights(module: nn.Module, scheme: str):
    """Apply weight initialization scheme to nn.Linear layers."""
    if not isinstance(module, nn.Linear):
        return
    if scheme == "kaiming_normal":
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
    elif scheme == "kaiming_uniform":
        nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
    elif scheme == "xavier_normal":
        nn.init.xavier_normal_(module.weight)
    elif scheme == "xavier_uniform":
        nn.init.xavier_uniform_(module.weight)
    elif scheme == "orthogonal":
        nn.init.orthogonal_(module.weight)
    elif scheme == "trunc_normal":
        nn.init.trunc_normal_(module.weight, std=0.02)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


# ---- Gene Variability Factor (GVF) ----

def _compute_data_fingerprint(
    data_path: str,
    target_sum: float,
    gene_names: List[str],
) -> Tuple[str, dict]:
    """Compute a content-aware fingerprint for GVF cache invalidation.

    The fingerprint captures:
    - File identity: resolved canonical paths, modification times, and byte sizes
      of every h5ad file. Detects added/removed/modified files in a folder.
    - Preprocessing: target_sum (library-size normalization target affects GVF
      because GVF is computed in NormalizePerCell + Log1p space).
    - Gene vocabulary: SHA-256 hash of the ordered gene list (gene alignment
      determines which genes appear in the output and in what order).

    Args:
        data_path: Path to h5ad file or folder of h5ad files.
        target_sum: Library-size normalization target.
        gene_names: Ordered gene vocabulary from the encoder checkpoint.

    Returns:
        (fingerprint_hash, metadata_dict) where fingerprint_hash is a 16-char
        hex string suitable for cache filenames, and metadata_dict contains
        the full fingerprint details for auditability.
    """
    data_path_resolved = os.path.realpath(data_path)

    file_entries = []
    if os.path.isdir(data_path_resolved):
        for root, _, files in os.walk(data_path_resolved):
            for fname in sorted(files):
                if fname.endswith('.h5ad'):
                    fpath = os.path.join(root, fname)
                    stat = os.stat(fpath)
                    file_entries.append({
                        'path': fpath,
                        'mtime_ns': stat.st_mtime_ns,
                        'size': stat.st_size,
                    })
        file_entries.sort(key=lambda x: x['path'])
    else:
        stat = os.stat(data_path_resolved)
        file_entries.append({
            'path': data_path_resolved,
            'mtime_ns': stat.st_mtime_ns,
            'size': stat.st_size,
        })

    if not file_entries:
        raise ValueError(f"No h5ad files found at {data_path}")

    gene_names_hash = hashlib.sha256('|'.join(gene_names).encode()).hexdigest()

    fingerprint_data = {
        'files': file_entries,
        'target_sum': target_sum,
        'gene_names_hash': gene_names_hash,
        'num_genes': len(gene_names),
        'num_files': len(file_entries),
    }

    fingerprint_json = json.dumps(fingerprint_data, sort_keys=True)
    fingerprint_hash = hashlib.sha256(fingerprint_json.encode()).hexdigest()[:16]

    return fingerprint_hash, fingerprint_data


def compute_gvf_chunked(
    dataset,
    cache_dir: Optional[str] = None,
    fingerprint_hash: Optional[str] = None,
    fingerprint_meta: Optional[dict] = None,
) -> torch.Tensor:
    """Compute Gene Variability Factor by streaming through dataset chunks.

    GVF is the per-gene coefficient of variation (std / mean) computed on
    NormalizePerCell + Log1p transformed expression, then normalized by its
    own mean so the average GVF is 1.0. This measures biological variability
    per gene in the preprocessed space the model actually trains on.

    Uses Welford's parallel/batch algorithm to compute per-gene mean and
    variance in a single pass without loading all data into memory.

    Caching: When cache_dir and fingerprint_hash are provided, results are
    stored as ``gvf_{hash}.npy`` with a metadata sidecar ``gvf_{hash}.json``.
    The fingerprint encodes file identity (paths + mtimes + sizes),
    target_sum, and gene vocabulary hash, so cached values are only reused
    when the underlying data is identical.

    Args:
        dataset: An iterable dataset yielding pre-batched tensors
            (already NormalizePerCell + Log1p transformed).
        cache_dir: Directory for GVF cache files.
        fingerprint_hash: Content-aware hash for cache filename.
        fingerprint_meta: Full fingerprint metadata to persist in sidecar JSON.

    Returns:
        GVF tensor of shape (num_genes,), normalized by its mean.
    """
    num_genes = dataset.num_genes
    cache_npy = None
    cache_json = None

    # Check cache
    if cache_dir and fingerprint_hash:
        cache_npy = os.path.join(cache_dir, f"gvf_{fingerprint_hash}.npy")
        cache_json = os.path.join(cache_dir, f"gvf_{fingerprint_hash}.json")

        if os.path.isfile(cache_npy):
            try:
                cached = np.load(cache_npy)
                if cached.shape[0] == num_genes:
                    logging.info(f"Loaded cached GVF from {cache_npy}")
                    logging.info(f"  Fingerprint: {fingerprint_hash}")
                    return torch.from_numpy(cached).float()
                else:
                    logging.warning(
                        f"Cached GVF shape {cached.shape[0]} != num_genes "
                        f"{num_genes}. Recomputing."
                    )
            except Exception as e:
                logging.warning(f"Failed to load cached GVF: {e}. Recomputing.")

    logging.info("Computing Gene Variability Factor (GVF) by streaming through data...")
    logging.info("  Space: NormalizePerCell + Log1p (matching training preprocessing)")
    n = 0
    mean = np.zeros(num_genes, dtype=np.float64)
    M2 = np.zeros(num_genes, dtype=np.float64)

    loader = DataLoader(dataset, batch_size=None, num_workers=0)
    n_chunks = len(dataset) if hasattr(dataset, '__len__') else None
    pbar = tqdm(
        loader, total=n_chunks,
        desc="Computing GVF",
        unit="chunk",
        dynamic_ncols=True,
    )
    for batch in pbar:
        # Datasets differ in what they yield, and GVF wants only the expression tensor:
        #   OccamChunkedDataset(track_presence=False) -> plain tensor
        #   OccamChunkedDataset(track_presence=True)  -> (data, mask_2D)
        #   OccamAdataDataset                           -> (data, label)
        # Element 0 is the data in every case, so unlike DecoderTrainer._unpack_batch
        # there is no need to disambiguate the second element here. Without this, a
        # `--data_mode in_memory` run dies with "'list' object has no attribute 'numpy'";
        # it went unnoticed because every production decoder run uses `optim`.
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        batch_np = batch.numpy().astype(np.float64)
        batch_n = batch_np.shape[0]
        batch_mean = batch_np.mean(axis=0)
        batch_var = batch_np.var(axis=0, ddof=0) * batch_n

        delta = batch_mean - mean
        new_n = n + batch_n
        mean = (n * mean + batch_n * batch_mean) / max(new_n, 1)
        M2 += batch_var + delta ** 2 * n * batch_n / max(new_n, 1)
        n = new_n
        pbar.set_postfix(samples=f"{n:,}")

    std = np.sqrt(M2 / max(n - 1, 1))
    gvf = std / (mean + 1e-8)
    gvf /= (gvf.mean() + 1e-8)

    logging.info(
        f"GVF computed over {n:,} samples. "
        f"Min: {gvf.min():.4f}, Max: {gvf.max():.4f}, Mean: {gvf.mean():.4f}"
    )

    # Save cache
    if cache_npy:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(cache_npy, gvf.astype(np.float32))
            sidecar = {
                'fingerprint_hash': fingerprint_hash,
                'fingerprint': fingerprint_meta or {},
                'gvf_stats': {
                    'n_samples': int(n),
                    'num_genes': int(num_genes),
                    'min': float(gvf.min()),
                    'max': float(gvf.max()),
                    'mean': float(gvf.mean()),
                },
                'computed_at': datetime.now().isoformat(),
            }
            with open(cache_json, 'w') as f:
                json.dump(sidecar, f, indent=2)
            logging.info(f"Saved GVF cache to {cache_npy}")
            logging.info(f"Saved GVF metadata to {cache_json}")
        except Exception as e:
            logging.warning(f"Failed to save GVF cache: {e}")

    return torch.from_numpy(gvf.astype(np.float32))


# ---- Augmentation Controller ----

class AugmentationController:
    """Multi-tier data augmentation with cosine-annealed parameters.

    Applies gene-level jitter and masking modulated by four tiers:
    1. Per-sample relative expression
    2. Per-gene batch median expression
    3. Global gene variability factor (GVF, CoV-based)
    4. Per-sample sparsity
    """

    def __init__(
        self,
        gvf_tensor: torch.Tensor,
        n_epochs: int,
        device: str,
        hparams_override: Optional[dict] = None,
    ):
        self.n_epochs = n_epochs
        self.device = device
        self.hparams = {**_DEFAULT_AUG_HPARAMS, **(hparams_override or {})}

        self.raw_gene_variability_factor = gvf_tensor.to(device)
        self.gene_variability_factor = torch.clone(self.raw_gene_variability_factor)

        logging.info(
            f"AugmentationController initialized. "
            f"GVF range: [{self.raw_gene_variability_factor.min():.2f}, "
            f"{self.raw_gene_variability_factor.max():.2f}]"
        )

        # Per-epoch state
        self.current_jitter_strength = 0.0
        self.current_mask_rate = 0.0
        self.current_sensitivity_p = 0.0
        self.current_mask_activation_threshold = 0.0
        self.current_sparsity_modulator_strength = 0.0
        self.current_cov_influence_weight = 0.0
        self.current_gvf_lower_clamp_val = 0.0
        self.current_gvf_upper_clamp_val = 0.0

    def update_epoch(self, epoch: int):
        """Update annealed augmentation parameters for the current epoch."""
        progress = epoch / max(self.n_epochs - 1, 1)
        cosine_term = 0.5 * (1 + math.cos(math.pi * progress))
        hp = self.hparams

        self.current_jitter_strength = (
            hp['final_gene_jitter_strength']
            + (hp['initial_gene_jitter_strength'] - hp['final_gene_jitter_strength']) * cosine_term
        )
        self.current_mask_rate = (
            hp['final_gene_mask_rate']
            + (hp['initial_gene_mask_rate'] - hp['final_gene_mask_rate']) * cosine_term
        )
        self.current_mask_activation_threshold = (
            hp['final_mask_activation_threshold']
            + (hp['initial_mask_activation_threshold'] - hp['final_mask_activation_threshold']) * cosine_term
        )
        self.current_sparsity_modulator_strength = (
            hp['final_sparsity_modulator_strength']
            + (hp['initial_sparsity_modulator_strength'] - hp['final_sparsity_modulator_strength']) * cosine_term
        )
        self.current_cov_influence_weight = (
            hp['final_cov_influence_weight']
            + (hp['initial_cov_influence_weight'] - hp['final_cov_influence_weight']) * cosine_term
        )

        # Phased annealing for sensitivity_p
        phase1_epochs = int(self.n_epochs * hp['sensitivity_p_phase1_epochs_ratio'])
        if epoch < phase1_epochs:
            self.current_sensitivity_p = hp['initial_sensitivity_p']
        else:
            remaining = self.n_epochs - phase1_epochs
            if remaining > 0:
                p2_progress = (epoch - phase1_epochs) / max(remaining - 1, 1)
                p2_cosine = 0.5 * (1 + math.cos(math.pi * p2_progress))
                self.current_sensitivity_p = (
                    hp['final_sensitivity_p']
                    + (hp['initial_sensitivity_p'] - hp['final_sensitivity_p']) * p2_cosine
                )
            else:
                self.current_sensitivity_p = hp['final_sensitivity_p']

        # GVF clamping
        with torch.no_grad():
            self.current_gvf_lower_clamp_val = torch.quantile(
                self.raw_gene_variability_factor, hp['gvf_lower_p']
            )
            self.current_gvf_upper_clamp_val = torch.quantile(
                self.raw_gene_variability_factor, hp['gvf_upper_p']
            )
            self.gene_variability_factor = self.raw_gene_variability_factor.clamp(
                min=self.current_gvf_lower_clamp_val,
                max=self.current_gvf_upper_clamp_val,
            ).unsqueeze(0)

    def apply(self, Y_batch: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to a batch of log1p-normalized expression."""
        hp = self.hparams
        apply_jitter = (
            torch.rand(1).item() < hp['gene_jitter_prob']
            and self.current_jitter_strength > 1e-6
        )
        apply_masking = (
            torch.rand(1).item() < hp['gene_mask_prob']
            and self.current_mask_rate > 1e-6
        )

        if not (apply_jitter or apply_masking):
            return Y_batch

        with torch.no_grad():
            augmented_counts = torch.expm1(Y_batch)

            # Tier 1: Per-sample relative expression
            sample_max_counts, _ = torch.max(augmented_counts, dim=1, keepdim=True)
            per_sample_relative = augmented_counts / sample_max_counts.clamp(min=1e-8)

            # Tier 2: Per-gene batch median expression
            raw_batch = augmented_counts.clone()
            raw_batch[raw_batch == 0] = float('nan')
            batch_gene_medians = torch.nanmedian(raw_batch, dim=0).values
            batch_gene_medians.nan_to_num_(0.0)
            max_median = torch.max(batch_gene_medians).clamp(min=1e-8)
            batch_gene_modulator = batch_gene_medians / max_median

            # Tier 3: Global GVF
            cov_modulator = 1 + self.current_cov_influence_weight * (
                self.gene_variability_factor - 1
            )

            # Tier 4: Per-sample sparsity
            sample_sparsity = (augmented_counts == 0).float().mean(dim=1, keepdim=True)
            avg_sparsity = sample_sparsity.mean()
            sparsity_effect = (sample_sparsity - avg_sparsity) / (avg_sparsity + 1e-8)
            sparsity_factor = 1 + self.current_sparsity_modulator_strength * sparsity_effect

            # Combine
            composite = (
                cov_modulator
                * batch_gene_modulator
                * per_sample_relative.pow(self.current_sensitivity_p)
                * sparsity_factor
            )

            # Gene masking
            if apply_masking:
                per_element_mask = (self.current_mask_rate * composite).clamp_(
                    max=hp['max_effective_mask_rate']
                )
                dropout_mask = torch.rand_like(augmented_counts) < per_element_mask
                above_threshold = augmented_counts > self.current_mask_activation_threshold
                augmented_counts[above_threshold & dropout_mask] = 0.0

            # Log-normal jitter
            if apply_jitter:
                std_base = self.current_jitter_strength / 3.0
                std_per_element = std_base * composite
                log_noise = torch.randn_like(augmented_counts) * std_per_element
                augmented_counts.mul_(torch.exp(log_noise))

            Y_batch = torch.log1p(augmented_counts.clamp_(min=0.0))
        return Y_batch

    def get_wandb_dict(self) -> dict:
        """Return current augmentation parameters for wandb logging."""
        return {
            "aug/jitter_strength": self.current_jitter_strength,
            "aug/mask_rate": self.current_mask_rate,
            "aug/sensitivity_p": self.current_sensitivity_p,
            "aug/mask_activation_threshold": self.current_mask_activation_threshold,
            "aug/sparsity_modulator_strength": self.current_sparsity_modulator_strength,
            "aug/cov_influence_weight": self.current_cov_influence_weight,
            "aug/gvf_lower_clamp_val": float(self.current_gvf_lower_clamp_val),
            "aug/gvf_upper_clamp_val": float(self.current_gvf_upper_clamp_val),
        }


# ---- Decoder Evaluation Metrics ----
