"""Reconstruction metrics, the BMDB benchmark, and the cell-eval v1 runner.

Extracted from ``decoder_training_pipeline.py:836-1202``.

``run_cell_eval`` is the shared cell-eval **v1** subprocess wrapper. The
virtual-cell panel module (``occam.vcell.panel.celleval_v1``) calls it rather
than carrying a second implementation, so there is exactly one place in the repo
that knows how to invoke cell-eval.

⚠️ ``combined_score`` (and therefore best-checkpoint selection) includes
Spearman over zero-filled padding genes, which collapses for any head that does
not emit exact zeros. The collapse is a monotone function of the zero-fill
fraction -- TREK at 1.4% padding loses 0.69 Spearman, nadig at 80.2% loses
90.46 -- while Pearson and MAE barely move. Select with
``eval_present_genes_only=True``, or on Pearson/MAE/cell-eval instead.
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

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
RECALL_THR_PAIRS = [(0.001, 0.999), (0.01, 0.99), (0.05, 0.95), (0.1, 0.9)]


def _get_bmdb_dir():
    """Directory holding the BMDB gene-relationship databases.

    The research tree implemented this four separate times; it now delegates to
    the single resolver in :mod:`occam.paths`, which defaults to the copy
    packaged with the wheel so the benchmark works with zero configuration.
    """
    from occam import paths
    return paths.bmdb_dir()


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
__all__ = [
    "compute_all_metrics", "compute_pearson_per_sample", "compute_spearman_per_sample",
    "run_decoder_benchmark", "run_cell_eval",
]


def _pearson_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample Pearson correlation on GPU tensors. Returns 1-D tensor."""
    pred_c = pred - pred.mean(dim=1, keepdim=True)
    target_c = target - target.mean(dim=1, keepdim=True)
    num = (pred_c * target_c).sum(dim=1)
    den = (pred_c.pow(2).sum(dim=1) * target_c.pow(2).sum(dim=1)).sqrt()
    return num / den.clamp(min=1e-8)


def _spearman_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample Spearman correlation on GPU tensors (rank-Pearson). Returns 1-D tensor."""
    arange = torch.arange(pred.shape[1], device=pred.device, dtype=pred.dtype).unsqueeze(0)
    ranks_p = torch.empty_like(pred)
    ranks_p.scatter_(1, pred.argsort(dim=1), arange.expand_as(pred))
    ranks_t = torch.empty_like(target)
    ranks_t.scatter_(1, target.argsort(dim=1), arange.expand_as(target))
    return _pearson_torch(ranks_p, ranks_t)


def compute_pearson_per_sample(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Vectorized per-sample Pearson correlation (numpy fallback)."""
    pred_c = pred - pred.mean(axis=1, keepdims=True)
    target_c = target - target.mean(axis=1, keepdims=True)
    num = np.sum(pred_c * target_c, axis=1)
    den = np.sqrt(np.sum(pred_c ** 2, axis=1) * np.sum(target_c ** 2, axis=1))
    return np.divide(num, den, out=np.zeros_like(num), where=den != 0)


def compute_spearman_per_sample(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Vectorized per-sample Spearman correlation (numpy fallback)."""
    arange = np.arange(pred.shape[1], dtype=np.float32)
    order_p = np.argsort(pred, axis=1)
    ranks_p = np.empty_like(pred)
    np.put_along_axis(ranks_p, order_p, arange, axis=1)
    order_t = np.argsort(target, axis=1)
    ranks_t = np.empty_like(target)
    np.put_along_axis(ranks_t, order_t, arange, axis=1)
    return compute_pearson_per_sample(ranks_p, ranks_t)


def compute_all_metrics(pred_log1p: np.ndarray, target_log1p: np.ndarray) -> Dict[str, float]:
    """Compute all decoder evaluation metrics in log1p space.

    Returns dict with pearson_mean, spearman_mean, mae_mean, mse_mean,
    mae_inverted, mse_inverted, combined_score (all %-scale where applicable).
    """
    pearson = compute_pearson_per_sample(pred_log1p, target_log1p)
    spearman = compute_spearman_per_sample(pred_log1p, target_log1p)
    mae = np.mean(np.abs(pred_log1p - target_log1p), axis=1)
    mse = np.mean((pred_log1p - target_log1p) ** 2, axis=1)

    mean_pearson = float(np.mean(pearson)) * 100
    mean_spearman = float(np.mean(spearman)) * 100
    mean_mae = float(np.mean(mae))
    mean_mse = float(np.mean(mse))

    mae_inv = 1.0 / (1.0 + mean_mae) * 100
    mse_inv = 1.0 / (1.0 + mean_mse) * 100

    mse_weight = 0.1
    combined = (mean_pearson + mean_spearman + mse_weight * mse_inv) / (2 + mse_weight)

    return {
        "combined_score": combined,
        "pearson_mean": mean_pearson,
        "spearman_mean": mean_spearman,
        "mae_mean": mean_mae,
        "mse_mean": mean_mse,
        "mae_inverted": mae_inv,
        "mse_inverted": mse_inv,
        "n_samples": pred_log1p.shape[0],
        "n_genes": pred_log1p.shape[1],
    }


# ---- BMDB Decoder Benchmark ----


def _extract_decoder_map(
    weights: np.ndarray,
    gene_names: List[str],
    n_components: int = 128,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract decoder map: PCA on weights → cosine similarity matrix."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity

    # Deduplicate
    seen = {}
    for i, g in enumerate(gene_names):
        if g not in seen:
            seen[g] = i
    dedup_indices = list(seen.values())
    if len(dedup_indices) < len(gene_names):
        print(f"  Deduplicated {len(gene_names) - len(dedup_indices)} duplicate gene symbols")
        gene_names = [gene_names[i] for i in dedup_indices]
        weights = weights[dedup_indices]

    weights_scaled = StandardScaler().fit_transform(weights)
    pca = PCA(n_components=min(n_components, weights_scaled.shape[1]))
    gene_vectors = pca.fit_transform(weights_scaled)
    sim_matrix = cosine_similarity(gene_vectors)

    return gene_vectors, sim_matrix, gene_names


def _get_benchmark_relationships(benchmark_dir: str, source: str) -> pd.DataFrame:
    """Load and deduplicate benchmark relationships."""
    df = pd.read_csv(Path(benchmark_dir) / f"{source}.txt")
    sorted_pairs = np.sort(df[['entity1', 'entity2']].values, axis=1)
    df = pd.DataFrame(sorted_pairs, columns=['entity1', 'entity2'])
    return df[df['entity1'] != df['entity2']].drop_duplicates()


def _compute_recall(
    null_dist: np.ndarray,
    query_dist: np.ndarray,
    threshold_pairs: List[Tuple[float, float]],
) -> Dict[str, float]:
    """Percentile-rank recall computation."""
    sorted_null = np.sort(null_dist)
    ranks_left = np.searchsorted(sorted_null, query_dist, side="left") / len(sorted_null)
    ranks_right = np.searchsorted(sorted_null, query_dist, side="right") / len(sorted_null)
    metrics = {}
    for left_thr, right_thr in threshold_pairs:
        left_thr, right_thr = min(left_thr, right_thr), max(left_thr, right_thr)
        recalled = np.sum((ranks_right <= left_thr) | (ranks_left >= right_thr))
        metrics[f"recall_{left_thr}_{right_thr}"] = recalled / len(query_dist)
    return metrics


def _run_bmdb_benchmark(
    sim_matrix: np.ndarray,
    gene_names: List[str],
    benchmark_dir: Optional[str] = None,
    sources: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run benchmark evaluation against BMDB databases."""
    if benchmark_dir is None:
        benchmark_dir = _get_bmdb_dir()
    bench_path = Path(benchmark_dir)
    if sources is None:
        sources = [f.stem for f in bench_path.glob("*.txt")]

    cossim_df = pd.DataFrame(sim_matrix, index=gene_names, columns=gene_names)
    null_dist = sim_matrix[np.triu_indices(len(gene_names), k=1)]

    metrics_list = []
    for source in sources:
        filepath = bench_path / f"{source}.txt"
        if not filepath.exists():
            continue
        rels = _get_benchmark_relationships(benchmark_dir, source)
        rels = rels[rels.entity1.isin(cossim_df.index) & rels.entity2.isin(cossim_df.index)]
        if len(rels) == 0:
            continue

        query_cossim = np.array([
            cossim_df.loc[e1, e2] for e1, e2 in rels.itertuples(index=False)
        ])
        metrics = _compute_recall(null_dist, query_cossim, RECALL_THR_PAIRS)
        metrics["source"] = source
        metrics["valid_pairs"] = len(rels)
        metrics_list.append(metrics)

    return pd.DataFrame(metrics_list) if metrics_list else pd.DataFrame()


def run_cell_eval(
    pred_path: str,
    gt_path: str,
    output_dir: str,
    num_threads: int = 16,
    profile: str = "full",
    timeout: int = 3600,
) -> Dict[str, float]:
    """Run ARC Institute cell-eval via subprocess (jupyter pyenv); parse agg_results.csv.

    Returns {metric_name: mean_value} or {} on any failure (non-fatal). Mirrors the
    helper in virtual_cell/pipeline.py. The cell-eval CLI auto-detects log1p input.
    """
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "cell-eval", "run",
        "-ap", pred_path,
        "-ar", gt_path,
        "-o", output_dir,
        "--num-threads", str(num_threads),
        "--profile", profile,
    ]
    env = os.environ.copy()
    env["PYENV_VERSION"] = "jupyter"
    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            print(f"  cell-eval failed (exit {result.returncode})")
            if result.stderr:
                # Tail, not head: cell-eval logs INFO lines to stderr, so the first 500
                # chars are banner noise and the actual traceback is at the end.
                print(f"  stderr (tail): ...{result.stderr[-3000:]}")
            return {}
    except subprocess.TimeoutExpired:
        print(f"  cell-eval TIMED OUT after {timeout}s ({timeout/3600:.1f}h) — "
              f"raise --celleval_timeout and/or --celleval_num_threads")
        return {}
    except FileNotFoundError:
        print("  cell-eval not found (is pyenv jupyter available?)")
        return {}

    agg_path = os.path.join(output_dir, "agg_results.csv")
    if not os.path.exists(agg_path):
        print(f"  agg_results.csv not found at {agg_path}")
        return {}
    try:
        with open(agg_path) as f:
            rows = {row["statistic"]: row for row in csv.DictReader(f)}
        mean_row = rows.get("mean", {})
        metrics = {}
        for k, v in mean_row.items():
            if k == "statistic":
                continue
            try:
                metrics[k] = float(v)
            except (ValueError, TypeError):
                pass
        print(f"  cell-eval done: {len(metrics)} metrics parsed")
        return metrics
    except Exception as e:
        print(f"  Failed to parse agg_results.csv: {e}")
        return {}


def run_decoder_benchmark(
    model: nn.Module,
    gene_names: List[str],
    n_components: int = 128,
) -> Dict:
    """Run BMDB benchmark on Occam Decoder weights.

    Returns dict with 'results_df', 'mean_recalls', 'n_genes_benchmarked'.
    """
    print("\n--- Occam Decoder Benchmark Evaluation ---")

    # Extract output layer weights
    weights = model.output_head[2].weight.detach().cpu().numpy()
    print(f"Decoder weight shape: {weights.shape}")

    # The decoder's gene_names are used as-is. The research tree also supported
    # rewriting them -- Ensembl IDs to symbols via a mapping file, then filtering
    # to a gene panel -- both read from absolute paths on one cluster. Neither is
    # carried here: the vocabulary a checkpoint was trained on is the vocabulary
    # its decoder weights are indexed by, and rewriting it at benchmark time only
    # creates a way for the two to disagree.
    print("Extracting decoder map...")
    gene_vectors, sim_matrix, final_names = _extract_decoder_map(
        weights, list(gene_names), n_components=n_components,
    )
    print(f"Gene vectors shape: {gene_vectors.shape}")

    bmdb_dir = _get_bmdb_dir()
    if not os.path.isdir(bmdb_dir):
        print(f"BMDB directory not found: {bmdb_dir}. Skipping benchmark.")
        return {"results_df": pd.DataFrame(), "mean_recalls": {}, "n_genes_benchmarked": 0}

    print("Running benchmark evaluation (percentile-rank recall)...")
    results_df = _run_bmdb_benchmark(sim_matrix, final_names, benchmark_dir=bmdb_dir)

    mean_recalls = {}
    if not results_df.empty:
        recall_cols_all = [c for c in results_df.columns if c.startswith("recall_")]
        # Cross-source mean EXCLUDES the "All" union source: it is a superset of the
        # other databases, so averaging it in would double-count and would also break
        # comparability with historical 5-source runs. "All" is reported separately.
        if "source" in results_df.columns:
            indep = results_df[results_df["source"] != "All"]
            all_row = results_df[results_df["source"] == "All"]
        else:
            indep, all_row = results_df, results_df.iloc[0:0]
        for col in recall_cols_all:
            if not indep.empty:
                mean_recalls[f"bmdb_{col}"] = float(indep[col].mean())
        # Per-"All" union recalls as their own metrics (bmdb_All_recall_*)
        if not all_row.empty:
            for col in recall_cols_all:
                mean_recalls[f"bmdb_All_{col}"] = float(all_row.iloc[0][col])
        print("\nBenchmark Results:")
        recall_cols = [c for c in results_df.columns if c.startswith("recall_")]
        meta_cols = ["source", "valid_pairs"]
        display_cols = [c for c in meta_cols if c in results_df.columns] + recall_cols
        print(results_df[display_cols].to_string(index=False))

    return {
        "results_df": results_df,
        "mean_recalls": mean_recalls,
        "n_genes_benchmarked": len(final_names),
    }


# ---- Decoder Trainer ----
