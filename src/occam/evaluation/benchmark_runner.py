"""End-of-training comprehensive benchmark runner.

Runs 6 evaluation tasks on 3 hardcoded datasets at the very end of
training/finetuning, logging all metrics to the active wandb run.
"""

import gc
import math
import os
import sys
import traceback
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import from Occam parent package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from occam.evaluation.perturbation import (
    tvn_on_controls,
    centerscale_on_controls,
    aggregate,
    known_relationship_benchmark,
    pert_signal_consistency_benchmark,
    DEFAULT_RECALL_THR_PAIRS,
)
from occam.data.datasets import OccamAdataDataset
from occam.transforms import NormalizePerCell, Log1p

from .ilisi import TorchILISIMetric
from .knn import WeightedKNNClassifier
from .linear_probing import run_linear_probing
from .reconstruction import run_reconstruction
from .data_split import train_test1_test2_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Dataset locations come from the path registry (configs/paths.yaml), not from
# hardcoded absolute paths. Resolved lazily so that importing this module does
# not require a registry -- only running the benchmark does.


def _benchmark_datasets():
    """[(name, path)] for the three perturbation benchmark datasets."""
    from occam import paths
    return paths.eval_datasets()




PERT_COL = "gene_name"
CONTROL_COL = "is_control"
CONTROL_KEY = True
BATCH_COL = "dataset_batch_num"
BMDB_SOURCES = ["CORUM", "HuMAP", "Reactome", "SIGNOR", "StringDB"]
SEEDS = [42, 43]
SPLIT_SEED = 42  # Data split always uses seed 42 (matching biomodal pre-computed splits)

# Metric names matching the biomodal benchmark notebook
TASK_NAMES = [
    "Gene Relationship Recall",
    "Perturbation Consistency",
    "iLISI",
    "KNN Separability",
    "Linear Separability",
    "Linear Invertibility",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_bmdb_dir():
    """Locate the BMDB database directory.

    Delegates to :func:`occam.paths.bmdb_dir`, which is the single resolver for
    what the research tree implemented four separate times. Returns ``None`` on
    failure so the caller can skip the gene-relationship task rather than abort
    the whole benchmark.
    """
    from occam import paths
    try:
        return paths.bmdb_dir()
    except paths.PathRegistryError:
        return None


def _get_gene_col_for_eval(adata, gene_names):
    """Find the best gene column by overlap with training gene names."""
    train_genes_upper = set(str(g).strip().upper() for g in gene_names)
    candidate_cols = ['gene_name', 'gene_symbol', 'gene_symbols', 'symbol',
                      'feature_name', 'feature_names', 'Gene', 'ensembl_gene_id', 'gene_id']
    best_col, best_overlap = None, -1
    for col in candidate_cols:
        if col not in adata.var.columns:
            continue
        try:
            eval_names = set(str(x).strip().upper() for x in adata.var[col].tolist())
        except Exception:
            continue
        overlap = len(train_genes_upper & eval_names)
        if overlap > best_overlap:
            best_overlap = overlap
            best_col = col
    return best_col if best_overlap > 0 else None


def _generate_embeddings(trainer, eval_ds, batch_size=2048):
    """Run model inference to generate embeddings."""
    device = trainer.device
    base_model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    was_training = trainer.model.training
    trainer.model.eval()

    loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False,
                        pin_memory=torch.cuda.is_available())
    amp_enabled = getattr(trainer, 'amp_enabled', False)
    _gpm = getattr(trainer.config, 'gene_presence_mask', False) if hasattr(trainer, 'config') else False
    _pshift = getattr(trainer.config, 'presence_shift', 1.0) if hasattr(trainer, 'config') else 1.0
    embeddings_list = []
    n_batches = len(loader)
    try:
        with torch.no_grad():
            for i, batch in enumerate(loader):
                bx = batch[0].to(device)
                if _gpm:
                    bx = bx + _pshift
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp_enabled):
                    be = base_model.feature(bx)
                embeddings_list.append(be.detach().cpu().numpy())
                if (i + 1) % 50 == 0 or (i + 1) == n_batches:
                    print(f"  [Embeddings] {i + 1}/{n_batches} batches ({(i + 1) * 100 // n_batches}%)", flush=True)
    finally:
        if was_training:
            trainer.model.train()

    return np.concatenate(embeddings_list, axis=0)


def _downsample_indices(mask, labels, max_samples=100000, seed=42):
    """Downsample indices while maintaining label distribution."""
    rng = np.random.RandomState(seed)
    indices = np.where(mask)[0]
    if len(indices) <= max_samples:
        return indices
    label_vals = labels[indices]
    unique_labels, counts = np.unique(label_vals, return_counts=True)
    proportions = counts / counts.sum()
    samples_per_label = np.maximum(1, np.round(proportions * max_samples).astype(int))
    selected = []
    for label, n_samples in zip(unique_labels, samples_per_label):
        label_indices = indices[label_vals == label]
        n = min(n_samples, len(label_indices))
        selected.append(rng.choice(label_indices, n, replace=False))
    return np.concatenate(selected)


# ---------------------------------------------------------------------------
# Per-task evaluation functions
# ---------------------------------------------------------------------------

def _eval_bmdb(tvn_embeddings, metadata, bmdb_dir):
    """Task 1: Gene Relationship Recall using TVN embeddings."""
    # Include all cells in aggregation (matching interval eval in training_pipeline.py)
    map_data = aggregate(
        tvn_embeddings,
        metadata,
        pert_col=PERT_COL,
        keys_to_remove=[],
    )
    metrics_df = known_relationship_benchmark(
        map_data,
        pert_col=PERT_COL,
        benchmark_sources=BMDB_SOURCES,
        recall_thr_pairs=[(0.05, 0.95)],
        benchmark_data_dir=bmdb_dir,
        min_req_entity_cnt=20,
        log_stats=True,
    )
    if metrics_df is not None and not metrics_df.empty and 'recall_0.05_0.95' in metrics_df.columns:
        gene_rel_recall = float(metrics_df['recall_0.05_0.95'].sum()) / 5.0 * 100.0
    else:
        gene_rel_recall = float('nan')
    return gene_rel_recall


def _eval_bmdb_precision(tvn_embeddings, metadata):
    """Task 2: Perturbation Consistency using TVN embeddings."""
    vc = metadata[PERT_COL].value_counts()
    keys_to_drop = vc[vc == 1].index.tolist()
    neg_ctrl_perts = metadata.loc[metadata[CONTROL_COL] == CONTROL_KEY, PERT_COL].unique().tolist()

    np.random.seed(SPLIT_SEED)  # Always seed 42, matching biomodal hardcoded seed
    cons_res = pert_signal_consistency_benchmark(
        tvn_embeddings,
        metadata,
        pert_col=PERT_COL,
        neg_ctrl_perts=neg_ctrl_perts,
        keys_to_drop=keys_to_drop,
        stat="cossim",
        num_null_cells=min(50000, tvn_embeddings.shape[0]),
    )
    denom = np.sum(~pd.isna(cons_res["pval"]))
    consistency = float(np.sum(cons_res.pval <= 0.01)) / float(denom if denom > 0 else 1) * 100.0
    return consistency


def _eval_ilisi(cs_embeddings, metadata):
    """Task 3: iLISI on center_scaled embeddings (full dataset, per-perturbation)."""
    all_perturbations = metadata[PERT_COL].unique()
    all_batches = metadata[BATCH_COL].unique()
    PERPLEXITY = 3

    ilisi_scores = []
    for pert in all_perturbations:
        pert_mask = (metadata[PERT_COL] == pert).values
        n_cells = pert_mask.sum()
        if n_cells < 56:
            continue

        ilisi_metric = TorchILISIMetric(
            unique_labels=list(all_batches),
            perplexity=PERPLEXITY,
            use_pca=False,
            seed=SPLIT_SEED,
        )

        sub_emb = torch.tensor(cs_embeddings[pert_mask], dtype=torch.float32)
        sub_batches = metadata.loc[pert_mask, BATCH_COL]
        ilisi_metric.update(sub_emb, sub_batches)
        score = ilisi_metric.compute().item()
        ilisi_scores.append(score)

    if len(ilisi_scores) > 0:
        mean_ilisi = np.mean(ilisi_scores)
    else:
        mean_ilisi = 0.0

    # Normalize: divide by perplexity (3), scale to percentage
    normalized_ilisi = mean_ilisi / 3.0 * 100.0
    return normalized_ilisi


def _eval_knn(raw_embeddings, labels, train_mask, test1_mask, seed=42):
    """Task 4: KNN Separability on raw embeddings (matching biomodal evaluation_config.yaml)."""
    train_labels = labels[train_mask]
    test1_labels = labels[test1_mask]
    all_labels = list(set(train_labels) | set(test1_labels))

    # Downsample if needed
    train_indices = np.where(train_mask)[0]
    if len(train_indices) > 100000:
        train_indices = _downsample_indices(train_mask, labels, max_samples=100000, seed=seed)

    test1_indices = np.where(test1_mask)[0]

    train_emb = raw_embeddings[train_indices]
    train_lbl = labels[train_indices]
    test1_emb = raw_embeddings[test1_indices]
    test1_lbl = labels[test1_indices]

    # Matching biomodal evaluation_config.yaml: k_initial=251, distance_fx=euclidean, use_pca=true
    k = 251

    knn = WeightedKNNClassifier(
        unique_labels=all_labels,
        k=k,
        T=0.07,
        distance_fx="euclidean",
        use_pca=True,
    )

    knn.update(
        train_features=torch.tensor(train_emb, dtype=torch.float32),
        train_targets=pd.Series(train_lbl),
    )
    knn.update(
        test_features=torch.tensor(test1_emb, dtype=torch.float32),
        test_targets=pd.Series(test1_lbl),
    )

    top1, top5 = knn.compute()

    del knn
    torch.cuda.empty_cache()

    knn_separability = (top1 + top5) / 2.0
    return knn_separability


def _eval_linear(raw_embeddings, labels, train_mask, test1_mask, device, seed=42):
    """Task 5: Linear Separability on raw embeddings."""
    train_lbl = labels[train_mask]
    test1_lbl = labels[test1_mask]
    unique = list(set(train_lbl) | set(test1_lbl))

    result = run_linear_probing(
        train_embeddings=raw_embeddings[train_mask],
        train_labels=train_lbl,
        test_embeddings=raw_embeddings[test1_mask],
        test_labels=test1_lbl,
        unique_labels=unique,
        num_epochs=250,
        lr=0.001,
        weight_decay=1e-6,
        batch_size=2048,
        warmup_epochs=10,
        seed=seed,
        device=device,
    )

    linear_sep = (result["test_acc1"] + result["test_acc5"]) / 2.0
    return linear_sep


def _eval_reconstruct(raw_embeddings, expression_data, metadata, train_mask, test2_mask, device, seed=42):
    """Task 6: Linear Invertibility (Reconstruction) on raw embeddings."""
    batch_ids_test2 = metadata.loc[test2_mask, BATCH_COL].values
    control_labels_test2 = metadata.loc[test2_mask, CONTROL_COL].values

    result = run_reconstruction(
        train_embeddings=raw_embeddings[train_mask],
        train_expressions=expression_data[train_mask],
        test_embeddings=raw_embeddings[test2_mask],
        test_expressions=expression_data[test2_mask],
        batch_ids_test=batch_ids_test2,
        control_labels_test=control_labels_test2,
        control_label=CONTROL_KEY,
        num_epochs=30,
        batch_size=512,
        lr=0.0001,
        weight_decay=1e-6,
        model_depth=3,
        warmup_epochs=10,
        seed=seed,
        device=device,
    )

    spearman = result.get("Average Spearman Correlation", float('nan'))
    structural_dist = result.get("structural_distance", float('nan'))

    # Scale to percentage (matching notebook)
    linear_inv = (spearman + structural_dist) / 2.0 * 100.0
    return linear_inv


# ---------------------------------------------------------------------------
# Per-dataset benchmark
# ---------------------------------------------------------------------------

def _benchmark_single_dataset(dataset_name, dataset_path, trainer, shuffle_embeddings=False):
    """Run all 6 tasks on a single dataset. Returns dict of metric_name -> value."""
    import scanpy as sc

    print(f"\n[Benchmark] === Processing dataset: {dataset_name} ===")
    if shuffle_embeddings:
        print("[Benchmark] *** SHUFFLE MODE: embeddings will be randomly permuted ***")
    results = {}

    # Load the AnnData for metadata
    adata = sc.read_h5ad(dataset_path)
    metadata = adata.obs.copy().reset_index(drop=True)
    print(f"[Benchmark] Loaded {adata.n_obs} cells x {adata.n_vars} genes")

    # Find gene column and create aligned dataset
    gene_col = _get_gene_col_for_eval(adata, trainer._gene_names)
    _cfg = trainer.config if hasattr(trainer, 'config') and trainer.config else None
    _skip_norm = getattr(_cfg, 'skip_normalize', False) if _cfg else False
    if _skip_norm:
        transforms = [Log1p()]
    elif hasattr(trainer, '_transforms') and trainer._transforms is not None:
        transforms = trainer._transforms
    else:
        transforms = [NormalizePerCell(1e4), Log1p()]
    eval_ds = OccamAdataDataset(
        adata_path=dataset_path,
        mode="in_memory",
        transforms=transforms,
        gene_name_col=gene_col,
        gene_vocabulary=trainer._gene_names,
        compact=True,
    )
    print(f"[Benchmark] Aligned dataset: {len(eval_ds)} cells, {eval_ds.num_genes} genes")

    # Get expression data for reconstruction from the ORIGINAL adata (not zero-padded).
    # This uses only the genes actually present in the dataset file, matching the biomodal benchmark.
    expression_data = adata.X
    if hasattr(expression_data, 'toarray'):
        expression_data = expression_data.toarray()
    expression_data = np.array(expression_data, dtype=np.float32)
    if not _skip_norm:
        row_sums = expression_data.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        target_sum = next((t.target_sum for t in transforms if hasattr(t, 'target_sum')), 1e4)
        expression_data = expression_data / row_sums * target_sum
    expression_data = np.log1p(expression_data)
    # Limit to first 15,000 genes (matching biomodal custom_dataloaders.py:123)
    if expression_data.shape[1] > 15000:
        expression_data = expression_data[:, :15000]
    print(f"[Benchmark] Reconstruction target: {expression_data.shape[0]} cells x {expression_data.shape[1]} genes")

    # Generate embeddings
    print("[Benchmark] Generating embeddings...")
    _eval_bs = getattr(trainer.config, 'eval_batch_size', 2048) if hasattr(trainer, 'config') else 2048
    raw_embeddings = _generate_embeddings(trainer, eval_ds, batch_size=_eval_bs)
    print(f"[Benchmark] Raw embeddings shape: {raw_embeddings.shape}")

    if shuffle_embeddings:
        rng = np.random.RandomState(42)
        perm = rng.permutation(raw_embeddings.shape[0])
        raw_embeddings = raw_embeddings[perm]
        print(f"[Benchmark] Shuffled {raw_embeddings.shape[0]} embeddings (seed=42)")

    # Post-processing
    print("[Benchmark] Computing center-scaled embeddings...")
    cs_embeddings = centerscale_on_controls(
        raw_embeddings, metadata, pert_col=CONTROL_COL, control_key=CONTROL_KEY, batch_col=BATCH_COL,
    )

    print("[Benchmark] Computing TVN embeddings...")
    with warnings.catch_warnings():
        _cw = getattr(np, 'ComplexWarning', DeprecationWarning)
        warnings.filterwarnings("ignore", category=_cw)
        tvn_embeddings = tvn_on_controls(
            raw_embeddings, metadata, pert_col=CONTROL_COL, control_key=CONTROL_KEY, batch_col=BATCH_COL,
        )
    # CORAL step can produce complex values when batch covariance is ill-conditioned;
    # numpy auto-casts to real (discarding imaginary part) which may leave NaN/inf.
    nan_count = int(np.isnan(tvn_embeddings).sum())
    if nan_count > 0:
        print(f"[Benchmark] Warning: TVN produced {nan_count} NaN values, replacing with 0")
        np.nan_to_num(tvn_embeddings, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Task 1: BMDB ---
    bmdb_dir = _find_bmdb_dir()
    if bmdb_dir:
        try:
            print("[Benchmark] Running BMDB (Gene Relationship Recall)...")
            results["Gene Relationship Recall"] = _eval_bmdb(tvn_embeddings, metadata, bmdb_dir)
            print(f"[Benchmark]   Gene Relationship Recall: {results['Gene Relationship Recall']:.4f}")
        except Exception as e:
            print(f"[Benchmark]   BMDB failed: {e}")
            results["Gene Relationship Recall"] = float('nan')
    else:
        print("[Benchmark]   bmdb_database not found, skipping BMDB")
        results["Gene Relationship Recall"] = float('nan')

    # --- Task 2: BMDB Precision ---
    try:
        print("[Benchmark] Running BMDB Precision (Perturbation Consistency)...")
        results["Perturbation Consistency"] = _eval_bmdb_precision(tvn_embeddings, metadata)
        print(f"[Benchmark]   Perturbation Consistency: {results['Perturbation Consistency']:.4f}")
    except Exception as e:
        print(f"[Benchmark]   BMDB Precision failed: {e}")
        results["Perturbation Consistency"] = float('nan')

    # Free TVN embeddings
    del tvn_embeddings

    # --- Task 3: iLISI ---
    try:
        print("[Benchmark] Running iLISI (per-perturbation, perplexity=3)...")
        results["iLISI"] = _eval_ilisi(cs_embeddings, metadata)
        print(f"[Benchmark]   iLISI: {results['iLISI']:.4f}")
    except Exception as e:
        print(f"[Benchmark]   iLISI failed: {e}")
        results["iLISI"] = float('nan')

    # Free center-scaled embeddings (no longer needed after iLISI)
    del cs_embeddings

    # --- Data split (always uses SPLIT_SEED=42, matching biomodal pre-computed splits) ---
    print("[Benchmark] Performing in-memory train/test1/test2 split...")
    train_mask, test1_mask, test2_mask = train_test1_test2_split(
        metadata, batch_column=BATCH_COL, perturb_column=PERT_COL,
        control_column=CONTROL_COL, control_key=CONTROL_KEY,
        train_size=0.7, random_state=SPLIT_SEED,
    )
    labels = metadata[PERT_COL].values
    print(f"[Benchmark] Split: train={train_mask.sum()}, test1={test1_mask.sum()}, test2={test2_mask.sum()}")

    # Precompute control mask for linear probing filtering
    non_control = ~(metadata[CONTROL_COL] == CONTROL_KEY).values
    linear_train_mask = train_mask & non_control
    linear_test1_mask = test1_mask & non_control

    # --- Multi-seed loop for seed-dependent tasks (KNN, Linear, Reconstruction) ---
    device = trainer.device
    seed_knn = []
    seed_linear = []
    seed_recon = []

    for seed in SEEDS:
        print(f"\n[Benchmark] --- Running seed-dependent tasks with seed={seed} ---")

        # --- Task 4: KNN ---
        try:
            print(f"[Benchmark] Running KNN Separability (k=251, euclidean, PCA=32, seed={seed})...")
            knn_val = _eval_knn(raw_embeddings, labels, train_mask, test1_mask, seed=seed)
            print(f"[Benchmark]   KNN Separability (seed={seed}): {knn_val:.4f}")
            seed_knn.append(knn_val)
        except Exception as e:
            print(f"[Benchmark]   KNN failed (seed={seed}): {e}")
            seed_knn.append(float('nan'))

        # --- Task 5: Linear (with controls filtered out) ---
        try:
            print(f"[Benchmark] Running Linear Separability (250 epochs, seed={seed})...")
            lin_val = _eval_linear(raw_embeddings, labels, linear_train_mask, linear_test1_mask, device, seed=seed)
            print(f"[Benchmark]   Linear Separability (seed={seed}): {lin_val:.4f}")
            seed_linear.append(lin_val)
        except Exception as e:
            print(f"[Benchmark]   Linear probing failed (seed={seed}): {e}")
            traceback.print_exc()
            seed_linear.append(float('nan'))

        # --- Task 6: Reconstruction ---
        try:
            print(f"[Benchmark] Running Reconstruction / Linear Invertibility (30 epochs, seed={seed})...")
            recon_val = _eval_reconstruct(
                raw_embeddings, expression_data, metadata, train_mask, test2_mask, device, seed=seed,
            )
            print(f"[Benchmark]   Linear Invertibility (seed={seed}): {recon_val:.4f}")
            seed_recon.append(recon_val)
        except Exception as e:
            print(f"[Benchmark]   Reconstruction failed (seed={seed}): {e}")
            traceback.print_exc()
            seed_recon.append(float('nan'))

    # Average across seeds
    results["KNN Separability"] = float(np.nanmean(seed_knn))
    results["Linear Separability"] = float(np.nanmean(seed_linear))
    results["Linear Invertibility"] = float(np.nanmean(seed_recon))

    print(f"\n[Benchmark] Averaged across {len(SEEDS)} seeds:")
    print(f"[Benchmark]   KNN Separability: {results['KNN Separability']:.4f}")
    print(f"[Benchmark]   Linear Separability: {results['Linear Separability']:.4f}")
    print(f"[Benchmark]   Linear Invertibility: {results['Linear Invertibility']:.4f}")

    # Cleanup
    del raw_embeddings, expression_data, eval_ds, adata
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_end_of_training_benchmark(trainer, wandb_module=None, shuffle_embeddings=False):
    """Run comprehensive benchmark on datasets at end of training/finetuning.

    Args:
        trainer: OccamTrainer instance (must have model, device, _gene_names, _transforms)
        wandb_module: the wandb module for logging, or None
        shuffle_embeddings: if True, randomly permute embeddings to break label correspondence

    Returns:
        dict with all benchmark results, or empty dict on failure
    """
    print("\n" + "=" * 60)
    print("[Benchmark] Starting end-of-training comprehensive benchmark")
    print("=" * 60)

    if trainer._gene_names is None:
        print("[Benchmark] No gene names on trainer; cannot run benchmark.")
        return {}

    datasets = list(_benchmark_datasets())

    all_results = {}
    per_dataset_scores = {}

    for dataset_name, dataset_path in datasets:
        if not os.path.exists(dataset_path):
            print(f"[Benchmark] Dataset not found: {dataset_path} — skipping")
            continue

        try:
            results = _benchmark_single_dataset(dataset_name, dataset_path, trainer,
                                                shuffle_embeddings=shuffle_embeddings)
        except Exception as e:
            print(f"[Benchmark] Dataset {dataset_name} failed entirely: {e}")
            traceback.print_exc()
            results = {name: float('nan') for name in TASK_NAMES}

        all_results[dataset_name] = results

        # Log individual metrics to wandb
        if wandb_module is not None:
            log_dict = {}
            for task_name, value in results.items():
                log_dict[f"benchmark/{dataset_name}/{task_name}"] = value
            wandb_module.log(log_dict)

        # Compute per-dataset aggregate (mean of non-nan task metrics)
        valid_values = [v for v in results.values() if not np.isnan(v)]
        dataset_score = np.mean(valid_values) if valid_values else float('nan')
        per_dataset_scores[dataset_name] = dataset_score

        if wandb_module is not None:
            wandb_module.log({f"benchmark/{dataset_name}/aggregate_score": dataset_score})

        print(f"\n[Benchmark] {dataset_name} aggregate score: {dataset_score:.4f}")
        for task_name, value in results.items():
            print(f"[Benchmark]   {task_name}: {value:.4f}")

    # Overall score (mean across datasets)
    valid_dataset_scores = [v for v in per_dataset_scores.values() if not np.isnan(v)]
    overall_score = np.mean(valid_dataset_scores) if valid_dataset_scores else float('nan')

    if wandb_module is not None:
        wandb_module.log({"benchmark/overall_score": overall_score})

    print(f"\n[Benchmark] === OVERALL SCORE: {overall_score:.4f} ===")
    for ds_name, ds_score in per_dataset_scores.items():
        print(f"[Benchmark]   {ds_name}: {ds_score:.4f}")

    print("=" * 60)
    print("[Benchmark] End-of-training benchmark completed")
    print("=" * 60 + "\n")

    return {
        "per_dataset": all_results,
        "per_dataset_scores": per_dataset_scores,
        "overall_score": overall_score,
    }
