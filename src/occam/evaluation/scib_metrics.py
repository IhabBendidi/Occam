"""scib-based cell-type benchmarks for Occam embeddings.

Two evaluation tiers:
  - Fast (periodic): ASW label, NMI, ARI, ASW batch on pbmc_12k only (~10-15 sec)
  - Full (end-of-training): All 11 scib metrics on 3 datasets (~5-10 min)

Requires: pip install scib
"""

import gc
import os
import sys
import time as _time
import traceback
import warnings

import numpy as np
import psutil as _psutil
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from occam.data.datasets import OccamAdataDataset
from occam.transforms import NormalizePerCell, Log1p

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Dataset locations come from the path registry (configs/paths.yaml). Resolved
# lazily so importing this module does not require a registry.


def _scib_datasets():
    """[(name, path, cell_type_col, batch_col)] for the three scIB datasets."""
    from occam import paths
    return paths.scib_datasets()


def _scib_fast_dataset():
    """The single small dataset used for the fast in-training scIB check."""
    from occam import paths
    return paths.scib_datasets(["pbmc_12k"])[0]

EMBED_KEY = "X_occam"

BIO_WEIGHT = 0.6
BATCH_WEIGHT = 0.4

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_gene_col(adata, gene_names):
    """Auto-detect gene column with best overlap against training vocabulary."""
    train_upper = set(str(g).strip().upper() for g in gene_names)
    candidates = [
        "gene_name", "gene_symbol", "gene_symbols", "symbol",
        "feature_name", "feature_names", "Gene",
        "ensembl_gene_id", "gene_id",
    ]
    best_col, best_overlap = None, -1
    for col in candidates:
        if col not in adata.var.columns:
            continue
        eval_names = set(str(x).strip().upper() for x in adata.var[col].tolist())
        overlap = len(train_upper & eval_names)
        if overlap > best_overlap:
            best_overlap = overlap
            best_col = col

    # Also check var index
    idx_names = set(str(x).strip().upper() for x in adata.var_names.tolist())
    idx_overlap = len(train_upper & idx_names)
    if idx_overlap > best_overlap:
        best_col = "index"
        best_overlap = idx_overlap

    if best_overlap <= 0:
        return None
    return best_col


def _generate_embeddings(trainer, adata, dataset_path, gene_col):
    """Generate Occam embeddings for an AnnData dataset.

    Returns numpy array of shape (n_cells, hidden_dim).
    """
    _cfg = trainer.config if hasattr(trainer, 'config') and trainer.config else None
    _skip_norm = getattr(_cfg, 'skip_normalize', False) if _cfg else False
    if _skip_norm:
        transforms = [Log1p()]
    elif trainer._transforms is not None:
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

    batch_size = getattr(trainer.config, 'eval_batch_size', None) or 2048
    loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False,
                        pin_memory=False)

    base_model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    was_training = trainer.model.training
    trainer.model.eval()

    amp_enabled = getattr(trainer, 'amp_enabled', False)
    _gpm = getattr(trainer.config, 'gene_presence_mask', False)
    _pshift = getattr(trainer.config, 'presence_shift', 1.0)

    embeddings_list = []
    n_batches = len(loader)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            bx = batch[0].to(trainer.device)
            if _gpm:
                bx = bx + _pshift
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp_enabled):
                be = base_model.feature(bx)
            embeddings_list.append(be.detach().cpu().numpy())
            if (i + 1) % 50 == 0 or (i + 1) == n_batches:
                print(f"  [scIB Embeddings] {i + 1}/{n_batches} batches ({(i + 1) * 100 // n_batches}%)", flush=True)

    if was_training:
        trainer.model.train()

    result = np.concatenate(embeddings_list, axis=0)
    del eval_ds, loader, embeddings_list
    return result


def _prepare_adata(trainer, dataset_path, label_key, batch_key, shuffle_embeddings=False):
    """Load dataset, generate embeddings, build neighbor graph + Leiden clusters.

    Returns (adata, gene_col) with embeddings in adata.obsm[EMBED_KEY]
    and optimal Leiden clusters in adata.obs['leiden_scib'].
    """
    import scanpy as sc

    _proc = _psutil.Process()
    _t0 = _time.time()
    def _smem():
        return _proc.memory_info().rss / 1024**3

    print(f"  [scib⏱] START | RSS {_smem():.1f}GB")
    adata = sc.read_h5ad(dataset_path)
    gene_col = _get_gene_col(adata, trainer._gene_names)
    print(f"  [scib] Loaded {adata.n_obs} cells x {adata.n_vars} genes (gene_col={gene_col})")

    # scib internals expect categorical label/batch columns
    adata.obs[label_key] = adata.obs[label_key].astype("category")
    adata.obs[batch_key] = adata.obs[batch_key].astype("category")

    print(f"  [scib⏱] read_h5ad: {_time.time()-_t0:.1f}s | RSS {_smem():.1f}GB")

    _te = _time.time()
    embeddings = _generate_embeddings(trainer, adata, dataset_path, gene_col)
    if shuffle_embeddings:
        rng = np.random.RandomState(42)
        perm = rng.permutation(embeddings.shape[0])
        embeddings = embeddings[perm]
        print(f"  [scib] Shuffled {embeddings.shape[0]} embeddings (seed=42)")
    adata.obsm[EMBED_KEY] = embeddings
    print(f"  [scib] Embeddings: {embeddings.shape}")
    print(f"  [scib⏱] embeddings: {_time.time()-_te:.1f}s | RSS {_smem():.1f}GB")

    _tn = _time.time()
    sc.pp.neighbors(adata, use_rep=EMBED_KEY, n_neighbors=15)
    print(f"  [scib⏱] neighbors: {_time.time()-_tn:.1f}s | RSS {_smem():.1f}GB")

    _tl = _time.time()
    import scib
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scib.me.cluster_optimal_resolution(
            adata,
            label_key=label_key,
            cluster_key="leiden_scib",
            use_rep=EMBED_KEY,
            verbose=False,
        )
    print(f"  [scib⏱] leiden_optimal: {_time.time()-_tl:.1f}s | RSS {_smem():.1f}GB")
    print(f"  [scib⏱] TOTAL _prepare_adata: {_time.time()-_t0:.1f}s")

    return adata, gene_col


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _compute_scib_fast(adata, label_key, batch_key):
    """Compute fast scib metrics (ASW label, NMI, ARI, ASW batch).

    Returns dict of metric_name -> float.
    """
    import scib

    results = {}

    _t = _time.time()
    try:
        results["ASW_label"] = float(scib.me.silhouette(adata, label_key, EMBED_KEY))
    except Exception as e:
        print(f"  [scib] ASW_label failed: {e}")
        results["ASW_label"] = float("nan")
    print(f"  [scib⏱] ASW_label: {_time.time()-_t:.1f}s")

    _t = _time.time()
    try:
        results["NMI"] = float(scib.me.nmi(adata, "leiden_scib", label_key))
    except Exception as e:
        print(f"  [scib] NMI failed: {e}")
        results["NMI"] = float("nan")
    print(f"  [scib⏱] NMI: {_time.time()-_t:.1f}s")

    _t = _time.time()
    try:
        results["ARI"] = float(scib.me.ari(adata, "leiden_scib", label_key))
    except Exception as e:
        print(f"  [scib] ARI failed: {e}")
        results["ARI"] = float("nan")
    print(f"  [scib⏱] ARI: {_time.time()-_t:.1f}s")

    _t = _time.time()
    try:
        results["ASW_batch"] = float(scib.me.silhouette_batch(adata, batch_key, label_key, EMBED_KEY, verbose=False))
    except Exception as e:
        print(f"  [scib] ASW_batch failed: {e}")
        results["ASW_batch"] = float("nan")
    print(f"  [scib⏱] ASW_batch: {_time.time()-_t:.1f}s")

    return results


def _compute_scib_full(adata, label_key, batch_key, dataset_path):
    """Compute all 11 scib metrics.

    Returns dict of metric_name -> float.
    """
    import scanpy as sc
    import scib

    results = _compute_scib_fast(adata, label_key, batch_key)

    # --- Additional bio conservation metrics ---

    try:
        results["isolated_label_ASW"] = float(
            scib.me.isolated_labels_asw(adata, label_key, batch_key, EMBED_KEY, verbose=False)
        )
    except Exception as e:
        print(f"  [scib] isolated_label_ASW failed: {e}")
        results["isolated_label_ASW"] = float("nan")

    try:
        results["isolated_label_F1"] = float(
            scib.me.isolated_labels_f1(adata, label_key, batch_key, EMBED_KEY, verbose=False)
        )
    except Exception as e:
        print(f"  [scib] isolated_label_F1 failed: {e}")
        results["isolated_label_F1"] = float("nan")

    try:
        results["cLISI"] = float(
            scib.me.clisi_graph(adata, label_key, type_="embed", use_rep=EMBED_KEY, verbose=False)
        )
    except Exception as e:
        print(f"  [scib] cLISI failed: {e}")
        results["cLISI"] = float("nan")

    # --- Batch correction metrics ---

    try:
        results["graph_connectivity"] = float(scib.me.graph_connectivity(adata, label_key))
    except Exception as e:
        print(f"  [scib] graph_connectivity failed: {e}")
        results["graph_connectivity"] = float("nan")

    try:
        results["iLISI"] = float(
            scib.me.ilisi_graph(adata, batch_key, type_="embed", use_rep=EMBED_KEY, verbose=False)
        )
    except Exception as e:
        print(f"  [scib] iLISI failed: {e}")
        results["iLISI"] = float("nan")

    try:
        # PCR comparison needs a "before" adata with PCA on raw expression
        adata_pre = sc.read_h5ad(dataset_path)
        sc.pp.normalize_total(adata_pre)
        sc.pp.log1p(adata_pre)
        sc.pp.pca(adata_pre)
        results["PCR_batch"] = float(
            scib.me.pcr_comparison(adata_pre, adata, covariate=batch_key, embed=EMBED_KEY, verbose=False)
        )
        del adata_pre
    except Exception as e:
        print(f"  [scib] PCR_batch failed: {e}")
        results["PCR_batch"] = float("nan")

    return results


def _aggregate_scores(results):
    """Compute bio, batch, and overall aggregate scores from full results."""
    bio_keys = ["ASW_label", "NMI", "ARI", "isolated_label_ASW", "isolated_label_F1", "cLISI"]
    batch_keys = ["ASW_batch", "graph_connectivity", "iLISI", "PCR_batch"]

    bio_vals = [results[k] for k in bio_keys if k in results and not np.isnan(results[k])]
    batch_vals = [results[k] for k in batch_keys if k in results and not np.isnan(results[k])]

    bio_score = float(np.mean(bio_vals)) if bio_vals else float("nan")
    batch_score = float(np.mean(batch_vals)) if batch_vals else float("nan")

    if not np.isnan(bio_score) and not np.isnan(batch_score):
        overall = BIO_WEIGHT * bio_score + BATCH_WEIGHT * batch_score
    elif not np.isnan(bio_score):
        overall = bio_score
    elif not np.isnan(batch_score):
        overall = batch_score
    else:
        overall = float("nan")

    return bio_score, batch_score, overall


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_scib_fast(trainer, wandb_module=None):
    """Run fast scib metrics on pbmc_12k (periodic evaluation).

    Returns dict with results, or empty dict on failure.
    """
    try:
        import scib  # noqa: F401
    except ImportError:
        print("[scib] scib package not installed — skipping cell-type benchmark. Install with: pip install scib")
        return {}

    ds_name, ds_path, label_key, batch_key = _scib_fast_dataset()

    if not os.path.exists(ds_path):
        print(f"[scib] Dataset not found: {ds_path} — skipping fast eval")
        return {}

    _proc = _psutil.Process()
    _t_total = _time.time()
    print(f"\n[scib] Fast cell-type eval on {ds_name} | RSS {_proc.memory_info().rss/1024**3:.1f}GB")

    # scib dependencies (leidenalg, igraph, scanpy) mutate global thread
    # settings which causes 100x+ slowdown in subsequent PyTorch DataLoader ops
    _saved_torch_threads = torch.get_num_threads()
    _saved_omp = os.environ.get('OMP_NUM_THREADS')
    _saved_mkl = os.environ.get('MKL_NUM_THREADS')

    adata = None
    try:
        adata, _ = _prepare_adata(trainer, ds_path, label_key, batch_key)
        results = _compute_scib_fast(adata, label_key, batch_key)
    except Exception as e:
        print(f"[scib] Fast eval failed: {e}")
        traceback.print_exc()
        return {}
    finally:
        del adata
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            import ctypes as _ctypes
            _ctypes.CDLL('libc.so.6').malloc_trim(0)
        except Exception:
            pass

        # Restore threading state
        if torch.get_num_threads() != _saved_torch_threads:
            torch.set_num_threads(_saved_torch_threads)
        if _saved_omp is not None:
            os.environ['OMP_NUM_THREADS'] = _saved_omp
        elif 'OMP_NUM_THREADS' in os.environ:
            del os.environ['OMP_NUM_THREADS']
        if _saved_mkl is not None:
            os.environ['MKL_NUM_THREADS'] = _saved_mkl
        elif 'MKL_NUM_THREADS' in os.environ:
            del os.environ['MKL_NUM_THREADS']

    print(f"[scib⏱] TOTAL run_scib_fast: {_time.time()-_t_total:.1f}s | RSS {_proc.memory_info().rss/1024**3:.1f}GB", flush=True)

    # Log to wandb
    if wandb_module is not None:
        log_dict = {}
        for metric_name, value in results.items():
            log_dict[f"scib_fast/{ds_name}/{metric_name}"] = value
        wandb_module.log(log_dict)

    print(f"[scib] Fast results ({ds_name}):")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    return results


def run_scib_full(trainer, wandb_module=None, shuffle_embeddings=False):
    """Run full scib benchmark on all datasets (end-of-training).

    Returns dict with all results, or empty dict on failure.
    """
    try:
        import scib  # noqa: F401
    except ImportError:
        print("[scib] scib package not installed — skipping cell-type benchmark. Install with: pip install scib")
        return {}

    print("\n" + "=" * 60)
    print("[scib] Starting full cell-type benchmark (scib)")
    print("=" * 60)

    # Snapshot threading state (scib deps mutate it)
    _saved_torch_threads = torch.get_num_threads()
    _saved_omp = os.environ.get('OMP_NUM_THREADS')
    _saved_mkl = os.environ.get('MKL_NUM_THREADS')

    all_results = {}
    aggregate_bio, aggregate_batch, aggregate_overall = [], [], []

    for ds_name, ds_path, label_key, batch_key in _scib_datasets():
        if not os.path.exists(ds_path):
            print(f"[scib] Dataset not found: {ds_path} — skipping {ds_name}")
            continue

        print(f"\n[scib] Evaluating {ds_name}...")
        adata = None
        try:
            adata, _ = _prepare_adata(trainer, ds_path, label_key, batch_key,
                                      shuffle_embeddings=shuffle_embeddings)
            results = _compute_scib_full(adata, label_key, batch_key, ds_path)
        except Exception as e:
            print(f"[scib] {ds_name} failed: {e}")
            traceback.print_exc()
            results = {}
        finally:
            del adata
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not results:
            continue

        bio_score, batch_score, overall = _aggregate_scores(results)
        results["bio_score"] = bio_score
        results["batch_score"] = batch_score
        results["overall_score"] = overall
        all_results[ds_name] = results

        if not np.isnan(bio_score):
            aggregate_bio.append(bio_score)
        if not np.isnan(batch_score):
            aggregate_batch.append(batch_score)
        if not np.isnan(overall):
            aggregate_overall.append(overall)

        # Log per-dataset metrics to wandb
        if wandb_module is not None:
            log_dict = {f"scib/{ds_name}/{k}": v for k, v in results.items()}
            wandb_module.log(log_dict)

        print(f"[scib] {ds_name}: bio={bio_score:.4f}  batch={batch_score:.4f}  overall={overall:.4f}")
        for k, v in results.items():
            if k not in ("bio_score", "batch_score", "overall_score"):
                print(f"  {k}: {v:.4f}")

    # Aggregate across datasets
    agg_bio = float(np.mean(aggregate_bio)) if aggregate_bio else float("nan")
    agg_batch = float(np.mean(aggregate_batch)) if aggregate_batch else float("nan")
    agg_overall = float(np.mean(aggregate_overall)) if aggregate_overall else float("nan")

    if wandb_module is not None:
        wandb_module.log({
            "scib/aggregate_bio_score": agg_bio,
            "scib/aggregate_batch_score": agg_batch,
            "scib/aggregate_overall_score": agg_overall,
        })

    print(f"\n[scib] === AGGREGATE: bio={agg_bio:.4f}  batch={agg_batch:.4f}  overall={agg_overall:.4f} ===")
    print("=" * 60)
    print("[scib] Cell-type benchmark completed")
    print("=" * 60 + "\n")

    all_results["aggregate"] = {
        "bio_score": agg_bio,
        "batch_score": agg_batch,
        "overall_score": agg_overall,
    }

    # Restore threading state
    if torch.get_num_threads() != _saved_torch_threads:
        torch.set_num_threads(_saved_torch_threads)
    if _saved_omp is not None:
        os.environ['OMP_NUM_THREADS'] = _saved_omp
    elif 'OMP_NUM_THREADS' in os.environ:
        del os.environ['OMP_NUM_THREADS']
    if _saved_mkl is not None:
        os.environ['MKL_NUM_THREADS'] = _saved_mkl
    elif 'MKL_NUM_THREADS' in os.environ:
        del os.environ['MKL_NUM_THREADS']

    return all_results
