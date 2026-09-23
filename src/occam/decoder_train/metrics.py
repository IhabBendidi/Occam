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
    symbol_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract decoder map: PCA on weights → cosine similarity matrix.

    ``symbol_mapping`` translates an Ensembl-ID vocabulary to the gene symbols
    the BMDB databases are keyed by. Rows with no symbol are dropped, and the
    weight matrix is subset with the SAME indices -- the gene axis and the
    weight rows are only meaningful together.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity

    if symbol_mapping:
        from occam.gene_ids import to_symbols

        before = len(gene_names)
        gene_names, keep = to_symbols(gene_names, symbol_mapping)
        weights = weights[keep]
        print(f"  Mapped {len(gene_names)}/{before} Ensembl IDs to gene symbols")

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


EMISSION_MODES = ("raw", "int_libnorm", "int_nolibnorm")


def apply_emission(x: np.ndarray, mode: str, target_sum: float) -> Tuple[np.ndarray, Dict[str, float]]:
    """Re-emit a log1p prediction matrix as integer counts.

    Predictions are continuous in log1p space; a real measurement is a count.
    This stage asks whether emitting them as counts changes what cell-eval
    sees, and separates two things that are easy to conflate:

    ``raw``
        Leave them alone. The baseline.
    ``int_libnorm``
        ``expm1 -> round -> rescale each cell to `target_sum` -> log1p``.
        Integerisation AND library-size alignment.
    ``int_nolibnorm``
        ``expm1 -> round -> log1p``. Integerisation only; rounding loses mass,
        so each cell keeps whatever total survives.

    ``int_libnorm`` vs ``int_nolibnorm`` isolates the library-size term given
    integerisation; ``raw`` vs ``int_nolibnorm`` isolates integerisation given
    no rescaling.

    ⚠️ This is an EMISSION choice applied uniformly to every row of a
    prediction file, injected real controls included. Applying it to the
    perturbed rows only would make the perturbed/control contrast partly an
    artifact of emission, which is exactly what the differential-expression
    test would then report as signal.

    ⚠️ It is lossy by construction at small ``target_sum``. Normalised to 2,000
    over ~8,900 present genes, the mean count per gene is ~0.23, so rounding
    sends most genes to exactly 0. The TARGET is not integer either -- it is
    counts divided by real depth and rescaled -- so this does not make
    predictions more like the target; it makes them more like a measurement.
    Whether that helps is the question, not the assumption.

    Returns the matrix and diagnostics for the record.
    """
    if mode not in EMISSION_MODES:
        raise ValueError(f"unknown emission mode {mode!r}; expected one of {EMISSION_MODES}")
    stats = {
        "zero_frac_before": float((x == 0.0).mean()),
        "mean_rowsum_counts_before": float(np.expm1(np.clip(x, 0, None)).sum(axis=1).mean()),
    }
    if mode == "raw":
        stats.update({"zero_frac_after": stats["zero_frac_before"],
                      "mean_rowsum_counts_after": stats["mean_rowsum_counts_before"],
                      "empty_rows": 0.0})
        return x, stats

    counts = np.rint(np.expm1(np.clip(x, 0, None))).astype(np.float32)
    empty = int((counts.sum(axis=1) == 0).sum())
    if mode == "int_libnorm":
        totals = counts.sum(axis=1, keepdims=True)
        # An all-zero row cannot be rescaled to anything; it stays all-zero and
        # is counted, rather than being turned into NaN or into a silent 1/N.
        np.divide(counts, np.maximum(totals, 1.0), out=counts)
        counts *= target_sum
    out = np.log1p(counts, out=counts)
    stats.update({
        "zero_frac_after": float((out == 0.0).mean()),
        "mean_rowsum_counts_after": float(np.expm1(out).sum(axis=1).mean()),
        "empty_rows": float(empty),
    })
    return out, stats


def sha256_file(path: str, chunk: int = 64 * 1024 * 1024) -> str:
    """Full sha256 of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def real_de_cache_path(cache_dir: str, target_path: str) -> Tuple[str, str]:
    """Where this target's real-side DE lives, keyed by the target's OWN hash.

    Returns ``(path, digest)``. Keying on the content hash rather than on a
    panel name is what makes reuse safe: a target built from different data,
    different transforms or a different gene subset hashes differently and
    simply misses the cache. There is no way to silently score arm B against
    arm A's ground truth.
    """
    digest = sha256_file(target_path)
    return os.path.join(cache_dir, f"real_de_{digest[:16]}.csv"), digest


def run_cell_eval(
    pred_path: str,
    gt_path: str,
    output_dir: str,
    num_threads: int = 16,
    profile: str = "full",
    timeout: int = 3600,
    de_batch_size: Optional[int] = None,
    de_real: Optional[str] = None,
    allow_discrete: bool = False,
) -> Dict[str, float]:
    """Run ARC Institute cell-eval via subprocess (jupyter pyenv); parse agg_results.csv.

    Returns {metric_name: mean_value} or {} on any failure (non-fatal). Mirrors the
    helper in virtual_cell/pipeline.py. The cell-eval CLI auto-detects log1p input.

    ``de_batch_size`` is cell-eval's ``--batch-size``: how many
    (perturbation, gene) pairs one differential-expression task covers. It
    matters more than it looks. cell-eval's default is 100, and the work is
    perturbations x genes, so a panel with 2,394 perturbations over 8,873
    genes becomes ~212,000 pool tasks -- each of which re-attaches to the one
    POSIX shared-memory segment holding the expression matrix. On a panel that
    size that run died with ``FileNotFoundError: /psm_...``: the segment was
    gone before a task could attach (deleted by systemd-logind RemoveIPC, see
    the F-infra1 comment below -- not the CPython tracker bug first blamed).
    Raising the batch size cuts the task count proportionally. Left at None,
    cell-eval's own default applies.
    """
    os.makedirs(output_dir, exist_ok=True)
    # ⛔ F-infra1, ROOT CAUSE (2026-09-23). The `FileNotFoundError: '/psm_<hex>'` is not a
    # CPython resource-tracker race: pdex's pool workers are forked and share the parent's
    # tracker, so that bug cannot fire inside one cell-eval. The segment is deleted from
    # OUTSIDE. systemd-logind on the GPU nodes runs with RemoveIPC=yes (live value, although
    # logind.conf does not mention it), and when a user's last logind session on a node ends
    # it deletes every /dev/shm object that user owns. Slurm jobs are not logind sessions, but
    # any `ssh <node>` is -- so one monitoring ssh wipes the live pdex segment of every
    # cell-eval we have running on that node, ~10 s after it logs out. Reproduced on hop01: a
    # /dev/shm/psm_* file vanished after `ssh hop01 true`. More jobs per node = more victims
    # per wipe, which is the concurrency dependence; retries only re-roll the timing.
    #
    # Fix: scripts/cell_eval_shmsafe.py runs the same cell-eval CLI with pdex's SharedMemory
    # swapped for an anonymous MAP_SHARED mmap inherited by the forked workers -- no name in
    # /dev/shm, so nothing can unlink it. Same bytes in, same DE out (verified identical).
    # OCCAM_CELLEVAL_SHMSAFE=0 restores the stock `cell-eval` executable.
    if os.environ.get("OCCAM_CELLEVAL_SHMSAFE", "1") != "0":
        launcher = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "..", "scripts", "cell_eval_shmsafe.py")
        launcher = os.path.normpath(launcher)
        # The launcher must run under the SAME interpreter as the `cell-eval` executable (the
        # jupyter env, which has cell_eval + pdex). A bare "python" inside a Slurm job resolves
        # to the system interpreter and fails with `No module named 'pdex'` (2026-09-23).
        import shutil as _sh
        _ce = _sh.which("cell-eval")
        _py = None
        if _ce:
            try:
                _real = os.path.realpath(_ce)
                with open(_real, "rb") as _fh:
                    _first = _fh.readline().decode(errors="ignore").strip()
                if _first.startswith("#!") and "python" in _first:
                    _py = _first[2:].split()[0]
                else:  # pyenv shim: ask pyenv for the real executable
                    import subprocess as _sp
                    _real = _sp.run(["pyenv", "which", "cell-eval"], capture_output=True,
                                    text=True, env={**os.environ, "PYENV_VERSION": "jupyter"}
                                    ).stdout.strip()
                    with open(_real, "rb") as _fh:
                        _first = _fh.readline().decode(errors="ignore").strip()
                    _py = _first[2:].split()[0] if _first.startswith("#!") else None
            except Exception:  # noqa: BLE001
                _py = None
        exe = [_py, launcher] if (_py and os.path.isfile(launcher)) else ["cell-eval"]
        if exe == ["cell-eval"]:
            print(f"  ⚠️ {launcher} not found; running stock cell-eval (shm-unsafe)")
    else:
        exe = ["cell-eval"]
    cmd = exe + [
        "run",
        "-ap", pred_path,
        "-ar", gt_path,
        "-o", output_dir,
        "--num-threads", str(num_threads),
        "--profile", profile,
    ]
    if de_batch_size:
        cmd += ["--batch-size", str(de_batch_size)]
    if allow_discrete:
        # Without this, cell-eval decides whether to normalize_total + log1p the input by
        # inspecting 100 randomly chosen cells with an UNSEEDED np.random.choice. Our input
        # is already normalised and log1p'd, so the guess should come out right -- but it is
        # a stochastic gate on a destructive transform, and sc.pp.normalize_total with no
        # target_sum normalises to each FILE's own median, so a false trigger would rescale
        # the real and predicted files differently and corrupt every delta. Passing the flag
        # makes "do not transform" deterministic.
        cmd += ["--allow-discrete"]
    if de_real:
        # The real-side differential expression depends only on the target file,
        # never on the decoder, so it is identical for every arm scored against
        # the same target -- and it is the dominant cost. The caller is
        # responsible for proving the target is the same one; see
        # `real_de_cache_path`.
        cmd += ["--de-real", de_real]
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

    # The BMDB databases are keyed by gene SYMBOLS (CORUM.txt starts
    # `SUB1,GTF3C3`) while the released vocabulary is 44,560 Ensembl IDs, so the
    # two have to be translated for this benchmark. Measured on the released
    # checkpoint: CORUM's 3,505 symbols are covered by 4 raw-vocabulary entries
    # and by 3,500 mapped ones. Without the translation the benchmark reported
    # `valid_pairs: 1`.
    #
    # Which direction to translate is decided from the data -- how many names
    # match the map's Ensembl keys versus its symbol values -- so a checkpoint
    # trained on symbols (e.g. an HVG panel) is used directly. The translation
    # is confined to this function: `_extract_decoder_map` subsets the weight
    # rows with the same indices, and no other code path sees a rewritten
    # vocabulary.
    from occam.gene_ids import detect_id_type, load_ensembl_to_symbol

    symbol_mapping = None
    mapping = load_ensembl_to_symbol()
    if not mapping:
        print("Gene-ID map not found (assets/gene_ids/ensembl_to_symbol.tsv); "
              "using the vocabulary as-is. If it holds Ensembl IDs, BMDB will "
              "match almost nothing -- rebuild it with "
              "scripts/build_gene_id_map.py.")
    else:
        kind, ens_hits, sym_hits = detect_id_type(gene_names, mapping)
        n = max(len(gene_names), 1)
        if kind == "ensembl":
            print(f"  Vocabulary is Ensembl IDs ({ens_hits}/{n} match Ensembl "
                  f"keys vs {sym_hits}/{n} symbols) — translating for BMDB")
            symbol_mapping = mapping
        else:
            print(f"  Vocabulary is already gene symbols ({sym_hits}/{n} match "
                  f"symbols vs {ens_hits}/{n} Ensembl) — no translation")

    print("Extracting decoder map...")
    gene_vectors, sim_matrix, final_names = _extract_decoder_map(
        weights, list(gene_names), n_components=n_components,
        symbol_mapping=symbol_mapping,
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
