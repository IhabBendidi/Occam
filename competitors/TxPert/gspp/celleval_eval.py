"""cell-eval scoring, matched to STATE's `tx predict`.

TxPert natively reports its own ``pearson_delta`` over a GEARS-style test loader, which is not
comparable to the numbers we have for the other arms. This module runs the *same* cell-eval
evaluation STATE runs, so every competitor produces directly comparable CSVs.

Copied verbatim from ``../PerturbDiff/src/apps/sampling/sampling_eval_celleval.py``, which is
itself a deliberate port of ``../STATE/state/src/state/_cli/_tx/_predict.py:896-946``. Every
argument that affects a metric is matched: the ``[0, 14]`` clip applied before scoring, the
per-cell-type split, ``pdex_kwargs={"exp_post_agg": True, "is_log1p": True}``, the cgroup-aware
thread count, and the skipped metrics. Changing any of them here silently breaks comparability,
so keep this file in sync with STATE's.

(``exp_post_agg`` is in fact inert -- no such argument exists in pdex 0.2.2 or 0.3.0, so it is
accepted and ignored -- and ``is_log1p=True`` is already cell-eval's default when
``allow_discrete`` is false. Both are passed anyway to stay literally identical to STATE.)

Outputs land in ``<output_dir>/<prefix>_*.csv``, mirroring
``../STATE/state/test_replogle_zarr/hepg2_holdout_ram/eval_last.ckpt/``::

    HepG2_results.csv
    HepG2_pred_de.csv
    HepG2_real_de.csv
    HepG2_inject_ctrl_agg_results.csv
    HepG2_rcstrct_ctrl_agg_results.csv
"""

import logging
import os
from pathlib import Path

import numpy as np
from scipy import sparse as sp

logger = logging.getLogger(__name__)

# Matches STATE: expression is log1p-scaled, and cell-eval's scale checks reject
# values outside this range.
CLIP_MIN, CLIP_MAX = 0.0, 14.0
SKIP_METRICS = ["pearson_edistance", "clustering_agreement"]
PDEX_KWARGS = dict(exp_post_agg=True, is_log1p=True)

# Control label written into the scored AnnData. Matches ../PerturbDiff's
# `control_pert: empty` and hooke-tx's HOOKE_CONTROL_PERT, so all arms name controls
# identically. gspp's internal sentinel is "ctrl"; the translation happens at export.
CONTROL_OUT = "empty"


def effective_cpu_count() -> int:
    """cgroup-aware CPU count (port of ``_predict.py:5-16``).

    cell-eval's ``num_threads=-1`` resolves through ``mp.cpu_count()``, which
    ignores SLURM/Docker cgroup limits and returns the host count;
    ``numba.set_num_threads(N)`` then raises when N exceeds the allocation.
    ``os.sched_getaffinity`` returns the allowed set on Linux.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def clip_anndata_values(adata, max_value: float = CLIP_MAX, min_value: float = CLIP_MIN) -> None:
    """Clip ``adata.X`` in place, keeping sparsity intact (port of _predict.py:174-183)."""
    if sp.issparse(adata.X):
        if adata.X.data.size:
            np.clip(adata.X.data, min_value, max_value, out=adata.X.data)
            if hasattr(adata.X, "eliminate_zeros"):
                adata.X.eliminate_zeros()
    else:
        np.clip(adata.X, min_value, max_value, out=adata.X)


def run_cell_eval(
    pred_adata,
    true_adata,
    *,
    output_dir,
    pert_col: str,
    cell_type_key: str,
    control_pert: str,
    profile: str = "full",
    logger=logger,
):
    """Score predictions with cell-eval, one evaluation per cell type.

    :param pred_adata: predictions, control rows carrying *real* expression
    :param true_adata: ground truth, row-aligned with ``pred_adata``
    :param output_dir: directory for the metric CSVs
    :param pert_col: obs column holding the perturbation (STATE: ``gene``)
    :param cell_type_key: obs column holding the cell type (STATE: ``cell_type``)
    :param control_pert: control label in ``pert_col`` (zarr: ``empty``)
    :param profile: cell-eval profile -- ``full`` / ``minimal`` / ``de`` / ``anndata``
    """
    from cell_eval import MetricsEvaluator
    from cell_eval.utils import split_anndata_on_celltype

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if pert_col not in pred_adata.obs:
        raise KeyError(
            f"pert_col {pert_col!r} not in obs (have {list(pred_adata.obs.columns)})"
        )
    if cell_type_key not in pred_adata.obs:
        raise KeyError(
            f"cell_type_key {cell_type_key!r} not in obs (have {list(pred_adata.obs.columns)})"
        )
    n_control = int((pred_adata.obs[pert_col].astype(str) == str(control_pert)).sum())
    if n_control == 0:
        raise ValueError(
            f"no control rows (obs[{pert_col!r}] == {control_pert!r}) in the adata; "
            f"cell-eval computes differential expression against them."
        )

    clip_anndata_values(pred_adata)
    clip_anndata_values(true_adata)
    logger.info(
        "cell-eval: clipped X to [%.1f, %.1f]; %d cells, %d control",
        CLIP_MIN, CLIP_MAX, pred_adata.n_obs, n_control,
    )

    ct_real = split_anndata_on_celltype(adata=true_adata, celltype_col=cell_type_key)
    ct_pred = split_anndata_on_celltype(adata=pred_adata, celltype_col=cell_type_key)
    if len(ct_real) != len(ct_pred):
        raise AssertionError(
            f"cell type count differs between real and pred: {len(ct_real)} != {len(ct_pred)}"
        )

    n_threads = effective_cpu_count()
    for cell_type in ct_real:
        logger.info("cell-eval: scoring %s (%d threads) ...", cell_type, n_threads)
        evaluator = MetricsEvaluator(
            adata_pred=ct_pred[cell_type],
            adata_real=ct_real[cell_type],
            control_pert=control_pert,
            pert_col=pert_col,
            outdir=str(output_dir),
            prefix=str(cell_type),
            pdex_kwargs=PDEX_KWARGS,
            num_threads=n_threads,
        )
        evaluator.compute(profile=profile, skip_metrics=SKIP_METRICS)

    written = sorted(p.name for p in output_dir.glob("*.csv"))
    logger.info("cell-eval wrote %d file(s) to %s: %s", len(written), output_dir, written)
    return output_dir


__all__ = ["run_cell_eval", "clip_anndata_values", "effective_cpu_count"]
