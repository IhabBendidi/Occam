"""Reusable cell-eval scoring for STATE.

Lifted out of ``_cli/_tx/_predict.py``, where the same ~45 lines appeared TWICE -- once in the
pseudobulk branch and once in the per-cell branch -- inside a 950-line CLI function. Training-time
validation scoring needs the same logic, and a third copy would guarantee drift.

Deliberately in ``state.tx.eval`` rather than in the CLI module: a Lightning callback importing
``_predict`` would pull in the whole CLI import block at module load.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Metrics averaged into the checkpoint-selection score. Must stay in lockstep with
# the upstream tree.eval.evaluation.SELECTION_SCORE_COMPONENTS and PerturbDiff's copy -- changing the
# components changes what "best checkpoint" means and silently makes arms non-comparable.
SELECTION_SCORE_COMPONENTS: tuple[str, ...] = ("pearson_delta", "discrimination_score_l1")
SELECTION_SCORE_NAME = "selection_score"

# STATE's convention, matching what _predict.py passed inline.
PDEX_KWARGS: dict[str, Any] = {"exp_post_agg": True, "is_log1p": True}
SKIP_METRICS: list[str] = ["pearson_edistance", "clustering_agreement"]


def effective_cpu_count() -> int:
    """Cgroup-aware CPU count.

    cell-eval's ``num_threads=-1`` resolves via ``mp.cpu_count()``, which ignores SLURM/Docker
    cgroup limits; ``numba.set_num_threads`` then raises when that exceeds the allocation.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def inject_real_controls(adata_pred, adata_real, pert_col: str, control_pert: str) -> int:
    """Replace predicted control rows in ``adata_pred`` with the real measured values.

    cell-eval computes each perturbation's differential expression against the control cells found
    *inside* each adata. STATE predicts control rows like any other row, so by default the
    predicted baseline is itself a prediction -- the "reconstruct controls" convention. This makes
    the baseline real and identical on both sides while perturbation rows stay predicted, i.e. the
    "inject controls" convention.

    Requires the two adatas to be row-aligned. Returns the number of control rows injected.
    """
    import numpy as np
    import scipy.sparse as sp

    if adata_pred.n_obs != adata_real.n_obs:
        raise ValueError(
            f"row counts differ: pred {adata_pred.n_obs} vs real {adata_real.n_obs}; "
            "control injection requires row-aligned adatas."
        )

    mask = adata_pred.obs[pert_col].astype(str).values == str(control_pert)
    n = int(mask.sum())
    if n == 0:
        raise ValueError(
            f"no control rows (obs[{pert_col!r}] == {control_pert!r}) to inject. cell-eval needs "
            "controls inside the adata to compute differential expression, so this would silently "
            "score against nothing."
        )

    if sp.issparse(adata_pred.X) or sp.issparse(adata_real.X):
        pred_dense = adata_pred.X.toarray() if sp.issparse(adata_pred.X) else np.asarray(adata_pred.X)
        real_dense = adata_real.X.toarray() if sp.issparse(adata_real.X) else np.asarray(adata_real.X)
        pred_dense[mask] = real_dense[mask]
        adata_pred.X = pred_dense
    else:
        adata_pred.X[mask] = adata_real.X[mask]

    for key in list(adata_pred.obsm.keys()):
        if key in adata_real.obsm:
            obsm_p = np.asarray(adata_pred.obsm[key])
            obsm_p[mask] = np.asarray(adata_real.obsm[key])[mask]
            adata_pred.obsm[key] = obsm_p

    return n


def score_celleval_pair(
    adata_pred,
    adata_real,
    *,
    pert_col: str,
    cell_type_key: str,
    control_pert: str,
    embed_key: str | None = None,
    outdir: str | Path | None = None,
    profile: str = "full",
    write_csv: bool = True,
    num_threads: int | None = None,
) -> dict[str, float]:
    """Score one (pred, real) pair with cell-eval, per cell type, and return the mean metrics.

    ``write_csv=False`` keeps everything in memory -- worth using for training-time validation,
    where the per-perturbation and DE CSVs are ~120 MB per call and would be written once per
    validation pass.

    Returns ``{metric: value}`` averaged over cell types, matching
    ``merge_celltype_celleval_metrics`` on the the upstream benchmark side.
    """
    import numpy as np
    from cell_eval import MetricsEvaluator
    from cell_eval.utils import split_anndata_on_celltype

    if pert_col not in adata_pred.obs:
        raise KeyError(f"pert_col {pert_col!r} not in obs (have {list(adata_pred.obs.columns)})")
    if cell_type_key not in adata_pred.obs:
        raise KeyError(f"cell_type_key {cell_type_key!r} not in obs")

    ct_real = split_anndata_on_celltype(adata=adata_real, celltype_col=cell_type_key)
    ct_pred = split_anndata_on_celltype(adata=adata_pred, celltype_col=cell_type_key)
    if len(ct_real) != len(ct_pred):
        raise AssertionError(
            f"cell type count differs between real and pred: {len(ct_real)} != {len(ct_pred)}"
        )

    # Only pass embed-key-dependent metric configs when the model works in a latent space; with
    # embed_key='X_hvg' (every replogle config) the defaults are correct.
    use_embed = bool(embed_key) and embed_key != "X_hvg"
    metric_configs = (
        {
            "discrimination_score": {"embed_key": embed_key},
            "pearson_edistance": {"embed_key": embed_key, "n_jobs": -1},
        }
        if use_embed
        else {}
    )

    threads = effective_cpu_count() if num_threads is None else num_threads
    per_ct: list[dict[str, float]] = []
    for ct in ct_real:
        evaluator = MetricsEvaluator(
            adata_pred=ct_pred[ct],
            adata_real=ct_real[ct],
            control_pert=control_pert,
            pert_col=pert_col,
            outdir=str(outdir) if outdir is not None else "./cell-eval-outdir",
            prefix=str(ct),
            # ⚠️ allow_discrete=True DISABLES cell-eval's own scale check
            # (`guess_is_lognorm`), which raises
            #     Invalid scale: max value 17.29 exceeds log1p threshold of 15.0
            # on any prediction that leaves log1p range. Unconstrained heads do that: STATE's
            # output is a plain `relu` (state_transition.py:516), non-negative but unbounded,
            # and one validation pass hit 17.29 -- so that pass published NO score and the
            # checkpoint could not be selected, for a reason that is about the model's output
            # rather than about the scoring being invalid.
            #
            # Set at the user's explicit instruction (2026-09-18): score high-max checkpoints
            # rather than skipping them. This also MATCHES the reported-number path, which
            # already passes allow_discrete=True (`celleval_v1.run_celleval_v1`) -- so
            # selection and reporting now disable the same check instead of only one of them.
            allow_discrete=True,
            pdex_kwargs=PDEX_KWARGS,
            num_threads=threads,
        )
        _results, agg = evaluator.compute(
            profile=profile,
            metric_configs=metric_configs,
            skip_metrics=SKIP_METRICS,
            write_csv=write_csv,
        )
        # agg is a polars frame: rows are statistics, columns are metrics.
        df = agg.to_pandas().set_index(agg.columns[0])
        if "mean" not in df.index:
            raise ValueError(f"cell-eval aggregate has no 'mean' row (index: {list(df.index)})")
        per_ct.append({
            str(k): float(v) for k, v in df.loc["mean"].items() if np.isfinite(float(v))
        })

    keys = set().union(*(set(d) for d in per_ct))
    return {
        k: float(np.mean([d[k] for d in per_ct if k in d]))
        for k in sorted(keys)
        if any(k in d for d in per_ct)
    }


def read_agg_metrics(eval_dir: Path) -> dict[str, float]:
    """Mean cell-eval metrics from the ``*_agg_results.csv`` files already on disk in ``eval_dir``."""
    import numpy as np
    import pandas as pd

    files = sorted(Path(eval_dir).glob("*_agg_results.csv"))
    if not files:
        raise FileNotFoundError(f"no *_agg_results.csv in {eval_dir}")
    per_file = []
    for f in files:
        df = pd.read_csv(f, index_col=0)
        if "mean" not in df.index:
            raise ValueError(f"{f} has no 'mean' row (index: {list(df.index)})")
        per_file.append({
            str(k): float(v) for k, v in df.loc["mean"].items() if np.isfinite(float(v))
        })
    keys = set().union(*(set(d) for d in per_file))
    return {
        k: float(np.mean([d[k] for d in per_file if k in d]))
        for k in sorted(keys)
        if any(k in d for d in per_file)
    }


def selection_score(metrics: dict[str, float]) -> float | None:
    """Mean of the selection components, or ``None`` if any is missing or non-finite.

    Returning ``None`` rather than a partial mean is deliberate: a score composed of a different
    subset is not comparable across validation passes, and monitoring it would corrupt selection.
    """
    import numpy as np

    vals = [metrics.get(n) for n in SELECTION_SCORE_COMPONENTS]
    if any(v is None or not np.isfinite(v) for v in vals):
        return None
    return float(np.mean(vals))
