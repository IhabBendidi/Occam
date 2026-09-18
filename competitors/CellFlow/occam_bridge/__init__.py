"""Upstream-Tx dump → CellFlow AnnData bridge.

Reads a canonical Upstream training directory (``obs.parquet`` / ``var.parquet`` /
``X.zarr`` or ``X.npy``), applies optional library-size + log1p transforms, maps
obs into CellFlow's control / gene-covariate schema, and fits PCA for
``sample_rep="X_pca"`` training.
"""

from __future__ import annotations

from .load import (
    control_row_indices,
    dataset_pert_tokens,
    load_dump_anndata,
    load_dump_sources,
    read_obs_columns,
    resolve_gene_list,
    write_synthetic_dump,
)
from .pca import (
    fit_and_project_pca,
    load_pca_artifacts,
    reconstruct_from_pca,
    save_pca_artifacts,
    write_scorer_pca_state,
)
from .schema import apply_cellflow_gene_schema
from .splits import load_fold
from .transforms import apply_normalize_log1p

__all__ = [
    "apply_cellflow_gene_schema",
    "apply_normalize_log1p",
    "control_row_indices",
    "dataset_pert_tokens",
    "fit_and_project_pca",
    "load_fold",
    "load_dump_anndata",
    "load_dump_sources",
    "load_pca_artifacts",
    "read_obs_columns",
    "reconstruct_from_pca",
    "resolve_gene_list",
    "save_pca_artifacts",
    "write_scorer_pca_state",
    "write_synthetic_dump",
]
