"""PCA fit on train / project other splits — CellFlow ``X_pca`` contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np


def fit_and_project_pca(
    adata_train: ad.AnnData,
    other_adatas: dict[str, ad.AnnData] | None = None,
    *,
    n_comps: int = 50,
    keep_centered_data: bool = False,
) -> dict[str, Any]:
    """Fit CellFlow ``centered_pca`` on train; project other splits in place.

    Returns
    -------
    dict
        ``{"n_comps": int, "means": ndarray, "pcs": ndarray}`` for persistence.
    """
    from cellflow.preprocessing import centered_pca, project_pca

    n_genes = adata_train.n_vars
    n_comps_eff = min(n_comps, n_genes, max(1, adata_train.n_obs - 1))
    centered_pca(
        adata_train,
        n_comps=n_comps_eff,
        keep_centered_data=keep_centered_data,
        copy=False,
    )
    means = np.asarray(adata_train.varm["X_mean"], dtype=np.float32)
    pcs = np.asarray(adata_train.varm["PCs"], dtype=np.float32)

    for name, adata in (other_adatas or {}).items():
        if adata.n_vars != n_genes:
            raise ValueError(f"{name}: n_vars {adata.n_vars} != train {n_genes}")
        project_pca(adata, ref_adata=adata_train, copy=False)

    return {"n_comps": int(n_comps_eff), "means": means, "pcs": pcs}


def save_pca_artifacts(artifacts: dict[str, Any], out_dir: Path | str) -> Path:
    """Write ``pca_means.npy`` / ``pca_pcs.npy`` / ``pca_meta.npz`` under ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "pca_means.npy", np.asarray(artifacts["means"]))
    np.save(out_dir / "pca_pcs.npy", np.asarray(artifacts["pcs"]))
    np.savez(out_dir / "pca_meta.npz", n_comps=int(artifacts["n_comps"]))
    return out_dir


def load_pca_artifacts(out_dir: Path | str) -> dict[str, Any]:
    """Load artifacts written by :func:`save_pca_artifacts`."""
    out_dir = Path(out_dir)
    means = np.load(out_dir / "pca_means.npy")
    pcs = np.load(out_dir / "pca_pcs.npy")
    meta = np.load(out_dir / "pca_meta.npz")
    return {"n_comps": int(meta["n_comps"]), "means": means, "pcs": pcs}


def reconstruct_from_pca(
    x_pca: np.ndarray,
    *,
    means: np.ndarray,
    pcs: np.ndarray,
    clip_min: float | None = 0.0,
    clip_max: float | None = None,
) -> np.ndarray:
    """Map PCA coordinates back to gene space: ``X @ PCs.T + means``.

    Matches :func:`cellflow.preprocessing.reconstruct_pca` / the Metrics callback.
    """
    x_pca = np.asarray(x_pca, dtype=np.float32)
    means = np.asarray(means, dtype=np.float32).reshape(-1)
    pcs = np.asarray(pcs, dtype=np.float32)
    x = x_pca @ pcs.T + means
    if clip_min is not None or clip_max is not None:
        np.clip(x, clip_min, clip_max, out=x)
    return x.astype(np.float32, copy=False)


def write_scorer_pca_state(artifacts: dict, path: Path | str) -> Path:
    """Write CellFlow's PCA basis in the npz layout the shared scorer's decode expects.

    ``hooke_tx/eval/celleval_artifacts.py --reconstruct-pred-controls`` loads
    ``mean / components / comp_mean / comp_std`` (``_pca_state_from_npz``) and reconstructs with
    ``z = (X - mean) @ components.T`` then ``z @ components + mean``
    (``hooke_tx/eval/adapters/celleval.pca_reconstruct``). So its ``components`` is
    ``(n_comps, n_genes)`` -- the TRANSPOSE of CellFlow's ``varm["PCs"]``, which this module
    stores as ``(n_genes, n_comps)``.

    ``comp_mean`` / ``comp_std`` are required keys but are only read when ``standardize`` is true.
    CellFlow's ``centered_pca`` centers and does not standardize, so they are written as the
    identity (0 / 1) and ``standardize=False``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcs = np.asarray(artifacts["pcs"], dtype=np.float32)      # (n_genes, n_comps)
    mean = np.asarray(artifacts["means"], dtype=np.float32).reshape(-1)
    n_comps = pcs.shape[1]
    if pcs.shape[0] != mean.shape[0]:
        raise ValueError(f"PCs cover {pcs.shape[0]} genes but mean has {mean.shape[0]}")
    np.savez(
        path,
        mean=mean,
        components=pcs.T.astype(np.float32),                  # (n_comps, n_genes)
        comp_mean=np.zeros(n_comps, dtype=np.float32),
        comp_std=np.ones(n_comps, dtype=np.float32),
        standardize=False,
    )
    return path
