"""Load a Hooke-Tx onboarding dump into AnnData."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import zarr

from .transforms import apply_normalize_log1p

_GENE_COL = "ensembl_gene_id"


def _read_selected_genes(path_or_ids: Path | Sequence[str] | None) -> list[str] | None:
    if path_or_ids is None:
        return None
    if isinstance(path_or_ids, (str, Path)):
        path = Path(path_or_ids)
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return [str(g) for g in path_or_ids]


def resolve_gene_list(
    var_gene_ids: Sequence[str],
    selected_genes: Sequence[str] | None,
    *,
    on_missing: Literal["error", "intersect"] = "error",
) -> list[str]:
    """Return ordered gene IDs to keep, validating against ``var``."""
    if selected_genes is None:
        return [str(g) for g in var_gene_ids]
    present = {str(g) for g in var_gene_ids}
    missing = [g for g in selected_genes if g not in present]
    if missing and on_missing == "error":
        raise ValueError(f"{len(missing)} selected genes missing from var, e.g. {missing[:3]}")
    kept = [g for g in selected_genes if g in present]
    if not kept:
        raise ValueError("no selected genes remain after intersecting with var")
    return kept


def _open_expression(data_dir: Path, feature: Literal["auto", "zarr", "npy"]):
    npy = data_dir / "X.npy"
    zarr_path = data_dir / "X.zarr"
    if feature == "auto":
        if npy.exists():
            feature = "npy"
        elif zarr_path.exists():
            feature = "zarr"
        else:
            raise FileNotFoundError(f"{data_dir}: need X.npy or X.zarr")
    if feature == "npy":
        if not npy.exists():
            raise FileNotFoundError(npy)
        return np.load(npy, mmap_mode="r"), "npy"
    if feature == "zarr":
        if not zarr_path.exists():
            raise FileNotFoundError(zarr_path)
        return zarr.open(str(zarr_path), mode="r"), "zarr"
    raise ValueError(f"Unknown feature={feature!r}")


def _gather_rows(store, row_idx: np.ndarray, col_idx: np.ndarray | None) -> np.ndarray:
    """Materialize a dense float32 block ``(n_rows, n_cols)`` from zarr/npy."""
    if col_idx is None:
        # Fancy indexing on mmap/zarr for arbitrary rows.
        if hasattr(store, "oindex"):
            return np.asarray(store.oindex[row_idx, :], dtype=np.float32)
        return np.asarray(store[row_idx], dtype=np.float32)

    if hasattr(store, "oindex"):
        return np.asarray(store.oindex[row_idx, col_idx], dtype=np.float32)
    # NumPy mmap: advanced indexing copies.
    return np.asarray(store[np.ix_(row_idx, col_idx)], dtype=np.float32)


def load_hooke_anndata(
    data_dir: Path | str,
    *,
    selected_genes: Path | Sequence[str] | None = None,
    feature: Literal["auto", "zarr", "npy"] = "auto",
    max_cells: int | None = None,
    row_indices: np.ndarray | Sequence[int] | None = None,
    target_sum: float | None = None,
    log1p: bool = False,
    on_missing_genes: Literal["error", "intersect"] = "error",
    seed: int = 0,
) -> ad.AnnData:
    """Load ``obs.parquet`` + ``var.parquet`` + ``X.{zarr,npy}`` into AnnData.

    Parameters
    ----------
    data_dir:
        Hooke dump directory (as in ``DATA_SOURCES``).
    selected_genes:
        Optional Ensembl ID list (path or sequence). Columns of ``X`` are reordered
        to this list; missing IDs raise.
    feature:
        Expression backend. ``auto`` prefers ``X.npy`` then ``X.zarr``.
    max_cells / row_indices:
        Optional row subset (for smokes). ``row_indices`` wins if both given.
    target_sum / log1p:
        Optional Hooke-style transforms applied after load.
    """
    data_dir = Path(data_dir)
    obs_path = data_dir / "obs.parquet"
    var_path = data_dir / "var.parquet"
    if not obs_path.exists() or not var_path.exists():
        raise FileNotFoundError(f"{data_dir}: need obs.parquet and var.parquet")

    obs = pd.read_parquet(obs_path)
    var = pd.read_parquet(var_path)
    if _GENE_COL not in var.columns:
        raise KeyError(f"{var_path}: missing {_GENE_COL!r} column")

    store, backend = _open_expression(data_dir, feature)
    n_cells, n_genes = int(store.shape[0]), int(store.shape[1])
    if len(obs) != n_cells:
        raise ValueError(f"obs rows {len(obs)} != X rows {n_cells}")
    if len(var) != n_genes:
        raise ValueError(f"var rows {len(var)} != X cols {n_genes}")

    var_ids = var[_GENE_COL].astype(str).tolist()
    requested = _read_selected_genes(selected_genes)
    try:
        genes = resolve_gene_list(var_ids, requested, on_missing=on_missing_genes)
    except ValueError as exc:
        raise ValueError(f"{data_dir.name}: {exc}") from exc
    if requested is None and genes == var_ids:
        col_idx = None
    else:
        gene_to_pos = {g: i for i, g in enumerate(var_ids)}
        col_idx = np.asarray([gene_to_pos[g] for g in genes], dtype=np.int64)

    if row_indices is not None:
        row_idx = np.asarray(row_indices, dtype=np.int64)
    else:
        row_idx = np.arange(n_cells, dtype=np.int64)
        if max_cells is not None and n_cells > max_cells:
            rng = np.random.default_rng(seed)
            row_idx = np.sort(rng.choice(row_idx, size=max_cells, replace=False))

    x = _gather_rows(store, row_idx, col_idx)
    obs_sub = obs.iloc[row_idx].reset_index(drop=True)
    var_sub = pd.DataFrame(index=pd.Index(genes, name=_GENE_COL))

    adata = ad.AnnData(X=x, obs=obs_sub, var=var_sub)
    adata.uns["hooke_bridge"] = {
        "data_dir": str(data_dir),
        "feature": backend,
        "n_cells_full": n_cells,
        "n_genes_full": n_genes,
    }
    if target_sum is not None or log1p:
        apply_normalize_log1p(adata, target_sum=target_sum, log1p=log1p, inplace=True)
    return adata


def read_obs_columns(data_dir: Path | str, columns: Sequence[str]) -> pd.DataFrame:
    """Read a few ``obs.parquet`` columns without touching the expression store."""
    return pd.read_parquet(Path(data_dir) / "obs.parquet", columns=list(columns))


def dataset_pert_tokens(data_dirs: Sequence[Path | str]) -> list[str]:
    """Every perturbation token present in ``data_dirs``, regardless of split.

    Used to widen the one-hot vocabulary to the whole dataset before training, because the
    vocabulary fixes the model's condition-input width and a perturbation without a column
    cannot be predicted at all (see ``schema.apply_cellflow_gene_schema``'s ``extra_tokens``).

    Reads ``gene_pert`` only, so it costs one parquet column per dump and needs no expression
    store. Tokens are the Ensembl ids exactly as the schema emits them; controls contribute
    nothing, since they are represented by the pad token.
    """
    tokens: set[str] = set()
    for data_dir in data_dirs:
        obs = read_obs_columns(data_dir, ["gene_pert"])
        for value in obs["gene_pert"].tolist():
            for entry in _as_tuple_list_for_tokens(value):
                tokens.add(str(entry))
    return sorted(tokens)


def _as_tuple_list_for_tokens(value) -> list[str]:
    """Ensembl ids out of one ``gene_pert`` cell value, tolerating the stored shapes."""
    if value is None:
        return []
    out: list[str] = []
    for entry in value:
        # Stored as (pert_type, ensembl_id, effect_class) tuples; be tolerant of plain ids.
        if isinstance(entry, (list, tuple, np.ndarray)):
            if len(entry) >= 2:
                out.append(str(entry[1]))
        else:
            out.append(str(entry))
    return out


def control_row_indices(
    data_dir: Path | str,
    *,
    within: Sequence[int] | np.ndarray | None = None,
    max_cells: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Dump-row indices of ``negative_control`` cells, optionally restricted to ``within``."""
    obs = read_obs_columns(data_dir, ["negative_control"])
    candidates = (
        np.arange(len(obs), dtype=np.int64)
        if within is None
        else np.asarray(within, dtype=np.int64)
    )
    mask = obs.iloc[candidates]["negative_control"].astype(bool).to_numpy()
    ctrl_idx = candidates[mask]
    if len(ctrl_idx) == 0:
        raise ValueError(f"no negative_control cells found in {data_dir}")
    if max_cells is not None and len(ctrl_idx) > max_cells:
        rng = np.random.default_rng(seed)
        ctrl_idx = np.sort(rng.choice(ctrl_idx, size=max_cells, replace=False))
    return ctrl_idx


def load_hooke_sources(
    sources: Sequence[dict],
    *,
    selected_genes: Path | Sequence[str] | None = None,
    feature: Literal["auto", "zarr", "npy"] = "auto",
    target_sum: float | None = None,
    log1p: bool = False,
    on_missing_genes: Literal["error", "intersect"] = "error",
    seed: int = 0,
) -> ad.AnnData:
    """Concatenate several Hooke dumps that share a gene space into one AnnData.

    Each entry of ``sources`` is ``{"data_dir": ..., "row_indices": ... | None}``.
    Adds ``obs['source_dir']`` and preserves each dump's own ``cell_type`` column,
    which is what CellFlow splits control populations on.
    """
    if not sources:
        raise ValueError("sources is empty")

    parts: list[ad.AnnData] = []
    for spec in sources:
        data_dir = Path(spec["data_dir"])
        part = load_hooke_anndata(
            data_dir,
            selected_genes=selected_genes,
            feature=spec.get("feature", feature),
            row_indices=spec.get("row_indices"),
            target_sum=target_sum,
            log1p=log1p,
            on_missing_genes=on_missing_genes,
            seed=seed,
        )
        part.obs["source_dir"] = str(data_dir)
        parts.append(part)

    reference = list(parts[0].var_names)
    for spec, part in zip(sources, parts, strict=True):
        if list(part.var_names) != reference:
            raise ValueError(f"{spec['data_dir']}: gene space differs from {sources[0]['data_dir']}")

    if len(parts) == 1:
        return parts[0]

    combined = ad.concat(parts, axis=0, join="outer", index_unique="-")
    combined.uns["hooke_bridge"] = {
        "sources": [str(Path(s["data_dir"])) for s in sources],
        "n_obs_per_source": [int(p.n_obs) for p in parts],
    }
    return combined


def write_synthetic_hooke_dump(
    out_dir: Path | str,
    *,
    n_cells: int = 40,
    n_genes: int = 8,
    seed: int = 0,
    with_combos: bool = True,
    cell_type: str = "K562",
) -> Path:
    """Write a tiny Hooke-shaped dump for unit tests (zarr + parquet)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    gene_ids = [f"ENSG{i:011d}" for i in range(n_genes)]
    var = pd.DataFrame({"ensembl_gene_id": gene_ids})

    is_control = np.zeros(n_cells, dtype=bool)
    is_control[: n_cells // 4] = True
    gene_pert: list = []
    for i in range(n_cells):
        if is_control[i]:
            gene_pert.append([])
        elif with_combos and i % 5 == 0:
            gene_pert.append(
                [
                    ["genetic", gene_ids[0], "ko"],
                    ["genetic", gene_ids[1], "ko"],
                ]
            )
        else:
            gene_pert.append([["genetic", gene_ids[i % n_genes], "ko"]])

    obs = pd.DataFrame(
        {
            "cell_type": [cell_type] * n_cells,
            "experiment_label": ["plate-1"] * n_cells,
            "well_address": ["A01"] * n_cells,
            "assay_type": ["perturb_seq"] * n_cells,
            "batch_center": ["batch-1"] * n_cells,
            "negative_control": is_control,
            "gene_only": ~is_control,
            "mol_only": False,
            "gene_pert": gene_pert,
            "mol_pert": [[] for _ in range(n_cells)],
        }
    )
    x = rng.integers(0, 50, size=(n_cells, n_genes)).astype(np.float32)

    obs.to_parquet(out_dir / "obs.parquet", index=False)
    var.to_parquet(out_dir / "var.parquet", index=False)
    z = zarr.open_array(
        str(out_dir / "X.zarr"),
        mode="w",
        shape=x.shape,
        dtype="float32",
        chunks=(min(16, n_cells), n_genes),
    )
    z[:] = x
    return out_dir
