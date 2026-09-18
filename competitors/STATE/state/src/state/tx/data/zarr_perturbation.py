"""Native-zarr perturbation data module for STATE.

A drop-in alternative to ``cell_load``'s ``PerturbationDataModule`` that reads the
**the upstream benchmark canonical zarr format** (``X.zarr`` + ``obs.parquet`` + ``var.parquet``)
instead of ``.h5ad``, and uses **index-based** splits (a JSON of row indices) instead
of perturbation-name lists in a TOML.

It exposes the same public surface (``setup``, ``train_dataloader`` /
``val_dataloader`` / ``test_dataloader``, ``get_var_dims``, ``get_control_pert``,
``get_shared_perturbations``, ``save_state`` / ``load_state``, the ``*_onehot_map``
attributes) and yields the same per-batch dict, so the StateTransition model, the
training loop, and ``_predict.py`` work unchanged.

Design notes
------------
* Reads the **canonical** ``obs.parquet`` schema (stable across every the upstream benchmark
  dataset): perturbation derived from ``gene_pert`` / ``mol_pert`` via
  :func:`encode_obs_pert`, control via ``negative_control``, and context/batch
  from the configured ``cell_type_key`` / ``batch_col`` columns. No dependence
  on source-specific ``obs_raw.parquet`` columns.
* ``source_format="anndata"`` reads an ``.h5ad`` instead, using direct obs columns
  (``pert_col`` / ``batch_col`` / ``cell_type_key`` name real columns).
* Normalization is config-driven (``target_sum`` / ``log1p``), mirroring
  ``the upstream tree.data.dataset.TaskDataset.transform``: pre-normalized data uses
  ``target_sum=None, log1p=False`` (no-op); raw counts use ``target_sum=300000,
  log1p=True``.
* Gene selection via ``selected_ensembl_gene_ids_path`` subsets ``X`` columns
  through ``var.parquet.ensembl_gene_id`` (mirrors ``TaskDataset.gene_index_map``).
* Obs handling is vectorized (Polars expressions + boolean masks) so 100M-row
  sources like Tahoe stay tractable; split membership uses masks rather than
  per-cell-type Python sets of row indices.
* The sentence/meta-batch sampler reproduces ``cell_load.PerturbationBatchSampler``:
  group cells by ``(cell_type, pert)`` codes, split into sentences of
  ``cell_sentence_len``, pack into flattened meta-batches of
  ``batch_size * cell_sentence_len`` (replacement-fill partial sentences, except
  in test mode).
"""

from __future__ import annotations

import json
import logging
import os
import tomllib
from pathlib import Path

import numpy as np
import polars as pl
import torch
import zarr
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler

logger = logging.getLogger(__name__)

# Sentinel used for control rows in the derived perturbation string (canonical mode).
CONTROL_SENTINEL = "empty"

_SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# Small helpers (mirror the upstream tree + cell_load behavior)
# --------------------------------------------------------------------------- #
def encode_pert(gene_pert) -> str:
    """Convert a canonical ``gene_pert`` entry into a perturbation string.

    Mirrors ``the upstream tree.eval.adapters.celleval.encode_gene_pert`` (plus empty):
      - empty            -> ``"empty"`` (control sentinel)
      - single genetic   -> Ensembl id (e.g. ``"ENSG00000141510"``)
      - multiple genetic -> sorted ids joined by ``"+"``
    """
    if gene_pert is None or len(gene_pert) == 0:
        return CONTROL_SENTINEL
    ids = []
    for p in gene_pert:
        if p[0] != "genetic":
            raise NotImplementedError(
                f"encode_pert only supports genetic perturbations, got {p[0]!r}"
            )
        ids.append(str(p[1]))
    return "+".join(sorted(ids))


# nM in mol_pert[2] → µM in Arc ``drugname_drugconc`` (see tahoe_100m.UM_TO_NM).
_UM_TO_NM = 1_000.0


def encode_mol_pert(mol_pert) -> str:
    """Convert a canonical ``mol_pert`` entry into a perturbation string.

    Matches Arc STATE / cell-load ``drugname_drugconc``: Python ``repr`` of a
    list of ``(name, µM, 'uM')`` tuples. ``mol_pert[2]`` is nM in the Upstream dump.
    Short tuples without name/dose fall back to ``p[1]``.

    Empty rows are handled by :func:`encode_obs_pert`; this expects non-empty.
    """
    if mol_pert is None or len(mol_pert) == 0:
        raise ValueError("encode_mol_pert expects a non-empty mol_pert list")
    tuples: list[tuple] = []
    for p in mol_pert:
        if p[0] != "compound":
            raise NotImplementedError(
                f"encode_mol_pert only supports compound perturbations, got {p[0]!r}"
            )
        tuples.append(_mol_arc_tuple(p))
    return repr(tuples)


def _mol_arc_tuple(p) -> tuple:
    """``(name, µM, 'uM')`` or ``(p[1],)`` if dose/name are missing."""
    name = str(p[3]) if len(p) >= 4 and p[3] not in (None, "") else str(p[1])
    if len(p) >= 3 and p[2] not in (None, ""):
        return (name, float(p[2]) / _UM_TO_NM, "uM")
    return (name,)


def encode_obs_pert(gene_pert=None, mol_pert=None, *, negative_control: bool = False) -> str:
    """Encode perturbation identity from canonical ``gene_pert`` / ``mol_pert``.

    Thin local mirror of ``the upstream tree.eval.adapters.celleval.encode_obs_pert``.
    Genetic-only datasets (empty ``mol_pert``) keep prior ``encode_pert`` behavior.
    Mixed gene+mol on one cell is rejected.
    """
    gene_pert = gene_pert or []
    mol_pert = mol_pert or []
    if negative_control or not (gene_pert or mol_pert):
        return CONTROL_SENTINEL
    if gene_pert and mol_pert:
        raise NotImplementedError(
            "zarr loader does not support mixed gene+mol perturbations on one cell"
        )
    if gene_pert:
        return encode_pert(gene_pert)
    return encode_mol_pert(mol_pert)


def generate_onehot_map(keys) -> dict[str, torch.Tensor]:
    """Sorted, deterministic one-hot map (same indexing rule as cell_load)."""
    unique_keys = sorted(set(keys))
    onehots = torch.eye(len(unique_keys))
    return {k: onehots[i].clone() for i, k in enumerate(unique_keys)}


def load_external_perturbation_map(
    path: str | Path,
    required_perturbations: set[str],
) -> dict[str, torch.Tensor]:
    """Load a ``{perturbation_id: 1D tensor}`` map with one shared dimension."""
    feature_path = Path(path)
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)
    loaded = torch.load(feature_path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or not loaded:
        raise TypeError("External perturbation features must be a non-empty dictionary")

    features: dict[str, torch.Tensor] = {}
    feature_dim: int | None = None
    for perturbation, value in loaded.items():
        if not isinstance(perturbation, str):
            raise TypeError(f"External perturbation feature key must be a string, got {type(perturbation).__name__}")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"External perturbation feature for {perturbation!r} must be a tensor")
        feature = value.detach().to(dtype=torch.float32).clone()
        if feature.ndim != 1 or feature.numel() == 0:
            raise ValueError(f"Feature for {perturbation!r} must be a non-empty 1D tensor")
        if not torch.isfinite(feature).all():
            raise ValueError(f"Feature for {perturbation!r} contains non-finite values")
        if feature_dim is None:
            feature_dim = feature.numel()
        elif feature.numel() != feature_dim:
            raise ValueError(
                f"Feature for {perturbation!r} has dimension {feature.numel()}, expected {feature_dim}"
            )
        features[perturbation] = feature

    missing = sorted(required_perturbations - features.keys())
    if missing:
        raise ValueError(f"External perturbation features are missing {len(missing)} IDs (e.g. {missing[:5]})")
    return {perturbation: features[perturbation] for perturbation in sorted(required_perturbations)}


def _holds_pert_tuples(dtype) -> bool:
    """False when a pert column carries no tuples (Tahoe ``gene_pert`` is ``List(Null)``)."""
    inner = getattr(dtype, "inner", None)
    if inner is None or inner == pl.Null:
        return False
    return getattr(inner, "inner", None) != pl.Null


def _pert_ids_expr(obs: pl.DataFrame, col: str) -> pl.Expr:
    """Vectorized genetic ids: sorted ``p[1]`` joined by ``"+"``, ``""`` if empty."""
    if col not in obs.columns or not _holds_pert_tuples(obs.schema[col]):
        return pl.lit("")
    return (
        pl.col(col)
        .list.eval(pl.element().list.get(1).cast(pl.Utf8))
        .list.sort()
        .list.join("+")
        .fill_null("")
    )


def _mol_ids_series(obs: pl.DataFrame) -> pl.Series:
    """Arc ``drugname_drugconc`` strings via :func:`encode_mol_pert` on unique tuples."""
    n = obs.height
    if "mol_pert" not in obs.columns or not _holds_pert_tuples(obs.schema["mol_pert"]):
        return pl.repeat("", n, dtype=pl.Utf8)
    filled = obs["mol_pert"].fill_null([])
    uniq = filled.unique()
    encoded = [encode_mol_pert(v) if v else "" for v in uniq.to_list()]
    lut = pl.DataFrame({"mol_pert": uniq, "_mol_id": encoded})
    return filled.to_frame("mol_pert").join(lut, on="mol_pert", how="left")["_mol_id"]


def _dense_codes(values: pl.Series) -> tuple[np.ndarray, int]:
    """Integer group codes for a label column, hashed in Rust (no 100M-element Python sort)."""
    codes = values.cast(pl.Utf8).cast(pl.Categorical).to_physical().to_numpy()
    codes = codes.astype(np.int64, copy=False)
    return codes, (int(codes.max()) + 1 if codes.size else 0)


def _mask_from_indices(indices, n: int) -> np.ndarray:
    """Boolean mask of length ``n`` from split row indices; out-of-range ids are ignored."""
    idx = np.fromiter(indices, dtype=np.int64)
    mask = np.zeros(n, dtype=bool)
    mask[idx[(idx >= 0) & (idx < n)]] = True
    return mask


def _transform_expression(
    x: np.ndarray, target_sum: int | None, log1p: bool
) -> np.ndarray:
    """target_sum + log1p transform; mirrors TaskDataset.transform (no-op when both off)."""
    x = x.astype(np.float32, copy=False)
    if float(x.sum()) == 0.0:
        return x
    if target_sum is not None:
        x = (x / x.sum()) * np.float32(target_sum)
    if log1p:
        x = np.log1p(x)
    return x


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class ZarrExperimentConfig:
    """Parsed zarr TOML.

    ``[datasets]``  name -> data dir (zarr) or .h5ad path (anndata).
    ``[training]``  name -> "train" (contributes to the train split).
    ``[splits]``    name -> index-based split JSON path (optional; else train-only).
    ``[split_fold]`` name -> fold index (optional, default 0).
    """

    def __init__(self, datasets, training, splits, split_fold, occam=False):
        self.datasets = datasets
        self.training = training
        self.splits = splits
        self.split_fold = split_fold
        # occam mode has no TOML and no on-disk dataset paths: a `[datasets]` value is a
        # cell-line NAME. validate() must not warn that "HEPG2" is a missing file.
        self.occam = occam

    @classmethod
    def from_toml(cls, path: str | Path) -> ZarrExperimentConfig:
        with open(path, "rb") as f:
            doc = tomllib.load(f)
        return cls(
            datasets=dict(doc.get("datasets", {})),
            training=dict(doc.get("training", {})),
            splits=dict(doc.get("splits", {})),
            split_fold=dict(doc.get("split_fold", {})),
        )

    @classmethod
    def from_occam(cls, bench_dir: str | Path) -> ZarrExperimentConfig:
        """Our frozen benchmark, in their TOML's vocabulary.

        One `[datasets]` entry per cell line, ALL of them `[training] = "train"`, and NO
        `[splits]` entries -- the split does not live in a JSON of row indices here, it is
        derived live per line by `_read_split`'s occam branch (HepG2 splits three ways;
        Jurkat and RPE1 are wholly training data, which is the positive half of the
        few-shot exclusion).

        ⚠️ `training` must mark every line, including HepG2. `setup()` only falls back to
        "whole source is training" when `fold is None`, and in occam mode a fold is always
        returned -- but a line missing from `training` would still be a silent trap for any
        later change to that branch.
        """
        lines = occam_datasets(str(bench_dir))
        return cls(
            datasets={ln: ln for ln in lines},
            training={ln: "train" for ln in lines},
            splits={},
            split_fold={},
            occam=True,
        )

    def get_all_datasets(self) -> list[str]:
        return list(self.datasets.keys())

    def validate(self) -> None:
        if not self.datasets:
            raise ValueError("zarr TOML has no [datasets] entries.")
        for name, path in self.datasets.items():
            if not self.occam and not Path(path).exists():
                logger.warning("Dataset path does not exist: %s -> %s", name, path)
        for name, path in self.splits.items():
            if name not in self.datasets:
                raise ValueError(f"[splits] references unknown dataset {name!r}")
            if not Path(path).exists():
                logger.warning("Split file does not exist: %s -> %s", name, path)


from .occam_source import (
    OCCAM_FORMAT, occam_datasets, occam_fold_for_line, read_occam_source,
)


# --------------------------------------------------------------------------- #
# Per-source reader: loads obs (derived fields) + an X row-fetcher + gene map
# --------------------------------------------------------------------------- #
class _Source:
    """Lazily reads expression rows + holds derived metadata for one data source."""

    def __init__(
        self,
        name: str,
        path: str | Path,
        *,
        source_format: str,
        pert_col: str,
        batch_col: str,
        cell_type_key: str,
        control_pert: str,
        selected_genes: list[str] | None,
        embed_key: str | None,
        feature_loading: str = "ram",
        occam_bench_dir: str | None = None,
        occam_target_sum: float = 10_000.0,
        occam_log1p: bool = True,
        occam_verify_sources: bool = True,
        occam_max_cells_per_line: int | None = None,
    ):
        self.name = name
        self._occam = None
        self.path = Path(path)
        self.source_format = source_format
        self.embed_key = embed_key
        self.feature_loading = feature_loading
        self._x = None  # lazily materialized expression matrix (numpy / zarr / backed)
        self._adata = None
        # Expression artifact paths (zarr mode), mirrors the upstream benchmark X.zarr / X.npy pair.
        self._zarr_path = str(self.path / "X.zarr")
        self._npy_path = str(self.path / "X.npy")

        if source_format == "zarr":
            obs_path = self.path / "obs.parquet"
            obs_schema = pl.read_parquet_schema(obs_path)
            obs_cols = [cell_type_key, batch_col] + [
                c for c in ("gene_pert", "mol_pert", "negative_control") if c in obs_schema
            ]
            obs = pl.read_parquet(obs_path, columns=obs_cols)
            var = pl.read_parquet(self.path / "var.parquet")
            var_ensembl = [str(g) for g in var["ensembl_gene_id"].to_list()]
            var_symbol = (
                [str(g) for g in var["gene_symbol"].to_list()]
                if "gene_symbol" in var.columns
                else var_ensembl
            )
            self.pert_str, self.is_control, self.cell_type, self.batch = (
                _derive_canonical(
                    obs,
                    batch_col=batch_col,
                    cell_type_key=cell_type_key,
                    control_pert=control_pert,
                )
            )
            self.n_rows = obs.height
            self.cell_type_code, self.n_cell_types = _dense_codes(obs[cell_type_key])
        elif source_format == "anndata":
            import anndata

            a = anndata.read_h5ad(self.path, backed="r")
            obs = a.obs
            if "ensembl_gene_id" in a.var.columns:
                var_ensembl = [str(g) for g in a.var["ensembl_gene_id"]]
            else:
                var_ensembl = [str(g) for g in a.var_names]
            var_symbol = [str(g) for g in a.var_names]
            self.pert_str, self.is_control, self.cell_type, self.batch = (
                _derive_direct(obs, pert_col, batch_col, cell_type_key, control_pert)
            )
            self.n_rows = a.n_obs
            self.cell_type_code, self.n_cell_types = _dense_codes(pl.Series(self.cell_type))
            # Close the metadata handle; expression is (re)opened in _ensure_open.
            try:
                a.file.close()
            except Exception:
                pass
        elif source_format == OCCAM_FORMAT:
            # Occam's `rhj_fewshot`: OUR h5ads, OUR frozen 6,983-gene axis, OUR splits, read
            # in place. Normalisation happens in occam_data on the FULL source panel row sum
            # -- see occam_source.py for why that denominator, and not the 6,983-gene one, is
            # the load-bearing choice in this port.
            self._occam = read_occam_source(
                str(self.path), bench_dir=occam_bench_dir, target_sum=occam_target_sum,
                log1p=occam_log1p, control_pert=control_pert, pert_col=pert_col,
                cell_type_key=cell_type_key, batch_col=batch_col,
                selected_genes=selected_genes,
                verify_sources=occam_verify_sources,
                max_cells=occam_max_cells_per_line)
            var_ensembl = list(self._occam.gene_ensembl)
            var_symbol = list(self._occam.gene_symbols)
            self.pert_str = self._occam.pert_str
            self.is_control = self._occam.is_control
            self.cell_type = self._occam.cell_type
            self.batch = self._occam.batch
            self.n_rows = self._occam.n_rows
            self.cell_type_code, self.n_cell_types = _dense_codes(
                pl.Series(self.cell_type))
            # The matrix is already in RAM, in axis order, and already normalised.
            self._x = self._occam.X
        else:
            raise ValueError(f"Unknown source_format {source_format!r}")

        # Gene selection map (into var/X columns). None selection -> identity.
        if selected_genes is None:
            self.gene_index_map = np.arange(len(var_ensembl), dtype=np.int64)
            self.gene_symbols = var_symbol
        else:
            pos = {g: i for i, g in enumerate(var_ensembl)}
            missing = [g for g in selected_genes if g not in pos]
            if missing:
                raise ValueError(
                    f"[{name}] {len(missing)} selected genes missing from var "
                    f"(e.g. {missing[:5]})"
                )
            self.gene_index_map = np.array(
                [pos[g] for g in selected_genes], dtype=np.int64
            )
            self.gene_symbols = [var_symbol[pos[g]] for g in selected_genes]

    # -- materialize the expression matrix (mirrors the upstream benchmark feature_loading) #
    def _ensure_open(self):
        """Materialize ``self._x`` according to ``feature_loading``.

        Mirrors ``the upstream tree.data.dataset.TaskDataset.setup_details``:
          - ``ram``  : full ``X.npy`` in RAM if present, else full ``X.zarr``.
          - ``npy``  : require ``X.npy`` and mmap it.
          - ``zarr`` : lazy ``zarr.open`` (per-row chunk reads; slowest).

        Safe to call in the main process (preload) or lazily inside a worker.
        """
        if self._x is not None:
            return
        fl = self.feature_loading

        if self.source_format == OCCAM_FORMAT:
            # read_occam_source already put the whole matrix in RAM, and __getstate__ drops
            # it for worker pickling, so re-read rather than hand a worker an empty source.
            if self._occam is None:
                raise RuntimeError(
                    "an occam source lost its matrix across a process boundary. Run with "
                    "num_workers=0, or keep feature_loading='ram' in the main process.")
            self._x = self._occam.X
        elif self.source_format == "zarr":
            if fl == "npy":
                if not os.path.exists(self._npy_path):
                    raise FileNotFoundError(
                        f"feature_loading='npy' requires {self._npy_path} to exist."
                    )
                self._x = np.load(self._npy_path, mmap_mode="r")
            elif fl == "ram":
                if os.path.exists(self._npy_path):
                    self._x = np.load(self._npy_path)
                else:
                    self._x = np.array(zarr.open(self._zarr_path, mode="r"))
            elif fl == "zarr":
                self._x = zarr.open(self._zarr_path, mode="r")
            else:
                raise ValueError(
                    f"Invalid feature_loading={fl!r}; must be 'ram', 'zarr', or 'npy'."
                )
        else:  # anndata
            import anndata

            if fl == "ram":
                a = anndata.read_h5ad(self.path)  # in-memory
                X = a.X
                self._x = np.asarray(
                    X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32
                )
            else:  # zarr / npy => lazy backed row access
                self._adata = anndata.read_h5ad(self.path, backed="r")
                self._x = self._adata.X

    def fetch_full_row(self, row: int) -> np.ndarray:
        """Full-width expression row (before gene selection / transform)."""
        self._ensure_open()
        x = self._x[row]
        if hasattr(x, "toarray"):
            x = x.toarray()
        return np.asarray(x, dtype=np.float32).reshape(-1)

    def __getstate__(self):
        # Don't pickle open file handles across worker processes.
        state = self.__dict__.copy()
        state["_x"] = None
        state["_adata"] = None
        # ⚠️ Drop the occam matrix as well. Without this, every DataLoader worker would be
        # sent several GB of expression through a pickle; with it, an occam source simply
        # cannot cross a process boundary and says so (see _ensure_open). Upstream's own
        # STATE launcher passes num_workers=0, so this costs nothing.
        state["_occam"] = None
        return state


def _derive_canonical(
    obs: pl.DataFrame,
    *,
    batch_col: str,
    cell_type_key: str,
    control_pert: str,
):
    """Derive STATE fields from configurable canonical ``obs.parquet`` columns.

    Vectorized equivalent of :func:`encode_obs_pert` applied row-wise, so a
    Tahoe-scale ``obs`` never materializes as 100M nested Python tuples.
    """
    missing = [column for column in (cell_type_key, batch_col) if column not in obs.columns]
    if missing:
        raise ValueError(f"Canonical obs missing configured columns: {sorted(set(missing))}")

    gene_ids = _pert_ids_expr(obs, "gene_pert")
    obs = obs.with_columns(_mol_ids_series(obs).alias("_mol_id"))
    mol_ids = pl.col("_mol_id")
    unperturbed = (gene_ids == "") & (mol_ids == "")
    if "negative_control" in obs.columns:
        control = pl.col("negative_control").fill_null(False)
    else:
        control = unperturbed  # equivalent to the old pert_str == CONTROL_SENTINEL

    derived = obs.select(
        control.alias("is_control"),
        # Control rows take the sentinel so the pert string stays consistent.
        pl.when(control | unperturbed)
        .then(pl.lit(control_pert))
        # Mixed gene+mol on one cell is unsupported; the null is rejected below.
        .when((gene_ids != "") & (mol_ids != ""))
        .then(pl.lit(None, dtype=pl.Utf8))
        .when(gene_ids != "")
        .then(gene_ids)
        .otherwise(mol_ids)
        .alias("pert_str"),
        pl.col(cell_type_key).cast(pl.Utf8).alias("cell_type"),
        pl.col(batch_col).cast(pl.Utf8).alias("batch"),
    )
    if derived["pert_str"].null_count():
        raise NotImplementedError(
            "zarr loader does not support mixed gene+mol perturbations on one cell"
        )
    return (
        derived["pert_str"].to_numpy(),
        derived["is_control"].to_numpy(),
        derived["cell_type"].to_numpy(),
        derived["batch"].to_numpy(),
    )


def _derive_direct(obs, pert_col, batch_col, cell_type_key, control_pert):
    """Derive fields from direct obs columns (anndata mode)."""
    pert_str = np.array([str(p) for p in obs[pert_col].to_numpy()], dtype=object)
    is_control = pert_str == str(control_pert)
    cell_type = np.array([str(c) for c in obs[cell_type_key].to_numpy()], dtype=object)
    batch = np.array([str(b) for b in obs[batch_col].to_numpy()], dtype=object)
    return pert_str, is_control, cell_type, batch


# --------------------------------------------------------------------------- #
# Dataset: holds one split (across all sources) + control pools
# --------------------------------------------------------------------------- #
class ZarrPerturbationDataset(Dataset):
    """One split's samples (perturbed + control rows) across all sources."""

    def __init__(
        self,
        *,
        split: str,
        index_map: list[tuple[str, int]],
        sources: dict[str, _Source],
        pert_onehot_map: dict[str, torch.Tensor],
        batch_onehot_map: dict[str, torch.Tensor],
        cell_type_onehot_map: dict[str, torch.Tensor],
        control_pert: str,
        embed_key: str | None,
        target_sum: int | None,
        log1p: bool,
        basal_mapping_strategy: str,
        random_seed: int,
    ):
        self.split = split
        self.index_map = index_map
        self.sources = sources
        self.pert_onehot_map = pert_onehot_map
        self.batch_onehot_map = batch_onehot_map
        self.cell_type_onehot_map = cell_type_onehot_map
        self.control_pert = control_pert
        self.embed_key = embed_key
        self.target_sum = target_sum
        self.log1p = log1p
        self.basal_mapping_strategy = basal_mapping_strategy
        self.rng = np.random.default_rng(random_seed)

        n = len(index_map)
        # Integer codes (arbitrary but consistent) for grouping in the sampler.
        pert_keys = sorted(pert_onehot_map.keys())
        ct_keys = sorted(cell_type_onehot_map.keys())
        batch_keys = sorted(batch_onehot_map.keys())
        pert_code_of = {k: i for i, k in enumerate(pert_keys)}
        ct_code_of = {k: i for i, k in enumerate(ct_keys)}
        batch_code_of = {k: i for i, k in enumerate(batch_keys)}

        self.pert_str = np.empty(n, dtype=object)
        self.cell_type = np.empty(n, dtype=object)
        self.batch = np.empty(n, dtype=object)
        self.is_control = np.zeros(n, dtype=bool)
        self.cell_type_codes = np.empty(n, dtype=np.int64)
        self.pert_codes = np.empty(n, dtype=np.int64)
        self.batch_codes = np.empty(n, dtype=np.int64)

        for i, (src_name, row) in enumerate(index_map):
            s = sources[src_name]
            self.pert_str[i] = s.pert_str[row]
            self.cell_type[i] = s.cell_type[row]
            self.batch[i] = s.batch[row]
            self.is_control[i] = s.is_control[row]
            self.cell_type_codes[i] = ct_code_of[s.cell_type[row]]
            self.pert_codes[i] = pert_code_of[s.pert_str[row]]
            self.batch_codes[i] = batch_code_of[s.batch[row]]

        # Control pools (positions in index_map) for basal pairing.
        ctrl_positions = np.flatnonzero(self.is_control)
        self.pool_by_ct: dict[int, np.ndarray] = {}
        for ct in np.unique(self.cell_type_codes[ctrl_positions]) if ctrl_positions.size else []:
            self.pool_by_ct[int(ct)] = ctrl_positions[
                self.cell_type_codes[ctrl_positions] == ct
            ]
        self.pool_by_batch_ct: dict[tuple[int, int], np.ndarray] = {}
        if basal_mapping_strategy == "batch" and ctrl_positions.size:
            for pos in ctrl_positions:
                key = (int(self.batch_codes[pos]), int(self.cell_type_codes[pos]))
                self.pool_by_batch_ct.setdefault(key, []).append(int(pos))
            self.pool_by_batch_ct = {
                k: np.asarray(v, dtype=np.int64) for k, v in self.pool_by_batch_ct.items()
            }

    def __len__(self) -> int:
        return len(self.index_map)

    def _pick_control(self, i: int) -> int | None:
        ct = int(self.cell_type_codes[i])
        if self.basal_mapping_strategy == "batch":
            pool = self.pool_by_batch_ct.get((int(self.batch_codes[i]), ct))
            if pool is None or len(pool) == 0:
                pool = self.pool_by_ct.get(ct)
        else:
            pool = self.pool_by_ct.get(ct)
        if pool is None or len(pool) == 0:
            return None
        return int(pool[self.rng.integers(len(pool))])

    def _expr(self, i: int) -> torch.Tensor:
        src_name, row = self.index_map[i]
        s = self.sources[src_name]
        x = s.fetch_full_row(row)[s.gene_index_map]
        x = _transform_expression(x, self.target_sum, self.log1p)
        return torch.from_numpy(np.ascontiguousarray(x))

    def __getitem__(self, i: int) -> dict:
        pert_expr = self._expr(i)
        ctrl_pos = self._pick_control(i)
        ctrl_expr = self._expr(ctrl_pos) if ctrl_pos is not None else torch.zeros_like(pert_expr)

        pert_name = self.pert_str[i]
        cell_type = self.cell_type[i]
        batch_name = self.batch[i]
        return {
            "pert_cell_emb": pert_expr,
            "ctrl_cell_emb": ctrl_expr,
            "pert_emb": self.pert_onehot_map[pert_name],
            "pert_name": pert_name,
            "dataset_name": self.index_map[i][0],
            "batch_name": batch_name,
            "batch": self.batch_onehot_map[batch_name],
            "cell_type": cell_type,
            "cell_type_onehot": self.cell_type_onehot_map[cell_type],
        }


# --------------------------------------------------------------------------- #
# Sampler: mirrors cell_load.PerturbationBatchSampler (flattened meta-batches)
# --------------------------------------------------------------------------- #
class SentenceBatchSampler(Sampler):
    def __init__(
        self,
        dataset: ZarrPerturbationDataset,
        batch_size: int,
        cell_sentence_len: int,
        *,
        test: bool = False,
        use_batch: bool = False,
        drop_last: bool = False,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = 1 if test else batch_size
        self.cell_sentence_len = cell_sentence_len
        self.test = test
        self.use_batch = use_batch
        self.drop_last = drop_last
        self._rng = np.random.RandomState(seed)

        self.sentences = self._create_sentences()
        self.tot_num = int(sum(len(s) for s in self.sentences))
        self.batches = self._create_batches()

    def _create_sentences(self) -> list[list[int]]:
        ct = self.dataset.cell_type_codes
        pert = self.dataset.pert_codes
        if self.use_batch:
            batch = self.dataset.batch_codes
            keys = np.ravel_multi_index(
                (batch, ct, pert),
                (batch.max() + 1, ct.max() + 1, pert.max() + 1),
            ) if len(ct) else np.array([], dtype=np.int64)
        else:
            keys = np.ravel_multi_index(
                (ct, pert), (ct.max() + 1, pert.max() + 1)
            ) if len(ct) else np.array([], dtype=np.int64)

        sentences: list[list[int]] = []
        if len(keys):
            order = np.argsort(keys, kind="stable")
            sorted_keys = keys[order]
            uniq, starts = np.unique(sorted_keys, return_index=True)
            starts = np.r_[starts, len(sorted_keys)]
            for s, e in zip(starts[:-1], starts[1:]):
                group = order[s:e].copy()
                self._rng.shuffle(group)
                for i in range(0, len(group), self.cell_sentence_len):
                    sent = group[i : i + self.cell_sentence_len]
                    if len(sent) < self.cell_sentence_len and self.drop_last:
                        continue
                    sentences.append(sent.tolist())
        self._rng.shuffle(sentences)
        return sentences

    def _create_batches(self) -> list[list[int]]:
        all_batches: list[list[int]] = []
        current: list[int] = []
        for sentence in self.sentences:
            if len(sentence) < self.cell_sentence_len and not self.test:
                new_sentence = self._rng.choice(
                    sentence, size=self.cell_sentence_len, replace=True
                ).tolist()
            else:
                new_sentence = list(sentence)
            sentence_len = len(new_sentence) if self.test else self.cell_sentence_len
            if len(current) + len(new_sentence) <= self.batch_size * sentence_len:
                current.extend(new_sentence)
            else:
                if current:
                    all_batches.append(current)
                current = new_sentence
        if current and not self.drop_last:
            all_batches.append(current)
        return all_batches

    def __iter__(self):
        if not self.test:
            self.batches = self._create_batches()
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


def collate_fn(batch: list[dict]) -> dict:
    """Stack tensors, keep name fields as lists (matches cell_load contract)."""
    return {
        "pert_cell_emb": torch.stack([b["pert_cell_emb"] for b in batch]),
        "ctrl_cell_emb": torch.stack([b["ctrl_cell_emb"] for b in batch]),
        "pert_emb": torch.stack([b["pert_emb"] for b in batch]),
        "pert_name": [b["pert_name"] for b in batch],
        "cell_type": [b["cell_type"] for b in batch],
        "cell_type_onehot": torch.stack([b["cell_type_onehot"] for b in batch]),
        "batch": torch.stack([b["batch"] for b in batch]),
        "batch_name": [b["batch_name"] for b in batch],
        "dataset_name": [b["dataset_name"] for b in batch],
    }


# --------------------------------------------------------------------------- #
# Data module
# --------------------------------------------------------------------------- #
class ZarrPerturbationDataModule(LightningDataModule):
    """Native-zarr drop-in replacement for cell_load's PerturbationDataModule."""

    # kwargs accepted by save_state/load_state round-trip.
    _STATE_KEYS = (
        "toml_config_path", "batch_size", "num_workers", "random_seed",
        "pert_col", "batch_col", "cell_type_key", "control_pert", "embed_key",
        "output_space", "basal_mapping_strategy", "n_basal_samples",
        "should_yield_control_cells", "cell_sentence_len", "source_format",
        "feature_loading", "selected_ensembl_gene_ids_path", "target_sum", "log1p",
        "occam_bench_dir", "occam_target_sum", "occam_log1p", "occam_verify_sources",
        "occam_max_cells_per_line",
        "barcode", "drop_last", "is_log1p", "split_idx",
        "pert_feature_mode", "pert_feature_path",
    )

    def __init__(
        self,
        toml_config_path: str,
        batch_size: int = 128,
        num_workers: int = 8,
        random_seed: int = 42,
        pert_col: str = "gene",
        batch_col: str = "batch_center",
        cell_type_key: str = "cell_type",
        control_pert: str = CONTROL_SENTINEL,
        embed_key: str | None = "X_hvg",
        output_space: str = "gene",
        basal_mapping_strategy: str = "random",
        n_basal_samples: int = 1,
        should_yield_control_cells: bool = True,
        cell_sentence_len: int = 512,
        source_format: str = "zarr",
        feature_loading: str = "ram",
        selected_ensembl_gene_ids_path: str | None = None,
        target_sum: int | None = None,
        log1p: bool = False,
        occam_bench_dir: str | None = None,
        occam_target_sum: float = 10_000.0,
        occam_log1p: bool = True,
        occam_verify_sources: bool = True,
        occam_max_cells_per_line: int | None = None,
        barcode: bool = False,
        drop_last: bool = False,
        is_log1p: bool = True,
        split_idx: int = 0,
        pert_feature_mode: str = "onehot",
        pert_feature_path: str | None = None,
        **kwargs,
    ):
        super().__init__()
        self.toml_config_path = toml_config_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.pert_col = pert_col
        self.batch_col = batch_col
        self.cell_type_key = cell_type_key
        self.control_pert = control_pert
        self.embed_key = embed_key
        self.output_space = output_space
        self.basal_mapping_strategy = basal_mapping_strategy
        self.n_basal_samples = n_basal_samples
        self.should_yield_control_cells = should_yield_control_cells
        self.cell_sentence_len = cell_sentence_len
        self.source_format = source_format
        self.feature_loading = feature_loading
        self.selected_ensembl_gene_ids_path = selected_ensembl_gene_ids_path
        self.target_sum = target_sum
        self.log1p = log1p
        self.occam_bench_dir = occam_bench_dir
        self.occam_target_sum = occam_target_sum
        self.occam_log1p = occam_log1p
        self.occam_verify_sources = occam_verify_sources
        self.occam_max_cells_per_line = occam_max_cells_per_line
        self.barcode = barcode
        self.drop_last = drop_last
        self.is_log1p = is_log1p
        self.split_idx = split_idx
        if pert_feature_mode not in {"onehot", "external"}:
            raise ValueError(f"pert_feature_mode must be 'onehot' or 'external', got {pert_feature_mode!r}")
        if pert_feature_mode == "external" and not pert_feature_path:
            raise ValueError("pert_feature_path is required when pert_feature_mode='external'")
        self.pert_feature_mode = pert_feature_mode
        self.pert_feature_path = pert_feature_path

        if source_format == OCCAM_FORMAT:
            if not occam_bench_dir:
                raise ValueError(
                    "source_format='occam' needs occam_bench_dir, the path to our frozen "
                    "benchmark (data/benchmarks/rhj_fewshot). It carries the gene axis, the "
                    "splits, the exclusions and the source sha256s.")
            if target_sum is not None or log1p:
                # `_transform_expression` divides by the row sum AFTER gene selection, i.e.
                # over 6,983 genes. occam_data has already normalised on the FULL source
                # panel, which is what makes the 10,000 -> 2,000 emission map exact. Doing
                # both would silently renormalise on the wrong denominator.
                raise ValueError(
                    f"source_format='occam' requires target_sum=null and log1p=false (got "
                    f"{target_sum!r} / {log1p!r}). The data arrives already normalised at "
                    f"target_sum={occam_target_sum:g} on the FULL source panel row sum; a "
                    f"second transform would divide by the 6,983-gene sum instead and break "
                    f"both `mae` and the prediction bridge's emission map.")

        if source_format == OCCAM_FORMAT:
            # There is no TOML in occam mode -- the benchmark directory is the manifest.
            # Passing one anyway would mean two disagreeing source lists, so refuse.
            if toml_config_path:
                raise ValueError(
                    f"source_format='occam' takes its sources from occam_bench_dir, not a "
                    f"TOML; got toml_config_path={toml_config_path!r}. Two source lists "
                    f"cannot both be authoritative -- drop one.")
            self.config = ZarrExperimentConfig.from_occam(occam_bench_dir)
        else:
            self.config = ZarrExperimentConfig.from_toml(toml_config_path)
        self.config.validate()

        self.selected_genes = None
        if selected_ensembl_gene_ids_path:
            with open(selected_ensembl_gene_ids_path) as f:
                self.selected_genes = [ln.strip() for ln in f if ln.strip()]

        self.train_datasets: list[ZarrPerturbationDataset] = []
        self.val_datasets: list[ZarrPerturbationDataset] = []
        self.test_datasets: list[ZarrPerturbationDataset] = []

        self.pert_onehot_map: dict[str, torch.Tensor] | None = None
        self.batch_onehot_map: dict[str, torch.Tensor] | None = None
        self.cell_type_onehot_map: dict[str, torch.Tensor] | None = None
        self._sources: dict[str, _Source] = {}
        self._gene_symbols: list[str] | None = None

        self._setup_global_maps()

    # alias used by _train.py
    @property
    def celltype_onehot_map(self):
        return self.cell_type_onehot_map

    # ----------------------------------------------------------------- maps #
    def _load_source(self, name: str) -> _Source:
        if name not in self._sources:
            self._sources[name] = _Source(
                name,
                self.config.datasets[name],
                source_format=self.source_format,
                pert_col=self.pert_col,
                batch_col=self.batch_col,
                cell_type_key=self.cell_type_key,
                control_pert=self.control_pert,
                selected_genes=self.selected_genes,
                embed_key=self.embed_key,
                feature_loading=self.feature_loading,
                occam_bench_dir=self.occam_bench_dir,
                occam_target_sum=self.occam_target_sum,
                occam_log1p=self.occam_log1p,
                occam_verify_sources=self.occam_verify_sources,
                occam_max_cells_per_line=self.occam_max_cells_per_line,
            )
        return self._sources[name]

    def _setup_global_maps(self) -> None:
        all_perts, all_batches, all_celltypes = set(), set(), set()
        for name in self.config.get_all_datasets():
            s = self._load_source(name)
            all_perts.update(s.pert_str.tolist())
            all_batches.update(s.batch.tolist())
            all_celltypes.update(s.cell_type.tolist())
            if self._gene_symbols is None:
                self._gene_symbols = list(s.gene_symbols)
        all_perts.add(self.control_pert)
        if self.pert_feature_mode == "external":
            assert self.pert_feature_path is not None
            self.pert_onehot_map = load_external_perturbation_map(
                self.pert_feature_path,
                all_perts,
            )
        else:
            self.pert_onehot_map = generate_onehot_map(all_perts)
        self.batch_onehot_map = generate_onehot_map(all_batches)
        self.cell_type_onehot_map = generate_onehot_map(all_celltypes)
        logger.info(
            "Global maps: %d perts (%s features), %d batches, %d cell types",
            len(self.pert_onehot_map),
            self.pert_feature_mode,
            len(self.batch_onehot_map),
            len(self.cell_type_onehot_map),
        )

    # --------------------------------------------------------------- splits #
    def _read_split(self, name: str) -> dict | None:
        if self.source_format == OCCAM_FORMAT:
            # Our split is not a JSON of row indices; it is the frozen benchmark, read live.
            # HepG2 splits three ways, Jurkat and RPE1 are wholly finetune -- our
            # per-(file, perturbation) exclusion in their vocabulary. Returned in the exact
            # shape setup() already consumes, so no new machinery touches the training loop.
            return occam_fold_for_line(
                self._load_source(name)._occam, bench_dir=self.occam_bench_dir,
                verify_sources=False)

        path = self.config.splits.get(name)
        if path is None:
            return None
        with open(path) as f:
            doc = json.load(f)
        folds = doc.get("folds")
        fold_idx = int(self.config.split_fold.get(name, self.split_idx))
        if not isinstance(folds, list) or fold_idx >= len(folds):
            raise ValueError(f"split file {path} missing folds or fold {fold_idx}")
        return folds[fold_idx]

    def setup(self, stage: str | None = None) -> None:
        if self.train_datasets or self.val_datasets or self.test_datasets:
            return

        train_map: list[tuple[str, int]] = []
        val_map: list[tuple[str, int]] = []
        test_map: list[tuple[str, int]] = []

        for name in self.config.get_all_datasets():
            s = self._load_source(name)
            is_training = self.config.training.get(name) == "train"
            fold = self._read_split(name)

            is_ctrl = s.is_control
            is_pert = ~is_ctrl

            if fold is not None:
                ct_code = s.cell_type_code
                for split_map, key in (
                    (train_map, "finetune"),
                    (val_map, "validation"),
                    (test_map, "test"),
                ):
                    perts = is_pert & _mask_from_indices(fold.get(key, ()), s.n_rows)
                    # Controls of a cell type join a split only when that type has a
                    # perturbed cell there (cell_load semantics).
                    keep_ct = np.zeros(s.n_cell_types, dtype=bool)
                    keep_ct[ct_code[perts]] = True
                    rows = np.flatnonzero(perts | (is_ctrl & keep_ct[ct_code]))
                    split_map.extend((name, int(r)) for r in rows)
            elif is_training:
                train_map.extend((name, r) for r in range(s.n_rows))

        self.train_datasets = [self._make_dataset("train", train_map)] if train_map else []
        self.val_datasets = [self._make_dataset("val", val_map)] if val_map else []
        self.test_datasets = [self._make_dataset("test", test_map)] if test_map else []
        logger.info(
            "Done! Train / Val / Test sizes: %d / %d / %d",
            len(train_map), len(val_map), len(test_map),
        )

        # Preload expression matrices in the MAIN process for ram/npy modes so
        # forked DataLoader workers share them copy-on-write (avoids each worker
        # re-loading ~GBs). Lazy "zarr" mode is intentionally not preloaded.
        if self.feature_loading in ("ram", "npy"):
            used = {src for split in (train_map, val_map, test_map) for src, _ in split}
            for name in used:
                logger.info("Preloading expression (%s) for %s", self.feature_loading, name)
                self._sources[name]._ensure_open()

    def _make_dataset(self, split: str, index_map) -> ZarrPerturbationDataset:
        return ZarrPerturbationDataset(
            split=split,
            index_map=index_map,
            sources=self._sources,
            pert_onehot_map=self.pert_onehot_map,
            batch_onehot_map=self.batch_onehot_map,
            cell_type_onehot_map=self.cell_type_onehot_map,
            control_pert=self.control_pert,
            embed_key=self.embed_key,
            target_sum=self.target_sum,
            log1p=self.log1p,
            basal_mapping_strategy=self.basal_mapping_strategy,
            random_seed=self.random_seed,
        )

    # ---------------------------------------------------------- dataloaders #
    def _create_dataloader(self, ds: ZarrPerturbationDataset, *, test: bool, batch_size=None):
        batch_size = batch_size or (1 if test else self.batch_size)
        sampler = SentenceBatchSampler(
            ds,
            batch_size=batch_size,
            cell_sentence_len=self.cell_sentence_len,
            test=test,
            use_batch=(self.basal_mapping_strategy == "batch"),
            drop_last=self.drop_last,
            seed=self.random_seed,
        )
        return DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=bool(self.num_workers > 0 and not test),
            prefetch_factor=4 if (not test and self.num_workers > 0) else None,
        )

    def train_dataloader(self, test: bool = False):
        if not self.train_datasets:
            raise ValueError("No training datasets. Call setup() first.")
        return self._create_dataloader(self.train_datasets[0], test=test)

    def val_dataloader(self):
        if not self.val_datasets:
            # This used to fall back to the TEST loader. That silently made val_loss a test metric,
            # so ModelCheckpoint(monitor="val_loss") selected its "best" checkpoint on test -- for
            # every run on a split whose `validation` list is empty, which includes the original
            # replogle_nadig_xcell.json. Failing loudly is the only safe behaviour: a benchmark that
            # selects on test is worse than one that refuses to start.
            splits = getattr(self.config, "splits", {}) or {}
            raise RuntimeError(
                "validation split is empty, so there is nothing to validate on. Point "
                "[splits] at a split file whose fold has a non-empty `validation` list "
                "(e.g. split_replogle_nadig_fewshot__v1.json). Refusing to fall back to the test "
                f"loader, which would select checkpoints on test. Current [splits]: {dict(splits)}"
            )
        return self._create_dataloader(self.val_datasets[0], test=False)

    def val_dataloader_for_scoring(self):
        """Validation loader with test-style batching, for prediction-based scoring.

        ``val_dataloader`` uses ``test=False`` because Lightning's validation loop wants the
        training-shaped sampler. Cell-eval scoring instead needs the one-cell-set-per-batch layout
        ``test_dataloader`` uses, or predictions do not line up row-for-row with the ground truth.
        """
        if not self.val_datasets:
            raise RuntimeError("validation split is empty; nothing to score.")
        return self._create_dataloader(self.val_datasets[0], test=True, batch_size=1)

    def test_dataloader(self):
        if not self.test_datasets:
            return []
        return self._create_dataloader(self.test_datasets[0], test=True, batch_size=1)

    def predict_dataloader(self):
        if not self.test_datasets:
            return []
        return self._create_dataloader(self.test_datasets[0], test=True)

    # --------------------------------------------------------------- dims & #
    def get_var_dims(self) -> dict:
        n_sel = len(self._gene_symbols) if self._gene_symbols is not None else 0
        pert_dim = next(iter(self.pert_onehot_map.values())).shape[0]
        batch_dim = next(iter(self.batch_onehot_map.values())).shape[0]
        return {
            "input_dim": n_sel,
            "gene_dim": n_sel,
            "hvg_dim": n_sel,
            "output_dim": n_sel,
            "pert_dim": pert_dim,
            "batch_dim": batch_dim,
            "gene_names": list(self._gene_symbols or []),
            "pert_names": list(self.pert_onehot_map.keys()),
        }

    def get_control_pert(self) -> str:
        return self.control_pert

    def get_shared_perturbations(self) -> set[str]:
        train = set(self.train_datasets[0].pert_str.tolist()) if self.train_datasets else set()
        test = set(self.test_datasets[0].pert_str.tolist()) if self.test_datasets else set()
        return train & test

    # ----------------------------------------------------------- persistence #
    def save_state(self, filepath) -> None:
        save_dict = {k: getattr(self, k) for k in self._STATE_KEYS}
        save_dict["data_module_class"] = "ZarrPerturbationDataModule"
        torch.save(save_dict, filepath)

    @classmethod
    def load_state(
        cls, filepath: str, *, toml_config_path: str | None = None
    ) -> ZarrPerturbationDataModule:
        save_dict = torch.load(filepath, map_location="cpu", weights_only=False)
        save_dict.pop("data_module_class", None)
        # Predict jobs often restage X.npy; the training TOML may point at a
        # vanished /tmp path. Override before __init__ opens the file.
        if toml_config_path is not None:
            save_dict["toml_config_path"] = toml_config_path
        return cls(**save_dict)

    def _setup_global_maps_public(self) -> None:  # parity with cell_load name
        self._setup_global_maps()
