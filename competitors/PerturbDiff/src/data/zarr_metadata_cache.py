"""Metadata cache for the hooke-tx canonical zarr format.

The h5ad path reads perturbation / cell-type / batch categoricals out of
``obs/<col>/{categories,codes}`` with h5py (:mod:`src.data.metadata_cache`). The
hooke-tx benchmark instead stores metadata in ``obs.parquet`` next to ``X.zarr``,
and does not store a perturbation *string* at all -- it is derived from the
canonical ``gene_pert`` / ``mol_pert`` / ``negative_control`` columns.

:class:`ZarrMetadataCache` reproduces the exact attribute surface that the rest
of the package consumes::

    pert_categories / pert_codes
    cell_type_categories / cell_type_codes
    batch_categories / batch_codes
    n_cells, control_pert_code, control_mask

so ``dataset_grouping``, ``split_strategy``, ``sampler`` and
``dataset_core.get_covarates`` work unchanged against zarr data.

The perturbation encoding is a deliberate port of STATE's
``tx/data/zarr_perturbation.py`` (``encode_pert`` / ``encode_mol_pert`` /
``encode_obs_pert``): the two models must derive *identical* perturbation
strings from the same rows, or the benchmark comparison is meaningless.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# Sentinel used for control rows in the derived perturbation string. Matches
# STATE's CONTROL_SENTINEL and the `control_pert: empty` benchmark convention.
CONTROL_SENTINEL = "empty"


# --------------------------------------------------------------------------- #
# Perturbation encoding (mirrors STATE's zarr_perturbation.py:62-120)
# --------------------------------------------------------------------------- #
def encode_pert(gene_pert) -> str:
    """Canonical ``gene_pert`` entry -> perturbation string.

    - empty            -> ``"empty"``
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


def encode_mol_pert(mol_pert) -> str:
    """Canonical ``mol_pert`` entry -> perturbation string (expects non-empty)."""
    if mol_pert is None or len(mol_pert) == 0:
        raise ValueError("encode_mol_pert expects a non-empty mol_pert list")
    ids = []
    for p in mol_pert:
        if p[0] != "compound":
            raise NotImplementedError(
                f"encode_mol_pert only supports compound perturbations, got {p[0]!r}"
            )
        ids.append(str(p[1]))
    return "+".join(sorted(ids))


def encode_obs_pert(gene_pert=None, mol_pert=None, *, negative_control: bool = False) -> str:
    """Encode perturbation identity from canonical ``gene_pert`` / ``mol_pert``."""
    gene_pert = gene_pert if gene_pert is not None else []
    mol_pert = mol_pert if mol_pert is not None else []
    if negative_control or not (len(gene_pert) or len(mol_pert)):
        return CONTROL_SENTINEL
    if len(gene_pert) and len(mol_pert):
        raise NotImplementedError(
            "zarr loader does not support mixed gene+mol perturbations on one cell"
        )
    if len(gene_pert):
        return encode_pert(gene_pert)
    return encode_mol_pert(mol_pert)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _factorize(values: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Sorted categories + int32 codes.

    Sorted order matches how the datamodule builds its global dictionaries
    (``{x: i for i, x in enumerate(sorted(...))}``) and STATE's
    ``generate_onehot_map``, so codes are stable across runs and machines.
    """
    categories = np.asarray(sorted(set(values)))
    code_of = {c: i for i, c in enumerate(categories.tolist())}
    codes = np.fromiter((code_of[v] for v in values), dtype=np.int32, count=len(values))
    return categories, codes


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
class ZarrMetadataCache:
    """Duck-typed twin of :class:`~src.data.metadata_cache.H5MetadataCache`.

    :param data_dir: directory holding ``X.zarr`` / ``obs.parquet`` / ``var.parquet``
    :param cell_type_key: obs column carrying the cell type (canonical: ``cell_type``)
    :param batch_col: obs column carrying the batch (canonical: ``batch_center``)
    :param control_pert: string to force onto control rows (canonical: ``empty``)
    """

    def __init__(
        self,
        data_dir,
        *,
        cell_type_key: str = "cell_type",
        batch_col: str = "batch_center",
        control_pert: str = CONTROL_SENTINEL,
        pert_col: Optional[str] = None,  # accepted for signature parity; unused
    ):
        self.data_dir = Path(data_dir)
        self.h5_path = str(self.data_dir)  # parity with H5MetadataCache.h5_path
        self.cell_type_key = cell_type_key
        self.batch_col = batch_col
        self.control_pert = control_pert

        obs_path = self.data_dir / "obs.parquet"
        if not obs_path.exists():
            raise FileNotFoundError(f"No obs.parquet under {self.data_dir}")
        obs = pl.read_parquet(obs_path)

        missing = [c for c in (cell_type_key, batch_col) if c not in obs.columns]
        if missing:
            raise ValueError(
                f"[{self.data_dir.name}] obs.parquet is missing configured column(s) "
                f"{sorted(missing)}; available: {obs.columns}"
            )

        n = obs.height
        self.n_cells = n

        # -- perturbation: derived, never read as a column --------------------
        gene_perts = obs["gene_pert"].to_list() if "gene_pert" in obs.columns else [None] * n
        mol_perts = obs["mol_pert"].to_list() if "mol_pert" in obs.columns else [None] * n
        has_neg = "negative_control" in obs.columns
        neg_ctrl = (
            [bool(v) for v in obs["negative_control"].to_list()] if has_neg else [False] * n
        )

        pert_str = np.array(
            [
                encode_obs_pert(g, m, negative_control=c)
                for g, m, c in zip(gene_perts, mol_perts, neg_ctrl)
            ],
            dtype=object,
        )
        is_control = (
            np.asarray(neg_ctrl, dtype=bool) if has_neg else (pert_str == CONTROL_SENTINEL)
        )
        # Force control rows onto the configured sentinel so the string is
        # consistent regardless of what gene_pert happened to contain.
        pert_str[is_control] = control_pert

        self.pert_categories, self.pert_codes = _factorize(pert_str.tolist())

        # -- cell type / batch ------------------------------------------------
        self.cell_type_categories, self.cell_type_codes = _factorize(
            [str(c) for c in obs[cell_type_key].to_list()]
        )
        self.batch_categories, self.batch_codes = _factorize(
            [str(b) for b in obs[batch_col].to_list()]
        )
        # Parity with H5MetadataCache, which sets this for the categorical branch.
        self.batch_is_categorical = True

        # -- control bookkeeping ---------------------------------------------
        idx = np.where(self.pert_categories == control_pert)[0]
        if idx.size == 0:
            raise ValueError(
                f"[{self.data_dir.name}] control_pert={control_pert!r} not present after "
                f"encoding; found {len(self.pert_categories)} categories, "
                f"e.g. {self.pert_categories[:3].tolist()}"
            )
        self.control_pert_code = int(idx[0])
        self.control_mask = is_control

        logger.info(
            "[%s] zarr obs: %d cells, %d perts (%d control), %d cell types, %d batches",
            self.data_dir.name,
            n,
            len(self.pert_categories),
            int(is_control.sum()),
            len(self.cell_type_categories),
            len(self.batch_categories),
        )

    # -- read-only accessors, parity with H5MetadataCache --------------------
    def get_batch_names(self, indices: np.ndarray) -> np.ndarray:
        """Return batch labels for the provided category indices."""
        return self.batch_categories[indices]

    def get_cell_type_names(self, indices: np.ndarray) -> np.ndarray:
        """Return cell-type labels for the provided category indices."""
        return self.cell_type_categories[indices]

    def get_pert_names(self, indices: np.ndarray) -> np.ndarray:
        """Return perturbation labels for the provided category indices."""
        return self.pert_categories[indices]


__all__ = [
    "CONTROL_SENTINEL",
    "ZarrMetadataCache",
    "encode_pert",
    "encode_mol_pert",
    "encode_obs_pert",
]
