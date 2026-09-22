"""Dataset classes for Virtual Cell Mode.

Provides:
- ControlMatcher / CellTypeControlMatcher: modular control-perturbation matching
- PerturbationBatch: collated batch dataclass
- VirtualCellDataset: in-memory eval dataset (standard Dataset)
- VirtualCellStreamingDataset: optim training dataset (IterableDataset)
"""

from __future__ import annotations

import gc
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import anndata
import h5py
import numpy as np
import torch
from scipy.sparse import issparse
from torch.utils.data import Dataset, IterableDataset


# ---------------------------------------------------------------------------
# Batch dataclass
# ---------------------------------------------------------------------------

@dataclass
class PerturbationBatch:
    """Collated batch for virtual cell training/evaluation."""

    control_expression: torch.Tensor  # (B, num_encoder_genes) log1p
    perturbed_expression: torch.Tensor  # (B, num_encoder_genes) log1p, zero-filled
    pert_indices: List[List[int]]  # pert2id indices per sample
    pert_names: List[str]  # perturbation gene names
    cell_types: List[str]  # cell type per sample
    gene_masks: torch.Tensor  # (B, num_encoder_genes) bool — True where gene exists
    is_sc: Optional[torch.Tensor] = None  # (B,) bool, True for SC samples


# ---------------------------------------------------------------------------
# Control matching
# ---------------------------------------------------------------------------

class ControlMatcher(ABC):
    """Protocol for sampling control cells to pair with perturbation cells.

    Subclass this to implement custom matching strategies (e.g. by batch,
    by embedding similarity, etc.).
    """

    @abstractmethod
    def sample_controls(self, cell_type: str, n: int = 5) -> torch.Tensor:
        """Return averaged control expression (num_genes,) for given cell_type.

        Args:
            cell_type: Cell type/line to match.
            n: Number of controls to sample and average.

        Returns:
            Averaged control expression vector (num_encoder_genes,).
        """
        ...


class CellTypeControlMatcher(ControlMatcher):
    """Match controls by cell type only. Average N random controls with replacement.

    Args:
        control_pool: {cell_type: Tensor(n_controls, num_encoder_genes)} in log1p space.
        seed: Random seed for reproducible control sampling.
    """

    def __init__(
        self,
        control_pool: Dict[str, torch.Tensor],
        seed: int = 42,
        group_pool: Optional[Dict[str, torch.Tensor]] = None,
        line_to_group: Optional[Dict[str, str]] = None,
        force_group: bool = False,
        batch_pool: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.control_pool = control_pool
        # {cell_line__batch: Tensor} for stream_batch_match (F97). Empty = unavailable,
        # in which case batch matching degrades to line matching and reports it.
        self.batch_pool = batch_pool or {}
        # Fallback pools, keyed by the coarse group (e.g. `cell_type`). Consulted ONLY
        # for a cell line that has no pool of its own. Feng's genome-wide screens ship
        # 12 and 36 NTCs in TOTAL across 34 and 24 donor lines, so without this they can
        # only be trained on where a donor line also appears in the targeted screen —
        # 46% and 33% of their perturbed cells respectively.
        self.group_pool = group_pool or {}
        self.line_to_group = line_to_group or {}
        # ABLATION ONLY: draw from the group pool even when the line has controls of its
        # own, i.e. abandon cell-line matching entirely in favour of the coarse group.
        # This is the degenerate end of the matching spectrum and exists to bracket the
        # hierarchical setting from the permissive side; it is never a default.
        self.force_group = force_group
        self._embedding_pool: Optional[Dict[str, torch.Tensor]] = None
        self._embedding_group_pool: Optional[Dict[str, torch.Tensor]] = None
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        for ct, pool in control_pool.items():
            if pool.shape[0] == 0:
                raise ValueError(f"No controls for cell_type='{ct}'")

    def has_pool(self, cell_type: str) -> bool:
        """True when this cell line can be paired at all — directly or by fallback."""
        if cell_type in self.control_pool:
            return True
        return self.line_to_group.get(cell_type) in self.group_pool

    def pool_for(self, cell_type: str, use_embeddings: bool = False) -> torch.Tensor:
        """Per-line pool if it exists, else the pooled controls of its group."""
        direct = self._embedding_pool if use_embeddings else self.control_pool
        group = self.line_to_group.get(cell_type)
        fallback = self._embedding_group_pool if use_embeddings else self.group_pool
        if self.force_group and group is not None and fallback and group in fallback:
            return fallback[group]
        if direct is not None and cell_type in direct:
            return direct[cell_type]
        if group is not None and fallback is not None and group in fallback:
            return fallback[group]
        raise KeyError(
            f"No control pool for cell line '{cell_type}' and no fallback group "
            f"(group='{group}'). Pass --control_fallback_col or drop the line."
        )

    def has_batch_pool(self, batch_id: str) -> bool:
        """True when this exact (cell_line, batch) has controls of its own."""
        return batch_id in self.batch_pool

    def pool_for_batch(self, batch_id: str) -> torch.Tensor:
        """Controls from the perturbed cell's OWN batch. Caller checks has_batch_pool."""
        return self.batch_pool[batch_id]

    def set_control_embeddings(self, embeddings: Dict[str, torch.Tensor]):
        """Set pre-computed encoder embeddings for controls.

        When set, sample_controls() returns averaged embeddings instead of
        averaged raw expression, and the model can skip the encoder.
        """
        self._embedding_pool = embeddings

    @property
    def has_embeddings(self) -> bool:
        return self._embedding_pool is not None

    def sample_controls(self, cell_type: str, n: int = 5) -> torch.Tensor:
        pool = self.pool_for(cell_type, use_embeddings=self._embedding_pool is not None)
        indices = self.rng.choice(pool.shape[0], size=n, replace=True)
        return pool[indices].mean(dim=0)


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

@dataclass
class FileMetadata:
    """Per-file metadata loaded at init time."""

    path: str
    cell_line: str
    n_obs: int
    n_vars: int
    var_ensembl_ids: List[str]  # ensembl gene IDs from var
    # Per-row obs metadata
    is_control: np.ndarray  # bool (n_obs,)
    pert_gene_names: np.ndarray  # str (n_obs,) — gene symbol from obs.gene_name
    # Gene alignment to encoder vocabulary
    src_indices: np.ndarray  # file gene positions that map to encoder vocab
    dst_indices: np.ndarray  # encoder vocab positions for those genes
    gene_mask: torch.Tensor  # (num_encoder_genes,) bool — genes present in this file
    batch_ids: Optional[np.ndarray] = None  # str (n_obs,) — batch IDs, loaded when pseudobulk=True
    obs_cell_lines: Optional[np.ndarray] = None  # str (n_obs,) — actual cell line per row
    # str (n_obs,) — coarse biological context used ONLY as a control-pool fallback key
    # (see `control_fallback_col`). Defaults to the cell line, i.e. no fallback.
    obs_control_groups: Optional[np.ndarray] = None
    # When True this file contributes its CONTROLS to the pools but none of its
    # perturbations to training. Used to lend one screen's NTCs to another screen that
    # has none of its own (Feng genome-wide ships 12-36 NTCs in total).
    controls_only: bool = False
    # Perturbations excluded from TRAINING in THIS FILE, by name. A (file, perturbation)
    # property, one level finer than `excluded_pert_lines` below and for the same reason: the
    # rhj_fewshot benchmark holds 724 perturbations out of HepG2 while the SAME symbols must
    # keep training from Jurkat and RPE1, so a global name-level exclusion would delete two
    # contexts of training data.
    #
    # ⚠️ Consumed ONLY by the training predicates (`_count_pert_cells`, the streaming
    # `__iter__`, and StreamingDatasetV2's static eligibility mask). The eval builders must
    # NOT consult it -- reading exactly these perturbations back is what makes them val/test.
    # That is also why this is not applied as a row filter in `_load_file_metadata`: a row
    # filter would delete the validation data too.
    excluded_perts: Optional[frozenset] = None
    # Cell lines whose perturbations IN THIS FILE must be skipped. Set by
    # require_control_batch_overlap, which is a property of a (file, cell line) pair, not
    # of a cell line: one donor line appears in both the targeted and the genome-wide
    # Feng screens, and only the targeted rows share a batch with that line's controls.
    # A line-level exclusion would therefore be a no-op — the line passes on its targeted
    # rows while its borrowed-control genome-wide rows sail through.
    excluded_pert_lines: Optional[set] = None


# Tokens accepted when `is_control` is stored as text rather than as a real bool.
_TRUE_TOKENS = {"true", "t", "1", "1.0", "yes", "y"}
_FALSE_TOKENS = {"false", "f", "0", "0.0", "no", "n", "", "nan", "none"}


def _coerce_is_control(col, path: str) -> np.ndarray:
    """Coerce an `is_control` obs column to a real boolean array.

    ⚠️ This exists because of a silent, total data loss. `feng_targeted_screen.h5ad`
    and `feng_genome_wide_nonfitness.h5ad` store `is_control` as a pandas CATEGORICAL
    OF THE STRINGS 'False'/'True'. The previous code did `.values.astype(bool)`, and
    `bool("False") is True` — so every row in both files was classified as a control
    and 725,055 perturbed cells across 5,335 perturbations were dropped from training
    while simultaneously polluting the NTC pools they were added to. Every "Feng"
    result before 2026-08-27 was really essential + `feng_genome_wide_fitness` alone,
    the one file of the three that happens to ship a numpy bool.

    Numeric and boolean columns keep their previous meaning exactly. Text is mapped by
    token, and an unrecognised token RAISES rather than guessing — a wrong guess here
    is invisible in every downstream number.
    """
    values = getattr(col, "values", col)
    # Categorical: decide on the CATEGORIES, then index, so we never stringify n_obs rows.
    categories = getattr(getattr(col, "cat", None), "categories", None)
    if categories is not None:
        mapped = np.array([_token_to_bool(c, path, "is_control") for c in categories])
        codes = np.asarray(col.cat.codes)
        if (codes < 0).any():
            raise ValueError(f"{path}: obs['is_control'] contains missing values.")
        return mapped[codes]
    values = np.asarray(values)
    if values.dtype == bool:
        return values
    if values.dtype.kind in "iuf":
        return values != 0
    return np.array([_token_to_bool(v, path, "is_control") for v in values], dtype=bool)


def _token_to_bool(value, path: str, what: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return value != 0
    token = str(value).strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    raise ValueError(
        f"{path}: cannot interpret obs['{what}'] value {value!r} as a boolean. "
        f"Recognised: {sorted(_TRUE_TOKENS)} / {sorted(_FALSE_TOKENS)}."
    )


# Transform-skip flags, set once per run by the pipeline (mirror the decoder/encoder
# skip_normalize/skip_log1p convention). Module-level because the transforms live in
# several standalone loaders with many call sites; one run = one process = one config.
_SKIP_NORMALIZE = False
_SKIP_LOG1P = False


def set_transform_flags(skip_normalize: bool, skip_log1p: bool) -> None:
    """Set whether data loaders skip NormalizePerCell / Log1p (data already transformed)."""
    global _SKIP_NORMALIZE, _SKIP_LOG1P
    _SKIP_NORMALIZE = bool(skip_normalize)
    _SKIP_LOG1P = bool(skip_log1p)
    if _SKIP_NORMALIZE or _SKIP_LOG1P:
        print(f"[dataset] transform-skip: skip_normalize={_SKIP_NORMALIZE}, "
              f"skip_log1p={_SKIP_LOG1P}")


# Combinatorial-perturbation decomposition. Off by default so every
# existing dataset keeps treating a label as one atomic token.
_DECOMPOSE_COMBOS = False


def set_combo_decomposition(enabled: bool) -> None:
    """Enable splitting 'GENE_A+GENE_B' labels into two separate pert2id indices."""
    global _DECOMPOSE_COMBOS
    _DECOMPOSE_COMBOS = bool(enabled)
    if _DECOMPOSE_COMBOS:
        print("[dataset] combo decomposition ON: 'A+B' -> [id_A, id_B]")


def resolve_pert_indices(pert_name, pert2id):
    """Map a perturbation label to a list of pert2id indices.

    Returns None when the label cannot be resolved, which callers treat as
    "skip this sample" (the pre-existing behaviour for unknown perturbations).

    Without decomposition an 'A+B' label is looked up whole and will almost always
    miss, silently dropping every combinatorial sample -- which is exactly what
    happened on a combinatorial corpus before this existed.
    """
    if pert_name in pert2id:
        return [pert2id[pert_name]]
    if _DECOMPOSE_COMBOS and "+" in str(pert_name):
        parts = [p for p in str(pert_name).split("+") if p and p != "ctrl"]
        if parts and all(p in pert2id for p in parts):
            return [pert2id[p] for p in parts]
    return None


def _load_file_metadata(
    path: str,
    encoder_gene_names: List[str],
    encoder_gene_to_idx: Dict[str, int],
    gene_name_col: str = "ensembl_gene_id",
    pert_gene_col: str = "gene_name",
    batch_col: Optional[str] = None,
    cell_line_cols: Optional[List[str]] = None,
    control_fallback_col: Optional[str] = None,
    controls_only: bool = False,
    restrict_gene_ids: Optional[Set[str]] = None,
) -> FileMetadata:
    """Load obs metadata and compute gene alignment for one h5ad file.

    `cell_line_cols` synthesizes the `cell_line` column by joining other obs columns
    with "_". The CD4 T-cell screen needs this: it ships `donor` (D1-D4) and `condition`
    (Rest/Stim8hr/Stim48hr) but no `cell_line`, and each donor x condition is a distinct
    biological context that must have its own control pool. `--cell_line_cols donor
    condition` gives `D1_Rest`, `D1_Stim8hr`, ... without rewriting a 1 TB dataset.
    Missing columns are skipped, so one setting works across a heterogeneous file list.

    `restrict_gene_ids` pins the gene axis to a fixed, pre-computed set (a frozen benchmark
    axis). It restricts BOTH the input scatter and `gene_mask`, i.e. both what the encoder is
    shown and what the loss counts, because `src_indices`/`dst_indices` drive both. The point
    is cross-context comparability: every file then presents the encoder with the same non-zero
    columns and the same zeros, so cell line is the only thing that differs between them.
    A file that cannot supply the whole axis is an error, not a warning -- a silently truncated
    axis changes the loss normalisation and makes two runs incomparable.
    """
    adata = anndata.read_h5ad(path, backed="r")

    # Expression gene names. Fall back to the var index when the configured column is
    # absent or explicitly requested via "__index__"/"index"/"var_names" (e.g. replogle
    # HVG data where gene symbols are the var index, not a column).
    if gene_name_col in ("__index__", "index", "var_names") or gene_name_col not in adata.var.columns:
        var_ensembl_ids = adata.var.index.astype(str).tolist()
    else:
        var_ensembl_ids = adata.var[gene_name_col].tolist()

    # Obs metadata. Derive is_control from the perturbation column when there is no
    # explicit is_control column (replogle: controls labelled "non-targeting" in obs.gene).
    pert_gene_names = adata.obs[pert_gene_col].values.astype(str)
    if "is_control" in adata.obs.columns:
        is_control = _coerce_is_control(adata.obs["is_control"], path)
    else:
        is_control = (pert_gene_names == "non-targeting")
    # A training file with no perturbations is never intentional — it is the signature
    # of a mis-parsed control flag or a wrong `pert_gene_col`, and it costs the whole
    # file silently. Declared controls-only lenders are the one legitimate case.
    if not controls_only and not (~is_control).any():
        raise ValueError(
            f"{path}: 0 of {len(is_control)} cells are perturbed — every row was read as "
            f"a control. Check obs['is_control'] and pert_gene_col='{pert_gene_col}'. "
            f"If this file is meant to lend only its controls, pass it to "
            f"--control_only_data_paths instead of --data_paths."
        )
    # Synthesize `cell_line` ONLY when the file does not ship one. This is a FALLBACK,
    # never an override: the Feng files carry both a real `cell_line` and a `donor`
    # column, so overriding whenever the source columns happen to exist would silently
    # relabel Feng's 19 donor lines by `donor` and break its control pooling — while
    # looking like it worked.
    if cell_line_cols and "cell_line" not in adata.obs.columns:
        present = [c for c in cell_line_cols if c in adata.obs.columns]
        if present:
            synth = adata.obs[present[0]].values.astype(str)
            for c in present[1:]:
                synth = np.char.add(np.char.add(synth.astype(str), "_"),
                                    adata.obs[c].values.astype(str))
            adata.obs["cell_line"] = synth
            print(f"  synthesized cell_line from {present}: "
                  f"{len(set(synth))} value(s), e.g. {sorted(set(synth))[:3]}")
    if "cell_line" not in adata.obs.columns:
        raise ValueError(
            f"{path} has no 'cell_line' column and none could be synthesized. "
            f"Pass --cell_line_cols with columns present in obs "
            f"(available: {list(adata.obs.columns)})."
        )
    cell_lines = adata.obs["cell_line"].unique()
    cell_line = str(cell_lines[0])
    if len(cell_lines) > 1:
        obs_cell_lines = adata.obs["cell_line"].values.astype(str)
        print(f"  Multi-cell-line file: {sorted(str(c) for c in cell_lines)}")
    else:
        obs_cell_lines = np.full(adata.n_obs, cell_line, dtype=object)

    # Coarse control-pool fallback key. Defaults to the cell line itself, which makes the
    # fallback a no-op; set `control_fallback_col` (e.g. `cell_type`) to let a cell line
    # with no NTCs of its own draw from the pooled controls of its biological group.
    if control_fallback_col and control_fallback_col in adata.obs.columns:
        obs_control_groups = adata.obs[control_fallback_col].values.astype(str)
    else:
        if control_fallback_col:
            print(f"  ⚠️  control_fallback_col='{control_fallback_col}' absent from "
                  f"{Path(path).name}; its lines get no fallback (cell-line matching only)")
        obs_control_groups = np.asarray(obs_cell_lines, dtype=str)

    n_obs = adata.n_obs
    n_vars = adata.n_vars

    # Batch IDs (for pseudobulk mode)
    batch_ids = None
    if batch_col is not None:
        if batch_col not in adata.obs.columns:
            raise ValueError(
                f"batch_col='{batch_col}' not found in obs columns of {path}. "
                f"Available: {list(adata.obs.columns)}"
            )
        raw_batch = adata.obs[batch_col].values.astype(str)
        # Prefix with per-row cell_line to ensure global uniqueness across files
        batch_ids = np.array([
            f"{cl}__{b}" for cl, b in zip(obs_cell_lines, raw_batch)
        ])

    # Gene alignment: data genes → encoder vocabulary, optionally pinned to a frozen axis
    allowed = None if restrict_gene_ids is None else set(restrict_gene_ids)
    src_indices = []
    dst_indices = []
    for j, gene in enumerate(var_ensembl_ids):
        if gene in encoder_gene_to_idx and (allowed is None or gene in allowed):
            src_indices.append(j)
            dst_indices.append(encoder_gene_to_idx[gene])
    src_indices = np.array(src_indices, dtype=np.int64)
    dst_indices = np.array(dst_indices, dtype=np.int64)

    gene_mask = torch.zeros(len(encoder_gene_names), dtype=torch.bool)
    gene_mask[dst_indices] = True

    if allowed is not None:
        # The whole point of a frozen axis is that every context presents the SAME columns.
        # If one file is short, the masks differ, the pooled loss normalisation differs, and
        # the contexts stop being comparable -- silently. So this is a raise.
        got = int(gene_mask.sum())
        if got != len(allowed):
            present = {var_ensembl_ids[j] for j in src_indices}
            missing = sorted(allowed - present)
            raise ValueError(
                f"{path}: the frozen benchmark axis has {len(allowed)} genes but only {got} "
                f"align here ({len(missing)} missing, e.g. {missing[:10]}). A partial axis "
                f"would give this context a different gene mask from the others and make the "
                f"loss normalisation incomparable. Rebuild the axis with "
                f"scripts/build_rhj_fewshot.py over exactly the files you intend to train on."
            )

    del adata
    gc.collect()

    return FileMetadata(
        path=path,
        cell_line=cell_line,
        n_obs=n_obs,
        n_vars=n_vars,
        var_ensembl_ids=var_ensembl_ids,
        is_control=is_control,
        pert_gene_names=pert_gene_names,
        src_indices=src_indices,
        dst_indices=dst_indices,
        gene_mask=gene_mask,
        batch_ids=batch_ids,
        obs_cell_lines=obs_cell_lines,
        obs_control_groups=obs_control_groups,
        controls_only=controls_only,
    )


def _load_expression_rows(
    path: str,
    row_indices: np.ndarray,
    n_vars: int,
    src_indices: np.ndarray,
    dst_indices: np.ndarray,
    num_encoder_genes: int,
    target_sum: float,
) -> torch.Tensor:
    """Load specific rows from h5ad, apply NormalizePerCell+Log1p, align to encoder vocab.

    Returns:
        Tensor (n_rows, num_encoder_genes) in log1p space.
    """
    with h5py.File(path, "r") as f:
        X = f["X"]
        # Read rows in sorted order for contiguous I/O
        sorted_idx = np.sort(row_indices)
        # NOT hasattr(X, "indptr"): an h5py Group exposes CSR components via
        # __getitem__ (X["indptr"]), never as attributes, so hasattr is ALWAYS False and
        # every CSR file silently took the dense path and died in read_direct. Latent
        # because all four datasets used so far are dense `array`; the CD4 T-cell screen
        # is csr_matrix and would have failed on the first read.
        if isinstance(X, h5py.Group) and "indptr" in X:
            # Sparse CSR — read as dense
            raw = np.zeros((len(sorted_idx), n_vars), dtype=np.float32)
            for i, ridx in enumerate(sorted_idx):
                _ip = _unwrap_indptr(np.asarray(X["indptr"][ridx:ridx + 2],
                                                 dtype=np.int64))
                start, end = int(_ip[0]), int(_ip[1])
                cols = X["indices"][start:end]
                vals = X["data"][start:end]
                raw[i, cols] = vals
        else:
            # Dense
            raw = np.empty((len(sorted_idx), n_vars), dtype=np.float32)
            for i, ridx in enumerate(sorted_idx):
                X.read_direct(raw, np.s_[ridx : ridx + 1, :], np.s_[i : i + 1, :])

    # NormalizePerCell + Log1p (skipped when data is already transformed)
    if not _SKIP_NORMALIZE:
        row_sums = raw.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        raw = raw / row_sums * target_sum
    if not _SKIP_LOG1P:
        raw = np.log1p(raw)

    # Align to encoder vocabulary
    out = np.zeros((len(sorted_idx), num_encoder_genes), dtype=np.float32)
    out[:, dst_indices] = raw[:, src_indices]

    # Restore original row order if needed
    if not np.array_equal(row_indices, sorted_idx):
        sort_map = np.argsort(np.argsort(row_indices))
        # sorted_idx maps to row_indices via inverse sort
        inv = np.empty_like(sort_map)
        inv[np.argsort(row_indices)] = np.arange(len(row_indices))
        out = out[inv]

    return torch.from_numpy(out)


def _unwrap_indptr(a: np.ndarray) -> np.ndarray:
    """Repair int32-wrapped CSR `indptr` values.

    ALL TWELVE CD4 T-cell files ship a corrupt `indptr`: declared int64, final value
    correct, but a contiguous band of entries stored as int32 two's-complement and
    therefore NEGATIVE. Measured:

        D1_Rest      1,481 negatives at rows 566,756..568,236
        D3_Rest      2,070 negatives at rows 547,739..549,808
        D2_Stim8hr     448 negatives at rows 471,555..472,002

    Each band sits exactly where the cumulative nnz crosses 2**31, so the true value is
    `v + 2**32`. Applying that elementwise restores a strictly monotonic array whose last
    entry equals `len(indices)` exactly — verified against that ground truth on three
    files, with per-row nnz landing in a sane 2,000-9,547.

    Left unrepaired this reads as a chunk spanning 4,327,513,523 entries (indptr[0] =
    -2,143,255,827), which is what killed the big-corpus runs. The correction is a pure
    per-value function, so it is safe to apply to any slice regardless of whether the
    slice straddles the band.
    """
    if a.size and (a < 0).any():
        a = np.where(a < 0, a + (1 << 32), a)
    return a


def _load_expression_chunk(
    path: str,
    start: int,
    end: int,
    n_vars: int,
    src_indices: np.ndarray,
    dst_indices: np.ndarray,
    num_encoder_genes: int,
    target_sum: float,
    buf: Optional[np.ndarray] = None,
    keep_local: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Load a contiguous chunk of rows from h5ad, normalize, align.

    Optimized for sequential reading — uses read_direct for zero-copy I/O.

    `keep_local` is an optional array of CHUNK-LOCAL row offsets to retain. Rows are
    dropped BEFORE the gene-alignment expansion, which is the expensive step: the output
    is `num_encoder_genes` (44,728) wide, so a full 8192-row chunk allocates ~1.5 GB even
    when the caller is about to discard most of it. With a per-perturbation cap of 16 the
    streaming loader keeps roughly an eighth of the perturbed rows, so filtering first
    removes most of that allocation.

    Returns:
        Tensor (n_kept, num_encoder_genes) in log1p space.
    """
    n_rows = end - start
    with h5py.File(path, "r") as f:
        X = f["X"]
        # NOT hasattr(X, "indptr"): an h5py Group exposes CSR components via
        # __getitem__ (X["indptr"]), never as attributes, so hasattr is ALWAYS False and
        # every CSR file silently took the dense path and died in read_direct. Latent
        # because all four datasets used so far are dense `array`; the CD4 T-cell screen
        # is csr_matrix and would have failed on the first read.
        if isinstance(X, h5py.Group) and "indptr" in X:
            # Sparse CSR — THREE h5py reads for the whole chunk, then one vectorised
            # scatter. The previous per-row loop issued three reads PER ROW
            # (indptr, indices, data), i.e. ~24,000 separate HDF5 calls for an 8192-row
            # chunk. That was the dominant cost of the big-corpus run: 168 samples/s
            # against ~10,000 on the cached path, and 6 h for one epoch.
            ip = _unwrap_indptr(np.asarray(X["indptr"][start:end + 1], dtype=np.int64))
            if len(ip) != n_rows + 1:
                raise RuntimeError(
                    f"CSR indptr slice returned {len(ip)} entries for rows "
                    f"[{start}:{end}] (expected {n_rows + 1}) in {path}"
                )
            s0, e0 = int(ip[0]), int(ip[-1])
            span = e0 - s0
            raw = np.zeros((n_rows, n_vars), dtype=np.float32)
            if span > 0:
                # A chunk of `chunk_size` rows over `n_vars` genes cannot hold more than
                # n_rows*n_vars nonzeros. A span above that means the indptr slice is not
                # what we think it is, and the read below would silently return the wrong
                # thing — an 8192-row chunk once claimed a 4,327,513,523-element span
                # (just above 2**32) while the index read came back EMPTY, which then
                # surfaced as an unhelpful broadcast error in the scatter.
                if span > n_rows * n_vars:
                    raise RuntimeError(
                        f"CSR span {span:,} exceeds the maximum possible "
                        f"{n_rows * n_vars:,} for rows [{start}:{end}] of {path} "
                        f"(indptr[0]={s0:,}, indptr[-1]={e0:,}). The indptr slice is "
                        f"corrupt or misaligned."
                    )
                cols = np.asarray(X["indices"][s0:e0])
                vals = np.asarray(X["data"][s0:e0], dtype=np.float32)
                # HDF5 can return short/empty for very large hyperslabs; catch it here
                # rather than letting the scatter fail with a shape-mismatch message that
                # says nothing about which file or offset was involved.
                if len(cols) != span or len(vals) != span:
                    raise RuntimeError(
                        f"CSR short read in {path} rows [{start}:{end}]: asked for "
                        f"{span:,} entries at offset {s0:,}, got indices={len(cols):,} "
                        f"data={len(vals):,}."
                    )
                rows = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(ip))
                raw[rows, cols] = vals
        else:
            # Dense — use pre-allocated buffer if provided
            if buf is not None and buf.shape[0] >= n_rows and buf.shape[1] == n_vars:
                raw = buf[:n_rows]
                X.read_direct(raw, np.s_[start:end, :])
            else:
                raw = X[start:end].astype(np.float32)

    # Drop unwanted rows BEFORE normalisation and gene alignment. Both are O(rows), and
    # the alignment allocates a num_encoder_genes-wide array, so filtering here is where
    # the saving is. Library-size normalisation is per row, so subsetting first is exact.
    if keep_local is not None:
        raw = raw[keep_local]
        n_rows = raw.shape[0]

    # NormalizePerCell + Log1p (skipped when data is already transformed)
    if not _SKIP_NORMALIZE:
        row_sums = raw.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        raw = raw / row_sums * target_sum
    if not _SKIP_LOG1P:
        np.log1p(raw, out=raw)

    # Align to encoder vocabulary
    out = np.zeros((n_rows, num_encoder_genes), dtype=np.float32)
    out[:, dst_indices] = raw[:, src_indices]

    return torch.from_numpy(out)


# ---------------------------------------------------------------------------
# Pseudobulk aggregation
# ---------------------------------------------------------------------------


def _aggregate_pseudobulk_for_file(
    fm: FileMetadata,
    target_sum: float,
    num_encoder_genes: int,
    chunk_size: int = 8192,
) -> Tuple[Dict[str, np.ndarray], Dict[Tuple[str, str], np.ndarray]]:
    """Read raw counts, aggregate per (batch, pert/control), normalize, align.

    Accumulates raw count sums in file-native gene space (n_vars), then
    normalizes each mean profile (library-size → log1p) and aligns to
    encoder vocabulary.  Uses float64 sums for precision.

    Args:
        fm: FileMetadata with batch_ids populated.
        target_sum: Library-size normalization target.
        num_encoder_genes: Width of encoder vocabulary.
        chunk_size: Rows per HDF5 read.

    Returns:
        ctrl_profiles: {batch_id: aligned_log1p_profile (num_encoder_genes,)}
        pert_profiles: {(batch_id, gene_name): aligned_log1p_profile}
    """
    assert fm.batch_ids is not None, (
        f"batch_ids not loaded for {fm.path}. "
        "Pass batch_col to _load_file_metadata."
    )

    # Accumulators in file-native gene space
    ctrl_sums: Dict[str, np.ndarray] = {}
    ctrl_counts: Dict[str, int] = {}
    pert_sums: Dict[Tuple[str, str], np.ndarray] = {}
    pert_counts: Dict[Tuple[str, str], int] = {}

    with h5py.File(fm.path, "r") as f:
        X = f["X"]
        is_sparse = isinstance(X, h5py.Group) and "indptr" in X  # see note above

        for start in range(0, fm.n_obs, chunk_size):
            end = min(start + chunk_size, fm.n_obs)
            n_rows = end - start

            # Read raw counts (no normalization, no log1p)
            if is_sparse:
                raw = np.zeros((n_rows, fm.n_vars), dtype=np.float32)
                for i in range(n_rows):
                    ridx = start + i
                    s, e = X["indptr"][ridx], X["indptr"][ridx + 1]
                    cols = X["indices"][s:e]
                    vals = X["data"][s:e]
                    raw[i, cols] = vals
            else:
                raw = X[start:end].astype(np.float32)

            chunk_batch = fm.batch_ids[start:end]
            chunk_ctrl = fm.is_control[start:end]
            chunk_pnames = fm.pert_gene_names[start:end]

            # --- Controls: group by batch, vectorized sum ---
            ctrl_mask = chunk_ctrl
            if ctrl_mask.any():
                cb = chunk_batch[ctrl_mask]
                cr = raw[ctrl_mask]
                for bid in np.unique(cb):
                    bid_mask = cb == bid
                    contribution = cr[bid_mask].sum(axis=0)
                    if bid not in ctrl_sums:
                        ctrl_sums[bid] = contribution.astype(np.float64)
                        ctrl_counts[bid] = int(bid_mask.sum())
                    else:
                        ctrl_sums[bid] += contribution
                        ctrl_counts[bid] += int(bid_mask.sum())

            # --- Perturbations: group by (batch, gene_name), vectorized sum ---
            pert_mask = ~chunk_ctrl
            if pert_mask.any():
                pb = chunk_batch[pert_mask]
                pn = chunk_pnames[pert_mask]
                pr = raw[pert_mask]

                # Build composite key for fast grouping (null byte separator)
                keys = np.array(
                    [f"{b}\0{n}" for b, n in zip(pb, pn)], dtype=object
                )
                order = np.argsort(keys)
                sorted_keys = keys[order]
                sorted_raw = pr[order]

                unique_keys, group_starts = np.unique(
                    sorted_keys, return_index=True
                )
                group_ends = np.append(group_starts[1:], len(sorted_keys))

                for g in range(len(unique_keys)):
                    s, e = int(group_starts[g]), int(group_ends[g])
                    bid, gname = unique_keys[g].split("\0", 1)
                    contribution = sorted_raw[s:e].sum(axis=0)
                    key = (bid, gname)
                    if key not in pert_sums:
                        pert_sums[key] = contribution.astype(np.float64)
                        pert_counts[key] = e - s
                    else:
                        pert_sums[key] += contribution
                        pert_counts[key] += e - s

    # --- Compute means → normalize → log1p → align ---
    def _finalize(raw_sum: np.ndarray, count: int) -> np.ndarray:
        mean = (raw_sum / count).astype(np.float32)
        if not _SKIP_NORMALIZE:
            total = mean.sum()
            if total > 0:
                mean = mean / total * target_sum
        if not _SKIP_LOG1P:
            mean = np.log1p(mean)
        aligned = np.zeros(num_encoder_genes, dtype=np.float32)
        aligned[fm.dst_indices] = mean[fm.src_indices]
        return aligned

    ctrl_profiles = {
        bid: _finalize(s, ctrl_counts[bid])
        for bid, s in ctrl_sums.items()
    }
    pert_profiles = {
        key: _finalize(s, pert_counts[key])
        for key, s in pert_sums.items()
    }

    return ctrl_profiles, pert_profiles


class PseudobulkDataset(Dataset):
    """In-memory dataset with pre-aggregated pseudobulk profiles.

    Each sample is a deterministic (batch-control, batch-perturbation) pair.
    No random sampling — all profiles are pre-computed from raw count averages.
    """

    def __init__(
        self,
        control_profiles: torch.Tensor,
        perturbed_profiles: torch.Tensor,
        pert_gene_names: List[str],
        pert_indices: List[List[int]],
        cell_types: List[str],
        gene_masks: torch.Tensor,
        batch_ids: Optional[List[str]] = None,
    ):
        self.control_profiles = control_profiles      # (N, num_encoder_genes)
        self.perturbed_profiles = perturbed_profiles  # (N, num_encoder_genes)
        self.pert_gene_names = list(pert_gene_names)
        self.pert_indices = list(pert_indices)
        self.cell_types = list(cell_types)
        self.gene_masks = gene_masks
        self.batch_ids = batch_ids
        self.is_sc_flags: Optional[List[bool]] = None

    def __len__(self) -> int:
        return self.control_profiles.shape[0]

    def __getitem__(self, idx: int) -> Tuple:
        base = (
            self.control_profiles[idx],
            self.perturbed_profiles[idx],
            self.pert_indices[idx],
            self.pert_gene_names[idx],
            self.cell_types[idx],
            self.gene_masks[idx],
        )
        if self.is_sc_flags is not None:
            return base + (self.is_sc_flags[idx],)
        return base

    @staticmethod
    def collate_fn(batch) -> PerturbationBatch:
        if len(batch[0]) == 7:
            controls, perts, pert_idxs, names, cell_types, masks, is_sc_flags = zip(*batch)
            return PerturbationBatch(
                control_expression=torch.stack(controls),
                perturbed_expression=torch.stack(perts),
                pert_indices=list(pert_idxs),
                pert_names=list(names),
                cell_types=list(cell_types),
                gene_masks=torch.stack(masks),
                is_sc=torch.tensor(is_sc_flags, dtype=torch.bool),
            )
        controls, perts, pert_idxs, names, cell_types, masks = zip(*batch)
        return PerturbationBatch(
            control_expression=torch.stack(controls),
            perturbed_expression=torch.stack(perts),
            pert_indices=list(pert_idxs),
            pert_names=list(names),
            cell_types=list(cell_types),
            gene_masks=torch.stack(masks),
        )

    # Compatibility properties for baselines.py
    @property
    def _n_pert(self) -> int:
        return self.perturbed_profiles.shape[0]

    @property
    def expression_data(self) -> torch.Tensor:
        return self.perturbed_profiles


# ---------------------------------------------------------------------------
# In-memory eval dataset
# ---------------------------------------------------------------------------

class VirtualCellDataset(Dataset):
    """In-memory dataset for virtual cell evaluation.

    Loads all perturbation cells for a given split into memory.
    Controls come from the shared ControlMatcher (in-memory pool).

    Args:
        expression_data: (N, num_encoder_genes) tensor in log1p space.
        pert_gene_names: List of perturbation gene names (symbols) per sample.
        pert_indices: List of pert2id integer indices per sample.
        cell_types: List of cell type strings per sample.
        gene_masks: (N, num_encoder_genes) bool tensor — genes present in source file.
        control_matcher: ControlMatcher for sampling controls.
        n_controls: Number of controls to average per sample.
        include_controls_as_training: If True, add control identity samples.
        control_expression: (C, num_encoder_genes) control data for identity samples.
        control_cell_types: List of cell types for control samples.
        control_gene_mask: Gene mask for controls.
    """

    def __init__(
        self,
        expression_data: torch.Tensor,
        pert_gene_names: List[str],
        pert_indices: List[List[int]],
        cell_types: List[str],
        gene_masks: torch.Tensor,
        control_matcher: ControlMatcher,
        n_controls: int = 5,
        include_controls_as_training: bool = False,
        control_expression: Optional[torch.Tensor] = None,
        control_cell_types: Optional[List[str]] = None,
        control_gene_mask: Optional[torch.Tensor] = None,
    ):
        self.expression_data = expression_data
        self.pert_gene_names = list(pert_gene_names)
        self.pert_indices = list(pert_indices)
        self.cell_types = list(cell_types)
        self.gene_masks = gene_masks
        self.control_matcher = control_matcher
        self.n_controls = n_controls

        self._n_pert = expression_data.shape[0]

        # Optionally append control samples as identity-mapping training data
        if include_controls_as_training and control_expression is not None:
            n_ctrl = control_expression.shape[0]
            self.expression_data = torch.cat(
                [self.expression_data, control_expression], dim=0
            )
            self.pert_gene_names.extend(["non-targeting"] * n_ctrl)
            self.pert_indices.extend([[-1]] * n_ctrl)
            self.cell_types.extend(control_cell_types or ["unknown"] * n_ctrl)
            if control_gene_mask is not None:
                ctrl_masks = control_gene_mask.unsqueeze(0).expand(n_ctrl, -1)
            else:
                ctrl_masks = torch.ones(n_ctrl, gene_masks.shape[1], dtype=torch.bool)
            self.gene_masks = torch.cat([self.gene_masks, ctrl_masks], dim=0)

    def __len__(self) -> int:
        return self.expression_data.shape[0]

    def __getitem__(self, idx: int) -> Tuple:
        cell_type = self.cell_types[idx]
        control_expr = self.control_matcher.sample_controls(cell_type, self.n_controls)
        pert_expr = self.expression_data[idx]
        pert_idx = self.pert_indices[idx]
        pert_name = self.pert_gene_names[idx]
        gene_mask = self.gene_masks[idx]

        return control_expr, pert_expr, pert_idx, pert_name, cell_type, gene_mask

    @staticmethod
    def collate_fn(batch) -> PerturbationBatch:
        controls, perts, pert_idxs, names, cell_types, masks = zip(*batch)
        return PerturbationBatch(
            control_expression=torch.stack(controls),
            perturbed_expression=torch.stack(perts),
            pert_indices=list(pert_idxs),
            pert_names=list(names),
            cell_types=list(cell_types),
            gene_masks=torch.stack(masks),
        )


# ---------------------------------------------------------------------------
# Streaming (optim) training dataset
# ---------------------------------------------------------------------------

def _stable_ctrl_index(file_idx: np.ndarray, row_idx: np.ndarray,
                       n_pool: int, seed: int) -> np.ndarray:
    """Deterministic control index per perturbed cell — splitmix64 over (file, row).

    Used by `fixed_pairs`. It must depend ONLY on the cell's identity: not on the epoch,
    not on the chunk visiting order, and not on how many cells happened to share the batch.
    A plain `rng.choice` fails all three, which is what makes the streaming pairing
    re-randomise every epoch.

    uint64 arithmetic wraps silently in numpy, which is exactly the mixing this needs.
    """
    with np.errstate(over="ignore"):     # wrap-around IS the mixing, not an error
        x = (np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)) \
            ^ (file_idx << np.uint64(40)) ^ row_idx
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return (x % np.uint64(n_pool)).astype(np.int64)


class VirtualCellStreamingDataset(IterableDataset):
    """Optim-mode streaming dataset for virtual cell training. SINGLE-CELL ONLY.

    Perturbation cells are streamed from h5ad files in chunks; only the control pools
    live in memory. This is what makes corpora that cannot be cached tractable — the
    non-streaming `build_mixed_dataset` path holds every pair in RAM (909 GB of cache
    for four datasets), which does not survive adding the ~20 M-cell CD4 T-cell screen.

    **No averaging, anywhere.** `n_controls` must be 1: each perturbed cell is paired
    with exactly ONE control cell, matching the single-cell convention the H1 work is
    built on. Averaging n>1 controls would make the input a pseudobulk and make every
    number incomparable with §5g-§5i of `VCCv2/H1_HELDOUT_TRANSFER.md`.

    **Controls match on `cell_line` only, never on batch.** Batch matching is
    deliberately dropped: it is infeasible across the Feng files (the targeted screen's
    controls carry `feng_ts-*` batch ids while the genome-wide perturbations carry
    `feng_nfg-*`/`feng_fg-*`, with zero overlap), and matching on cell line is what
    lets ~7,000 genome-wide Feng perturbations borrow the targeted screen's controls.
    The cost is that a Feng genome-wide delta carries a whole-experiment offset; this
    is accepted knowingly rather than corrected.

    Args:
        file_metas: List of FileMetadata for each h5ad file.
        pert2id: Gene symbol → integer ID mapping.
        split_pert_names: Set of perturbation gene names in the current split.
        control_matcher: ControlMatcher with in-memory control pools.
        encoder_gene_names: Ordered encoder gene vocabulary.
        target_sum: Library-size normalization target.
        n_controls: MUST be 1. Any other value raises.
        batch_size: Samples per yielded batch.
        chunk_size: Rows per HDF5 read.
        buffer_chunks: Chunks to combine for shuffling.
        shuffle: Whether to shuffle.
        seed: Random seed.
        include_controls_as_training: Add control identity samples.
        control_identity_mode: "exact" (input cell IS the target cell) or "paired"
            (a different control from the same line). `exact` was unreachable in this
            class before — it always drew a random control — and `ess_exact` is the
            best-measured arm, so it is now explicit.
        train_cell_lines: Cell lines used for training (filter files).
        max_cells_per_pert: Cap on perturbed cells kept per (cell_line, perturbation)
            per epoch, 0 = unlimited. The T-cell screen contributes ~20 M perturbed
            cells against ~540 k samples/epoch today; without a cap one epoch is ~40x
            longer. Applied by hashing (cell_line, pert, epoch) so the retained subset
            differs each epoch — over several epochs the model still sees the whole
            corpus, it just does not see all of it at once.
        balance_sources: Interleave chunks across files so each buffer group — and
            therefore each batch — draws from many sources rather than whichever file
            the shuffled chunk order happened to land on. Without this the T-cell
            screen, at ~95% of all chunks, would dominate nearly every batch.
    """

    def __init__(
        self,
        file_metas: List[FileMetadata],
        pert2id: Dict[str, int],
        split_pert_names: Set[str],
        control_matcher: ControlMatcher,
        encoder_gene_names: List[str],
        target_sum: float = 1e4,
        n_controls: int = 5,
        batch_size: int = 64,
        chunk_size: int = 8192,
        buffer_chunks: int = 4,
        shuffle: bool = True,
        seed: int = 42,
        include_controls_as_training: bool = True,
        train_cell_lines: Optional[List[str]] = None,
        control_identity_mode: str = "paired",
        max_cells_per_pert: int = 0,
        min_cells_per_pert: int = 0,
        balance_sources: bool = True,
        extra_samples: Optional[Tuple] = None,
        fixed_pairs: bool = False,
        batch_match: bool = False,
    ):
        if n_controls != 1:
            raise ValueError(
                f"VirtualCellStreamingDataset is single-cell only: n_controls must be 1, "
                f"got {n_controls}. Averaging controls would make the input a pseudobulk."
            )
        self.control_identity_mode = control_identity_mode
        self.fixed_pairs = fixed_pairs
        self.batch_match = batch_match
        # Counted and reported per epoch, mirroring the cached path's own
        # "N batch-matched, M cell-line-matched" line so the two are directly comparable.
        self._bm_matched = 0
        self._bm_fellback = 0
        self.max_cells_per_pert = max_cells_per_pert
        # DEPTH FLOOR — drop (cell_line, perturbation) groups with too few cells.
        # Measured median cells per (line, pert): rpe1 72 · hepg2 45 · jurkat 83 ·
        # k562_e 121 · feng_targeted 74 — but feng_genome_wide_nonfitness **1** (74% of
        # its groups have <=2 cells) and feng_genome_wide_fitness **2** (58%). A 1-cell
        # group's "perturbation delta" is one cell minus one control, i.e. the effect plus
        # full single-cell noise — and for the Feng genome-wide files the control is
        # BORROWED from a different experiment, adding a whole-experiment offset on top.
        # Those pairs are the prime suspect for why adding Feng made H1 worse, and this
        # floor is how that is tested.
        self.min_cells_per_pert = min_cells_per_pert
        self.balance_sources = balance_sources
        # Pre-paired samples that live in no h5ad on the file list — specifically the H1
        # control-in/control-out identity rows. H1 is STRUCTURALLY excluded from
        # data_paths (that is the holdout), so its controls cannot be streamed; but they
        # must still be trained on, since "you get the unseen line's NTCs and nothing
        # else" is the task. On the cached path this was _append_h1_control_identity;
        # streaming needs its own channel or H1 controls silently vanish from training.
        # Tuple layout matches build_h1_identity_samples:
        #   (ctrl, tgt, pert_names, pert_indices, cell_types, gene_mask)
        self.extra_samples = extra_samples
        self._n_extra = 0 if extra_samples is None else len(extra_samples[0])
        self.file_metas = file_metas
        self.pert2id = pert2id
        self.split_pert_names = split_pert_names
        self.control_matcher = control_matcher
        self.num_encoder_genes = len(encoder_gene_names)
        self.target_sum = target_sum
        self.n_controls = n_controls
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.buffer_chunks = buffer_chunks
        self.shuffle = shuffle
        self.seed = seed
        self.include_controls_as_training = include_controls_as_training
        self.train_cell_lines = train_cell_lines

        # Filter files to training cell lines (keep files with any overlap)
        if train_cell_lines is not None:
            train_cl_set = set(train_cell_lines)
            self.active_metas = [
                fm for fm in file_metas
                if set(fm.obs_cell_lines).intersection(train_cl_set)
            ]
        else:
            self.active_metas = list(file_metas)

        # Build global chunk list: (meta_idx, chunk_start, chunk_end)
        self._chunks = []
        for mi, fm in enumerate(self.active_metas):
            for start in range(0, fm.n_obs, chunk_size):
                end = min(start + chunk_size, fm.n_obs)
                self._chunks.append((mi, start, end))

        # Pre-allocate read buffers
        max_vars = max(fm.n_vars for fm in self.active_metas)
        self._read_buf = np.empty((chunk_size, max_vars), dtype=np.float32)

        # Depth floor is epoch-INDEPENDENT (it is a property of the data), so resolve it
        # once here rather than per epoch. `_row_ok[mi][row]` is False for perturbed rows
        # whose (cell_line, perturbation) group is under the floor.
        self._row_ok: Dict[int, np.ndarray] = {}
        if self.min_cells_per_pert > 0:
            tot_drop = tot_keep = 0
            for mi, fm in enumerate(self.active_metas):
                pert = ~np.asarray(fm.is_control, dtype=bool)
                keys = np.char.add(
                    np.char.add(np.asarray(fm.obs_cell_lines, dtype=str), "|"),
                    np.asarray(fm.pert_gene_names, dtype=str))
                uniq, inv, cnt = np.unique(keys[pert], return_inverse=True, return_counts=True)
                ok = np.ones(fm.n_obs, dtype=bool)               # controls always kept
                ok[np.flatnonzero(pert)] = cnt[inv] >= self.min_cells_per_pert
                self._row_ok[mi] = ok
                d = int((~ok).sum()); tot_drop += d; tot_keep += int(pert.sum()) - d
                if d:
                    print(f"  [min_cells={self.min_cells_per_pert}] "
                          f"{Path(fm.path).name}: dropped {d:,}/{int(pert.sum()):,} "
                          f"perturbed cells in under-depth groups")
            print(f"  [min_cells={self.min_cells_per_pert}] TOTAL kept {tot_keep:,}, "
                  f"dropped {tot_drop:,} perturbed cells")

        # Every cell line that will actually be streamed must be pairable BEFORE the first
        # batch, not at the point of a KeyError 40 minutes into an epoch. Silently
        # dropping the row instead would reproduce the F68 failure mode: a source that
        # contributes nothing while the run looks healthy.
        self._assert_every_streamed_line_is_pairable()

        self._epoch = 0
        self._total_pert_cells = self._count_pert_cells()
        if self._total_pert_cells == 0:
            raise ValueError(
                "Streaming dataset resolved 0 perturbed cells. Check the corpus filters "
                "(train_cell_lines, split_pert_names, min_cells_per_pert) — a training "
                "set with no perturbations is never intentional."
            )
        print(f"  Streaming corpus: {self._total_pert_cells:,} perturbed cells/epoch "
              f"before the per-epoch cap, over {len(self.active_metas)} file(s)")

    def _assert_every_streamed_line_is_pairable(self) -> None:
        train_cl_set = set(self.train_cell_lines) if self.train_cell_lines else None
        missing: Dict[str, int] = {}
        for fm in self.active_metas:
            if fm.controls_only:
                continue
            lines = np.asarray(fm.obs_cell_lines, dtype=str)
            pert = ~np.asarray(fm.is_control, dtype=bool)
            for cl in np.unique(lines[pert]):
                if train_cl_set is not None and cl not in train_cl_set:
                    continue
                if fm.excluded_pert_lines and cl in fm.excluded_pert_lines:
                    continue
                if not self.control_matcher.has_pool(cl):
                    missing[cl] = missing.get(cl, 0) + int((lines == cl).sum())
        if missing:
            raise ValueError(
                f"{len(missing)} cell line(s) would be streamed but have no control pool "
                f"and no fallback: {sorted(missing)[:10]}. Either raise the corpus filters "
                f"so they are excluded from the split, or supply controls for them."
            )

    def audit_eligible_perts(self) -> Dict[str, Dict[str, int]]:
        """{file basename: {perturbation: n eligible rows}}, applying the same predicates
        as `__iter__`'s row loop. See StreamingDatasetV2.audit_eligible_perts."""
        train_cl_set = set(self.train_cell_lines) if self.train_cell_lines else None
        out: Dict[str, Dict[str, int]] = {}
        for fmi, fm in enumerate(self.active_metas):
            per: Dict[str, int] = {}
            if not fm.controls_only:
                keep = ~np.asarray(fm.is_control, dtype=bool)
                if fm.excluded_pert_lines:
                    keep &= ~np.isin(fm.obs_cell_lines, list(fm.excluded_pert_lines))
                if fm.excluded_perts:
                    keep &= ~np.isin(np.asarray(fm.pert_gene_names, dtype=str),
                                     list(fm.excluded_perts))
                if self.min_cells_per_pert > 0:
                    keep &= self._row_ok[fmi]
                if train_cl_set is not None:
                    keep &= np.isin(fm.obs_cell_lines, list(train_cl_set))
                names = np.asarray(fm.pert_gene_names, dtype=str)
                keep &= np.isin(names, list(self.split_pert_names))
                for nm, cnt in zip(*np.unique(names[keep], return_counts=True)):
                    per[str(nm)] = int(cnt)
            out[Path(fm.path).name] = per
        return out

    def _count_pert_cells(self) -> int:
        """Vectorised row count. The previous Python loop ran once per row, which is
        ~20 M iterations with the T-cell screen — minutes of dead time at __init__
        before a single batch is produced."""
        train_cl_set = set(self.train_cell_lines) if self.train_cell_lines else None
        split = self.split_pert_names
        count = 0
        for fmi, fm in enumerate(self.active_metas):
            if fm.controls_only:
                continue
            keep = ~np.asarray(fm.is_control, dtype=bool)
            if fm.excluded_pert_lines:
                keep &= ~np.isin(fm.obs_cell_lines, list(fm.excluded_pert_lines))
            if fm.excluded_perts:
                # Must be ANDed here, while `keep` is still full length -- the
                # `keep[keep] = ...` compaction below cannot take a full-length mask.
                keep &= ~np.isin(np.asarray(fm.pert_gene_names, dtype=str),
                                 list(fm.excluded_perts))
            if self.min_cells_per_pert > 0:
                keep &= self._row_ok[fmi]
            if train_cl_set is not None:
                keep &= np.isin(fm.obs_cell_lines, list(train_cl_set))
            if keep.any():
                keep[keep] = np.isin(np.asarray(fm.pert_gene_names)[keep], list(split))
            count += int(keep.sum())
        return count

    def _ordered_chunks(self, rng) -> List[int]:
        """Chunk visiting order for one epoch.

        With `balance_sources`, per-file chunk lists are shuffled independently and then
        INTERLEAVED, so consecutive chunks come from different files. A buffer group of
        `buffer_chunks` consecutive chunks therefore spans that many sources, and the
        batches assembled from it are source-mixed. A plain global shuffle does not
        achieve this: the T-cell screen owns ~95% of all chunks, so a random group of 4
        is almost always 4 T-cell chunks, and the model would see long stretches of a
        single context.
        """
        by_file: Dict[int, List[int]] = {}
        for ci, (mi, _, _) in enumerate(self._chunks):
            by_file.setdefault(mi, []).append(ci)
        if not self.balance_sources:
            order = [ci for lst in by_file.values() for ci in lst]
            if self.shuffle:
                rng.shuffle(order)
            return order
        lists = []
        for mi in sorted(by_file):
            lst = list(by_file[mi])
            if self.shuffle:
                rng.shuffle(lst)
            lists.append(lst)
        # round-robin over files; longer files simply keep contributing once the
        # shorter ones are exhausted, so no data is dropped
        order, i = [], 0
        while True:
            added = False
            for lst in lists:
                if i < len(lst):
                    order.append(lst[i]); added = True
            if not added:
                break
            i += 1
        return order

    def _pert_cap_mask(self, fm, rows: np.ndarray, epoch: int) -> np.ndarray:
        """Keep at most `max_cells_per_pert` rows per (cell_line, perturbation).

        Selection varies with `epoch`, so successive epochs draw different cells and the
        full corpus is still covered over a run — the cap bounds epoch COST, it does not
        permanently discard data.
        """
        if self.max_cells_per_pert <= 0 or len(rows) == 0:
            return np.ones(len(rows), dtype=bool)
        keys = np.char.add(
            np.char.add(np.asarray(fm.obs_cell_lines[rows], dtype=str), "|"),
            np.asarray(fm.pert_gene_names[rows], dtype=str),
        )
        keep = np.zeros(len(rows), dtype=bool)
        # `fixed_pairs` drops `epoch` from the seed so the SAME cells survive the cap every
        # epoch. Without this the pairing is still frozen but the population it is drawn
        # from rotates, which reintroduces the very variability the flag exists to remove.
        cap_seed = self.seed if self.fixed_pairs else self.seed + epoch
        order = np.random.default_rng(cap_seed).permutation(len(rows))
        seen: Dict[str, int] = {}
        for j in order:
            k = keys[j]
            c = seen.get(k, 0)
            if c < self.max_cells_per_pert:
                keep[j] = True
                seen[k] = c + 1
        return keep

    def __len__(self) -> int:
        return self._total_pert_cells

    def set_epoch(self, epoch: int):
        # Report the epoch just finished in the SAME shape as build_mixed_dataset's
        # "N batch-matched, M cell-line-matched" line, so the cached and streaming paths
        # can be compared directly instead of by reading two different formats (F97).
        if self.batch_match and (self._bm_matched or self._bm_fellback):
            tot = self._bm_matched + self._bm_fellback
            print(f"  [stream] epoch {self._epoch}: {self._bm_matched:,} batch-matched, "
                  f"{self._bm_fellback:,} cell-line-matched "
                  f"({100.0 * self._bm_matched / max(tot, 1):.1f}% batch-matched)")
        self._bm_matched = 0
        self._bm_fellback = 0
        self._epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)

        # "exact" feeds the cell's own RAW expression as the control input, while a
        # precomputed embedding pool is hidden_size-dimensional. Mixing them yields a
        # ragged stack. Precomputed embeddings are an optimisation for frozen encoders;
        # exact identity simply cannot use them, so fail loudly rather than at torch.stack.
        if (self.control_identity_mode == "exact"
                and self.include_controls_as_training
                and self.control_matcher.has_embeddings):
            raise ValueError(
                "control_identity_mode='exact' is incompatible with precomputed control "
                "embeddings: the identity input is raw expression (num_genes) while the "
                "pool holds embeddings (hidden_size). Use control_identity_mode='paired', "
                "or disable control-embedding precompute for this run."
            )

        chunk_order = self._ordered_chunks(rng)

        # Per-epoch cap: decide up-front which rows of each file survive, so the chunk
        # loop only has to test membership. Computed once per epoch, not per chunk.
        cap_keep: Dict[int, set] = {}
        if self.max_cells_per_pert > 0:
            for mi, fm in enumerate(self.active_metas):
                rows = np.where(~np.asarray(fm.is_control, dtype=bool))[0]
                if len(rows) == 0:
                    cap_keep[mi] = set()
                    continue
                m = self._pert_cap_mask(fm, rows, self._epoch)
                cap_keep[mi] = set(rows[m].tolist())

        # Pre-index unique gene masks by meta index (only 3 unique masks)
        unique_masks = {
            mi: fm.gene_mask for mi, fm in enumerate(self.active_metas)
        }

        # Spread the extra (H1 identity) samples evenly over the buffer groups, so they
        # mix into batches throughout the epoch instead of arriving as one clump at the
        # end — a clump would give the model a run of pure null-token batches.
        n_groups = max(1, (len(chunk_order) + self.buffer_chunks - 1) // self.buffer_chunks)
        extra_order = (rng.permutation(self._n_extra) if self._n_extra else
                       np.empty(0, dtype=np.int64))
        per_group = int(np.ceil(self._n_extra / n_groups)) if self._n_extra else 0

        # Process chunks in groups of buffer_chunks
        for gi, buf_start in enumerate(range(0, len(chunk_order), self.buffer_chunks)):
            buf_end = min(buf_start + self.buffer_chunks, len(chunk_order))
            chunk_group = chunk_order[buf_start:buf_end]

            # Collect all perturbation samples from this group of chunks
            all_targets = []
            all_pert_indices = []
            all_pert_names = []
            all_cell_types = []
            all_mask_indices = []  # store meta index, expand at yield time
            all_exact_ctrl = []    # for control_identity_mode="exact": the row itself
            all_row_uid = []       # (file, global row) — the key `fixed_pairs` hashes on
            all_batch_id = []      # cell_line__batch, for `batch_match`

            for ci in chunk_group:
                mi, c_start, c_end = self._chunks[ci]
                fm = self.active_metas[mi]

                # Decide which rows of this chunk are wanted BEFORE reading expression.
                # All the predicates below are pure obs metadata, already in memory, so
                # this costs nothing — and it lets the loader skip the chunk entirely
                # when nothing is wanted, and skip the 44,728-wide expansion for rows
                # that would be discarded anyway.
                train_cl_set = set(self.train_cell_lines) if self.train_cell_lines else None
                wanted = []
                for local_i in range(c_end - c_start):
                    gi = c_start + local_i
                    if train_cl_set is not None and fm.obs_cell_lines[gi] not in train_cl_set:
                        continue
                    if fm.is_control[gi]:
                        if self.include_controls_as_training:
                            wanted.append(local_i)
                        continue
                    if fm.controls_only:
                        continue
                    if (fm.excluded_pert_lines
                            and fm.obs_cell_lines[gi] in fm.excluded_pert_lines):
                        continue
                    if fm.pert_gene_names[gi] not in self.split_pert_names:
                        continue
                    if fm.excluded_perts and fm.pert_gene_names[gi] in fm.excluded_perts:
                        continue
                    if self.min_cells_per_pert > 0 and not self._row_ok[mi][gi]:
                        continue
                    if self.max_cells_per_pert > 0 and gi not in cap_keep[mi]:
                        continue
                    wanted.append(local_i)
                if not wanted:
                    continue
                keep_local = np.asarray(wanted, dtype=np.int64)

                # Read the chunk expression data (already row-filtered)
                chunk_expr = _load_expression_chunk(
                    path=fm.path,
                    start=c_start,
                    end=c_end,
                    n_vars=fm.n_vars,
                    src_indices=fm.src_indices,
                    dst_indices=fm.dst_indices,
                    num_encoder_genes=self.num_encoder_genes,
                    target_sum=self.target_sum,
                    buf=self._read_buf,
                    keep_local=keep_local,
                )

                # chunk_expr is COMPACTED to `wanted`, so it is indexed by position in
                # that list, not by the original chunk-local offset.
                # The cell-line / split / cap predicates were already applied when
                # `wanted` was built — re-testing them here would be dead code. Only
                # pert2id resolution remains, because it needs the pert2id lookup.
                for pos, local_i in enumerate(wanted):
                    global_i = c_start + local_i
                    row_cl = fm.obs_cell_lines[global_i]

                    if fm.is_control[global_i]:
                        all_targets.append(chunk_expr[pos])
                        all_pert_indices.append([-1])
                        all_pert_names.append("non-targeting")
                        all_cell_types.append(row_cl)
                        all_mask_indices.append(mi)
                        # "exact" reproduces the cell from ITSELF; "paired" leaves
                        # None so a different control is drawn from the same line.
                        all_exact_ctrl.append(
                            chunk_expr[pos]
                            if self.control_identity_mode == "exact" else None
                        )
                        all_row_uid.append((mi, global_i))
                        all_batch_id.append(
                            str(fm.batch_ids[global_i]) if fm.batch_ids is not None else "")
                        continue

                    _pidx = resolve_pert_indices(fm.pert_gene_names[global_i], self.pert2id)
                    if _pidx is None:
                        continue

                    all_targets.append(chunk_expr[pos])
                    all_pert_indices.append(_pidx)
                    all_pert_names.append(fm.pert_gene_names[global_i])
                    all_cell_types.append(row_cl)
                    all_mask_indices.append(mi)
                    all_exact_ctrl.append(None)
                    all_row_uid.append((mi, global_i))
                    all_batch_id.append(
                        str(fm.batch_ids[global_i]) if fm.batch_ids is not None else "")

            if not all_targets:
                continue

            n_samples = len(all_targets)

            # Batch control sampling — one vectorized call per cell type
            all_controls = [None] * n_samples
            use_embeddings = self.control_matcher.has_embeddings

            # Grouping key per sample. Default: the cell LINE, which is all the streaming
            # path has ever matched on. With `batch_match` (F97): the perturbed cell's own
            # (cell_line, batch), falling back to the line when that batch has no controls
            # — the same hierarchy `build_mixed_dataset` uses on the cached path.
            # Batch pools hold raw expression, so a precomputed embedding pool disables it.
            bm = self.batch_match and not use_embeddings
            ct_sample_indices: Dict[Tuple[str, str], List[int]] = {}
            for i, ct in enumerate(all_cell_types):
                key = ("C", ct)
                if bm:
                    b = all_batch_id[i]
                    if b and self.control_matcher.has_batch_pool(b):
                        key = ("B", b)
                        if all_exact_ctrl[i] is None:
                            self._bm_matched += 1
                    elif all_exact_ctrl[i] is None:
                        self._bm_fellback += 1
                ct_sample_indices.setdefault(key, []).append(i)

            for (kind, ct), indices in ct_sample_indices.items():
                pool = (self.control_matcher.pool_for_batch(ct) if kind == "B"
                        else self.control_matcher.pool_for(ct, use_embeddings=use_embeddings))
                # ONE control per sample, no mean over n. n_controls is asserted to be 1
                # in __init__; taking a mean here would silently turn the model's input
                # into a pseudobulk.
                need = [si for si in indices if all_exact_ctrl[si] is None]
                if need:
                    if self.fixed_pairs:
                        # A perturbed cell always gets THE SAME control, every epoch:
                        # the index is a pure hash of (file, global row), so it depends on
                        # neither the epoch nor the chunk visiting order. See the
                        # `fixed_pairs` note in the class docstring for why.
                        ctrl_idx = _stable_ctrl_index(
                            np.array([all_row_uid[si][0] for si in need], dtype=np.uint64),
                            np.array([all_row_uid[si][1] for si in need], dtype=np.uint64),
                            pool.shape[0], self.seed)
                    else:
                        ctrl_idx = rng.choice(pool.shape[0], size=len(need), replace=True)
                    ctrl_batch = pool[ctrl_idx]          # (n, dim) — no averaging
                    for j, si in enumerate(need):
                        all_controls[si] = ctrl_batch[j]
            # exact-identity rows use their own expression as the control input
            for si, ex in enumerate(all_exact_ctrl):
                if ex is not None:
                    all_controls[si] = ex

            # Splice in this group's share of the pre-paired extras (H1 identity rows).
            # They arrive already paired, so no control pool is consulted and they cannot
            # leak into another cell line's sampling.
            if per_group:
                sl = extra_order[gi * per_group:(gi + 1) * per_group]
                if len(sl):
                    e_ctrl, e_tgt, e_names, e_pidx, e_cts, e_mask = self.extra_samples
                    for k in sl:
                        all_controls.append(e_ctrl[k])
                        all_targets.append(e_tgt[k])
                        all_pert_indices.append(e_pidx[k])
                        all_pert_names.append(e_names[k])
                        all_cell_types.append(e_cts[k])
                        all_mask_indices.append(-1)     # -1 => use the extras' own mask
                    n_samples = len(all_targets)
                    # build_h1_identity_samples returns `gene_mask.unsqueeze(0).expand(n, -1)`
                    # — one mask broadcast over every row — so a single registered entry
                    # is exact, not an approximation. Asserted rather than assumed,
                    # because a per-row mask here would silently mislabel present genes.
                    if e_mask.ndim == 2 and e_mask.shape[0] > 1:
                        if not bool((e_mask[0] == e_mask[-1]).all()):
                            raise ValueError(
                                "extra_samples carry per-row gene masks; the streaming "
                                "mask lookup assumes one shared mask for all extras."
                            )
                    unique_masks[-1] = e_mask[0] if e_mask.ndim == 2 else e_mask

            # Stack tensors
            ctrl_tensor = torch.stack(all_controls)
            tgt_tensor = torch.stack(all_targets)
            # Expand gene masks from indices — only 3 unique masks
            mask_idx_arr = np.array(all_mask_indices, dtype=np.int64)

            if self.shuffle:
                perm = rng.permutation(n_samples)
                ctrl_tensor = ctrl_tensor[perm]
                tgt_tensor = tgt_tensor[perm]
                mask_idx_arr = mask_idx_arr[perm]
                all_pert_indices = [all_pert_indices[p] for p in perm]
                all_pert_names = [all_pert_names[p] for p in perm]
                all_cell_types = [all_cell_types[p] for p in perm]
                # all_exact_ctrl is not permuted: it is already consumed into
                # all_controls above, before ctrl_tensor is stacked.

            # Yield batches
            for b_start in range(0, n_samples, self.batch_size):
                b_end = min(b_start + self.batch_size, n_samples)
                if b_end - b_start < self.batch_size and b_end < n_samples:
                    continue  # skip incomplete mid-buffer batches (shouldn't happen)
                # The TRAILING partial batch of each buffer group IS yielded (b_end ==
                # n_samples), and it can contain a single sample. The decoder's
                # BatchNorm then raises:
                #   "Expected more than 1 value per channel ... got input size [1, 2048]"
                # which killed the cap-32 big-corpus arm (17701491_1) 3.5 h in. Whether it
                # fires depends on n_samples % batch_size landing on exactly 1, so it is
                # intermittent — the sibling cap-16 and cap-64 arms ran for hours without
                # hitting it. Dropping <2-sample batches costs at most one cell per buffer
                # group.
                if b_end - b_start < 2:
                    continue

                # Expand gene masks for this batch only
                batch_mask_ids = mask_idx_arr[b_start:b_end]
                batch_masks = torch.stack(
                    [unique_masks[mi] for mi in batch_mask_ids]
                )

                yield PerturbationBatch(
                    control_expression=ctrl_tensor[b_start:b_end],
                    perturbed_expression=tgt_tensor[b_start:b_end],
                    pert_indices=all_pert_indices[b_start:b_end],
                    pert_names=all_pert_names[b_start:b_end],
                    cell_types=all_cell_types[b_start:b_end],
                    gene_masks=batch_masks,
                )


# ---------------------------------------------------------------------------
# High-level data preparation
# ---------------------------------------------------------------------------

def prepare_virtual_cell_data(
    data_paths: List[str],
    encoder_gene_names: List[str],
    pert2id: Dict[str, int],
    target_sum: float = 1e4,
    gene_name_col: str = "ensembl_gene_id",
    pert_gene_col: str = "gene_name",
    batch_col: Optional[str] = None,
    seed: int = 42,
    min_controls_per_cell_line: int = 0,
    cell_line_cols: Optional[List[str]] = None,
    max_controls_per_cell_line: int = 0,
    control_fallback_col: Optional[str] = None,
    restrict_gene_ids: Optional[Set[str]] = None,
    exclude_file_perts: Optional[Dict[str, Set[str]]] = None,
    control_only_paths: Optional[List[str]] = None,
    min_control_batches_per_cell_line: int = 0,
    require_control_batch_overlap: bool = False,
    control_force_group: bool = False,
    build_pools: bool = True,
) -> Tuple[List[FileMetadata], CellTypeControlMatcher]:
    """Load file metadata and build control pools.

    `build_pools=False` (stream_v2): apply every eligibility rule as usual but do NOT load the
    dense pools; return a matcher whose `pending_pools` = {line: [(file_meta, sorted control
    rows)]} so `virtual_cell.stream_v2.build_control_embedding_pool` can encode them chunk-wise.

    Args:
        data_paths: Paths to h5ad files (or a single folder path).
        encoder_gene_names: Ordered encoder gene vocabulary (ensembl IDs).
        pert2id: Gene symbol → integer ID mapping for GNN.
        target_sum: Library-size normalization target.
        gene_name_col: var column for expression gene IDs.
        pert_gene_col: obs column for perturbation gene symbols.
        batch_col: obs column for batch IDs (loaded when not None, for pseudobulk).
        seed: Random seed for control sampling reproducibility.
        min_controls_per_cell_line: Exclude cell lines with fewer controls than this.
        cell_line_cols: obs columns to join into a synthetic `cell_line` (see
            `_load_file_metadata`).
        max_controls_per_cell_line: Subsample each cell line's control pool to at most
            this many cells (0 = keep all). **This is a memory guard, not a modelling
            choice.** The pool is held as raw 44,728-gene float32, i.e. ~179 KB/cell, so
            the CD4 T-cell screen's 74,678 controls x 12 donor-condition contexts would
            need ~160 GB — against a 700 GB job that also holds read buffers, chunk
            groups and the model. At 3,000/line the same pools cost ~6.4 GB. Controls
            are sampled uniformly at random with a fixed seed, and since the matcher
            draws one control uniformly from the pool anyway, capping changes only how
            many distinct control cells can be drawn, not the distribution they are
            drawn from.

    Returns:
        (file_metas, control_matcher)
    """
    rng_cap = np.random.default_rng(seed)

    # Resolve paths. Controls-only lenders are appended after the training files so a
    # path given in both lists keeps its training role.
    def _resolve(paths):
        out = []
        for p in paths or []:
            pp = Path(p)
            out.extend(sorted(pp.rglob("*.h5ad")) if pp.is_dir() else [pp])
        return [str(p) for p in out]

    resolved_paths = _resolve(data_paths)
    lender_paths = [p for p in _resolve(control_only_paths) if p not in resolved_paths]
    controls_only_set = set(lender_paths)
    resolved_paths = resolved_paths + lender_paths
    if lender_paths:
        print(f"Controls-only lenders ({len(lender_paths)}): "
              + ", ".join(Path(p).name for p in lender_paths))

    encoder_gene_to_idx = {g: i for i, g in enumerate(encoder_gene_names)}
    num_encoder_genes = len(encoder_gene_names)

    # Load metadata for each file
    file_metas = []
    for path in resolved_paths:
        print(f"Loading metadata: {path}")
        fm = _load_file_metadata(
            path, encoder_gene_names, encoder_gene_to_idx,
            gene_name_col=gene_name_col, pert_gene_col=pert_gene_col,
            batch_col=batch_col, cell_line_cols=cell_line_cols,
            control_fallback_col=control_fallback_col,
            controls_only=path in controls_only_set,
            restrict_gene_ids=restrict_gene_ids,
        )
        file_metas.append(fm)
        unique_cls = sorted(set(fm.obs_cell_lines))
        cl_label = unique_cls[0] if len(unique_cls) == 1 else f"{len(unique_cls)} cell lines"
        role = " [CONTROLS-ONLY LENDER]" if fm.controls_only else ""
        print(
            f"  {cl_label}: {fm.n_obs} cells, {fm.n_vars} genes, "
            f"{(~fm.is_control).sum()} perturbed, {fm.is_control.sum()} controls, "
            f"{len(fm.src_indices)}/{fm.n_vars} genes aligned to encoder{role}"
        )

    # ---- Per-(file, perturbation) TRAINING exclusions ----
    # Keyed on file BASENAME, not cell line: the exclusion is a property of (file,
    # perturbation). Applied only in the training predicates, so the eval builders still see
    # these rows -- which is what makes them the val/test set.
    if exclude_file_perts:
        seen = set()
        for fm in file_metas:
            name = Path(fm.path).name
            ex = exclude_file_perts.get(name)
            if not ex:
                continue
            seen.add(name)
            fm.excluded_perts = frozenset(ex)
            n_cells = int(np.isin(np.asarray(fm.pert_gene_names, dtype=str), list(ex)).sum())
            print(f"  [bench-exclude] {name}: {len(ex)} perturbation(s), "
                  f"{n_cells:,} perturbed cells withheld from TRAINING")
        missing = sorted(set(exclude_file_perts) - seen)
        if missing:
            # The highest-consequence silent failure in the whole benchmark: a renamed or
            # relocated data file turns the exclusion into a no-op and the held-out
            # perturbations train normally, with no error and a flattering score.
            raise ValueError(
                f"exclusions name file(s) that are not in data_paths: {missing}. "
                f"Loaded: {sorted(Path(fm.path).name for fm in file_metas)}. "
                f"Refusing to train -- an exclusion that matches nothing would leak the "
                f"held-out perturbations into training silently."
            )

    # ---- Cell-line eligibility, decided BEFORE any expression is read ----
    # Three independent criteria, reported per line so none of them is ever silent:
    #   n_ctrl         cells available to pair against  (min_controls_per_cell_line)
    #   n_ctrl_batch   distinct batches those controls span (min_control_batches_...)
    #   overlap        do the controls share a batch with the perturbations they will be
    #                  paired to? (require_control_batch_overlap)
    # The three do NOT rank the sources the same way, and picking one by intuition gets
    # it backwards. Measured 2026-08-27: CD4 T-cell has exactly ONE control batch per
    # donor x condition — but the perturbations are in that same batch, which is the best
    # possible matching. Feng genome-wide has more batches yet ZERO overlap, because its
    # controls are borrowed from a different experiment. A bare ">=2 batches" rule would
    # therefore delete all 20 M CD4 cells and keep the borrowed-control pairs.
    ctrl_cells: Dict[str, int] = {}
    ctrl_batches: Dict[str, set] = {}
    pert_batches: Dict[str, set] = {}
    ctrl_cells_by_batch: Dict[str, Dict[str, int]] = {}
    for fm in file_metas:
        c = np.asarray(fm.is_control, dtype=bool)
        lines = np.asarray(fm.obs_cell_lines, dtype=str)
        for cl in np.unique(lines[c]):
            ctrl_cells[cl] = ctrl_cells.get(cl, 0) + int((lines[c] == cl).sum())
        if fm.batch_ids is None:
            continue
        b = np.asarray(fm.batch_ids, dtype=str)
        for cl in np.unique(lines):
            m = lines == cl
            ctrl_batches.setdefault(cl, set()).update(b[m & c].tolist())
            by = ctrl_cells_by_batch.setdefault(cl, {})
            bb, nn = np.unique(b[m & c], return_counts=True)
            for k, v in zip(bb.tolist(), nn.tolist()):
                by[k] = by.get(k, 0) + int(v)
            if not fm.controls_only:
                pert_batches.setdefault(cl, set()).update(b[m & ~c].tolist())
    need_batches = min_control_batches_per_cell_line > 0 or require_control_batch_overlap
    if need_batches and all(fm.batch_ids is None for fm in file_metas):
        raise ValueError(
            "min_control_batches_per_cell_line / require_control_batch_overlap need batch "
            "labels, but batch_col was not set. Pass --batch_col."
        )

    eligible: Dict[str, str] = {}
    for cl in sorted(set(ctrl_cells) | set(pert_batches)):
        n_c = ctrl_cells.get(cl, 0)
        n_b = len(ctrl_batches.get(cl, ()))
        ov = len(ctrl_batches.get(cl, set()) & pert_batches.get(cl, set()))
        if n_c < max(min_controls_per_cell_line, 1):
            eligible[cl] = f"only {n_c} controls (min {max(min_controls_per_cell_line, 1)})"
        elif n_b < min_control_batches_per_cell_line:
            eligible[cl] = f"only {n_b} control batch(es) (min {min_control_batches_per_cell_line})"
        elif require_control_batch_overlap and cl in pert_batches and ov == 0:
            eligible[cl] = "controls share NO batch with any of this line's perturbations"
    print(f"Cell-line eligibility: min_controls={min_controls_per_cell_line}, "
          f"min_control_batches={min_control_batches_per_cell_line}, "
          f"require_batch_overlap={require_control_batch_overlap}")
    keep_lines = sorted(set(ctrl_cells) - set(eligible))
    # Report "n/a" rather than "0b" when batch labels were never loaded — a printed 0
    # reads as "this line has no batches", which is a different and alarming claim.
    def _bstr(cl: str) -> str:
        return "n/a" if not ctrl_batches else f"{len(ctrl_batches.get(cl, ()))}b"
    print(f"  {len(keep_lines)} cell line(s) usable: " + ", ".join(
        f"{cl}({ctrl_cells[cl]}c/{_bstr(cl)})" for cl in keep_lines[:12])
        + (" ..." if len(keep_lines) > 12 else ""))
    if eligible:
        lost = sum(
            int(((np.asarray(fm.obs_cell_lines, dtype=str) == cl)
                 & ~np.asarray(fm.is_control, dtype=bool)).sum())
            for fm in file_metas if not fm.controls_only for cl in eligible
        )
        print(f"  {len(eligible)} cell line(s) EXCLUDED, costing {lost:,} perturbed cells:")
        for cl, why in sorted(eligible.items())[:20]:
            print(f"    - {cl}: {why}")
        if len(eligible) > 20:
            print(f"    ... and {len(eligible) - 20} more")
    excluded_lines = set(eligible)

    # Batch overlap is a property of a (FILE, cell line) pair, so it is resolved per file
    # after the line-level pass. Within one donor line the targeted screen's rows share a
    # batch with that line's controls while the genome-wide screen's rows do not, and only
    # the latter should be dropped. A line-level test alone is a NO-OP here.
    # Counted in CONTROL CELLS, not as a set intersection. A bare "the batch sets touch"
    # test is satisfied by a handful of cells and is therefore meaningless: Feng's
    # genome-wide files ship 1-3 NTCs per line in their own `feng_nfg-*`/`feng_fg-*`
    # batches, so a line whose 625-cell pool is 622 targeted-screen controls and 3
    # genome-wide controls "shares a batch" — while a random draw picks a
    # different-experiment control 99.5% of the time. Requiring min_controls same-batch
    # cells is the test that means what the flag says.
    if require_control_batch_overlap:
        dropped_total = 0
        need = max(min_controls_per_cell_line, 1)
        for fm in file_metas:
            if fm.controls_only:
                continue
            lines = np.asarray(fm.obs_cell_lines, dtype=str)
            pert = ~np.asarray(fm.is_control, dtype=bool)
            b = np.asarray(fm.batch_ids, dtype=str)
            bad = set()
            for cl in np.unique(lines[pert]):
                if cl in excluded_lines:
                    continue
                own = set(b[pert & (lines == cl)].tolist())
                if ctrl_cells_by_batch.get(cl, {}) and sum(
                        n for bb, n in ctrl_cells_by_batch[cl].items() if bb in own) >= need:
                    continue
                bad.add(cl)
                dropped_total += int((pert & (lines == cl)).sum())
            fm.excluded_pert_lines = bad
            if bad:
                print(f"  [batch-overlap] {Path(fm.path).name}: dropping perturbations of "
                      f"{len(bad)} cell line(s) with no shared control batch")
        print(f"  [batch-overlap] {dropped_total:,} perturbed cells dropped for using "
              f"borrowed (different-experiment) controls")

    # Load all controls into memory (split by actual cell line within each file)
    print("Loading control cells into memory..." if build_pools else
          "Resolving control rows (pools deferred to stream_v2 embeddings)...")
    pending_pools: Dict[str, list] = {}
    pending_counts: Dict[str, int] = {}
    control_pools: Dict[str, List[torch.Tensor]] = {}
    control_batch_ids: Dict[str, List[np.ndarray]] = {}
    for fm in file_metas:
        ctrl_indices = np.where(fm.is_control)[0]
        if len(ctrl_indices) == 0:
            continue

        # Group control indices by actual cell line
        ctrl_cls = fm.obs_cell_lines[ctrl_indices]
        for cl in sorted(np.unique(ctrl_cls)):
            # Decided above from obs alone, so an excluded line never costs an HDF5 read.
            if cl in excluded_lines:
                continue
            cl_mask = ctrl_cls == cl
            cl_indices = ctrl_indices[cl_mask]
            # Cap BEFORE reading — the point is to never materialise the full pool.
            if 0 < max_controls_per_cell_line < len(cl_indices):
                keep = rng_cap.choice(len(cl_indices),
                                      size=max_controls_per_cell_line, replace=False)
                print(f"  {cl}: capping {len(cl_indices)} -> "
                      f"{max_controls_per_cell_line} controls (memory guard)")
                cl_indices = np.sort(cl_indices[keep])
            if not build_pools:
                pending_pools.setdefault(cl, []).append((fm, np.sort(cl_indices)))
                pending_counts[cl] = pending_counts.get(cl, 0) + len(cl_indices)
                continue
            ctrl_data = _load_expression_rows(
                fm.path, cl_indices, fm.n_vars,
                fm.src_indices, fm.dst_indices,
                num_encoder_genes, target_sum,
            )
            if cl not in control_pools:
                control_pools[cl] = []
            control_pools[cl].append(ctrl_data)
            # Batch id per loaded control row, for stream_batch_match's per-batch pools.
            # Costs nothing: these rows are already in hand, this is only their label.
            if fm.batch_ids is not None:
                control_batch_ids.setdefault(cl, []).append(
                    np.asarray(fm.batch_ids, dtype=str)[cl_indices])
            print(f"  Loaded {len(cl_indices)} controls for {cl}")

    # Concatenate per cell type
    # Eligibility was already decided from obs; the only thing left is the cap, which can
    # only shrink a pool that was judged on its UNCAPPED size. Re-check so a cap below the
    # minimum cannot smuggle an under-powered pool through.
    if not build_pools:
        need = max(min_controls_per_cell_line, 1)
        for cl in [c for c, n in pending_counts.items() if n < need]:
            print(f"  EXCLUDED {cl}: {pending_counts[cl]} controls (min={min_controls_per_cell_line})")
            pending_pools.pop(cl, None)
        if control_fallback_col or control_force_group:
            raise ValueError("stream_v2 (build_pools=False) does not support control fallback groups")
        # 1-row placeholders so `has_pool()` (used by the split and by the streaming
        # dataset's pairability assertion) sees every deferred line; stream_v2 replaces
        # `control_pool` with the real embedding pools before the first batch.
        matcher = CellTypeControlMatcher(
            {cl: torch.zeros(1, 1) for cl in pending_pools}, seed=seed)
        matcher.pending_pools = pending_pools
        matcher.pools_deferred = True
        print(f"  Deferred control pools: {len(pending_pools)} cell line(s), "
              f"{sum(pending_counts[c] for c in pending_pools):,} control cells "
              f"(embedded once by stream_v2, no dense pool)")
        return file_metas, matcher
    control_pool_tensors = {}
    for ct, tensors in control_pools.items():
        pool = torch.cat(tensors, dim=0)
        if pool.shape[0] < max(min_controls_per_cell_line, 1):
            print(f"  EXCLUDED {ct}: {pool.shape[0]} controls after capping "
                  f"(min={min_controls_per_cell_line})")
            continue
        control_pool_tensors[ct] = pool
        print(f"  Control pool {ct}: {tuple(control_pool_tensors[ct].shape)}")

    excluded_cls = excluded_lines | (set(control_pools) - set(control_pool_tensors))
    if excluded_cls:
        print(f"  {len(excluded_cls)} cell line(s) excluded in total")

    # ---- Fallback pools, keyed by the coarse group ----
    # Every cell line that HAS a usable pool contributes it to its group's union. A line
    # with no pool of its own then draws from that union rather than being dropped. The
    # group is `control_fallback_col` (e.g. `cell_type`); with it unset the group key IS
    # the cell line, so a group pool only ever duplicates a per-line pool and the
    # fallback can never fire. Excluded lines do NOT contribute — an under-depth pool is
    # exactly the pool we judged untrustworthy.
    line_to_group: Dict[str, str] = {}
    for fm in file_metas:
        if fm.obs_control_groups is None:
            continue
        for cl, grp in zip(fm.obs_cell_lines, fm.obs_control_groups):
            line_to_group.setdefault(str(cl), str(grp))
    group_pool_tensors: Dict[str, torch.Tensor] = {}
    if control_fallback_col:
        by_group: Dict[str, List[torch.Tensor]] = {}
        for cl, pool in control_pool_tensors.items():
            by_group.setdefault(line_to_group.get(cl, cl), []).append(pool)
        for grp, tensors in by_group.items():
            group_pool_tensors[grp] = torch.cat(tensors, dim=0)
        borrowers = sorted(
            cl for cl in line_to_group
            if cl not in control_pool_tensors
            and line_to_group[cl] in group_pool_tensors
        )
        print(f"  Control fallback on '{control_fallback_col}': "
              f"{len(group_pool_tensors)} group pool(s) "
              + ", ".join(f"{g}={t.shape[0]}" for g, t in sorted(group_pool_tensors.items())))
        if borrowers:
            print(f"  {len(borrowers)} cell line(s) will BORROW group controls: "
                  f"{borrowers[:8]}{' ...' if len(borrowers) > 8 else ''}")

    if control_force_group and not group_pool_tensors:
        raise ValueError("control_force_group needs --control_fallback_col to define groups.")
    if control_force_group:
        print("  ⚠️  control_force_group: cell-line matching DISABLED, every line draws "
              "from its group pool (ablation only)")
    # ---- Per-(cell_line, batch) pools, for stream_batch_match (F97) ----
    # Sliced out of the per-line tensors already in memory, so no extra HDF5 reads. Only
    # built when batch ids were loaded; otherwise the matcher simply has no batch pools
    # and stream_batch_match falls back to line matching everywhere (and says so).
    batch_pool_tensors: Dict[str, torch.Tensor] = {}
    if control_batch_ids:
        for cl, pool in control_pool_tensors.items():
            chunks = control_batch_ids.get(cl)
            if not chunks:
                continue
            bids = np.concatenate(chunks)
            if len(bids) != pool.shape[0]:
                # A cap was applied per (file, line) but ids were recorded on the kept
                # rows, so these must agree. If they ever do not, drop batch pools for
                # this line rather than mis-pair silently.
                print(f"  [batch-pool] {cl}: {len(bids)} ids vs {pool.shape[0]} rows — "
                      f"skipping batch pools for this line")
                continue
            for b in np.unique(bids):
                sub = pool[torch.from_numpy((bids == b).nonzero()[0])]
                if sub.shape[0] > 0:
                    batch_pool_tensors[str(b)] = sub
        if batch_pool_tensors:
            sizes = np.array([t.shape[0] for t in batch_pool_tensors.values()])
            print(f"  Batch control pools: {len(batch_pool_tensors)} (cell_line, batch) "
                  f"pools, sizes min={sizes.min()} median={int(np.median(sizes))} "
                  f"max={sizes.max()}")

    control_matcher = CellTypeControlMatcher(
        control_pool_tensors, seed=seed,
        group_pool=group_pool_tensors, line_to_group=line_to_group,
        force_group=control_force_group,
        batch_pool=batch_pool_tensors,
    )

    return file_metas, control_matcher


def build_eval_dataset(
    file_metas: List[FileMetadata],
    pert2id: Dict[str, int],
    split_pert_names: Set[str],
    control_matcher: CellTypeControlMatcher,
    encoder_gene_names: List[str],
    target_sum: float = 1e4,
    n_controls: int = 5,
    cell_lines: Optional[List[str]] = None,
    include_controls_as_training: bool = False,
    exclude_cl_perts: Optional[Dict[str, Set[str]]] = None,
) -> VirtualCellDataset:
    """Build an in-memory VirtualCellDataset for a given split.

    Args:
        file_metas: All file metadata.
        pert2id: Gene symbol → integer ID.
        split_pert_names: Perturbation names in this split.
        control_matcher: Control sampling object.
        encoder_gene_names: Encoder vocabulary.
        target_sum: Normalization target.
        n_controls: Controls to average per sample.
        cell_lines: Restrict to these cell lines (or None for all).
        include_controls_as_training: Add control identity samples.
        exclude_cl_perts: {cell_line: {pert_names}} — exclude specific
            (cell_line, pert) combinations.

    Returns:
        VirtualCellDataset ready for DataLoader.
    """
    num_encoder_genes = len(encoder_gene_names)

    all_expr = []
    all_pert_names = []
    all_pert_indices = []
    all_cell_types = []
    all_gene_masks = []

    for fm in file_metas:
        # Skip files with no overlap with requested cell lines
        if cell_lines is not None:
            file_cls = set(fm.obs_cell_lines)
            if not file_cls.intersection(cell_lines):
                continue

        # Find perturbation cells in this split
        row_indices = []
        row_pert_names = []
        row_cell_types = []
        for i in range(fm.n_obs):
            if fm.is_control[i]:
                continue
            # Row-level cell line filtering
            row_cl = fm.obs_cell_lines[i]
            if cell_lines is not None and row_cl not in cell_lines:
                continue
            pname = fm.pert_gene_names[i]
            # ⚠️ `resolve_pert_indices`, NOT `pname in pert2id`. A combinatorial label
            # ('GENE_A+GENE_B') is not itself a graph node, so the bare membership test
            # dropped EVERY double and this builder then raised "No perturbation cells found
            # for this split" -- on the norman_combo val split, which is made entirely of
            # doubles. The resolver already existed and every other call site used it; this
            # one did not, which is the same consumer-without-the-gate shape as F-loss1.
            if pname in split_pert_names and resolve_pert_indices(pname, pert2id) is not None:
                if exclude_cl_perts and row_cl in exclude_cl_perts:
                    if pname in exclude_cl_perts[row_cl]:
                        continue
                row_indices.append(i)
                row_pert_names.append(pname)
                row_cell_types.append(row_cl)

        if not row_indices:
            continue

        unique_cls = sorted(set(row_cell_types))
        cl_label = unique_cls[0] if len(unique_cls) == 1 else f"{len(unique_cls)} cell lines"
        print(
            f"  Loading {len(row_indices)} perturbation cells from "
            f"{cl_label} ({Path(fm.path).name})"
        )
        row_indices = np.array(row_indices, dtype=np.int64)
        expr = _load_expression_rows(
            fm.path, row_indices, fm.n_vars,
            fm.src_indices, fm.dst_indices,
            num_encoder_genes, target_sum,
        )

        all_expr.append(expr)
        all_pert_names.extend(row_pert_names)
        # one entry per row, each a LIST of ids: [id] for a single, [id_A, id_B] for a double
        all_pert_indices.extend([resolve_pert_indices(pn, pert2id) for pn in row_pert_names])
        all_cell_types.extend(row_cell_types)
        all_gene_masks.append(fm.gene_mask.unsqueeze(0).expand(len(row_indices), -1))

    if not all_expr:
        raise ValueError("No perturbation cells found for this split")

    expression_data = torch.cat(all_expr, dim=0)
    gene_masks = torch.cat(all_gene_masks, dim=0)

    # Controls for identity mapping (if requested)
    ctrl_expr = None
    ctrl_cell_types = None
    ctrl_gene_mask = None
    if include_controls_as_training:
        ctrl_parts = []
        ctrl_cts = []
        ctrl_masks = []
        seen_cls = set()
        for fm in file_metas:
            # Iterate per actual cell line in the file
            for cl in sorted(set(fm.obs_cell_lines)):
                if cell_lines is not None and cl not in cell_lines:
                    continue
                if cl in seen_cls:
                    continue
                seen_cls.add(cl)
                pool = control_matcher.control_pool.get(cl)
                if pool is not None:
                    ctrl_parts.append(pool)
                    ctrl_cts.extend([cl] * pool.shape[0])
                    ctrl_masks.append(
                        fm.gene_mask.unsqueeze(0).expand(pool.shape[0], -1)
                    )
        if ctrl_parts:
            ctrl_expr = torch.cat(ctrl_parts, dim=0)
            ctrl_cell_types = ctrl_cts
            ctrl_gene_mask = torch.cat(ctrl_masks, dim=0)

    dataset = VirtualCellDataset(
        expression_data=expression_data,
        pert_gene_names=all_pert_names,
        pert_indices=all_pert_indices,
        cell_types=all_cell_types,
        gene_masks=gene_masks,
        control_matcher=control_matcher,
        n_controls=n_controls,
        include_controls_as_training=include_controls_as_training,
        control_expression=ctrl_expr,
        control_cell_types=ctrl_cell_types,
        control_gene_mask=ctrl_gene_mask,
    )

    print(
        f"  Built eval dataset: {len(dataset)} samples "
        f"({expression_data.shape[0]} perturbation"
        f"{f' + {ctrl_expr.shape[0]} control' if ctrl_expr is not None else ''})"
    )
    return dataset


def build_pseudobulk_dataset(
    file_metas: List[FileMetadata],
    pert2id: Dict[str, int],
    split_pert_names: Set[str],
    encoder_gene_names: List[str],
    target_sum: float = 1e4,
    cell_lines: Optional[List[str]] = None,
    exclude_cl_perts: Optional[Dict[str, Set[str]]] = None,
) -> PseudobulkDataset:
    """Build PseudobulkDataset: aggregate raw counts per (batch, pert/control).

    For each file, averages raw counts per (batch, perturbation) and per
    (batch, control), then normalizes (library-size → log1p) and aligns
    to encoder vocabulary.  Pairs each perturbation profile with the control
    from the same batch.

    Args:
        file_metas: All file metadata (must have batch_ids loaded).
        pert2id: Gene symbol → integer ID.
        split_pert_names: Perturbation names in this split.
        encoder_gene_names: Encoder vocabulary.
        target_sum: Normalization target.
        cell_lines: Restrict to these cell lines (or None for all).
        exclude_cl_perts: {cell_line: {pert_names}} — exclude specific
            (cell_line, pert) combinations. Used to exclude
            test perts from HEPG2 while keeping them in other cell lines.

    Returns:
        PseudobulkDataset ready for DataLoader.
    """
    num_encoder_genes = len(encoder_gene_names)

    all_ctrl = []
    all_pert = []
    all_pert_names = []
    all_pert_indices = []
    all_cell_types = []
    all_gene_masks = []
    all_batch_ids = []
    n_skipped_no_ctrl = 0
    n_skipped_no_pert2id = 0

    for fm in file_metas:
        # Skip files with no overlap with requested cell lines
        if cell_lines is not None:
            file_cls = set(fm.obs_cell_lines)
            if not file_cls.intersection(cell_lines):
                continue

        if fm.batch_ids is None:
            raise ValueError(
                f"batch_ids not loaded for {fm.path}. "
                "Ensure --pseudobulk and --batch_col are set."
            )

        unique_cls = sorted(set(fm.obs_cell_lines))
        if cell_lines is not None:
            unique_cls = [cl for cl in unique_cls if cl in cell_lines]
        cl_label = unique_cls[0] if len(unique_cls) == 1 else f"{len(unique_cls)} cell lines"
        print(
            f"  Aggregating pseudobulk for {cl_label} "
            f"({Path(fm.path).name})..."
        )
        ctrl_profiles, pert_profiles = _aggregate_pseudobulk_for_file(
            fm, target_sum, num_encoder_genes,
        )
        print(
            f"    {len(ctrl_profiles)} batch controls, "
            f"{len(pert_profiles)} (batch, pert) groups"
        )

        # Build paired samples: each pert profile matched with same-batch control
        for (batch_id, gene_name), pert_profile in pert_profiles.items():
            # Extract cell line from batch_id prefix (format: "cell_line__batch")
            sample_cl = batch_id.split("__", 1)[0]
            if cell_lines is not None and sample_cl not in cell_lines:
                continue
            if gene_name not in split_pert_names:
                continue
            if exclude_cl_perts and sample_cl in exclude_cl_perts:
                if gene_name in exclude_cl_perts[sample_cl]:
                    continue
            _pidx = resolve_pert_indices(gene_name, pert2id)
            if _pidx is None:
                n_skipped_no_pert2id += 1
                continue
            if batch_id not in ctrl_profiles:
                n_skipped_no_ctrl += 1
                continue

            all_ctrl.append(ctrl_profiles[batch_id])
            all_pert.append(pert_profile)
            all_pert_names.append(gene_name)
            all_pert_indices.append(_pidx)
            all_cell_types.append(sample_cl)
            all_gene_masks.append(fm.gene_mask.numpy())
            all_batch_ids.append(batch_id)

    if not all_pert:
        raise ValueError("No pseudobulk samples found for this split")

    if n_skipped_no_ctrl > 0:
        print(f"  Warning: {n_skipped_no_ctrl} samples skipped (no same-batch control)")
    if n_skipped_no_pert2id > 0:
        print(f"  Warning: {n_skipped_no_pert2id} samples skipped (pert not in GNN vocab)")

    # Stack into tensors
    ctrl_tensor = torch.from_numpy(np.stack(all_ctrl))
    pert_tensor = torch.from_numpy(np.stack(all_pert))
    mask_tensor = torch.from_numpy(np.stack(all_gene_masks))

    dataset = PseudobulkDataset(
        control_profiles=ctrl_tensor,
        perturbed_profiles=pert_tensor,
        pert_gene_names=all_pert_names,
        pert_indices=all_pert_indices,
        cell_types=all_cell_types,
        gene_masks=mask_tensor,
        batch_ids=all_batch_ids,
    )

    print(
        f"  Built pseudobulk dataset: {len(dataset)} samples "
        f"({len(set(all_batch_ids))} unique batches)"
    )
    return dataset


def _mixed_dataset_fingerprint(
    file_metas: List[FileMetadata],
    split_pert_names: Set[str],
    encoder_gene_names: List[str],
    target_sum: float,
    cell_lines: Optional[List[str]],
    exclude_cl_perts: Optional[Dict[str, Set[str]]],
    seed: int,
    max_sc_pairs_per_pert: int = 0,
    include_pb: bool = True,
) -> str:
    """Content-aware fingerprint for caching the mixed dataset."""
    import hashlib
    import os

    h = hashlib.sha256()
    for fm in file_metas:
        h.update(fm.path.encode())
        h.update(str(os.path.getmtime(fm.path)).encode())
        h.update(str(fm.n_obs).encode())
    h.update(",".join(sorted(split_pert_names)).encode())
    h.update(",".join(encoder_gene_names[:5] + encoder_gene_names[-5:]).encode())
    h.update(str(len(encoder_gene_names)).encode())
    h.update(str(target_sum).encode())
    h.update(str(sorted(cell_lines) if cell_lines else "all").encode())
    if exclude_cl_perts:
        for cl in sorted(exclude_cl_perts):
            h.update(f"{cl}:{sorted(exclude_cl_perts[cl])}".encode())
    h.update(str(seed).encode())
    h.update(f"maxsc{max_sc_pairs_per_pert}".encode())
    h.update(f"pb{int(include_pb)}".encode())
    return h.hexdigest()[:16]


def _load_rows_chunked(
    path: str,
    needed_rows: np.ndarray,
    n_vars: int,
    src_indices: np.ndarray,
    dst_indices: np.ndarray,
    num_encoder_genes: int,
    target_sum: float,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Load specific rows via sequential chunk reads (faster than per-row access).

    Reads the file in contiguous chunks and extracts needed rows on the fly,
    avoiding per-row random h5py access. Returns rows in the original order
    of needed_rows.
    """
    n_needed = len(needed_rows)
    out = np.zeros((n_needed, num_encoder_genes), dtype=np.float32)

    # Map each needed row to its position in the output
    row_set = set(needed_rows.tolist())
    row_to_out_pos: Dict[int, List[int]] = {}
    for out_idx, row_idx in enumerate(needed_rows):
        row_to_out_pos.setdefault(int(row_idx), []).append(out_idx)

    # Read file metadata
    with h5py.File(path, "r") as f:
        X = f["X"]
        is_sparse = isinstance(X, h5py.Group) and "indptr" in X  # see note above
        n_obs = X.shape[0] if not is_sparse else len(f["X"]["indptr"]) - 1

        for start in range(0, n_obs, chunk_size):
            end = min(start + chunk_size, n_obs)

            # Quick check: any needed rows in this chunk?
            chunk_needed = []
            for r in range(start, end):
                if r in row_set:
                    chunk_needed.append(r)
            if not chunk_needed:
                continue

            # Read entire chunk
            n_rows = end - start
            if is_sparse:
                raw = np.zeros((n_rows, n_vars), dtype=np.float32)
                indptr = X["indptr"][start:end + 1]
                for i in range(n_rows):
                    s, e = int(indptr[i]), int(indptr[i + 1])
                    if s < e:
                        cols = X["indices"][s:e]
                        vals = X["data"][s:e]
                        raw[i, cols] = vals
            else:
                raw = X[start:end].astype(np.float32)

            # Extract needed rows, normalize, align (skip transforms if already applied)
            for r in chunk_needed:
                local_idx = r - start
                row = raw[local_idx]
                if not _SKIP_NORMALIZE:
                    row_sum = row.sum()
                    if row_sum > 0:
                        row = row / row_sum * target_sum
                if not _SKIP_LOG1P:
                    row = np.log1p(row)
                aligned = np.zeros(num_encoder_genes, dtype=np.float32)
                aligned[dst_indices] = row[src_indices]
                for out_idx in row_to_out_pos[r]:
                    out[out_idx] = aligned

    return torch.from_numpy(out)


def _filter_sc_only(dataset: PseudobulkDataset, n_pb: int) -> PseudobulkDataset:
    """Keep only single-cell samples from a mixed PB+SC dataset."""
    filtered = PseudobulkDataset(
        control_profiles=dataset.control_profiles[n_pb:],
        perturbed_profiles=dataset.perturbed_profiles[n_pb:],
        pert_gene_names=dataset.pert_gene_names[n_pb:],
        pert_indices=dataset.pert_indices[n_pb:],
        cell_types=dataset.cell_types[n_pb:],
        gene_masks=dataset.gene_masks[n_pb:],
        batch_ids=dataset.batch_ids[n_pb:] if dataset.batch_ids else None,
    )
    filtered.is_sc_flags = [True] * len(filtered)
    return filtered


def precompute_paired_control_embeddings(
    dataset: PseudobulkDataset,
    encoder: torch.nn.Module,
    hidden_size: int,
    device,
    batch_size: int = 4096,
) -> PseudobulkDataset:
    """Replace a paired dataset's gene-space controls with frozen encoder embeddings.

    Valid ONLY with a frozen encoder, and then it is EXACT rather than an approximation:
    the checkpointed encoder is built with ``dropout=0`` and contains only Linear /
    LayerNorm / Mish, so it is deterministic and its output for a fixed input never
    changes across epochs. The pairs in a PseudobulkDataset are fixed too, so the model
    recomputes the identical embedding every epoch for 100 epochs.

    ``VirtualCellModel`` already dispatches on width -- ``control_expression.shape[1] ==
    encoder_hidden_size`` short-circuits the encoder -- so nothing else has to change, and
    the raw controls are needed nowhere downstream (they are only ever encoded).

    Measured on the whole-genome arm (975k pairs, 44,728 genes, hidden 768):
      * control tensor  174 GB -> 3.0 GB   (host RAM)
      * per-batch H2D   366 MB -> 6.3 MB   at batch_size 2048
      * one of the two per-step frozen encoder passes removed; the encoder is ~193M
        params against the decoder's 129M, so this is a large share of step time.
    """
    n, in_dim = dataset.control_profiles.shape
    if in_dim == hidden_size:
        return dataset                      # already embeddings
    was_training = encoder.training
    encoder.eval()
    out = torch.empty(n, hidden_size, dtype=torch.float32)
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            chunk = dataset.control_profiles[s:e].to(device, non_blocking=True)
            out[s:e] = encoder(chunk).float().cpu()
    if was_training:
        encoder.train()
    old_gb = dataset.control_profiles.numel() * 4 / 1e9
    dataset.control_profiles = out
    print(f"  Pre-computed control embeddings: ({n}, {in_dim}) -> ({n}, {hidden_size})  "
          f"[{old_gb:.1f} GB -> {out.numel() * 4 / 1e9:.1f} GB host RAM]")
    return dataset


def append_control_identity(
    dataset: PseudobulkDataset,
    file_metas: List[FileMetadata],
    encoder_gene_names: List[str],
    target_sum: float = 1e4,
    cell_lines: Optional[List[str]] = None,
    ratio: float = 0.0,
    mode: str = "exact",
    seed: int = 42,
) -> PseudobulkDataset:
    """Append control-in / control-out samples for the TRAINING cell lines.

    The perturbation pairs never teach the model what "no perturbation" looks like: every
    target is a perturbed cell, so nothing anchors the null-token output to the real
    control distribution. That matters whenever predictions are scored against injected
    real controls, because drift in the model's control state is then charged to every
    perturbation.

    Samples carry ``pert_indices=[-1]``, the model's control token, which zeroes the
    perturbation embedding (``pert_z = z_graph[0] * 0.0`` in model.py). They are the
    single-cell analogue of ``include_controls_as_training``, which only ever existed on
    the pseudobulk path and is therefore discarded entirely by ``--sc_only``.

    ``mode``:
      ``exact``  the input cell IS the target cell.
      ``paired`` a different control cell from the same batch, matching how perturbation
                 pairs are built (input and target are never the same cell).

    Runs AFTER the mixed-dataset cache so that switching modes does not invalidate it.
    """
    import time as _time

    if ratio <= 0:
        return dataset

    t0 = _time.time()
    rng = np.random.RandomState(seed)
    num_encoder_genes = len(encoder_gene_names)
    n_pert_pairs = len(dataset)
    total_want = int(round(n_pert_pairs * ratio))
    if total_want <= 0:
        return dataset

    # Split the budget across files in proportion to their control counts.
    avail = []
    for fm in file_metas:
        if cell_lines is not None and not set(fm.obs_cell_lines).intersection(cell_lines):
            avail.append(0)
            continue
        avail.append(int(fm.is_control.sum()) if fm.is_control is not None else 0)
    tot_avail = sum(avail)
    if tot_avail == 0:
        print("  No control cells found; skipping control identity samples")
        return dataset

    ctrl_parts, tgt_parts, names, indices, cts, masks = [], [], [], [], [], []
    for fi, fm in enumerate(file_metas):
        if avail[fi] == 0:
            continue
        n_want = int(round(total_want * avail[fi] / tot_avail))
        if n_want <= 0:
            continue
        ctrl_rows = np.where(fm.is_control)[0]
        pick = rng.choice(ctrl_rows, size=min(n_want, len(ctrl_rows)),
                          replace=n_want > len(ctrl_rows))

        if mode == "exact":
            src = pick
        elif mode == "paired":
            by_b: Dict[str, List[int]] = {}
            if fm.batch_ids is not None:
                for i in ctrl_rows:
                    by_b.setdefault(fm.batch_ids[i], []).append(int(i))
            src = np.empty_like(pick)
            for k, t in enumerate(pick):
                pool = np.asarray(by_b.get(fm.batch_ids[t] if fm.batch_ids is not None
                                           else "_", ctrl_rows), dtype=np.int64)
                cand = pool[pool != t]
                src[k] = cand[rng.randint(len(cand))] if len(cand) else t
        else:
            raise ValueError(f"unknown control identity mode {mode!r}")

        rows = np.unique(np.concatenate([src, pick]))
        data = _load_rows_chunked(fm.path, rows, fm.n_vars, fm.src_indices,
                                  fm.dst_indices, num_encoder_genes, target_sum)
        pos = {int(r): i for i, r in enumerate(rows)}
        ctrl_parts.append(data[[pos[int(r)] for r in src]])
        tgt_parts.append(data[[pos[int(r)] for r in pick]])
        names.extend(["non-targeting"] * len(pick))
        indices.extend([[-1]] * len(pick))
        cts.extend([str(fm.obs_cell_lines[i]) for i in pick])
        masks.append(torch.from_numpy(np.tile(fm.gene_mask.numpy(), (len(pick), 1))))
        print(f"    {Path(fm.path).name}: +{len(pick)} control identity ({mode})")
        del data

    if not ctrl_parts:
        return dataset

    dataset.control_profiles = torch.cat([dataset.control_profiles] + ctrl_parts)
    dataset.perturbed_profiles = torch.cat([dataset.perturbed_profiles] + tgt_parts)
    dataset.pert_gene_names = dataset.pert_gene_names + names
    dataset.pert_indices = dataset.pert_indices + indices
    dataset.cell_types = dataset.cell_types + cts
    dataset.gene_masks = torch.cat([dataset.gene_masks] + masks)
    if dataset.batch_ids is not None:
        dataset.batch_ids = dataset.batch_ids + [None] * len(names)
    if dataset.is_sc_flags is not None:
        dataset.is_sc_flags = dataset.is_sc_flags + [True] * len(names)
    print(f"  Control identity: {n_pert_pairs} -> {len(dataset)} samples "
          f"(+{len(names)}, mode={mode}, ratio={ratio}) in {_time.time() - t0:.1f}s")
    return dataset


def build_mixed_dataset(
    file_metas: List[FileMetadata],
    pert2id: Dict[str, int],
    split_pert_names: Set[str],
    encoder_gene_names: List[str],
    target_sum: float = 1e4,
    cell_lines: Optional[List[str]] = None,
    exclude_cl_perts: Optional[Dict[str, Set[str]]] = None,
    seed: int = 42,
    cache_dir: Optional[str] = None,
    sc_only: bool = False,
    max_sc_pairs_per_pert: int = 0,
    include_pb: bool = True,
) -> PseudobulkDataset:
    """Build dataset with both pseudobulk AND single-cell paired samples.

    Pseudobulk samples are built via build_pseudobulk_dataset() (unchanged).
    Single-cell samples pair each perturbation cell with a deterministic control
    cell (1:1 per perturbation). Matching priority: same-batch > same-cell-line.
    The matching is seeded and fixed across epochs.

    For a given (control_cell_i, pert_gene_j), the target is always the same
    perturbed_cell_k. A control cell can appear across different perturbations
    but within one perturbation each control maps to exactly one perturbed cell.

    ``max_sc_pairs_per_pert`` caps the pairs built per (cell_line, perturbation),
    summed across that cell line's batches. 0 = uncapped (previous behaviour).
    This is a MEMORY control, not a modelling one: pairs are held as dense
    (n_pairs, num_encoder_genes) float32 tensors for BOTH control and target, plus
    a tiled gene mask, so an uncapped 4-file whole-genome build measures ~1,023 GB.
    A cap of 32 brings it near 220 GB.

    The cap is deliberately NOT per (batch, perturbation): K562 whole-genome spreads
    1.91M perturbed cells over 267 batches x 9,867 genes, so a per-batch cell already
    holds ~8 cells and capping it at 8 removes only 0.04% of pairs. Aggregating across
    batches is also the more meaningful budget -- it balances perturbations against
    each other rather than balancing batches. Both the batch visiting order and the
    cells within a batch are shuffled under the run seed, so the cap subsamples
    uniformly rather than taking a prefix.

    Results are cached to disk for fast reload on subsequent runs.
    """
    import time as _time

    num_encoder_genes = len(encoder_gene_names)

    # --- Check cache ---
    # NOTE control_identity_* is deliberately NOT part of the fingerprint. The control
    # identity samples are appended AFTER the cache (see append_control_identity), so the
    # expensive perturbation-pair cache — up to 392 GB for a whole-genome build — is shared
    # by every arm of the exact/paired/none ablation instead of being rebuilt three times.
    fingerprint = _mixed_dataset_fingerprint(
        file_metas, split_pert_names, encoder_gene_names,
        target_sum, cell_lines, exclude_cl_perts, seed,
        max_sc_pairs_per_pert, include_pb,
    )
    if cache_dir is None:
        cache_dir = str(Path(file_metas[0].path).parent)
    cache_path = Path(cache_dir) / f"mixed_dataset_cache_{fingerprint}.pt"

    if cache_path.exists():
        print(f"Loading cached mixed dataset: {cache_path}")
        t0 = _time.time()
        cached = torch.load(cache_path, weights_only=False)
        dataset = PseudobulkDataset(
            control_profiles=cached["ctrl"],
            perturbed_profiles=cached["pert"],
            pert_gene_names=cached["names"],
            pert_indices=cached["indices"],
            cell_types=cached["cell_types"],
            gene_masks=cached["masks"],
            batch_ids=cached.get("batch_ids"),
        )
        dataset.is_sc_flags = [False] * cached["n_pb"] + [True] * cached["n_sc"]
        if sc_only:
            dataset = _filter_sc_only(dataset, cached["n_pb"])
        print(
            f"  Loaded {len(dataset)} samples in {_time.time() - t0:.1f}s "
            f"({cached['n_pb']} PB + {cached['n_sc']} SC"
            f"{', kept SC only' if sc_only else ''})"
        )
        return dataset

    t0_total = _time.time()

    # --- Step 1: Build pseudobulk samples (skipped entirely when include_pb=False) ---
    # With --sc_only the pseudobulk profiles were built and then thrown away by
    # _filter_sc_only. That aggregation is the slowest part of startup: on the K562
    # whole-genome file it means grouping 1.9M cells into (batch, perturbation) cells and
    # averaging every one of them, all to discard the result. include_pb=False skips it.
    if include_pb:
        pb_dataset = build_pseudobulk_dataset(
            file_metas=file_metas,
            pert2id=pert2id,
            split_pert_names=split_pert_names,
            encoder_gene_names=encoder_gene_names,
            target_sum=target_sum,
            cell_lines=cell_lines,
            exclude_cl_perts=exclude_cl_perts,
        )
        n_pb = len(pb_dataset)
    else:
        pb_dataset = None
        n_pb = 0

    # --- Step 2: Build single-cell paired samples ---
    print("Building single-cell paired samples...")
    rng = np.random.default_rng(seed)

    # Per-file collection: {file_idx: (ctrl_row_indices, pert_row_indices,
    #                                   pert_names, pert_indices, cell_types, gene_mask)}
    per_file_pairs: Dict[int, Dict] = {}
    n_batch_matched = 0
    n_cl_matched = 0

    for fi, fm in enumerate(file_metas):
        if cell_lines is not None:
            file_cls = set(fm.obs_cell_lines)
            if not file_cls.intersection(cell_lines):
                continue

        if fm.batch_ids is None:
            print(f"  Skipping {Path(fm.path).name}: no batch_ids")
            continue

        ctrl_indices_all = np.where(fm.is_control)[0]
        pert_indices_all = np.where(~fm.is_control)[0]

        if len(ctrl_indices_all) == 0 or len(pert_indices_all) == 0:
            continue

        # Group controls by batch_id and cell_line
        ctrl_bids = fm.batch_ids[ctrl_indices_all]
        ctrl_cls = fm.obs_cell_lines[ctrl_indices_all]

        ctrl_by_batch: Dict[str, np.ndarray] = {}
        for bid in np.unique(ctrl_bids):
            ctrl_by_batch[bid] = ctrl_indices_all[ctrl_bids == bid]

        ctrl_by_cl: Dict[str, np.ndarray] = {}
        for cl in np.unique(ctrl_cls):
            ctrl_by_cl[cl] = ctrl_indices_all[ctrl_cls == cl]

        # Build pert lookup: {(batch_id, gene_name): [row_indices]}
        pert_cls = fm.obs_cell_lines[pert_indices_all]
        pert_bids = fm.batch_ids[pert_indices_all]
        pert_gnames = fm.pert_gene_names[pert_indices_all]

        pert_by_batch_gene: Dict[Tuple[str, str], List[int]] = {}
        for idx in range(len(pert_indices_all)):
            cl = pert_cls[idx]
            gname = pert_gnames[idx]
            bid = pert_bids[idx]

            if cell_lines is not None and cl not in cell_lines:
                continue
            if gname not in split_pert_names:
                continue
            if exclude_cl_perts and cl in exclude_cl_perts:
                if gname in exclude_cl_perts[cl]:
                    continue
            if gname not in pert2id:
                continue

            pert_by_batch_gene.setdefault((bid, gname), []).append(
                int(pert_indices_all[idx])
            )

        unique_cls = sorted(set(fm.obs_cell_lines))
        if cell_lines is not None:
            unique_cls = [cl for cl in unique_cls if cl in cell_lines]
        cl_label = unique_cls[0] if len(unique_cls) == 1 else f"{len(unique_cls)} cell lines"

        file_ctrl_list = []
        file_pert_list = []
        file_names = []
        file_indices = []
        file_cts = []
        file_batch_matched = 0
        file_cl_matched = 0

        # The cap is per (cell_line, perturbation), NOT per (batch, perturbation). Measured:
        # K562 whole-genome spreads 1.91M perturbed cells over 267 batches x 9,867 genes, so
        # a per-batch cell already holds ~8 cells and capping it at 8 removes 0.04% of pairs.
        # Aggregating across batches is also the more meaningful budget — it balances
        # perturbations against each other rather than balancing batches.
        #
        # Keys are visited in a seeded random order so the cap subsamples batches uniformly;
        # dict order here is row-scan order, which would systematically favour whichever
        # batches appear first in the file.
        cl_gene_taken: Dict[Tuple[str, str], int] = {}
        _keys = list(pert_by_batch_gene.keys())
        if max_sc_pairs_per_pert > 0:
            rng.shuffle(_keys)

        for (bid, gname) in _keys:
            pert_row_list = pert_by_batch_gene[(bid, gname)]
            sample_cl = bid.split("__", 1)[0]
            pert_rows = np.array(pert_row_list)
            n_pert = len(pert_rows)

            if bid in ctrl_by_batch:
                avail_ctrl = ctrl_by_batch[bid].copy()
                batch_matched = True
            elif sample_cl in ctrl_by_cl:
                avail_ctrl = ctrl_by_cl[sample_cl].copy()
                batch_matched = False
            else:
                continue

            n_pairs = min(n_pert, len(avail_ctrl))
            if max_sc_pairs_per_pert > 0:
                room = max_sc_pairs_per_pert - cl_gene_taken.get((sample_cl, gname), 0)
                n_pairs = min(n_pairs, max(room, 0))
                if n_pairs == 0:
                    continue
                cl_gene_taken[(sample_cl, gname)] = (
                    cl_gene_taken.get((sample_cl, gname), 0) + n_pairs)

            rng.shuffle(pert_rows)
            rng.shuffle(avail_ctrl)

            file_pert_list.append(pert_rows[:n_pairs])
            file_ctrl_list.append(avail_ctrl[:n_pairs])
            file_names.extend([gname] * n_pairs)
            file_indices.extend([[pert2id[gname]]] * n_pairs)
            file_cts.extend([sample_cl] * n_pairs)

            if batch_matched:
                file_batch_matched += n_pairs
            else:
                file_cl_matched += n_pairs

        if file_ctrl_list:
            all_ctrl = np.concatenate(file_ctrl_list)
            all_pert = np.concatenate(file_pert_list)
            per_file_pairs[fi] = {
                "ctrl_rows": all_ctrl,
                "pert_rows": all_pert,
                "names": file_names,
                "indices": file_indices,
                "cell_types": file_cts,
                "gene_mask": fm.gene_mask.numpy(),
            }
            n_batch_matched += file_batch_matched
            n_cl_matched += file_cl_matched
            print(
                f"  {cl_label} ({Path(fm.path).name}): "
                f"{len(all_ctrl)} SC pairs ({file_batch_matched} batch-matched, "
                f"{file_cl_matched} cell-line-matched)"
            )

    # --- Step 3: Load expression data via chunked reads ---
    n_sc = sum(len(d["ctrl_rows"]) for d in per_file_pairs.values())
    if n_sc == 0:
        if pb_dataset is None:
            raise ValueError(
                "No single-cell pairs could be built and pseudobulk is disabled. Check "
                "that --batch_col matches the data and that the split has perturbations.")
        print("  No single-cell pairs found, returning pseudobulk-only dataset")
        return pb_dataset

    print(f"  Loading {n_sc} single-cell expression pairs (chunked reads)...")

    sc_ctrl_parts = []
    sc_pert_parts = []
    sc_names = []
    sc_indices = []
    sc_cts = []
    sc_masks = []

    for fi in sorted(per_file_pairs.keys()):
        fm = file_metas[fi]
        d = per_file_pairs[fi]
        t0 = _time.time()

        ctrl_data = _load_rows_chunked(
            fm.path, d["ctrl_rows"], fm.n_vars,
            fm.src_indices, fm.dst_indices,
            num_encoder_genes, target_sum,
        )
        pert_data = _load_rows_chunked(
            fm.path, d["pert_rows"], fm.n_vars,
            fm.src_indices, fm.dst_indices,
            num_encoder_genes, target_sum,
        )

        sc_ctrl_parts.append(ctrl_data)
        sc_pert_parts.append(pert_data)
        sc_names.extend(d["names"])
        sc_indices.extend(d["indices"])
        sc_cts.extend(d["cell_types"])

        n_file = len(d["ctrl_rows"])
        mask_tile = np.tile(d["gene_mask"], (n_file, 1))
        sc_masks.append(torch.from_numpy(mask_tile))

        elapsed = _time.time() - t0
        print(f"    {Path(fm.path).name}: {n_file} pairs in {elapsed:.1f}s")

    sc_ctrl_tensor = torch.cat(sc_ctrl_parts)
    sc_pert_tensor = torch.cat(sc_pert_parts)
    sc_mask_tensor = torch.cat(sc_masks)

    # --- Step 4: Concatenate PB + SC (SC only when include_pb=False) ---
    if pb_dataset is not None:
        combined_ctrl = torch.cat([pb_dataset.control_profiles, sc_ctrl_tensor])
        combined_pert = torch.cat([pb_dataset.perturbed_profiles, sc_pert_tensor])
        combined_names = pb_dataset.pert_gene_names + sc_names
        combined_indices = pb_dataset.pert_indices + sc_indices
        combined_cts = pb_dataset.cell_types + sc_cts
        combined_masks = torch.cat([pb_dataset.gene_masks, sc_mask_tensor])
        combined_bids = (pb_dataset.batch_ids or [None] * n_pb) + [None] * n_sc
    else:
        combined_ctrl, combined_pert = sc_ctrl_tensor, sc_pert_tensor
        combined_names, combined_indices = sc_names, sc_indices
        combined_cts, combined_masks = sc_cts, sc_mask_tensor
        combined_bids = [None] * n_sc

    dataset = PseudobulkDataset(
        control_profiles=combined_ctrl,
        perturbed_profiles=combined_pert,
        pert_gene_names=combined_names,
        pert_indices=combined_indices,
        cell_types=combined_cts,
        gene_masks=combined_masks,
        batch_ids=combined_bids,
    )
    dataset.is_sc_flags = [False] * n_pb + [True] * n_sc

    # --- Save cache (atomically) ---
    # The fingerprint depends only on (files, split, genes, target_sum, cell lines, seed,
    # cap), so ARMS THAT DIFFER ONLY IN A POST-CACHE SETTING SHARE A PATH and, when run
    # concurrently, previously interleaved their writes into one another. Observed: two
    # array tasks writing the same 249 GB file at once. A torch.load of that is either a
    # crash or, worse, silently wrong tensors.
    #
    # Write to a pid-unique temp then os.replace, which is atomic within a filesystem: a
    # reader sees either no file or a complete one, and a loser of the race simply discards
    # its own copy.
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f".{cache_path.name}.tmp{os.getpid()}")
    print(f"  Saving cache: {cache_path}")
    try:
        torch.save({
            "ctrl": combined_ctrl,
            "pert": combined_pert,
            "names": combined_names,
            "indices": combined_indices,
            "cell_types": combined_cts,
            "masks": combined_masks,
            "batch_ids": combined_bids,
            "n_pb": n_pb,
            "n_sc": n_sc,
        }, tmp_path)
        if cache_path.exists():
            print("    another process finished this cache first; discarding our copy")
            tmp_path.unlink(missing_ok=True)
        else:
            os.replace(tmp_path, cache_path)
    except Exception as exc:                                    # noqa: BLE001
        # A half-written cache is far more dangerous than no cache.
        tmp_path.unlink(missing_ok=True)
        print(f"    WARNING: cache write failed and was rolled back: {exc}")

    if sc_only:
        dataset = _filter_sc_only(dataset, n_pb)

    elapsed_total = _time.time() - t0_total
    print(
        f"  Built mixed dataset: {n_pb} pseudobulk + {n_sc} single-cell "
        f"= {len(dataset)} total samples "
        f"({n_batch_matched} batch-matched, {n_cl_matched} cell-line-matched) "
        f"in {elapsed_total:.1f}s"
        f"{' [SC only: kept ' + str(n_sc) + ' samples]' if sc_only else ''}"
    )
    return dataset
