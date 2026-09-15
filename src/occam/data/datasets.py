"""Dataset classes for Occam training."""
from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import anndata
import h5py
import numpy as np
import torch
from scipy.sparse import issparse
from torch.utils.data import Dataset, IterableDataset

from occam.data.chunked import OptimizedChunkedDataset
from occam.transforms import Log1p, NormalizePerCell, Scale, OccamTransform

logger = logging.getLogger(__name__)

# Common gene name columns to try during auto-detection, in priority order.
_GENE_NAME_CANDIDATES = [
    "gene_name",
    "gene_symbol",
    "gene_symbols",
    "symbol",
    "Gene",
    "feature_name",
    "feature_names",
    "ensembl_gene_id",
    "gene_id",
]

# AnnData attributes purged after loading to save memory.
_PURGE_ADATA_ATTRIBUTES = ("obsm", "varm", "obsp", "varp", "layers", "uns", "raw")


def _detect_gene_name_col(var_df, hint: str = None) -> str:
    """Auto-detect the gene name column in adata.var.

    Args:
        var_df: adata.var DataFrame.
        hint: Preferred column name to try first.

    Returns:
        Column name to use, or "index" if falling back to var.index.
    """
    # If caller explicitly requested a column, honour it immediately.
    if hint and hint in var_df.columns:
        return hint
    if hint and hint == "index":
        return "index"

    for col in _GENE_NAME_CANDIDATES:
        if col in var_df.columns:
            return col

    return "index"


# ---------------------------------------------------------------------------
# Multi-file folder support (optim mode)
# ---------------------------------------------------------------------------


def _h5ad_x_shape(f: h5py.File) -> tuple[int, int]:
    """Get the shape of X whether stored as a dense dataset or sparse group."""
    x = f["X"]
    if hasattr(x, "shape"):
        return tuple(x.shape)
    # Sparse matrix (CSR/CSC): shape is stored in group attributes
    return tuple(x.attrs["shape"])


@dataclass
class FileInfo:
    """Lightweight metadata for a single h5ad file in a multi-file folder."""

    path: str
    n_obs: int
    n_vars: int
    gene_names: list[str]
    chunks: list[tuple[int, int]] = field(default_factory=list)
    # Gene alignment to target vocabulary (populated by _compute_alignment_maps)
    src_indices: np.ndarray | None = None  # columns in this file to read
    dst_indices: np.ndarray | None = None  # target positions in master vocab
    is_identity: bool = False  # True if same genes, same order → skip temp buffer
    overlap_pct: float = 100.0  # % of target vocab genes present in this file


def _h5_read_column(item) -> np.ndarray:
    """Read an h5py Dataset or categorical Group into a flat numpy array.

    Categorical columns in h5ad are stored as Groups with ``categories``
    and ``codes`` sub-datasets.  Regular columns are plain Datasets.
    """
    if isinstance(item, h5py.Dataset):
        return item[:]
    # Categorical group
    categories = item["categories"][:]
    codes = item["codes"][:]
    return categories[codes]


def _read_gene_names_h5py(f: h5py.File, hint: str | None = None) -> list[str]:
    """Read gene names from an h5ad file via h5py (no anndata dependency).

    Uses the same priority-based detection as ``_detect_gene_name_col`` but
    operates on h5py groups directly.  Handles both plain and categorical
    columns.

    Returns:
        List of gene name strings.
    """
    var = f["var"]

    def _decode(raw):
        return [s.decode() if isinstance(s, bytes) else str(s) for s in raw]

    # If a specific column was requested, try it first
    if hint is not None and hint != "index":
        if hint in var:
            return _decode(_h5_read_column(var[hint]))

    # Try priority candidates
    for col in _GENE_NAME_CANDIDATES:
        if col in var:
            return _decode(_h5_read_column(var[col]))

    # Fallback: use the DataFrame index (whatever var.attrs["_index"] points to)
    index_col = var.attrs.get("_index", "_index")
    if isinstance(index_col, bytes):
        index_col = index_col.decode()
    if index_col != "_index" and index_col in var:
        raw = _h5_read_column(var[index_col])
    elif "_index" in var:
        raw = _h5_read_column(var["_index"])
    else:
        raise ValueError(
            f"Cannot find gene names in {f.filename}: "
            f"no candidate columns or _index dataset in /var"
        )
    return _decode(raw)


def _scan_h5ad_folder(
    folder_path: str,
    gene_name_col: str | None = None,
    chunk_size: int = 8192,
) -> list[FileInfo]:
    """Scan a folder recursively for .h5ad files and read metadata via h5py.

    Files are sorted by path for deterministic ordering across DDP ranks.

    Returns:
        List of :class:`FileInfo` with gene alignment fields unset.
    """
    folder = Path(folder_path)
    h5ad_files = sorted(folder.rglob("*.h5ad"))
    if not h5ad_files:
        raise ValueError(f"No .h5ad files found in {folder_path}")

    file_infos: list[FileInfo] = []
    for fp in h5ad_files:
        with h5py.File(fp, "r") as f:
            n_obs, n_vars = _h5ad_x_shape(f)
            if n_obs == 0:
                print(f"  WARNING: Skipping empty file {fp.name} (0 observations)")
                continue
            gene_names = _read_gene_names_h5py(f, hint=gene_name_col)

        # Compute local chunk boundaries
        chunks = []
        for start in range(0, n_obs, chunk_size):
            end = min(start + chunk_size, n_obs)
            chunks.append((start, end))

        file_infos.append(FileInfo(
            path=str(fp),
            n_obs=n_obs,
            n_vars=n_vars,
            gene_names=gene_names,
            chunks=chunks,
        ))

    if not file_infos:
        raise ValueError(f"All .h5ad files in {folder_path} are empty")

    print(f"Scanned {len(file_infos)} h5ad files from {folder_path} "
          f"({sum(fi.n_obs for fi in file_infos):,} total cells)")
    return file_infos


def _compute_alignment_maps(
    file_infos: list[FileInfo],
    target_genes: list[str],
) -> None:
    """Compute per-file gene alignment maps in-place.

    For each file, sets ``src_indices``, ``dst_indices``, and ``is_identity``
    on the :class:`FileInfo` object.
    """
    target_map = {gene: i for i, gene in enumerate(target_genes)}
    n_target = len(target_genes)

    for fi in file_infos:
        src, dst = [], []
        for j, gene in enumerate(fi.gene_names):
            if gene in target_map:
                src.append(j)
                dst.append(target_map[gene])
        fi.src_indices = np.array(src, dtype=np.intp)
        fi.dst_indices = np.array(dst, dtype=np.intp)

        # Identity: same length, same genes, same order
        n_common = len(src)
        fi.is_identity = (
            n_common == n_target == fi.n_vars
            and np.array_equal(fi.dst_indices, np.arange(n_target))
            and np.array_equal(fi.src_indices, np.arange(fi.n_vars))
        )

        overlap_pct = n_common / n_target * 100 if n_target > 0 else 0
        fi.overlap_pct = overlap_pct
        if overlap_pct < 10:
            print(f"  WARNING: {Path(fi.path).name} has only {n_common}/{n_target} "
                  f"genes in common with target vocabulary ({overlap_pct:.1f}%)")


def compute_global_chunk_info(
    data_path: str,
    chunk_size: int,
    gene_vocabulary: list[str] | None = None,
    gene_name_col: str | None = None,
    min_gene_overlap_pct: float | None = None,
) -> tuple[int, int]:
    """Return ``(total_n_obs, total_n_chunks)`` for a file or folder.

    For a single file, reads shape via h5py.
    For a folder, scans all .h5ad files and sums.

    When *gene_vocabulary* and *min_gene_overlap_pct* are provided for a
    folder, files below the overlap threshold are excluded from the count.
    """
    if os.path.isdir(data_path):
        if min_gene_overlap_pct is not None and gene_vocabulary is not None:
            file_infos = _scan_h5ad_folder(data_path, gene_name_col, chunk_size)
            _compute_alignment_maps(file_infos, gene_vocabulary)
            file_infos = [fi for fi in file_infos if fi.overlap_pct >= min_gene_overlap_pct]
            total_obs = sum(fi.n_obs for fi in file_infos)
            total_chunks = sum(len(fi.chunks) for fi in file_infos)
            return total_obs, total_chunks
        total_obs, total_chunks = 0, 0
        for fp in sorted(Path(data_path).rglob("*.h5ad")):
            with h5py.File(fp, "r") as f:
                n = _h5ad_x_shape(f)[0]
            if n == 0:
                continue
            total_obs += n
            total_chunks += (n + chunk_size - 1) // chunk_size
        return total_obs, total_chunks
    else:
        with h5py.File(data_path, "r") as f:
            n = _h5ad_x_shape(f)[0]
        return n, (n + chunk_size - 1) // chunk_size


class OccamAdataDataset(Dataset):
    """Unified dataset for Occam that reads from h5ad files.

    Inspired by TxAdataDataset from perturbseq-map. Supports two modes:
      - ``backed``: data stays on disk, loaded row-by-row (memory efficient).
      - ``in_memory``: entire data loaded and pretransformed at init (faster training).

    By default uses **all genes** in the anndata as input features.

    Args:
        adata_path: Path to an ``.h5ad`` file.
        mode: ``"backed"`` or ``"in_memory"``.
        transforms: List of :class:`OccamTransform` instances applied per-sample.
            Defaults to ``[NormalizePerCell(1e4), Log1p()]``.
        gene_name_col: Column in ``adata.var`` containing gene names.
            ``None`` for auto-detection.
        gene_vocabulary: Optional fixed gene list for alignment (e.g. from a
            training checkpoint). Genes missing from the file are zero-filled;
            extra genes in the file are ignored.
    """

    def __init__(
        self,
        adata_path: str,
        mode: str = "backed",
        transforms: list[OccamTransform] | None = None,
        gene_name_col: str | None = None,
        gene_vocabulary: list[str] | None = None,
        compact: bool = False,
    ):
        if mode not in ("backed", "in_memory"):
            raise ValueError(f"mode must be 'backed' or 'in_memory', got '{mode}'")
        if not str(adata_path).endswith(".h5ad"):
            raise ValueError(f"Expected .h5ad file, got {adata_path}")

        self.file_path = str(adata_path)
        self.mode = mode
        self._compact = compact
        self.transforms = transforms if transforms is not None else [NormalizePerCell(1e4), Log1p()]

        # -- Open h5ad (always open backed first to read metadata cheaply) --
        adata = anndata.read_h5ad(adata_path, backed="r")
        for attr in _PURGE_ADATA_ATTRIBUTES:
            if hasattr(adata, attr):
                try:
                    delattr(adata, attr)
                except Exception:
                    pass

        self.n_obs = adata.n_obs
        self.n_vars = adata.n_vars

        # -- Detect and store gene names --
        col = _detect_gene_name_col(adata.var, hint=gene_name_col)
        if col == "index":
            self._original_gene_labels: list[str] = adata.var.index.tolist()
        else:
            self._original_gene_labels = adata.var[col].tolist()
        self._gene_name_col_used = col

        # -- Gene vocabulary alignment --
        self._gene_vocabulary = gene_vocabulary
        if gene_vocabulary is not None:
            self._setup_gene_vocabulary(self._original_gene_labels, gene_vocabulary)
        else:
            self.gene_names: list[str] = list(self._original_gene_labels)
            self._gene_indices: torch.Tensor | None = None
            self._gene_in_dataset: torch.Tensor | None = None

        # -- Mode-specific init --
        self._pretransformed = False
        if mode == "in_memory":
            self._load_into_memory(adata)
            self.adata = None
        else:
            self.adata = adata
            self.X_data = None

        memory_info = ""
        if self.X_data is not None:
            memory_info = f", {self.X_data.nbytes / (1024**3):.2f} GB in memory"
        print(
            f"OccamAdataDataset initialized: {self.n_obs:,} cells x {self.num_genes:,} genes "
            f"(mode={mode}{memory_info})"
        )
        print(f"  Gene names source: adata.var['{self._gene_name_col_used}']")
        print(f"  Transforms: {[type(t).__name__ for t in self.transforms]}")

    # ------------------------------------------------------------------
    # Gene vocabulary alignment (borrowed from TxAdataDataset pattern)
    # ------------------------------------------------------------------

    def _setup_gene_vocabulary(self, original_labels: list[str], vocabulary: list[str]) -> None:
        """Map from *vocabulary* to *original_labels* so we can reindex."""
        index_map = {gene: i for i, gene in enumerate(original_labels)}
        index_list = [index_map.get(gene, -1) for gene in vocabulary]
        self._gene_indices = torch.tensor(index_list, dtype=torch.long)
        self._gene_in_dataset = self._gene_indices >= 0
        self.gene_names = list(vocabulary)

        n_common = int(self._gene_in_dataset.sum().item())
        print(
            f"  Gene vocabulary: {len(vocabulary)} requested, "
            f"{n_common} found in file ({len(vocabulary) - n_common} will be zero-filled)"
        )

    def _fill_observed_gene_vector(self, gene_vector: torch.Tensor) -> torch.Tensor:
        """Expand/reindex a vector from file gene order to vocabulary order."""
        if self._gene_indices is None:
            return gene_vector
        expanded = torch.zeros(len(self.gene_names), dtype=gene_vector.dtype)
        available_indices = self._gene_indices[self._gene_in_dataset]
        expanded[self._gene_in_dataset] = gene_vector[available_indices]
        return expanded

    # ------------------------------------------------------------------
    # In-memory loading
    # ------------------------------------------------------------------

    def _load_into_memory(self, adata) -> None:
        """Chunk-read the full X matrix, apply transforms vectorized, store result."""
        chunk_size = 10000
        n_obs = adata.n_obs

        # Determine output gene count
        out_genes = len(self.gene_names)

        # Read in chunks and apply transforms
        chunks = []
        for start in range(0, n_obs, chunk_size):
            end = min(start + chunk_size, n_obs)
            chunk = adata.X[start:end]
            if issparse(chunk):
                chunk = chunk.toarray()
            chunk = np.asarray(chunk, dtype=np.float32)

            # Apply gene vocabulary reindexing at raw level if needed
            if self._gene_indices is not None and not self._compact:
                reindexed = np.zeros((chunk.shape[0], out_genes), dtype=np.float32)
                mask = self._gene_in_dataset.numpy()
                source_indices = self._gene_indices[self._gene_in_dataset].numpy()
                reindexed[:, mask] = chunk[:, source_indices]
                chunk = reindexed

            chunks.append(chunk)

        self.X_data = np.vstack(chunks)

        # Apply transforms vectorized on the full matrix
        self._apply_transforms_vectorized()
        self._pretransformed = True

        # Ensure C-contiguous for tensor creation speed
        if not self.X_data.flags["C_CONTIGUOUS"]:
            self.X_data = np.ascontiguousarray(self.X_data)

        # Close the backed adata
        del adata
        gc.collect()

    def _apply_transforms_vectorized(self) -> None:
        """Apply all transforms to self.X_data in-place (vectorized numpy ops)."""
        X = self.X_data
        for t in self.transforms:
            if isinstance(t, NormalizePerCell):
                row_sums = X.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums == 0, 1.0, row_sums)
                X = X / row_sums * t.target_sum
            elif isinstance(t, Log1p):
                X = np.log1p(X)
            else:
                # Generic fallback: apply row-by-row with torch tensors
                rows = []
                for i in range(X.shape[0]):
                    rows.append(t(torch.tensor(X[i], dtype=torch.float32)).numpy())
                X = np.stack(rows)
        self.X_data = X.astype(np.float32)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_obs

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "in_memory":
            x = torch.tensor(self.X_data[idx], dtype=torch.float32)
            if self._compact and self._gene_indices is not None:
                x = self._fill_observed_gene_vector(x)
        else:
            row = self.adata.X[idx]
            if issparse(row):
                row = row.toarray().flatten()
            else:
                row = np.asarray(row).flatten()
            x = torch.tensor(row, dtype=torch.float32)

            # Apply transforms per-sample
            for t in self.transforms:
                x = t(x)

            # Vocabulary expansion
            if self._gene_indices is not None:
                x = self._fill_observed_gene_vector(x)

        return (x, torch.tensor(0, dtype=torch.int64))

    @property
    def num_genes(self) -> int:
        """Number of genes in the output vectors."""
        return len(self.gene_names)

    def get_dataset_info(self) -> dict:
        """Return metadata dict suitable for saving in a checkpoint."""
        return {
            "gene_names": list(self.gene_names),
            "gene_name_col": self._gene_name_col_used,
            "transforms": [type(t).__name__ for t in self.transforms],
            "target_sum": next(
                (t.target_sum for t in self.transforms if isinstance(t, NormalizePerCell)),
                None,
            ),
            "n_obs": self.n_obs,
            "n_genes": self.num_genes,
            "mode": self.mode,
            "file_path": self.file_path,
        }

    # ------------------------------------------------------------------
    # Worker init for backed mode (HDF5 is not fork-safe)
    # ------------------------------------------------------------------

    @staticmethod
    def worker_init_fn(worker_id: int) -> None:
        """Reopen the h5ad in each DataLoader worker process.

        Usage::

            DataLoader(dataset, worker_init_fn=OccamAdataDataset.worker_init_fn)
        """
        import torch.utils.data

        info = torch.utils.data.get_worker_info()
        if info is None:
            return
        dataset = info.dataset
        # Handle torch Subset wrapper
        while hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        if isinstance(dataset, OccamAdataDataset) and dataset.mode == "backed" and dataset.adata is not None:
            dataset.adata = anndata.read_h5ad(dataset.file_path, backed="r")
            for attr in _PURGE_ADATA_ATTRIBUTES:
                if hasattr(dataset.adata, attr):
                    try:
                        delattr(dataset.adata, attr)
                    except Exception:
                        pass


class OccamChunkedDataset(IterableDataset):
    """Fast, low-memory dataset wrapping :class:`OptimizedChunkedDataset`.

    Reads HDF5 in large chunks with prefetched I/O, applies batch-level
    transforms, and yields pre-batched tensors.  Designed as a drop-in
    ``data_mode="optim"`` alternative to :class:`OccamAdataDataset`.

    Supports both **single-file** and **folder** modes:

    * **Single file** (``adata_path`` is an ``.h5ad`` file): reads from one
      file, same behaviour as before.
    * **Folder** (``adata_path`` is a directory): recursively scans for
      ``.h5ad`` files, determines a master gene vocabulary (from the file
      with the most genes, or from ``gene_vocabulary`` if given), and aligns
      all files to that vocabulary per-chunk in the I/O thread.  Chunks from
      all files are interleaved and shuffled globally, so batches naturally
      mix data from multiple files.

    Args:
        adata_path: Path to an ``.h5ad`` file or a **folder** of h5ad files.
        batch_size: Samples per yielded batch.
        transforms: List of :class:`OccamTransform` instances.
        gene_name_col: Column in ``adata.var`` for gene names (auto-detect).
        gene_vocabulary: Optional fixed gene list for alignment.
        chunk_size: Rows per HDF5 read (default 8192).
        buffer_chunks: Chunks combined into a shuffle buffer (default 4).
        seed: Random seed for shuffling.
        shuffle: Whether to shuffle chunk order and within-buffer rows.
        drop_last: Drop the last incomplete batch.
        rank: DDP rank (0 for single-GPU).
        world_size: DDP world size (1 for single-GPU).
        chunk_indices: Optional list of chunk index integers to use.
    """

    def __init__(
        self,
        adata_path: str,
        batch_size: int = 512,
        transforms: list[OccamTransform] | None = None,
        gene_name_col: str | None = None,
        gene_vocabulary: list[str] | None = None,
        chunk_size: int = 8192,
        buffer_chunks: int = 4,
        seed: int = 42,
        shuffle: bool = True,
        drop_last: bool = False,
        rank: int = 0,
        world_size: int = 1,
        chunk_indices: list[int] | None = None,
        track_presence: bool = False,
        min_gene_overlap_pct: float | None = None,
    ):
        self.file_path = str(adata_path)
        self.batch_size = batch_size
        self.transforms = transforms if transforms is not None else [NormalizePerCell(1e4), Log1p()]
        self.chunk_size = chunk_size
        self.buffer_chunks = buffer_chunks
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        self._chunk_indices = chunk_indices
        self._track_presence = track_presence
        self._presence_table = None
        self._min_gene_overlap_pct = min_gene_overlap_pct
        self._epoch = 0

        self._is_folder = os.path.isdir(adata_path)

        if self._is_folder:
            self._init_folder(adata_path, gene_name_col, gene_vocabulary, chunk_size)
        else:
            self._init_single_file(adata_path, gene_name_col, gene_vocabulary)

        chunk_info = f"chunk_size={chunk_size}, buffer_chunks={buffer_chunks}"
        if chunk_indices is not None:
            chunk_info += f", using {len(chunk_indices)}/{self.compute_chunk_count()} chunks"
        mode_label = f"folder, {len(self._file_infos)} files" if self._is_folder else "single file"
        print(
            f"OccamChunkedDataset initialized ({mode_label}): "
            f"{self.n_obs:,} cells x {self.num_genes:,} genes "
            f"({chunk_info}, shuffle={shuffle}, rank={rank}/{world_size})"
        )
        print(f"  Transforms: {[type(t).__name__ for t in self.transforms]}")

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _init_single_file(
        self,
        adata_path: str,
        gene_name_col: str | None,
        gene_vocabulary: list[str] | None,
    ) -> None:
        """Initialise from a single h5ad file (existing behaviour)."""
        self._file_infos = None  # not used in single-file mode

        adata = anndata.read_h5ad(adata_path, backed="r")
        for attr in _PURGE_ADATA_ATTRIBUTES:
            if hasattr(adata, attr):
                try:
                    delattr(adata, attr)
                except Exception:
                    pass

        self.n_obs = adata.n_obs
        self.n_vars = adata.n_vars

        col = _detect_gene_name_col(adata.var, hint=gene_name_col)
        if col == "index":
            self._original_gene_labels: list[str] = adata.var.index.tolist()
        else:
            self._original_gene_labels = adata.var[col].tolist()
        self._gene_name_col_used = col

        self._gene_vocabulary = gene_vocabulary
        if gene_vocabulary is not None:
            index_map = {gene: i for i, gene in enumerate(self._original_gene_labels)}
            index_list = [index_map.get(gene, -1) for gene in gene_vocabulary]
            self._gene_indices = torch.tensor(index_list, dtype=torch.long)
            self._gene_in_dataset = self._gene_indices >= 0
            self.gene_names: list[str] = list(gene_vocabulary)
            n_common = int(self._gene_in_dataset.sum().item())
            print(
                f"  Gene vocabulary: {len(gene_vocabulary)} requested, "
                f"{n_common} found in file ({len(gene_vocabulary) - n_common} zero-filled)"
            )
        else:
            self.gene_names = list(self._original_gene_labels)
            self._gene_indices = None
            self._gene_in_dataset = None

        del adata
        gc.collect()

        print(f"  Gene names source: adata.var['{self._gene_name_col_used}']")

    def _init_folder(
        self,
        folder_path: str,
        gene_name_col: str | None,
        gene_vocabulary: list[str] | None,
        chunk_size: int,
    ) -> None:
        """Initialise from a folder of h5ad files with gene alignment."""
        self._file_infos = _scan_h5ad_folder(folder_path, gene_name_col, chunk_size)

        # Determine effective vocabulary
        if gene_vocabulary is not None:
            effective_vocab = list(gene_vocabulary)
        else:
            # Use genes from the file with the most genes
            master_idx = max(
                range(len(self._file_infos)),
                key=lambda i: self._file_infos[i].n_vars,
            )
            effective_vocab = self._file_infos[master_idx].gene_names
            print(f"  Master vocabulary from: {Path(self._file_infos[master_idx].path).name} "
                  f"({self._file_infos[master_idx].n_vars:,} genes)")

        # Compute per-file gene alignment maps
        _compute_alignment_maps(self._file_infos, effective_vocab)

        # Filter files by minimum gene overlap
        if self._min_gene_overlap_pct is not None:
            before = len(self._file_infos)
            excluded = [fi for fi in self._file_infos if fi.overlap_pct < self._min_gene_overlap_pct]
            self._file_infos = [fi for fi in self._file_infos if fi.overlap_pct >= self._min_gene_overlap_pct]
            if excluded:
                print(f"\n  Excluded {len(excluded)} file(s) with <{self._min_gene_overlap_pct:.1f}% gene overlap:")
                for fi in excluded:
                    print(f"    - {Path(fi.path).name} ({fi.overlap_pct:.1f}%, {fi.n_obs:,} cells)")
            if not self._file_infos:
                raise ValueError(
                    f"All {before} files excluded by min_gene_overlap_pct={self._min_gene_overlap_pct}%. "
                    f"Lower the threshold or check gene vocabulary."
                )

        self.n_obs = sum(fi.n_obs for fi in self._file_infos)
        self.n_vars = len(effective_vocab)
        self.gene_names = list(effective_vocab)
        self._original_gene_labels = list(effective_vocab)
        self._gene_name_col_used = "auto"
        self._gene_vocabulary = gene_vocabulary
        # Alignment is handled at the I/O level, not batch level
        self._gene_indices = None
        self._gene_in_dataset = None

        # Print per-file summary table
        self._print_folder_summary(effective_vocab)

        # Precompute per-file presence vectors for compact file-index encoding
        if self._track_presence:
            n = len(effective_vocab)
            self._presence_table = torch.zeros(len(self._file_infos), n)
            for i, fi in enumerate(self._file_infos):
                if fi.is_identity:
                    self._presence_table[i, :] = 1.0
                else:
                    self._presence_table[i, fi.dst_indices] = 1.0

    def _print_folder_summary(self, effective_vocab: list[str]) -> None:
        """Print a formatted table summarising all files in the folder."""
        n_vocab = len(effective_vocab)
        total_cells = sum(fi.n_obs for fi in self._file_infos)
        total_chunks = sum(len(fi.chunks) for fi in self._file_infos)

        # Column widths
        names = [Path(fi.path).name for fi in self._file_infos]
        w_name = max(len(n) for n in names)
        w_name = max(w_name, 4)  # min width for header "File"

        sep = "  "
        header = (
            f"  {'#':<4}{sep}"
            f"{'File':<{w_name}}{sep}"
            f"{'Cells':>10}{sep}"
            f"{'Genes':>7}{sep}"
            f"{'Overlap':>9}{sep}"
            f"{'Chunks':>7}{sep}"
            f"{'Alignment'}"
        )
        rule = "  " + "-" * (len(header) - 2)

        print(f"\n  Training data: {len(self._file_infos)} files, "
              f"{total_cells:,} cells, {n_vocab:,} master genes, "
              f"{total_chunks} chunks")
        print(rule)
        print(header)
        print(rule)

        for i, fi in enumerate(self._file_infos):
            n_common = len(fi.src_indices) if fi.src_indices is not None else 0
            overlap_pct = n_common / n_vocab * 100 if n_vocab else 0
            if fi.is_identity:
                align = "identity"
            elif overlap_pct >= 99.9:
                align = "reordered"
            else:
                align = f"{n_common:,}/{n_vocab:,}"
            print(
                f"  {i + 1:<4}{sep}"
                f"{names[i]:<{w_name}}{sep}"
                f"{fi.n_obs:>10,}{sep}"
                f"{fi.n_vars:>7,}{sep}"
                f"{overlap_pct:>8.1f}%{sep}"
                f"{len(fi.chunks):>7}{sep}"
                f"{align}"
            )

        print(rule)
        print(f"  {'':4}{sep}"
              f"{'TOTAL':<{w_name}}{sep}"
              f"{total_cells:>10,}{sep}"
              f"{n_vocab:>7,}{sep}"
              f"{'':>9}{sep}"
              f"{total_chunks:>7}")
        print()

    # ------------------------------------------------------------------
    # Properties and epoch handling
    # ------------------------------------------------------------------

    def compute_chunk_count(self) -> int:
        """Return the total number of chunks across all files."""
        if self._is_folder:
            return sum(len(fi.chunks) for fi in self._file_infos)
        return (self.n_obs + self.chunk_size - 1) // self.chunk_size

    @property
    def num_genes(self) -> int:
        return len(self.gene_names)

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __len__(self) -> int:
        """Approximate number of batches (for progress bars)."""
        if self._chunk_indices is not None:
            n_chunks = len(self._chunk_indices)
        else:
            n_chunks = self.compute_chunk_count()
        if self.world_size > 1:
            n_chunks = n_chunks // self.world_size
        total_rows = n_chunks * self.chunk_size
        return max(1, total_rows // self.batch_size)

    # ------------------------------------------------------------------
    # Batch transforms and gene reindexing
    # ------------------------------------------------------------------

    def _reindex_genes_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Reindex columns of a batch tensor to match gene vocabulary."""
        if self._gene_indices is None:
            return batch
        expanded = torch.zeros(batch.shape[0], len(self.gene_names), dtype=batch.dtype)
        available_indices = self._gene_indices[self._gene_in_dataset]
        expanded[:, self._gene_in_dataset] = batch[:, available_indices]
        return expanded

    def _apply_batch_transforms(self, batch: torch.Tensor) -> torch.Tensor:
        """Apply transforms vectorised on a (batch_size, n_genes) tensor."""
        for t in self.transforms:
            if isinstance(t, NormalizePerCell):
                row_sums = batch.sum(dim=1, keepdim=True)
                row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
                batch = batch / row_sums * t.target_sum
            elif isinstance(t, Log1p):
                batch = torch.log1p(batch)
            elif isinstance(t, Scale):
                batch = (batch - t.mean.unsqueeze(0)) / t.std.unsqueeze(0)
            else:
                batch = torch.stack([t(row) for row in batch])
        return batch

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[torch.Tensor]:
        if self._is_folder:
            yield from self._iter_multi_file()
        else:
            yield from self._iter_single_file()

    def _partition_chunks_for_ddp(self, all_chunks: list) -> list:
        """Partition chunks across DDP ranks using contiguous blocks.

        Contiguous assignment ensures each rank reads from mostly different
        files, avoiding GPFS lock contention from interleaved I/O patterns.

        Partial chunks (last chunk of each file, fewer than chunk_size rows)
        are dropped to guarantee identical batch counts across ranks —
        otherwise the rank with fewer rows finishes its training loop early,
        causing a DDP all_reduce deadlock (SIGABRT).
        """
        if self.world_size <= 1:
            return all_chunks
        # Drop partial chunks to guarantee equal batch counts across ranks
        if self._is_folder:
            all_chunks = [c for c in all_chunks if c[2] - c[1] == self.chunk_size]
        else:
            all_chunks = [c for c in all_chunks if c[1] - c[0] == self.chunk_size]
        # Truncate so every rank gets the same number of chunks
        usable = len(all_chunks) // self.world_size * self.world_size
        all_chunks = all_chunks[:usable]
        per_rank = usable // self.world_size
        start = self.rank * per_rank
        return all_chunks[start : start + per_rank]

    def _iter_single_file(self) -> Iterator:
        """Iterate from a single h5ad file (existing behaviour)."""
        ds = OptimizedChunkedDataset(
            h5ad_path=self.file_path,
            batch_size=self.batch_size,
            chunk_size=self.chunk_size,
            buffer_chunks=self.buffer_chunks,
            shuffle=self.shuffle,
            prefetch=2,
            seed=self.seed,
            drop_last=self.drop_last,
        )
        ds._epoch = self._epoch

        if self._chunk_indices is not None:
            all_chunks = ds._chunks
            ds._chunks = [all_chunks[i] for i in self._chunk_indices]

        ds._chunks = self._partition_chunks_for_ddp(ds._chunks)

        for batch in ds:
            batch = self._reindex_genes_batch(batch)
            batch = self._apply_batch_transforms(batch)
            if self._track_presence:
                yield (batch, None)
            else:
                yield batch

    def _iter_multi_file(self) -> Iterator:
        """Iterate from multiple h5ad files with gene alignment in I/O thread."""
        ds = OptimizedChunkedDataset(
            h5ad_path=None,
            batch_size=self.batch_size,
            chunk_size=self.chunk_size,
            buffer_chunks=self.buffer_chunks,
            shuffle=self.shuffle,
            prefetch=2,
            seed=self.seed,
            drop_last=self.drop_last,
            file_infos=self._file_infos,
            n_target_genes=len(self.gene_names),
            track_presence=self._track_presence,
        )
        ds._epoch = self._epoch

        if self._chunk_indices is not None:
            all_chunks = ds._chunks
            ds._chunks = [all_chunks[i] for i in self._chunk_indices]

        ds._chunks = self._partition_chunks_for_ddp(ds._chunks)

        for batch in ds:
            if self._track_presence:
                n = self.num_genes
                expr = batch[:, :n]
                file_idx = batch[:, n].long()
                pres = self._presence_table[file_idx]
                expr = self._apply_batch_transforms(expr)
                yield (expr, pres)
            else:
                batch = self._apply_batch_transforms(batch)
                yield batch

    # ------------------------------------------------------------------
    # Dataset info
    # ------------------------------------------------------------------

    def get_dataset_info(self) -> dict:
        """Return metadata dict suitable for saving in a checkpoint."""
        info = {
            "gene_names": list(self.gene_names),
            "gene_name_col": self._gene_name_col_used,
            "transforms": [type(t).__name__ for t in self.transforms],
            "target_sum": next(
                (t.target_sum for t in self.transforms if isinstance(t, NormalizePerCell)),
                None,
            ),
            "n_obs": self.n_obs,
            "n_genes": self.num_genes,
            "mode": "optim",
        }
        if self._is_folder:
            info["folder_path"] = self.file_path
            info["n_files"] = len(self._file_infos)
            info["file_paths"] = [fi.path for fi in self._file_infos]
        else:
            info["file_path"] = self.file_path
        return info

    def get_folder_summary(self) -> dict:
        """Return a dict summarising all files for wandb config logging.

        Returns a dict with per-file info and totals, e.g.::

            {
                "datasets": {"file1.h5ad": {"n_samples": 1000, "n_genes": 500}, ...},
                "total_samples": 5000,
                "total_genes": 600,
                "n_files": 3,
            }
        """
        datasets = {}
        if self._file_infos is None:
            return None
        for fi in self._file_infos:
            datasets[Path(fi.path).name] = {
                "n_samples": fi.n_obs,
                "n_genes": fi.n_vars,
            }
        return {
            "datasets": datasets,
            "total_samples": sum(fi.n_obs for fi in self._file_infos),
            "total_genes": len(self.gene_names),
            "n_files": len(self._file_infos),
        }


def optim_train_val_chunk_split(
    n_obs: int | None = None,
    chunk_size: int = 8192,
    val_fraction: float = 0.1,
    seed: int = 42,
    total_chunks: int | None = None,
) -> tuple[list[int], list[int]]:
    """Split chunk indices into train and val sets for optim mode.

    Returns two disjoint lists of chunk-index integers.  The split is
    deterministic for a given chunk count and ``seed``.

    Args:
        n_obs: Total number of observations (used to compute chunk count
            when *total_chunks* is not provided).
        chunk_size: Rows per chunk (must match the dataset).
        val_fraction: Fraction of chunks to reserve for validation.
        seed: Random seed for reproducibility.
        total_chunks: If provided, use this directly instead of computing
            from *n_obs* and *chunk_size*.  Useful for multi-file folders.

    Returns:
        ``(train_chunk_ids, val_chunk_ids)``
    """
    if total_chunks is None:
        if n_obs is None:
            raise ValueError("Either n_obs or total_chunks must be provided")
        total_chunks = (n_obs + chunk_size - 1) // chunk_size
    n_val = max(1, int(total_chunks * val_fraction))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(total_chunks)
    val_ids = sorted(perm[:n_val].tolist())
    train_ids = sorted(perm[n_val:].tolist())

    return train_ids, val_ids


# ---------------------------------------------------------------------------
# Masking / noise function (unchanged from original, used by training loop)
# ---------------------------------------------------------------------------


def apply_noise(X, p=0.4, mode='swap', gaussian_sigma=0.5):
    """Apply corruption to a batch of gene expression vectors.

    Args:
        X: Tensor of shape (batch_size, num_genes).
        p: Masking probability. Can be a scalar, a [low, high] pair,
           or a per-gene list of length num_genes.
        mode: Corruption type ('swap', 'zero_mask', 'gaussian').
        gaussian_sigma: Std dev of additive Gaussian noise in log1p space.

    Returns:
        Tuple of (corrupted_X, mask) tensors.
    """
    if torch.isnan(X).any() or torch.isinf(X).any():
        print(f"WARNING: Input X contains NaN or Inf values!")
        print(f"  NaN count: {torch.isnan(X).sum().item()}")
        print(f"  Inf count: {torch.isinf(X).sum().item()}")

    if isinstance(p, (list, tuple)):
        if len(p) == 2:
            prob = torch.empty(X.shape, device=X.device, dtype=torch.float32).uniform_(float(p[0]), float(p[1]))
        elif len(p) == X.shape[1]:
            prob_tensor = torch.tensor(p, dtype=torch.float32)
            if torch.isnan(prob_tensor).any() or torch.isinf(prob_tensor).any():
                print(f"WARNING: prob_tensor contains NaN or Inf!")
            prob = prob_tensor.unsqueeze(0).expand(X.shape[0], -1).contiguous()
            prob = prob.to(X.device)
        else:
            prob = torch.full(X.shape, float(p[0]), device=X.device, dtype=torch.float32)
    else:
        prob = torch.full(X.shape, float(p), device=X.device, dtype=torch.float32)

    if prob.numel() == 0:
        print(f"WARNING: Empty probability tensor! X shape: {X.shape}")

    prob = torch.clamp(prob, 0.0, 1.0)

    if prob.mean().item() == 0.0:
        mask = torch.zeros_like(X)
        return X.clone(), mask

    try:
        mask = torch.bernoulli(prob)
    except Exception as e:
        print(f"Error in bernoulli sampling: {e}")
        print(f"Prob tensor: min={prob.min()}, max={prob.max()}, has_nan={torch.isnan(prob).any()}")
        raise

    if mode == 'swap':
        perm = torch.randperm(X.shape[0], device=X.device)
        X_perm = X[perm]
        corrupted = torch.where(mask.bool(), X_perm, X)
    elif mode == 'zero_mask':
        corrupted = torch.where(mask.bool(), torch.zeros_like(X), X)
    elif mode == 'gaussian':
        noise = torch.randn_like(X) * gaussian_sigma
        corrupted = torch.where(mask.bool(), torch.clamp(X + noise, min=0.0), X)
    else:
        raise ValueError(f"Unknown corruption mode: {mode}")

    return corrupted, mask


# Backward compatibility aliases
OccamAdataDataset = OccamAdataDataset
OptimizedOccamDataset = OccamChunkedDataset
