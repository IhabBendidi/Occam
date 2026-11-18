"""Expression IO and gene selection for the the upstream benchmark canonical zarr format.

Zarr counterpart to :mod:`src.data.dataset.dataset_io` (row fetch) and to
``dataset_grouping.get_selected_gene_vars`` (gene panel resolution). Both live
here because they are two halves of one concern: mapping ``X`` columns onto the
selected gene panel.

Differences from the h5ad path that are deliberate:

* **Column order is resolved once, at setup.** ``dataset_io.retrieve_counts``
  rebuilds a ``{gene: position}`` dict and a column map on *every row read*.
  Here ``gene_index_map[ds_name]`` is computed once from ``var.parquet`` and the
  row read is a single fancy-index.
* **A missing gene is an error, not a zero column.** The h5ad path silently
  zero-fills genes it cannot find (``dataset_io.py:156-158``). For a benchmark
  that must use exactly the same 2000-gene panel as STATE, a silent partial
  panel would change the result without any signal, so we raise instead --
  matching STATE (``zarr_perturbation.py:266-271``).
* **No normalization here.** The canonical ``X.zarr`` is already
  ``log1p(1e4)``-normalized, which is why the STATE run passes
  ``target_sum=null, log1p=false``. Keep ``data.normalize_counts=false``.
"""

import logging
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
import torch

logger = logging.getLogger(__name__)


def load_selected_genes(selected_gene_file) -> List[str]:
    """Read the gene panel from a ``.txt`` (one id per line) or a pickle.

    The benchmark panel is
    ``configs/metadata/replogle_nadig_selected_ens_ids.txt`` -- the same file
    STATE passes as ``selected_ensembl_gene_ids_path`` -- so the two models
    provably share a gene set. Upstream PerturbDiff ships pickles of gene
    symbols; both are accepted.
    """
    path = Path(selected_gene_file)
    if not path.exists():
        raise FileNotFoundError(f"selected_gene_file does not exist: {path}")

    if path.suffix.lower() in (".txt", ".csv", ".tsv", ".list"):
        genes = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    else:
        import pickle

        with open(path, "rb") as fin:
            genes = pickle.load(fin)
        if isinstance(genes, set):
            genes = sorted(genes)
        genes = [str(g) for g in genes]

    if len(genes) != len(set(genes)):
        raise ValueError(f"selected_gene_file {path} contains duplicate ids")
    return genes


def get_selected_gene_vars_zarr(dataset):
    """Resolve the gene panel against each dataset's ``var.parquet``.

    Sets on ``dataset``:

    ``selected_genes``      the panel, in output column order (list[str])
    ``gene_index_map``      ds_name -> int64 array indexing X columns in panel order
    ``gene_symbols``        ds_name -> gene symbols in panel order (for embeddings)
    ``gene_vars``           ds_name -> panel ids as an array (h5ad-path parity)
    ``selected_gene_mask``  ds_name -> all-True mask (h5ad-path parity; unused here)
    """
    selected = load_selected_genes(dataset.data_args.selected_gene_file)
    dataset.selected_genes = selected

    dataset.gene_index_map = {}
    dataset.gene_symbols = {}
    dataset.gene_vars = {}
    dataset.selected_gene_mask = {}

    for ds_name in dataset._names:
        var_path = Path(dataset.dataset_path_map[ds_name]) / "var.parquet"
        if not var_path.exists():
            raise FileNotFoundError(f"[{ds_name}] no var.parquet at {var_path}")
        var = pl.read_parquet(var_path)
        if "ensembl_gene_id" not in var.columns:
            raise ValueError(
                f"[{ds_name}] var.parquet has no 'ensembl_gene_id' column "
                f"(found {var.columns})"
            )

        ensembl = [str(g) for g in var["ensembl_gene_id"].to_list()]
        symbols = (
            [str(g) for g in var["gene_symbol"].to_list()]
            if "gene_symbol" in var.columns
            else ensembl
        )

        position = {g: i for i, g in enumerate(ensembl)}
        missing = [g for g in selected if g not in position]
        if missing:
            raise ValueError(
                f"[{ds_name}] {len(missing)} of {len(selected)} selected genes are absent "
                f"from var.parquet (e.g. {missing[:5]}). The gene panel must match the "
                f"dataset, otherwise the benchmark is not comparable."
            )

        index_map = np.fromiter(
            (position[g] for g in selected), dtype=np.int64, count=len(selected)
        )
        dataset.gene_index_map[ds_name] = index_map
        dataset.gene_symbols[ds_name] = [symbols[i] for i in index_map]
        dataset.gene_vars[ds_name] = np.asarray(selected)
        dataset.selected_gene_mask[ds_name] = np.ones(len(selected), dtype=bool)

        dataset.py_logger.info(
            f"[{ds_name}] gene panel resolved: {len(selected)} genes "
            f"out of {len(ensembl)} in var.parquet"
        )


def retrieve_counts_zarr(dataset, handle, ds_name, local_idx):
    """Fetch one expression row, already ordered to the selected gene panel.

    Returns shape ``[1, G]`` to match ``dataset_io.retrieve_counts``; the collate
    takes ``counts[0]``.
    """
    row = handle[int(local_idx)]
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    row = row[dataset.gene_index_map[ds_name]]
    return torch.from_numpy(np.ascontiguousarray(row)).unsqueeze(0)


__all__ = [
    "load_selected_genes",
    "get_selected_gene_vars_zarr",
    "retrieve_counts_zarr",
]
