"""Load Occam's `rhj_fewshot` benchmark as a gspp-shaped AnnData.

The sibling of `zarr_loader.py`. Same output contract, different source: instead of a
the upstream benchmark zarr dump on a 2,000-gene panel, this reads **Occam's own h5ads** through
`competitors/shared/occam_data.py` -- our three source files, our frozen 6,983-gene axis, our
splits, in place. Nothing is converted and no second copy exists on disk.

WHY THE WIRING IS SMALL
-----------------------
Three things line up already, which is why this file is short rather than a rewrite:

1. **`CONTROL_NAME` is `"non-targeting"` in gspp too** (`zarr_loader.py:41`), which is exactly
   Occam's control label. No relabelling.
2. **`_load_zarr_splits` consumes a `{train, validation, test}` dict of perturbation NAMES**,
   which is precisely what `occam_data.occam_split_perturbations` returns. So that method is
   reused verbatim -- this loader just populates `self._zarr_split`.
3. **Its exclusion semantics are already ours.** Its own docstring: *"`_create_datasets` then
   excludes the held-out perturbations from HepG2 *only*, so the very same perturbation
   measured in jurkat / k562 / rpe1 still trains"*, and *"`condition_group["train"]`
   deliberately still lists them -- the row mask, not the name list, is what holds them out."*
   That is the `rhj_fewshot` design restated in gspp's vocabulary.

NORMALISATION
-------------
`X` is left on `log1p(target_sum=10_000)`, matching what the zarr path receives and what
every arm's config assumes (`target_sum=null, log1p=false`). Occam's h5ads are raw counts, so
`occam_data` does that normalisation -- **on the full source gene panel row sum, before the
axis restriction**, which is what makes the later conversion to Occam's own 2,000 evaluation
space exact. See `competitors/shared/emission_space.py`.

HVG selection is skipped, as in the zarr path: the axis is already the frozen 6,983 genes.

⚠️ CELL-LINE NAMING
-------------------
gspp's `PUBLIC_CELL_TYPES` is `["K562", "RPE1", "hepg2", "jurkat"]` -- lower case for the
Nadig lines, upper case for the Replogle ones. Occam uses `HEPG2` / `jurkat` / `RPE1`, so
`HEPG2` is mapped to `hepg2` and the other two pass through. Getting this wrong does not
crash; it silently produces a cell type gspp's vocabulary does not know.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy.sparse import csr_matrix

from gspp.constants import BATCH_RAW, CELL_TYPE, GENE_NAME, IS_CONTROL

# `competitors/shared` sits three levels up from gspp/data/.
_SHARED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "shared"))
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

# gspp's vocabulary casing. See the module docstring.
OCCAM_CELL_LINE_RENAME = {"HEPG2": "hepg2"}

__all__ = ["load_occam_datasets", "load_occam_split_perturbations", "OCCAM_CELL_LINE_RENAME"]


def load_occam_datasets(bench_dir: str, *,
                        cell_lines: Optional[List[str]] = None,
                        target_sum: float = 10_000.0,
                        verify_sources: bool = True,
                        max_cells_per_line: Optional[int] = None):
    """Occam's benchmark as one gspp-shaped AnnData, plus the benchmark handle.

    Returns `(adata, bench)`. The AnnData carries exactly the columns
    `_standardize_obs_format` requires -- `gene_name`, `is_control`, `batch_num`,
    `cell_line` -- plus `dataset` and `row_in_dataset`, matching `load_zarr_datasets`.
    """
    import anndata as sc_ad

    import occam_data as od

    bench = od.load_occam_benchmark(bench_dir, verify_sources=verify_sources)
    logger.info(f"occam: {bench.summary()}")

    lines = tuple(cell_lines) if cell_lines else od.OCCAM_CELL_LINES
    adata = od.load_occam_adata(
        bench, cell_lines=lines, target_sum=target_sum, log1p=True,
        # gspp's own control sentinel happens to equal ours, so no translation is needed.
        control_label=od.CONTROL_LABEL,
        max_cells_per_line=max_cells_per_line)

    ds = np.asarray(adata.obs["dataset_name"], dtype=str)
    frame = pd.DataFrame(index=pd.RangeIndex(adata.n_obs))
    frame[GENE_NAME] = np.asarray(adata.obs["target_gene"], dtype=str)
    frame[IS_CONTROL] = np.asarray(adata.obs["is_control"], dtype=bool)
    frame[BATCH_RAW] = np.asarray(adata.obs["batch_center"], dtype=str)
    frame[CELL_TYPE] = pd.Series(
        np.asarray(adata.obs["cell_line"], dtype=str)
    ).map(lambda c: OCCAM_CELL_LINE_RENAME.get(c, c)).to_numpy()
    frame["dataset"] = ds
    # per-dataset row index, so an index-based split file could still be resolved
    frame["row_in_dataset"] = np.concatenate(
        [np.arange(int((ds == d).sum()), dtype=np.int64) for d in pd.unique(ds)])
    frame.index = frame.index.astype(str)
    frame.index.name = None

    var = adata.var.copy()
    var[GENE_NAME] = var.index.astype(str)

    out = sc_ad.AnnData(X=csr_matrix(adata.X), obs=frame, var=var)
    out.var_names_make_unique()
    out.var[GENE_NAME] = out.var_names.astype(str)
    out.uns["occam"] = dict(adata.uns["occam"])

    logger.info(
        "occam: combined {:,} cells x {:,} genes | {:,} perturbations | "
        "cell lines {} | X max {:.2f}",
        out.n_obs, out.n_vars,
        frame.loc[~frame[IS_CONTROL], GENE_NAME].nunique(),
        sorted(frame[CELL_TYPE].unique()), float(out.X.max()),
    )
    return out, bench


def load_occam_split_perturbations(bench, adata) -> Dict[str, List[str]]:
    """`{train, validation, test}` perturbation-name lists, for `_load_zarr_splits`.

    ⚠️ `train` deliberately contains the held-out symbols. They legitimately train from
    Jurkat and RPE1, and `_create_datasets` holds them out of HepG2 with a ROW mask. Within
    HepG2 the three sets are disjoint, which is what gspp's straddle check requires.
    """
    import occam_data as od

    # `occam_split_perturbations` reads `target_gene` / `is_control`, which this loader
    # renamed; hand it a shim carrying the names it expects.
    class _Shim:
        pass

    shim = _Shim()
    shim.obs = pd.DataFrame({
        "target_gene": adata.obs[GENE_NAME].to_numpy(),
        "is_control": adata.obs[IS_CONTROL].to_numpy(),
    })
    names = od.occam_split_perturbations(bench, shim)
    logger.info(
        "occam: split names -> train {:,}, validation {:,}, test {:,}",
        len(names["train"]), len(names["validation"]), len(names["test"]))
    return names
