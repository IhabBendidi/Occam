"""Read Occam's `rhj_fewshot` benchmark as PerturbDiff datasets.

The sibling of `zarr_metadata_cache.py` + `zarr_store.py` + `dataset_io_zarr.py`, collapsed
into one module because the occam versions of all three are short. Same duck-typed surfaces,
different origin: instead of a hooke-tx canonical dump (`X.zarr` + `obs.parquet` +
`var.parquet`) on a 2,000-gene panel, this reads **Occam's own h5ads** through
`competitors/shared/occam_data.py`. Our three source files, our frozen 6,983-gene axis, our
splits, in place. Nothing is converted and no second copy exists on disk.

WHAT IT PROVIDES, AND WHAT IT DUCK-TYPES
----------------------------------------
| this module | the twin it matches |
|---|---|
| `OccamMetadataCache` | `ZarrMetadataCache` / `H5MetadataCache` |
| `OccamStore` | `ZarrStore` / `H5Store` |
| `get_selected_gene_vars_occam` | `get_selected_gene_vars_zarr` |
| `retrieve_counts_occam` | `retrieve_counts_zarr` |
| `split_index_occam` | `split_index_json` |

So `dataset_grouping`, `sampler`, `collate_fn` and `dataset_core.get_covarates` work
unchanged, which is the point: nothing in the training loop is aware of where the data
came from.

⚠️ NORMALISATION HAPPENS ONCE, IN `occam_data`, ON THE FULL SOURCE PANEL ROW SUM:
`log1p(counts / rowsum_FULL_PANEL * 10_000)`. Their zarr arrived pre-normalised at 1e4, which
is why every config on this benchmark passes `target_sum=null, log1p=false`; our h5ads are raw
counts, so we do that step. The FULL-panel denominator (rather than the 6,983-gene one) is
what makes the prediction bridge's 10,000 -> 2,000 map exact — a per-cell factor leaves
`pearson_delta` untouched but shifts `mae` and moves the DE calls. See
`docs/rhj_competitors.md` C1.

⚠️ THE PERTURBATION STRING IS A GENE SYMBOL, not an Ensembl id. Their zarr path derives an
Ensembl string from canonical `gene_pert` tuples, because that is what their dumps carry.
`rhj_fewshot` identifies perturbations by SYMBOL throughout — the splits, the exclusions, the
611-perturbation cohort and cell-eval's own `target_gene` column — so a symbol is the identity
here, and a perturbation target need not even be on the gene axis. Controls are forced onto
the configured `control_pert` sentinel exactly as `ZarrMetadataCache` does.

⚠️ ONE DATASET IS ONE CELL LINE. `[datasets]` names three h5ads; the split is asymmetric by
design: HepG2 splits three ways, jurkat and RPE1 are wholly training data. The very same
perturbation held out of HepG2 still TRAINS from jurkat and RPE1, which is what makes the task
few-shot-in-a-new-context rather than never-seen-at-all.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# `competitors/shared` is four levels up from PerturbDiff/src/data/.
_SHARED = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "shared"))
if os.path.isdir(_SHARED) and _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

CONTROL_SENTINEL = "empty"
OCCAM_FORMAT = "occam"
OCCAM_CELL_LINES = ("HEPG2", "jurkat", "RPE1")
EVALUATED_LINE = "HEPG2"

# One materialised matrix per cell line, process-global and NEVER pickled -- exactly the
# discipline `zarr_store.py` documents. One H5adSentenceDataset is built per split
# (train / validation / test) over the same files, so per-instance storage would hold three
# copies of every matrix; HepG2 alone is 145,473 x 6,983 float32 ~= 4.1 GB.
_MATRIX_CACHE: Dict[str, np.ndarray] = {}
_META_CACHE: Dict[str, "OccamMetadataCache"] = {}
_CACHE_LOCK = threading.RLock()

_ALIASES = {"hepg2": "HEPG2", "HEPG2": "HEPG2", "jurkat": "jurkat", "Jurkat": "jurkat",
            "rpe1": "RPE1", "RPE1": "RPE1", "rpe1_e": "RPE1"}

__all__ = [
    "OCCAM_FORMAT", "OCCAM_CELL_LINES", "EVALUATED_LINE", "OccamMetadataCache",
    "OccamStore", "get_selected_gene_vars_occam", "retrieve_counts_occam",
    "split_index_occam", "occam_line_from_path", "read_occam_line",
]


def occam_line_from_path(path: str) -> str:
    """The cell line a `[datasets]` entry refers to, refusing anything ambiguous.

    A source silently bound to the wrong cell line trains on the held-out context and the run
    looks fine, so this raises rather than guesses.
    """
    key = str(path)
    base = os.path.basename(key)
    if key in _ALIASES:
        return _ALIASES[key]
    hits = {v for k, v in _ALIASES.items() if k.lower() in base.lower()}
    if len(hits) == 1:
        return hits.pop()
    raise ValueError(
        f"cannot tell which Occam cell line {path!r} is. Name the [datasets] entry's value "
        f"one of {sorted(set(_ALIASES.values()))}, or point it at our h5ad whose filename "
        f"contains the line.")


def read_occam_line(path: str, *, bench_dir: str, target_sum: float = 10_000.0,
                    log1p: bool = True, verify_sources: bool = True,
                    max_cells: Optional[int] = None):
    """`(X, obs_frame, axis_ensembl, axis_symbols)` for one cell line, loaded at most once."""
    import occam_data as od

    line = occam_line_from_path(path)
    key = f"{line}|{target_sum}|{log1p}|{max_cells}"
    with _CACHE_LOCK:
        cached = _MATRIX_CACHE.get(key)
        if cached is not None:
            return cached

        bench = od.load_occam_benchmark(bench_dir, verify_sources=verify_sources)
        adata = od.load_occam_adata(
            bench, cell_lines=[line], target_sum=float(target_sum), log1p=log1p,
            control_label=CONTROL_SENTINEL, pert_col_out="pert",
            max_cells_per_line=max_cells)
        out = (np.asarray(adata.X, dtype=np.float32), adata.obs.copy(),
               [str(g) for g in bench.axis_ensembl], [str(g) for g in bench.axis_symbols])
        _MATRIX_CACHE[key] = out
        logger.info("[%s] occam: %d cells x %d genes in RAM", line, out[0].shape[0],
                    out[0].shape[1])
        return out


def _factorize(values: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Sorted categories + int32 codes, matching `zarr_metadata_cache._factorize`.

    Sorted order is what the datamodule's global dictionaries assume
    (`{x: i for i, x in enumerate(sorted(...))}`), so codes are stable across runs.
    """
    categories = np.asarray(sorted(set(values)))
    code_of = {c: i for i, c in enumerate(categories.tolist())}
    codes = np.fromiter((code_of[v] for v in values), dtype=np.int32, count=len(values))
    return categories, codes


class OccamMetadataCache:
    """Duck-typed twin of `ZarrMetadataCache`, for one cell line of our benchmark."""

    def __init__(self, path, *, bench_dir: str, cell_type_key: str = "cell_type",
                 batch_col: str = "batch_center", control_pert: str = CONTROL_SENTINEL,
                 target_sum: float = 10_000.0, log1p: bool = True,
                 verify_sources: bool = True, max_cells: Optional[int] = None,
                 pert_col: Optional[str] = None):
        del pert_col  # accepted for signature parity with the h5ad cache; unused
        self.data_dir = str(path)
        self.h5_path = str(path)          # parity with H5MetadataCache.h5_path
        self.line = occam_line_from_path(path)
        self.bench_dir = str(bench_dir)
        self.cell_type_key = cell_type_key
        self.batch_col = batch_col
        self.control_pert = control_pert

        X, obs, ens, sym = read_occam_line(
            path, bench_dir=bench_dir, target_sum=target_sum, log1p=log1p,
            verify_sources=verify_sources, max_cells=max_cells)
        self.n_cells = int(X.shape[0])
        self.axis_ensembl = ens
        self.axis_symbols = sym

        for col in (cell_type_key, batch_col):
            if col not in obs.columns:
                raise ValueError(
                    f"[{self.line}] occam_data did not emit {col!r}; "
                    f"available: {list(obs.columns)}")

        pert_str = np.asarray([str(p) for p in obs["target_gene"].to_numpy()], dtype=object)
        is_control = np.asarray(obs["is_control"].to_numpy(), dtype=bool)
        # Force controls onto the configured sentinel, as ZarrMetadataCache does.
        pert_str[is_control] = control_pert

        self.pert_categories, self.pert_codes = _factorize(pert_str.tolist())
        self.cell_type_categories, self.cell_type_codes = _factorize(
            [str(c) for c in obs[cell_type_key].to_numpy()])
        self.batch_categories, self.batch_codes = _factorize(
            [str(b) for b in obs[batch_col].to_numpy()])
        self.batch_is_categorical = True

        idx = np.where(self.pert_categories == control_pert)[0]
        if idx.size == 0:
            raise ValueError(
                f"[{self.line}] control_pert={control_pert!r} is absent after encoding; "
                f"found {len(self.pert_categories)} categories, e.g. "
                f"{self.pert_categories[:3].tolist()}")
        self.control_pert_code = int(idx[0])
        self.control_mask = is_control

        logger.info(
            "[%s] occam obs: %d cells, %d perts (%d control), %d cell types, %d batches",
            self.line, self.n_cells, len(self.pert_categories), int(is_control.sum()),
            len(self.cell_type_categories), len(self.batch_categories))


def get_occam_cache(path, **kwargs) -> OccamMetadataCache:
    """Process-global cache keyed by path, so the three splits share one instance."""
    key = str(path)
    with _CACHE_LOCK:
        if key not in _META_CACHE:
            _META_CACHE[key] = OccamMetadataCache(path, **kwargs)
        return _META_CACHE[key]


class OccamStore:
    """Path -> in-RAM expression matrix, with the `H5Store` / `ZarrStore` contract."""

    def __init__(self, dataset_path_map: Dict[str, str], *, bench_dir: str,
                 target_sum: float = 10_000.0, log1p: bool = True,
                 verify_sources: bool = True, max_cells: Optional[int] = None,
                 max_open: int = 16):
        del max_open  # accepted for parity with H5Store; nothing is opened lazily
        self.dataset_path_map = dict(dataset_path_map)
        self.bench_dir = str(bench_dir)
        self.target_sum = float(target_sum)
        self.log1p = bool(log1p)
        self.verify_sources = bool(verify_sources)
        self.max_cells = max_cells
        self._pid = os.getpid()

    def dataset_file(self, name: str):
        X, _, _, _ = read_occam_line(
            self.dataset_path_map[name], bench_dir=self.bench_dir,
            target_sum=self.target_sum, log1p=self.log1p,
            verify_sources=self.verify_sources, max_cells=self.max_cells)
        return X

    def preload(self):
        """Materialise every line in the CURRENT process, before workers fork.

        Under fork, a worker then inherits the populated module-level cache and shares the
        pages copy-on-write; nothing is re-read and nothing is serialised per worker.
        """
        for name in self.dataset_path_map:
            self.dataset_file(name)

    def close_all(self):
        """No-op: the matrices are shared process-wide and outlive any one store."""
        return

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pid"] = os.getpid()
        return state


def get_selected_gene_vars_occam(dataset) -> None:
    """The gene panel IS our frozen axis, so this is the identity map.

    Sets the same five attributes `get_selected_gene_vars_zarr` sets. There is nothing to
    resolve: `occam_data` already returned exactly the 6,983 axis genes in axis order, so a
    second subset could only reorder or shrink the matrix.

    ⚠️ `data.selected_gene_file` is REFUSED rather than ignored, because a silently ignored
    gene panel is how an arm ends up scored on a different axis than it trained on.
    """
    if getattr(dataset.data_args, "selected_gene_file", None):
        raise ValueError(
            "data.selected_gene_file must be null with source_format='occam': the matrix "
            "already carries exactly the frozen 6,983 genes in axis order, so a second "
            "panel could only reorder or shrink it.")

    bench_dir = str(dataset.data_args.occam_bench_dir)
    dataset.gene_index_map = {}
    dataset.gene_symbols = {}
    dataset.gene_vars = {}
    dataset.selected_gene_mask = {}
    selected: Optional[List[str]] = None

    for ds_name in dataset._names:
        _, _, ens, sym = read_occam_line(
            dataset.dataset_path_map[ds_name], bench_dir=bench_dir,
            target_sum=float(getattr(dataset.data_args, "occam_target_sum", 10_000.0)),
            log1p=bool(getattr(dataset.data_args, "occam_log1p", True)),
            verify_sources=False,
            max_cells=getattr(dataset.data_args, "occam_max_cells_per_line", None))
        if selected is None:
            selected = list(ens)
        elif list(ens) != selected:
            raise ValueError(
                f"[{ds_name}] gene axis differs from the first dataset's. Every line must "
                f"carry the same frozen axis in the same order; the prediction bridge is "
                f"positional, so a mismatch silently relabels genes.")
        n = len(ens)
        dataset.gene_index_map[ds_name] = np.arange(n, dtype=np.int64)
        dataset.gene_symbols[ds_name] = list(sym)
        dataset.gene_vars[ds_name] = np.asarray(ens)
        dataset.selected_gene_mask[ds_name] = np.ones(n, dtype=bool)

    dataset.selected_genes = selected


def retrieve_counts_occam(dataset, handle, ds_name, local_idx):
    """One expression row as `[1, G]`, matching `retrieve_counts_zarr`.

    The gene map is the identity here, so the row is already in output column order; it is
    still indexed through `gene_index_map` so the two paths stay literally the same shape of
    code and a future non-identity map would work without a second edit.
    """
    row = np.asarray(handle[int(local_idx)], dtype=np.float32).reshape(-1)
    row = row[dataset.gene_index_map[ds_name]]
    return torch.from_numpy(np.ascontiguousarray(row)).unsqueeze(0)


def _empty() -> np.ndarray:
    return np.empty(0, dtype=np.int64)


def split_index_occam(dm, cache, target_type, key_info, holdout_setname=None,
                      random_setname=None):
    """Our split, computed live from the frozen benchmark. No index JSON exists.

    Semantics are a deliberate match for `split_index_json`, so PerturbDiff and STATE see the
    same cells:

    * HepG2 splits three ways over PERTURBED rows; jurkat and RPE1 contribute every row to
      train (the positive half of the per-(file, perturbation) exclusion).
    * **All control rows are added to every non-empty split.** An empty split stays empty
      rather than becoming controls-only.

    ⚠️ Nothing is cached and nothing is written. The benchmark is the split, so there is no
    stale-index surface at all — which is the one failure mode a generated JSON would add.
    """
    del target_type, holdout_setname, random_setname

    import occam_data as od

    bench = od.load_occam_benchmark(str(key_info["occam_bench_dir"]),
                                    verify_sources=False)
    control_mask = np.asarray(cache.control_mask, dtype=bool)
    ctrl_rows = np.flatnonzero(control_mask)
    perts = np.asarray(cache.pert_categories)[np.asarray(cache.pert_codes)]
    perts = np.asarray([str(p) for p in perts], dtype=str)

    line = cache.line
    if line != EVALUATED_LINE:
        all_rows = np.arange(cache.n_cells, dtype=np.int64)
        logger.info("[%s] wholly training data: %d rows", line, len(all_rows))
        return {"train": all_rows, "validation": _empty(), "test": _empty()}

    val, test = bench.val_perts, bench.test_perts
    is_pert = ~control_mask
    v = np.flatnonzero(is_pert & np.isin(perts, list(val)))
    t = np.flatnonzero(is_pert & np.isin(perts, list(test)))
    held = np.zeros(cache.n_cells, dtype=bool)
    held[v] = True
    held[t] = True
    f = np.flatnonzero(is_pert & ~held)

    out = {}
    for name, rows in (("train", f), ("validation", v), ("test", t)):
        if rows.size:
            out[name] = np.union1d(rows, ctrl_rows).astype(np.int64)
        else:
            out[name] = _empty()

    if out["validation"].size == 0 or out["test"].size == 0:
        raise ValueError(
            f"[{line}] the validation or test split is EMPTY ({out['validation'].size} / "
            f"{out['test'].size}). On this benchmark HepG2 must carry both, or the arm has "
            f"nothing to select a checkpoint on and would silently keep its final weights.")

    logger.info("[%s] occam split: train %d / validation %d / test %d "
                "(incl. %d controls in each)", line, len(out["train"]),
                len(out["validation"]), len(out["test"]), len(ctrl_rows))
    return out
