"""Read Occam's `rhj_fewshot` benchmark as a STATE `_Source`, one cell line at a time.

The sibling of the `zarr` and `anndata` branches in `zarr_perturbation.py::_Source`. Same
output contract, different origin: instead of a the upstream benchmark zarr dump on a 2,000-gene panel, this
reads **Occam's own h5ads** through `competitors/shared/occam_data.py` -- our three source
files, our frozen 6,983-gene axis, our splits, in place. Nothing is converted and no second
copy exists on disk.

WHY A THIRD FORMAT AND NOT THE EXISTING `anndata` ONE
-----------------------------------------------------
The `anndata` branch reads an h5ad straight off disk and normalises, when asked, inside
`_transform_expression` -- which divides by `x.sum()` **after** gene selection. On our data
that would be the row sum over 6,983 genes, and the whole emission-space argument depends on
the denominator being the row sum over the file's **FULL** gene panel.

⚠️ THIS IS THE SINGLE MOST LOAD-BEARING DETAIL IN THE PORT. The two denominators differ by a
per-cell factor (measured: the axis carries ~86.6% of the library, so the median ratio is
about 1.15). A per-cell factor leaves `pearson_delta` untouched -- it is scale-invariant per
delta vector -- but it SHIFTS `mae` and MOVES the DE calls, and `mae` is one of our four
comparison metrics. It would also break the exact 10,000 -> 2,000 map the prediction bridge
relies on, because that map is only exact when both spaces share the same denominator.

So normalisation happens once, in `occam_data`, on the full panel, and STATE is configured
with `target_sum=null, log1p=false` exactly as upstream -- which in their pipeline meant "the
input arrives already normalised", and here means the same thing.

WHAT THIS PROVIDES
------------------
Everything `_Source` exposes, for ONE cell line:
`pert_str`, `is_control`, `cell_type`, `batch`, `n_rows`, `cell_type_code`, `n_cell_types`,
`gene_index_map`, `gene_symbols`, and a materialised `X`.

Gene selection is the IDENTITY here: `occam_data` already returns the frozen 6,983 genes in
axis order, so there is nothing left to subset and `selected_ensembl_gene_ids_path` is not
used. Passing one anyway is refused rather than silently ignored.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# `competitors/shared` is five levels up from state/src/state/tx/data/.
_SHARED = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "..", "..", "shared"))
if os.path.isdir(_SHARED) and _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

__all__ = ["OCCAM_FORMAT", "OccamSourceData", "read_occam_source", "occam_fold_for_line",
           "occam_line_from_path", "occam_datasets"]

OCCAM_FORMAT = "occam"

# STATE's configs name cell lines in their own casing; ours are HEPG2 / jurkat / RPE1.
_ALIASES = {"hepg2": "HEPG2", "HEPG2": "HEPG2", "jurkat": "jurkat", "Jurkat": "jurkat",
            "rpe1": "RPE1", "RPE1": "RPE1", "rpe1_e": "RPE1"}


class OccamSourceData:
    """One cell line's matrix plus the derived metadata `_Source` needs."""

    def __init__(self, line, X, pert_str, is_control, cell_type, batch,
                 gene_symbols, gene_ensembl, uns):
        self.line = line
        self.X = X
        self.pert_str = pert_str
        self.is_control = is_control
        self.cell_type = cell_type
        self.batch = batch
        self.gene_symbols = gene_symbols
        self.gene_ensembl = gene_ensembl
        self.uns = uns

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[0])


def occam_line_from_path(path: str, known: Optional[Sequence[str]] = None) -> str:
    """The cell line a `[datasets]` entry refers to.

    Accepts the line name itself (`hepg2`, `RPE1`, ...) or our h5ad path, from which the line
    is taken by substring. Refuses anything ambiguous rather than guessing: a source silently
    bound to the wrong cell line trains on the held-out context and the run looks fine.

    ⚠️ `known` is the set of lines THE BENCHMARK actually has, and it is checked BEFORE the
    alias table. The aliases only cover rhj_fewshot's three lines, so without this a
    perfectly valid line from another benchmark is rejected -- norman_combo's `K562` died
    here on the first launch with "cannot tell which Occam cell line 'K562' is". A line
    the benchmark itself lists is by definition a line we support.
    """
    key = str(path)
    base = os.path.basename(key)
    if known:
        for ln in known:
            if key == str(ln):
                return str(ln)
        hits = {str(ln) for ln in known if str(ln).lower() in base.lower()}
        if len(hits) == 1:
            return hits.pop()
    if key in _ALIASES:
        return _ALIASES[key]
    hits = {v for k, v in _ALIASES.items() if k.lower() in base.lower()}
    if len(hits) == 1:
        return hits.pop()
    raise ValueError(
        f"cannot tell which Occam cell line {path!r} is. Name the [datasets] entry's value "
        f"one of {sorted(set(known or _ALIASES.values()))}, or point it at our h5ad whose "
        f"filename contains the line.")


def occam_datasets(bench_dir: str, *, verify_sources: bool = False) -> List[str]:
    """The `[datasets]` entries for our benchmark: one per cell line, read from the frozen dir.

    STATE normally gets its sources from a TOML. In occam mode there is no TOML -- the
    benchmark directory IS the manifest -- so the dataset list is derived from it here and
    every line is one `[datasets]` entry whose value is the line name itself, which is what
    `occam_line_from_path` accepts.

    Order is the benchmark's own. `_setup_global_maps` fixes the gene axis from the FIRST
    source and checks every later one against it, so a stable order keeps that check
    meaningful across restarts.
    """
    import occam_data as od

    bench = od.load_occam_benchmark(bench_dir, verify_sources=verify_sources)
    lines = [str(f) for f in bench.files]
    if not lines:
        raise ValueError(
            f"{bench_dir} lists no source files. Rebuild the benchmark: without sources "
            f"there is nothing to train on and setup() would report 0 / 0 / 0 rather than "
            f"fail.")
    for ln in lines:
        # `known=lines` so a line the benchmark itself lists is accepted; the alias table
        # only knows rhj_fewshot's three.
        occam_line_from_path(ln, known=lines)   # refuse now, not 20 minutes in
    return lines


def read_occam_source(path: str, *, bench_dir: str, target_sum: float = 10_000.0,
                      log1p: bool = True, control_pert: str = "empty",
                      pert_col: str = "gene_pert",
                      cell_type_key: str = "cell_type",
                      batch_col: str = "batch_center",
                      selected_genes: Optional[List[str]] = None,
                      verify_sources: bool = True,
                      max_cells: Optional[int] = None) -> OccamSourceData:
    """One cell line of our benchmark, normalised on the FULL panel row sum."""
    import occam_data as od

    if selected_genes:
        raise ValueError(
            "selected_ensembl_gene_ids_path is not used with source_format='occam': the "
            "matrix already carries exactly the benchmark's frozen genes in axis order, "
            "so a "
            "second subset could only reorder or shrink it. Remove the option.")

    # The benchmark is loaded FIRST so its own line list can resolve `path`; the alias
    # table alone only knows rhj_fewshot's three lines.
    bench = od.load_occam_benchmark(bench_dir, verify_sources=verify_sources)
    line = occam_line_from_path(path, known=[str(f) for f in bench.files])
    adata = od.load_occam_adata(bench, cell_lines=[line], target_sum=target_sum,
                                log1p=log1p, control_label=control_pert,
                                pert_col_out=pert_col, max_cells_per_line=max_cells)

    obs = adata.obs
    for col in (pert_col, cell_type_key, batch_col):
        if col not in obs.columns:
            raise ValueError(
                f"occam_data did not emit {col!r} for {line}; have {list(obs.columns)}")

    X = np.asarray(adata.X, dtype=np.float32)
    return OccamSourceData(
        line=line, X=X,
        pert_str=np.asarray([str(p) for p in obs[pert_col].to_numpy()], dtype=object),
        is_control=np.asarray(obs["is_control"].to_numpy(), dtype=bool),
        cell_type=np.asarray([str(c) for c in obs[cell_type_key].to_numpy()], dtype=object),
        batch=np.asarray([str(b) for b in obs[batch_col].to_numpy()], dtype=object),
        gene_symbols=[str(g) for g in adata.var["gene_symbol"]],
        gene_ensembl=[str(g) for g in adata.var["ensembl_gene_id"]],
        uns=dict(adata.uns.get("occam", {})))


def occam_fold_for_line(src: OccamSourceData, *, bench_dir: str,
                        verify_sources: bool = False) -> Dict[str, List[int]]:
    """`{finetune, validation, test}` as ROW INDICES within this one cell line.

    STATE's `setup()` consumes exactly this shape -- it calls `_mask_from_indices` per split
    key -- so our split reaches it through their own mechanism rather than a new one.

    ⚠️ ASYMMETRY BY DESIGN. HepG2 splits three ways; Jurkat and RPE1 are wholly `finetune`.
    That is the `rhj_fewshot` design: the very same perturbation held out of HepG2 still
    TRAINS from Jurkat and RPE1, which is what makes the task few-shot-in-a-new-context
    rather than never-seen-at-all. The exclusion is per (file, perturbation), and here it is
    expressed as a per-source row mask -- their vocabulary, our semantics.

    ⚠️ This does NOT call `occam_data.occam_split_indices`. That function asserts over the
    CONCATENATED panel -- it requires `validation` to cover every val perturbation and every
    control to be in `finetune` -- and those invariants are false for a single line taken
    alone (Jurkat contributes no validation rows at all). Reusing it here would either fail
    on Jurkat or have to have its assertions weakened, which is the opposite of what they are
    for. The per-line invariants are asserted below instead.
    """
    import occam_data as od

    bench = od.load_occam_benchmark(bench_dir, verify_sources=verify_sources)
    val, test = bench.val_perts, bench.test_perts

    perts = src.pert_str.astype(str)
    is_ctrl = np.asarray(src.is_control, dtype=bool)
    n = src.n_rows

    # ⚠️ The EVALUATED line comes from the benchmark. `== "HEPG2"` is rhj_fewshot's
    # held-out line, and on norman_combo (K562 only) every source took the `else` branch
    # -- wholly training data, empty val and test -- and `val_dataloader` then refused
    # with "validation split is empty". Correctly: it will not fall back to the test
    # loader, which would select checkpoints on test.
    if src.line == bench.evaluated_line:
        v = np.flatnonzero(~is_ctrl & np.isin(perts, list(val)))
        t = np.flatnonzero(~is_ctrl & np.isin(perts, list(test)))
        held = np.zeros(n, dtype=bool)
        held[v] = True
        held[t] = True
        f = np.flatnonzero(~held)
    else:
        # the positive half of the exclusion: a held-out perturbation still trains here
        v = np.empty(0, dtype=np.int64)
        t = np.empty(0, dtype=np.int64)
        f = np.arange(n, dtype=np.int64)

    fold = {"finetune": f, "validation": v, "test": t}

    seen = np.concatenate([f, v, t])
    if len(np.unique(seen)) != len(seen):
        raise ValueError(f"{src.line}: split index sets overlap")
    if len(seen) != n:
        raise ValueError(
            f"{src.line}: the split covers {len(seen):,} of {n:,} rows. Every row must land "
            f"in exactly one of finetune/validation/test, or cells vanish silently.")
    if is_ctrl[v].any() or is_ctrl[t].any():
        raise ValueError(
            f"{src.line}: a control cell landed in validation or test. Controls are the "
            f"paired basal pool and must all stay in finetune.")
    if int(is_ctrl[f].sum()) != int(is_ctrl.sum()):
        raise ValueError(f"{src.line}: not every control is in finetune")
    if src.line == bench.evaluated_line:
        for name, want, rows in (("validation", val, v), ("test", test, t)):
            got = set(perts[rows].tolist())
            if got - want:
                raise ValueError(
                    f"{src.line} {name} carries {len(got - want)} perturbation(s) not in "
                    f"splits.json, e.g. {sorted(got - want)[:5]}")
    else:
        if v.size or t.size:
            raise ValueError(f"{src.line} must be wholly training data, got "
                             f"{v.size} val / {t.size} test rows")

    print(f"  [occam-state] {src.line}: finetune {len(f):,} "
          f"(incl. {int(is_ctrl[f].sum()):,} controls) · validation {len(v):,} "
          f"· test {len(t):,}")
    return {k: [int(i) for i in arr] for k, arr in fold.items()}
