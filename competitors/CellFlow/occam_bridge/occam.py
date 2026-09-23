"""Read Occam's `rhj_fewshot` benchmark as a CellFlow-shaped AnnData.

The sibling of `load.py`. Same output contract -- `-> ad.AnnData` with the obs columns
`apply_cellflow_gene_schema` requires -- but a different origin: instead of a the upstream benchmark dump
(`obs.parquet` + `var.parquet` + `X.zarr`) on a 2,000-gene panel, this reads **Occam's own
h5ads** through `competitors/shared/occam_data.py`. Our three source files, our frozen
6,983-gene axis, our splits, in place. Nothing is converted and no second copy exists on disk.

Like `load.py`, this module imports nothing from CellFlow, so it can be exercised without a
jax environment.

WHAT CELLFLOW NEEDS, AND WHAT IT GETS
-------------------------------------
`apply_cellflow_gene_schema` reads exactly three things from obs:

* **`gene_pert`** -- a LIST OF TUPLES per cell, `[(kind, token, effect), ...]`, empty for
  controls. Built here as one `("genetic", SYMBOL, "")` per COMPONENT gene, so a pair
  `A+B` is two tuples and reaches the model as the set `[A, B]` (see `pert_components`).
* **`negative_control`** -- bool. `occam_data` emits it.
* the **`sample_covariates` / `split_covariates`** columns, `["cell_type"]` on this
  benchmark, so a perturbation is matched to controls of its own cell line.

⚠️ THE PERTURBATION TOKEN IS A GENE SYMBOL, NOT AN ENSEMBL ID, and that is deliberate.
`load.py`'s docstring says Ensembl because their dumps carry Ensembl; the code treats the
token as an opaque string and one-hot encodes it. `rhj_fewshot` identifies perturbations by
SYMBOL -- the splits, the exclusions, the 611-perturbation cohort and cell-eval's own
`target_gene` column are all symbols -- so mapping them through Ensembl would add a lossy
join for no gain, and a perturbation target need not even be on the 6,983-gene axis. The
symbol IS the identity here.

⚠️ THE GENE AXIS IS ENSEMBL, in `var_names`, matching `load.py`'s convention. That is safe
because the bridge is POSITIONAL: `gene_axis.csv` lists Ensembl and symbol side by side in one
order, and `assert_axis_order` below refuses anything else. Our evaluation h5ads use symbols
in `var.index` so cell-eval can exclude each target gene by name; the two are the same 6,983
columns in the same order.

NORMALISATION
-------------
`log1p(counts / rowsum_FULL_PANEL * 10_000)`, done once in `occam_data`. CellFlow must
therefore run with `target_sum: null, log1p: false`, exactly as its official config already
does -- which in their pipeline meant "the source is already normalised", and here means the
same thing. See `docs/rhj_competitors.md` C1 for why the FULL-panel denominator, rather than
the 6,983-gene one, is the load-bearing choice.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# `competitors/shared` is three levels up from CellFlow/occam_bridge/.
_SHARED = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "shared"))
if os.path.isdir(_SHARED) and _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

__all__ = [
    "OCCAM_CELL_LINES", "occam_enabled", "occam_bench_dir", "load_occam_anndata",
    "load_occam_sources", "load_occam_validation", "occam_pert_tokens",
    "assert_axis_order", "pert_components", "canonical_pert_label",
]

# Our casing, which is what lands in obs["cell_type"] and what the config's
# sample_covariates / split_covariates match on.
OCCAM_CELL_LINES = ("HEPG2", "jurkat", "RPE1")
EVALUATED_LINE = "HEPG2"
SUPPORTING_LINES = ("jurkat", "RPE1")


def occam_enabled(data_cfg: Dict[str, Any]) -> bool:
    """True when the config asks for our benchmark instead of a the upstream benchmark dump."""
    return bool(data_cfg.get("occam_bench_dir"))


def occam_bench_dir(data_cfg: Dict[str, Any]) -> str:
    d = data_cfg.get("occam_bench_dir")
    if not d:
        raise KeyError(
            "data.occam_bench_dir is not set, so there is no benchmark to read. Point it at "
            "data/benchmarks/rhj_fewshot -- it carries the gene axis, the splits, the "
            "exclusions and the source sha256s.")
    return str(d)


def assert_axis_order(var_names: Sequence[str], bench) -> None:
    """The gene axis must be our axis, in our order. Refuse anything else.

    The prediction bridge maps columns POSITIONALLY: a reordered axis would produce a
    plausible matrix in which every gene is the wrong gene, and no metric would look odd.
    """
    got = [str(g) for g in var_names]
    want = [str(g) for g in bench.axis_ensembl]
    if got != want:
        if set(got) == set(want):
            raise ValueError(
                f"the gene axis has the right {len(want)} genes in the WRONG ORDER. The "
                f"prediction bridge is positional, so this silently relabels every gene.")
        raise ValueError(
            f"the gene axis is not ours: {len(got)} genes vs the frozen {len(want)} "
            f"(missing {sorted(set(want) - set(got))[:3]}, "
            f"extra {sorted(set(got) - set(want))[:3]})")


PAIR_SEP = "+"


def pert_components(label: str) -> List[str]:
    """`"A+B"` -> `["A", "B"]`, `"A"` -> `["A"]`, SORTED.

    ⚠️ THE ONE PLACE A PERTURBATION LABEL IS SPLIT INTO THE TOKENS CELLFLOW SEES. A
    combination is a SET of component genes, each its own one-hot token, which CellFlow's
    set encoder aggregates (`max_combination_length`, `pooling`). It used to be passed
    through as ONE opaque token `"A+B"`, so validation pairs were `["A+B", NULL]` -- a token
    no training cell ever activated -- while the export split test pairs into `[A, B]`.
    Selection therefore scored a different encoding from the one that was reported.

    Sorted, as `schema.apply_cellflow_gene_schema` sorts, so `A+B` and `B+A` are one
    condition. A label without `+` (every rhj_fewshot symbol) is returned unchanged.
    """
    parts = sorted(p for p in str(label).split(PAIR_SEP) if p)
    if not parts:
        raise ValueError(f"empty perturbation label {label!r}")
    return parts


def canonical_pert_label(label: str) -> str:
    """The sorted `A+B` form -- what `schema` writes to `obs["pert_str"]`."""
    return PAIR_SEP.join(pert_components(label))


def _to_cellflow_obs(adata, *, control_label: str) -> pd.DataFrame:
    """Add the `gene_pert` tuple column CellFlow's schema parses.

    One tuple PER COMPONENT gene: a single is `[("genetic", A, "")]` (the schema pads it to
    `[A, NULL]`), a pair `A+B` is `[("genetic", A, ""), ("genetic", B, "")]`.
    """
    perts = np.asarray(adata.obs["target_gene"], dtype=str)
    is_ctrl = np.asarray(adata.obs["is_control"], dtype=bool)

    obs = adata.obs.copy()
    obs["gene_pert"] = [
        [] if c else [("genetic", g, "") for g in pert_components(p)]
        for p, c in zip(perts, is_ctrl)
    ]
    obs["negative_control"] = is_ctrl
    # keep our own label visible too, for the manifest and for debugging
    obs["occam_target_gene"] = perts
    obs["pert_label"] = np.where(is_ctrl, control_label, perts)
    return obs


def load_occam_anndata(data_cfg: Dict[str, Any], *,
                       cell_lines: Optional[Sequence[str]] = None,
                       row_filter: Optional[str] = None,
                       max_cells_per_line: Optional[int] = None,
                       verify_sources: bool = True):
    """Our benchmark as one CellFlow-shaped AnnData.

    `row_filter` is `None` (everything), `"train"` (the finetune rows of every line) or
    `"validation"` (the HepG2 validation rows only). The split itself is read live from the
    frozen benchmark, so it cannot drift from `splits.json`.
    """
    import anndata as ad

    import occam_data as od

    bench_dir = occam_bench_dir(data_cfg)
    bench = od.load_occam_benchmark(bench_dir, verify_sources=verify_sources)

    target_sum = data_cfg.get("occam_target_sum", 10_000.0)
    log1p = bool(data_cfg.get("occam_log1p", True))
    if data_cfg.get("target_sum") is not None or bool(data_cfg.get("log1p", False)):
        raise ValueError(
            f"data.target_sum / data.log1p must stay null / false with occam_bench_dir set "
            f"(got {data_cfg.get('target_sum')!r} / {data_cfg.get('log1p')!r}). The data "
            f"arrives already normalised at target_sum={float(target_sum):g} on the FULL "
            f"source panel row sum; a second transform would renormalise on the "
            f"6,983-gene sum instead and break both `mae` and the bridge's emission map.")

    adata = od.load_occam_adata(
        # `None` must stay None so the reader uses every line the benchmark has;
        # `list(None)` raises, which is how the first fix half-landed.
        bench, cell_lines=(list(cell_lines) if cell_lines is not None else None),
        target_sum=float(target_sum), log1p=log1p,
        control_label="empty", pert_col_out="pert",
        max_cells_per_line=max_cells_per_line)

    if row_filter is not None:
        idx = od.occam_split_indices(bench, adata)
        key = {"train": "finetune", "validation": "validation", "test": "test"}[row_filter]
        keep = np.asarray(idx[key], dtype=np.int64)
        adata = adata[keep].copy()

    # CellFlow's convention: var_names are the gene ids. Ours are symbols for cell-eval, so
    # swap to Ensembl here and assert the ORDER, which is what the bridge relies on.
    var = pd.DataFrame(index=pd.Index(
        [str(g) for g in adata.var["ensembl_gene_id"]], name="ensembl_gene_id"))
    var["gene_symbol"] = [str(g) for g in adata.var["gene_symbol"]]
    assert_axis_order(var.index, bench)

    out = ad.AnnData(X=np.asarray(adata.X, dtype=np.float32),
                     obs=_to_cellflow_obs(adata, control_label="empty").reset_index(drop=True),
                     var=var)
    out.obs.index = out.obs.index.astype(str)
    out.uns["occam"] = dict(adata.uns.get("occam", {}))
    out.uns["occam_bridge"] = {
        "data_dir": bench_dir,
        "feature": "occam_h5ad",
        "n_cells_full": int(out.n_obs),
        "n_genes_full": int(out.n_vars),
        "row_filter": row_filter,
    }
    return out


def load_occam_sources(data_cfg: Dict[str, Any], *, seed: int = 0,
                       max_cells_per_line: Optional[int] = None,
                       verify_sources: bool = True):
    """The TRAINING AnnData: every line's finetune rows, concatenated.

    ⚠️ ASYMMETRY BY DESIGN. HepG2 contributes only its finetune rows; jurkat and RPE1
    contribute every cell. That is the `rhj_fewshot` design -- the very same perturbation held
    out of HepG2 still TRAINS from jurkat and RPE1, which is what makes the task
    few-shot-in-a-new-context rather than never-seen-at-all.

    Mirrors `load_dump_sources`'s output, including
    `uns["occam_bridge"]["n_obs_per_source"]`, so the caller's per-source logging works
    unchanged.
    """
    del seed  # the split is deterministic; nothing here samples
    adata = load_occam_anndata(data_cfg, row_filter="train",
                               max_cells_per_line=max_cells_per_line,
                               verify_sources=verify_sources)
    lines = np.asarray(adata.obs["cell_type"], dtype=str)
    # ⚠️ The ORDER of the lines, for a stable row order. It must come from the benchmark:
    # rhj_fewshot's trio is not a universal list, and using it on norman_combo asked for
    # HepG2 from a K562-only artifact.
    # `lines` is a PER-ROW array, not a list of line names -- `lines or []` on an ndarray
    # raises "truth value of an array is ambiguous", so the membership set is built from
    # its values.
    present = set(lines.tolist())
    order = [c for c in OCCAM_CELL_LINES if c in present] or sorted(present)
    adata.uns["occam_bridge"] = dict(adata.uns.get("occam_bridge", {}))
    adata.uns["occam_bridge"]["sources"] = order
    adata.uns["occam_bridge"]["n_obs_per_source"] = [
        int((lines == ln).sum()) for ln in order]
    return adata


def load_occam_validation(data_cfg: Dict[str, Any], *,
                          max_cells_per_line: Optional[int] = None,
                          verify_sources: bool = False):
    """The VALIDATION AnnData: HepG2's 103 held-out perturbations, plus nothing else.

    ⚠️ Controls are NOT here. Every control cell is in finetune by construction -- they are
    the paired basal pool. CellFlow's `_ensure_validation_controls` injects the real controls
    it needs, which is the same `injctrl` convention our own arms are scored under.
    """
    # ⚠️ `cell_lines=None`, not `[EVALUATED_LINE]`. EVALUATED_LINE is HEPG2, which is
    # rhj_fewshot's held-out line and does not exist on every benchmark -- on
    # norman_combo (K562 only) pinning it asks for a line the artifact does not have.
    # The row filter already restricts to the validation perturbations, and the
    # benchmark's own `eval_cell_lines` is what decides which line those live in, so
    # taking every line the benchmark has is both correct and benchmark-agnostic.
    return load_occam_anndata(data_cfg, cell_lines=None,
                              row_filter="validation",
                              max_cells_per_line=max_cells_per_line,
                              verify_sources=verify_sources)


def occam_pert_tokens(data_cfg: Dict[str, Any], *,
                      verify_sources: bool = False) -> List[str]:
    """Every perturbation symbol in the benchmark, for the one-hot vocabulary.

    ⚠️ THE VOCABULARY FIXES THE MODEL'S CONDITION-INPUT WIDTH, so it cannot grow after
    training: a test perturbation with no column makes `cf.predict` raise
    `Representation ... not found` and produce NO predictions at all. Upstream hit exactly
    this -- 8 of their 380 test perturbations had never been assayed in any supporting line.

    Read from the benchmark's own split lists rather than from the loaded cells, so a
    subsampled smoke run still builds the full-width vocabulary.
    """
    import occam_data as od

    bench = od.load_occam_benchmark(occam_bench_dir(data_cfg),
                                    verify_sources=verify_sources)
    tokens = set()
    for key in ("train", "val", "test_unseen_pert"):
        for p in (bench.split.get(key) or []):
            if str(p) in (od.CONTROL_LABEL, "empty"):
                continue
            # ⚠️ COMPONENT genes, not the `A+B` string: a pair is two tokens (see
            # `pert_components`). A no-op on rhj_fewshot, whose labels carry no `+`.
            tokens.update(pert_components(str(p)))
    tokens.discard(od.CONTROL_LABEL)
    tokens.discard("empty")
    return sorted(tokens)
