"""Read the the upstream benchmark canonical zarr format into the AnnData shape gspp expects.

The benchmark datasets are ``X.zarr`` (zarr v3) + ``obs.parquet`` + ``var.parquet``, one
directory per cell line, listed in a manifest TOML. This is the same data, the same manifest
and the same split file that ``../STATE/state`` and ``../PerturbDiff`` read; see
``STATE/state/src/state/tx/data/zarr_perturbation.py`` for the reference implementation.

Two things make this more than a format shim:

* **ID space.** The zarr is Ensembl end-to-end (``var.ensembl_gene_id``, and perturbations as
  Ensembl ids inside ``obs.gene_pert``). gspp is gene-symbol native: ``pert2id`` comes from
  ``data/gears_gene_set.csv`` and every knowledge graph is keyed by symbol. So perturbation ids
  and gene names are translated to symbols on load.
* **Splits.** The the upstream benchmark split file is *row-index* based, not perturbation-name based.
  :func:`load_split_perturbations` converts fold indices into the perturbation-name sets that
  ``PertDataModule.condition_group`` wants, and checks the two are disjoint (if a perturbation
  straddled the split, name-based exclusion downstream would silently leak).

No normalization happens here: ``X.zarr`` is already ``log1p(target_sum=1e4)`` and both sibling
competitors train on that scale untouched.
"""

from __future__ import annotations

import itertools
import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
import zarr
from loguru import logger
from scipy.sparse import csr_matrix

from gspp.constants import BATCH_RAW, CELL_TYPE, DATA_DIR, GENE_NAME, IS_CONTROL

# Sentinel gspp uses for untreated cells (``_standardize_obs_format`` maps it to "ctrl").
CONTROL_NAME = "non-targeting"

# obs.parquet columns we need. Reading only these keeps the load small -- obs also carries
# `perturbations`, `mol_pert`, `assay_type` and several boolean composition flags.
_OBS_COLUMNS = ["cell_type", "batch_center", "gene_pert"]


class GeneIDMapper:
    """Ensembl <-> symbol translation, from the map vendored in ``data/``.

    Port of ``the upstream tree/data/tx_utils.GeneIDMapper``, which reads the same parquet from a
    personal ``/rxrx/data/user`` directory. Using an identical mapping matters: the upstream benchmark's own
    adjacency-map embedder resolves graph symbols this way, so a different table would give the
    two arms different graph node sets.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        path = Path(path) if path is not None else DATA_DIR / "gene_ensembl_map.parquet"
        table = pl.read_parquet(path)
        self.symbol2ensembl = {gs: ens for gs, ens in table.iter_rows()}
        self.ensembl2symbol = {ens: gs for gs, ens in table.iter_rows()}

    def to_symbols(self, ensembl_ids) -> list[str | None]:
        return [self.ensembl2symbol.get(e) for e in ensembl_ids]


def load_manifest(toml_path: str | Path) -> dict:
    """Parse a dataset manifest TOML (same schema as STATE's ``ihabxcell_hepg2_zarr.toml``)."""
    with open(toml_path, "rb") as fh:
        cfg = tomllib.load(fh)
    if "datasets" not in cfg:
        raise ValueError(f"{toml_path}: manifest has no [datasets] table")
    return {
        "datasets": cfg["datasets"],
        "splits": cfg.get("splits", {}),
        "split_fold": cfg.get("split_fold", {}),
    }


def encode_pert_symbols(gene_pert, ens2sym: dict) -> str:
    """Encode one cell's ``gene_pert`` tuple list as a gspp perturbation name.

    ``gene_pert`` holds ``(pert_type, ensembl_id, effect_class)`` tuples: empty for controls,
    one entry for a single knockdown. Multi-gene perturbations are joined with "+", matching
    gspp's own ``condition`` convention (``_standardize_obs_format`` appends "+ctrl" to singles).

    Unmappable Ensembl ids fall back to the raw id so they stay distinguishable rather than
    silently collapsing together; they will simply not match any knowledge-graph node.
    """
    if gene_pert is None or len(gene_pert) == 0:
        return CONTROL_NAME
    symbols = sorted(ens2sym.get(p[1]) or str(p[1]) for p in gene_pert)
    return "+".join(symbols)


def _read_expression(path: Path, n_obs: int, n_var: int) -> np.ndarray:
    """Read ``X.zarr`` fully into RAM (~900 MB across the four benchmark cell lines)."""
    arr = zarr.open(str(path / "X.zarr"), mode="r")
    x = np.asarray(arr[:], dtype=np.float32)
    if x.shape != (n_obs, n_var):
        raise ValueError(f"{path}: X.zarr is {x.shape}, expected {(n_obs, n_var)}")
    return x


def load_zarr_datasets(
    manifest: dict,
    *,
    gene_mapper: GeneIDMapper | None = None,
    cell_line_rename: dict[str, str] | None = None,
) -> sc.AnnData:
    """Concatenate the manifest's zarr datasets into one gspp-shaped AnnData.

    The returned object carries the columns ``_standardize_obs_format`` requires
    (``gene_name``, ``is_control``, ``batch_num``, ``cell_line``) plus ``dataset`` and
    ``row_in_dataset``, which :func:`load_split_perturbations` uses to resolve index-based
    splits back to perturbations. ``X`` is left on its native ``log1p(1e4)`` scale.
    """
    gene_mapper = gene_mapper or GeneIDMapper()
    ens2sym = gene_mapper.ensembl2symbol
    cell_line_rename = cell_line_rename or {}

    var_ref: pd.DataFrame | None = None
    blocks, obs_frames = [], []

    for name, path in manifest["datasets"].items():
        path = Path(path)
        obs = pd.read_parquet(path / "obs.parquet", columns=_OBS_COLUMNS)
        var = pd.read_parquet(path / "var.parquet")

        if var_ref is None:
            var_ref = var
        elif not var["ensembl_gene_id"].equals(var_ref["ensembl_gene_id"]):
            # Concatenating along cells is only valid if every dataset shares one gene axis
            # in one order. The benchmark datasets are built from a single selected-gene list,
            # so a mismatch means the manifest is pointing somewhere unexpected.
            raise ValueError(
                f"{name}: var.parquet gene axis differs from the first dataset; "
                "all manifest datasets must share the same genes in the same order"
            )

        frame = pd.DataFrame(index=pd.RangeIndex(len(obs)))
        frame[GENE_NAME] = [encode_pert_symbols(gp, ens2sym) for gp in obs["gene_pert"]]
        frame[IS_CONTROL] = frame[GENE_NAME].eq(CONTROL_NAME)
        frame[BATCH_RAW] = obs["batch_center"].astype(str)
        ct = obs["cell_type"].astype(str)
        frame[CELL_TYPE] = ct.map(lambda c: cell_line_rename.get(c, c))
        frame["dataset"] = name
        frame["row_in_dataset"] = np.arange(len(obs), dtype=np.int64)

        blocks.append(_read_expression(path, len(obs), len(var)))
        obs_frames.append(frame)
        logger.info(
            "zarr: {} -> {:,} cells, {:,} controls, cell types {}",
            name, len(obs), int(frame[IS_CONTROL].sum()), sorted(frame[CELL_TYPE].unique()),
        )

    obs_all = pd.concat(obs_frames, ignore_index=True)
    obs_all.index = obs_all.index.astype(str)
    obs_all.index.name = None

    var_all = var_ref.copy()
    var_all[GENE_NAME] = var_all["gene_symbol"].astype(str).values
    var_all.index = pd.Index(var_all[GENE_NAME].astype(str))

    adata = sc.AnnData(X=csr_matrix(np.vstack(blocks)), obs=obs_all, var=var_all)
    adata.var_names_make_unique()
    adata.var[GENE_NAME] = adata.var_names.astype(str)

    logger.info(
        "zarr: combined {:,} cells x {:,} genes | {:,} perturbations | X max {:.2f}",
        adata.n_obs, adata.n_vars,
        obs_all.loc[~obs_all[IS_CONTROL], GENE_NAME].nunique(),
        float(adata.X.max()),
    )
    return adata


def load_split_perturbations(manifest: dict, adata: sc.AnnData) -> dict[str, list[str]]:
    """Resolve the index-based split file into train / validation / test perturbation-name sets.

    the upstream benchmark stores splits as ``{"folds": [{"finetune": [...], "validation": [...],
    "test": [...]}]}`` where the lists are **row indices into that dataset's obs.parquet**. gspp
    splits by perturbation *name*, so we map indices to names.

    That conversion is only faithful if no perturbation appears on two sides of the split --
    otherwise the downstream name-based exclusion in ``_create_datasets`` would drop training
    cells (or leak held-out ones). We assert it rather than assume it, pairwise over all three
    sides.

    ``validation`` is optional. ``replogle_nadig_xcell.json`` ships ``validation: []`` and then
    this returns a ``"validation"`` key holding an empty list, leaving train/test byte-identical
    to before the key existed. ``split_replogle_nadig_fewshot__v1.json`` ships 4,825 rows =
    60 whole perturbations.
    """
    test_names: set[str] = set()
    val_names: set[str] = set()
    train_names: set[str] = set()

    for name, split_path in manifest["splits"].items():
        fold = manifest["split_fold"].get(name, 0)
        with open(split_path) as fh:
            folds = json.load(fh)["folds"]
        if fold >= len(folds):
            raise ValueError(f"{name}: fold {fold} requested but split file has {len(folds)}")
        spec = folds[fold]

        sub = adata.obs[adata.obs["dataset"] == name]
        if sub.empty:
            raise ValueError(f"split references dataset {name!r}, absent from the manifest data")
        by_row = pd.Series(sub[GENE_NAME].values, index=sub["row_in_dataset"].values)

        def names_for(idx):
            picked = by_row.reindex(np.asarray(idx, dtype=np.int64))
            if picked.isna().any():
                raise ValueError(f"{name}: split indices out of range for {len(by_row)} rows")
            return set(picked.values) - {CONTROL_NAME}

        te = names_for(spec["test"])
        tr = names_for(spec.get("finetune", []))
        va = names_for(spec.get("validation", []))
        for (a, an), (b, bn) in itertools.combinations(
            ((te, "test"), (va, "validation"), (tr, "finetune")), 2
        ):
            overlap = a & b
            if overlap:
                raise ValueError(
                    f"{name}: {len(overlap)} perturbation(s) appear in BOTH {an} and {bn} "
                    f"(e.g. {sorted(overlap)[:5]}). gspp splits by name, so this cannot be "
                    "represented faithfully."
                )
        logger.info(
            "split {} fold {}: {:,} test / {:,} val / {:,} train rows "
            "-> {} test, {} val, {} train perts",
            name, fold, len(spec["test"]), len(spec.get("validation", [])),
            len(spec.get("finetune", [])), len(te), len(va), len(tr),
        )
        test_names |= te
        val_names |= va
        train_names |= tr

    # Datasets without a split entry are wholly training data (jurkat / k562 / rpe1 here).
    #
    # Validation names must be subtracted here as well as test. "Not test" used to mean "train",
    # which is precisely how a populated validation list would be swept back into training --
    # silently, since every name is observed somewhere. Subtracting both keeps this line doing its
    # real job (absorbing the unsplit cell lines) without overriding the split file.
    all_names = set(adata.obs.loc[~adata.obs[IS_CONTROL], GENE_NAME].unique())
    train_names |= (all_names - test_names - val_names)

    return {
        "train": sorted(train_names),
        "validation": sorted(val_names),
        "test": sorted(test_names),
    }
