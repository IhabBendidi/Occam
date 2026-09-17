"""Regression tests for the zarr benchmark path.

These exist because two of the three pandas-3 breakages found while building this arm failed
*silently* -- one left ``treatment_data`` completely empty, the other dropped a single gene from
the control means. Training still started in both cases. Assert the invariants directly.

Requires the benchmark data on disk (~30 s to load, no GPU):

    uv run pytest tests/test_zarr_pipeline.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

import gspp.constants as cs

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "manifests" / "ihabxcell_hepg2_zarr.toml"
CONFIG = ROOT / "configs" / "config-txpert-zarr-adjmap.yaml"
FEWSHOT_MANIFEST = ROOT / "configs" / "manifests" / "ihabxcell_hepg2_fewshot_zarr.toml"
FEWSHOT_CONFIG = ROOT / "configs" / "config-txpert-zarr-fewshot.yaml"

# Fixed properties of replogle_nadig fold 0, cross-checked against the split file and against
# hooke-tx's own reading of the same zarr datasets.
N_GENES = 2000
N_CELLS = 643_413
N_TEST_ROWS = 26_878
N_TEST_PERTS = 380
TEST_CELL_TYPE = "hepg2"

# Replogle Nadig Few Shot split. 60 whole perturbations carved out of finetune, HepG2 only.
N_VAL_ROWS = 4_825
N_VAL_PERTS = 60

pytestmark = pytest.mark.skipif(
    not Path("/rxrx/scratch/hooke_tx/public/ihabxcell_replogle_nadig_hepg2").exists(),
    reason="benchmark zarr datasets not available",
)


@pytest.fixture(scope="module")
def adata_and_manifest():
    from gspp.data.zarr_loader import GeneIDMapper, load_manifest, load_zarr_datasets

    manifest = load_manifest(MANIFEST)
    adata = load_zarr_datasets(
        manifest,
        gene_mapper=GeneIDMapper(),
        cell_line_rename={"HepG2": "hepg2", "Jurkat": "jurkat"},
    )
    return adata, manifest


@pytest.fixture(scope="module")
def datamodule():
    from gspp.data.datamodule_hybrid import HybridDataModuleWrapper
    from gspp.utils import modify_config

    # The config uses ${hydra:runtime.cwd} so that paths survive hydra chdir'ing into its
    # output directory. Outside a hydra run that resolver does not exist, so stub it with the
    # value hydra would supply when main.py is launched from the project root.
    if not OmegaConf.has_resolver("hydra"):
        OmegaConf.register_new_resolver("hydra", lambda _key: str(ROOT))

    cfg = modify_config(OmegaConf.load(CONFIG))
    args = OmegaConf.to_container(cfg.datamodule, resolve=True)
    args.pop("added_data_args", None)

    dm = HybridDataModuleWrapper(base_dm_args=args)
    dm.prepare_data()
    dm.setup("fit")
    return dm


@pytest.fixture(scope="module")
def fewshot_datamodule():
    """Same as `datamodule`, on the fewshot manifest -- i.e. with a real validation split."""
    from gspp.data.datamodule_hybrid import HybridDataModuleWrapper
    from gspp.utils import modify_config

    if not OmegaConf.has_resolver("hydra"):
        OmegaConf.register_new_resolver("hydra", lambda _key: str(ROOT))

    cfg = modify_config(OmegaConf.load(FEWSHOT_CONFIG))
    args = OmegaConf.to_container(cfg.datamodule, resolve=True)
    args.pop("added_data_args", None)

    dm = HybridDataModuleWrapper(base_dm_args=args)
    dm.prepare_data()
    dm.setup("fit")
    return dm


def test_manifest_matches_sibling_competitors():
    """All arms must read the same datasets, split file and fold."""
    import tomllib

    ours = tomllib.loads(MANIFEST.read_text())
    theirs = tomllib.loads((ROOT.parent / "STATE" / "state" / "ihabxcell_hepg2_zarr.toml").read_text())
    assert ours == theirs, "manifest has diverged from STATE's -- the arms would differ"


def test_shape_and_scale(adata_and_manifest):
    adata, _ = adata_and_manifest
    assert adata.shape == (N_CELLS, N_GENES)
    assert sorted(adata.obs[cs.CELL_TYPE].unique()) == ["K562", "RPE1", "hepg2", "jurkat"]
    # No renormalization: the zarr is log1p(1e4) and cell-eval's lognorm check rejects >= 15.
    assert 0.0 <= adata.X.min()
    assert adata.X.max() < 15.0, "expression looks renormalized or on a count scale"


def test_no_gene_dropped_by_symbol_collisions(adata_and_manifest):
    """var_names_make_unique must not change the gene count (guards a silent axis shift)."""
    adata, _ = adata_and_manifest
    assert adata.n_vars == N_GENES
    assert len(set(adata.var_names)) == N_GENES


def test_split_resolves_to_expected_perturbations(adata_and_manifest):
    from gspp.data.zarr_loader import load_split_perturbations

    adata, manifest = adata_and_manifest
    split = load_split_perturbations(manifest, adata)
    assert len(split["test"]) == N_TEST_PERTS
    # Disjointness is what makes an index-based split representable as gspp's name-based
    # condition_group at all; load_split_perturbations enforces it per dataset, this checks
    # the combined result.
    assert not (set(split["test"]) & set(split["train"]))
    assert len(split["test"]) + len(split["train"]) == len(
        set(adata.obs.loc[~adata.obs[cs.IS_CONTROL], cs.GENE_NAME].unique())
    )
    # The raw split file must agree on row counts.
    spec = json.load(open(manifest["splits"]["ihabxcell_hepg2"]))["folds"][0]
    assert len(spec["test"]) == N_TEST_ROWS
    assert spec["validation"] == [], "fold 0 is expected to have no validation split"


def test_all_test_perturbations_survive_vocabulary_filter(datamodule):
    """The regression that motivated extending pert2id.

    33 benchmark perturbations (9 of them in the test set) are missing from
    ``data/gears_gene_set.csv``. Without the extension they are silently dropped and TxPert is
    scored on 371 perturbations while the other arms score 380.
    """
    b = datamodule.base_dm
    retained = {p.replace("+ctrl", "") for p in b.condition_group["test"]}
    assert len(retained) == N_TEST_PERTS
    assert b.condition_group["val"] == []


def test_treatment_data_is_not_empty(datamodule):
    """pandas 3 `categorical.str.split` regression: this silently went to 0 rows."""
    b = datamodule.base_dm
    assert b.treatment_data.n_obs > 0
    assert b.treatment_data.n_obs == b.adata.n_obs - int(
        (b.adata.obs[cs.CONTROL].astype(int) == 1).sum()
    )


def test_no_holdout_leakage_into_training(datamodule):
    """Held-out HepG2 perturbations must not train, but must train in the other lines."""
    b = datamodule.base_dm
    tr = b.train_adata.obs
    test_conditions = set(b.condition_group["test"])

    leaked = int(((tr[cs.CELL_TYPE] == TEST_CELL_TYPE) & tr[cs.CONDITION].isin(test_conditions)).sum())
    assert leaked == 0, f"{leaked} held-out HepG2 cells leaked into training"

    elsewhere = int(((tr[cs.CELL_TYPE] != TEST_CELL_TYPE) & tr[cs.CONDITION].isin(test_conditions)).sum())
    assert elsewhere > 0, "test perturbations should still train in the other cell lines"

    assert len(b.test_data) == N_TEST_ROWS


def test_control_means_keep_every_gene(datamodule):
    """pandas 3 `groupby.apply` regression: control means lost one gene (2000 -> 1999)."""
    from gspp.metrics import CELL_TYPE_MEAN_CONTROL, compute_mean_control

    b = datamodule.base_dm
    CELL_TYPE_MEAN_CONTROL.pop(TEST_CELL_TYPE, None)
    compute_mean_control(b.adata, TEST_CELL_TYPE)
    overall, per_batch = CELL_TYPE_MEAN_CONTROL[TEST_CELL_TYPE]

    assert overall.shape[0] == N_GENES
    assert per_batch.shape[1] == N_GENES, "a gene was dropped from the per-batch control means"
    assert per_batch.shape[0] > 0
    assert np.isfinite(per_batch.to_numpy()).all()


def test_graph_top_k_matches_nlargest():
    """The `mode=top_N` rewrite in graphmodule must select exactly what nlargest did."""
    import pandas as pd

    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "regulator": rng.integers(0, 50, 4000),
        "target": rng.integers(0, 30, 4000),
        "weight": rng.choice([1.0, 2.0, 2.0, 3.0, 3.5], 4000),  # heavy ties
    })
    n = 20
    new = (
        df.sort_values("weight", ascending=False, kind="mergesort")
        .groupby("target", sort=False, group_keys=False)
        .head(n)
        .reset_index(drop=True)
    )
    ref = pd.concat([g.nlargest(n, ["weight"]) for _, g in df.groupby("target", sort=False)])
    key = lambda d: sorted(map(tuple, d[["regulator", "target", "weight"]].to_numpy().tolist()))
    assert key(new) == key(ref)


# ---------------------------------------------------------------------------------------------
# Replogle Nadig Few Shot split. The tests above pin the ORIGINAL manifest's behaviour (empty
# validation, val group empty) and must keep passing unchanged -- that manifest is byte-shared
# with STATE and PerturbDiff. These pin the fewshot manifest beside them.
# ---------------------------------------------------------------------------------------------


def test_fewshot_manifest_differs_only_in_the_split():
    """Everything except the hepg2 split path must match the original, or the arms diverge."""
    import tomllib

    ours = tomllib.loads(MANIFEST.read_text())
    fewshot = tomllib.loads(FEWSHOT_MANIFEST.read_text())
    assert ours["datasets"] == fewshot["datasets"]
    assert ours["split_fold"] == fewshot["split_fold"]
    assert set(ours["splits"]) == set(fewshot["splits"])
    assert ours["splits"]["ihabxcell_hepg2"] != fewshot["splits"]["ihabxcell_hepg2"]

    # And it must be the same split file PerturbDiff's fewshot arm reads.
    theirs = tomllib.loads(
        (ROOT.parent / "PerturbDiff" / "configs" / "manifests"
         / "ihabxcell_hepg2_fewshot_zarr.toml").read_text()
    )
    assert fewshot["splits"]["ihabxcell_hepg2"] == theirs["splits"]["ihabxcell_hepg2"]


def test_fewshot_split_resolves_to_whole_perturbations(adata_and_manifest):
    """The premise of a name-based split: no perturbation may straddle two sides.

    gspp splits by perturbation name, so a validation set built by sampling *cells* could not be
    represented at all. This asserts the fewshot split is 60 whole perturbations and that the
    names reproduce exactly the 4,825 rows the split file lists.
    """
    from gspp.data.zarr_loader import load_manifest, load_split_perturbations

    adata, _ = adata_and_manifest
    manifest = load_manifest(FEWSHOT_MANIFEST)
    split = load_split_perturbations(manifest, adata)

    assert len(split["validation"]) == N_VAL_PERTS
    assert len(split["test"]) == N_TEST_PERTS
    assert not (set(split["validation"]) & set(split["test"]))
    assert not (set(split["validation"]) & set(split["train"]))

    hepg2 = adata.obs[(adata.obs["dataset"] == "ihabxcell_hepg2") & ~adata.obs[cs.IS_CONTROL]]
    n_rows = int(hepg2[cs.GENE_NAME].isin(split["validation"]).sum())
    assert n_rows == N_VAL_ROWS, (
        f"the 60 validation names cover {n_rows} HepG2 rows, but the split file lists "
        f"{N_VAL_ROWS}. The name-based representation is lossy and must not be used."
    )

    spec = json.load(open(manifest["splits"]["ihabxcell_hepg2"]))["folds"][0]
    assert len(spec["validation"]) == N_VAL_ROWS
    assert len(spec["test"]) == N_TEST_ROWS


def test_fewshot_validation_dataset_is_built(fewshot_datamodule):
    b = fewshot_datamodule.base_dm
    assert len(b.condition_group["val"]) == N_VAL_PERTS
    assert b.val_data is not None, "no validation dataset; selection would have nothing to rank on"
    assert len(b.val_data) == N_VAL_ROWS
    assert len(b.test_data) == N_TEST_ROWS


def test_fewshot_validation_holdout_is_hepg2_only(fewshot_datamodule):
    """Validation follows the same rule as test: out of HepG2 training, still training elsewhere.

    This is the fairness condition. hooke-tx, STATE and PerturbDiff all carve validation out of
    the HepG2 rows and keep those perturbations in the other three cell lines; excluding them
    everywhere would train TxPert on strictly less data than the arms it is compared against.
    """
    b = fewshot_datamodule.base_dm
    tr = b.train_adata.obs
    val_conditions = set(b.condition_group["val"])

    leaked = int(
        ((tr[cs.CELL_TYPE] == TEST_CELL_TYPE) & tr[cs.CONDITION].isin(val_conditions)).sum()
    )
    assert leaked == 0, f"{leaked} HepG2 validation cells leaked into training"

    elsewhere = int(
        ((tr[cs.CELL_TYPE] != TEST_CELL_TYPE) & tr[cs.CONDITION].isin(val_conditions)).sum()
    )
    assert elsewhere > 0, "validation perturbations should still train in the other cell lines"


def test_val_dataloader_does_not_fall_back_to_test(datamodule):
    """It used to return test_dataloader() when no validation dataset existed.

    Every `val_*` metric on the zarr arm was therefore a TEST metric, and anything selected on one
    was selected with test knowledge. STATE shipped the identical fallback. A hard error is the
    only safe behaviour: a silent one cannot be noticed from the logs.
    """
    assert datamodule.base_dm.val_data is None  # original manifest, empty validation list
    with pytest.raises(RuntimeError, match="no validation dataset"):
        datamodule.val_dataloader()


def test_fewshot_val_dataloader_yields_the_validation_split(fewshot_datamodule):
    loader = fewshot_datamodule.val_dataloader()
    assert sum(len(b.x) for b in loader) == N_VAL_ROWS


def test_control_token_is_not_a_graph_node(fewshot_datamodule):
    """`ctrl` maps to pert id -1, which is Python negative indexing into the embedding table.

    It lands on row ``num_perts - 1``, which works only because no real gene owns that id: the row
    is isolated in every knowledge graph, so its embedding is the MLP bias -- a de-facto learned
    control token. A graph edge whose symbol were literally "ctrl" would write into that row and
    silently contaminate it.
    """
    b = fewshot_datamodule.base_dm
    assert b.pert2id[cs.CONTROL_LABEL] == -1
    # -1 must be the only negative id, and the max real id must be below num_perts - 1.
    ids = [i for i in b.pert2id.values()]
    assert min(ids) == -1 and sorted(i for i in ids if i < 0) == [-1]
    assert max(ids) == len(b.pert2id) - 2, (
        "the control token's row is no longer unowned; z_p[-1] would collide with a real gene"
    )
