"""Unit tests for the Hooke → CellFlow AnnData bridge."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hooke_bridge import (  # noqa: E402
    apply_cellflow_gene_schema,
    apply_normalize_log1p,
    dataset_pert_tokens,
    fit_and_project_pca,
    load_fold,
    load_hooke_anndata,
    load_pca_artifacts,
    resolve_gene_list,
    save_pca_artifacts,
    write_synthetic_hooke_dump,
)


@pytest.fixture()
def dump_dir(tmp_path: Path) -> Path:
    return write_synthetic_hooke_dump(tmp_path / "dump", n_cells=40, n_genes=8, with_combos=True)


def test_load_hooke_anndata_from_zarr(dump_dir: Path):
    adata = load_hooke_anndata(dump_dir, feature="zarr")
    assert adata.n_obs == 40
    assert adata.n_vars == 8
    assert adata.X.dtype == np.float32
    assert "gene_pert" in adata.obs.columns


def test_load_with_gene_subset_and_normalize(dump_dir: Path):
    genes = [f"ENSG{i:011d}" for i in range(4)]
    adata = load_hooke_anndata(
        dump_dir,
        selected_genes=genes,
        target_sum=1_000.0,
        log1p=True,
    )
    assert adata.n_vars == 4
    assert list(adata.var_names) == genes
    # After normalize+log1p, row sums of expm1(X) ≈ target_sum for non-empty cells.
    recovered = np.expm1(np.asarray(adata.X)).sum(axis=1)
    nonzero = recovered > 0
    np.testing.assert_allclose(recovered[nonzero], 1_000.0, rtol=1e-4)


def test_missing_selected_gene_fails(dump_dir: Path):
    with pytest.raises(ValueError, match="missing"):
        load_hooke_anndata(dump_dir, selected_genes=["ENSG_NOT_REAL"])


def test_intersect_missing_genes(dump_dir: Path):
    genes = [f"ENSG{i:011d}" for i in range(3)] + ["ENSG_NOT_REAL"]
    adata = load_hooke_anndata(
        dump_dir,
        selected_genes=genes,
        on_missing_genes="intersect",
    )
    assert list(adata.var_names) == genes[:3]


def test_resolve_gene_list_intersect():
    kept = resolve_gene_list(["a", "b"], ["b", "c", "a"], on_missing="intersect")
    assert kept == ["b", "a"]


def test_load_fold(tmp_path: Path):
    path = tmp_path / "split.json"
    path.write_text(
        '{"folds":[{"finetune":[0,1],"validation":[2],"test":[3]},'
        '{"finetune":[3],"validation":[],"test":[0]}]}'
    )
    fold0 = load_fold(path, fold=0)
    assert fold0["finetune"] == [0, 1]
    assert fold0["validation"] == [2]
    with pytest.raises(IndexError):
        load_fold(path, fold=2)


def test_cellflow_schema_norman_style(dump_dir: Path):
    adata = load_hooke_anndata(dump_dir)
    apply_cellflow_gene_schema(adata, max_combination_length=2)
    assert "is_control" in adata.obs
    assert "gene_1" in adata.obs and "gene_2" in adata.obs
    assert "pert_str" in adata.obs
    assert adata.obs.loc[adata.obs["is_control"], "pert_str"].eq("empty").all()
    # Combo rows encode sorted ENSG+ENSG
    combos = adata.obs.loc[adata.obs["pert_str"].str.contains(r"\+", regex=True), "pert_str"]
    assert len(combos) >= 1
    assert all("+" in s for s in combos)
    schema = adata.uns["cellflow_schema"]
    assert schema["perturbation_covariates"]["gene"] == ("gene_1", "gene_2")
    assert "NULL" in adata.uns["gene_embeddings"]


def test_cellflow_schema_rejects_too_short_max_len(dump_dir: Path):
    adata = load_hooke_anndata(dump_dir)
    with pytest.raises(ValueError, match="max_combination_length"):
        apply_cellflow_gene_schema(adata, max_combination_length=1)


def test_pca_fit_project_and_persist(dump_dir: Path, tmp_path: Path):
    full = load_hooke_anndata(dump_dir, target_sum=1000.0, log1p=True)
    apply_cellflow_gene_schema(full, max_combination_length=2)
    train = full[:30].copy()
    test = full[30:].copy()
    artifacts = fit_and_project_pca(train, {"test": test}, n_comps=4)
    assert "X_pca" in train.obsm and "X_pca" in test.obsm
    assert train.obsm["X_pca"].shape[1] == 4
    assert test.obsm["X_pca"].shape[1] == 4
    out = save_pca_artifacts(artifacts, tmp_path / "pca")
    loaded = load_pca_artifacts(out)
    np.testing.assert_array_equal(loaded["means"], artifacts["means"])
    np.testing.assert_array_equal(loaded["pcs"], artifacts["pcs"])
    # Round-trip reconstruct ≈ centered projection identity on train PCs.
    from hooke_bridge import reconstruct_from_pca

    recon = reconstruct_from_pca(train.obsm["X_pca"], means=loaded["means"], pcs=loaded["pcs"], clip_min=None)
    assert recon.shape == train.X.shape
    assert np.isfinite(recon).all()


def test_train_set_overrides():
    from train_cellflow import _parse_set_item, _set_path

    cfg: dict = {"model": {"pooling": "mean", "probability_path": {"constant_noise": 0.5}}}
    key, value = _parse_set_item("model.pooling=attention_token")
    _set_path(cfg, key, value)
    key, value = _parse_set_item("model.probability_path.constant_noise=1.5")
    _set_path(cfg, key, value)
    key, value = _parse_set_item("model.linear_projection_before_concatenation=true")
    _set_path(cfg, key, value)
    assert cfg["model"]["pooling"] == "attention_token"
    assert cfg["model"]["probability_path"]["constant_noise"] == 1.5
    assert cfg["model"]["linear_projection_before_concatenation"] is True


def test_train_config_environment_expansion(monkeypatch: pytest.MonkeyPatch):
    from train_cellflow import _expand_env_vars

    monkeypatch.setenv("USER", "cellflow-user")
    expanded = _expand_env_vars({"output": ["/scratch/${USER}", 100_000]})
    assert expanded == {"output": ["/scratch/cellflow-user", 100_000]}


def test_normalize_helper_inplace():
    from anndata import AnnData

    x = np.ones((2, 3), dtype=np.float32)
    adata = AnnData(X=x.copy())
    apply_normalize_log1p(adata, target_sum=30.0, log1p=True)
    assert adata.X.shape == (2, 3)
    np.testing.assert_allclose(np.expm1(adata.X).sum(axis=1), 30.0, rtol=1e-5)


# ---------------------------------------------------------------------------------------------
# Few-shot split support. These pin the two bridge defects that made
# split_replogle_nadig_fewshot__v1.json -- the split every other competitor arm is benchmarked
# on -- unusable: a split loader that rejected the file outright, and a one-hot vocabulary built
# from the training cells only.
# ---------------------------------------------------------------------------------------------


def test_load_fold_skips_scalar_metadata(tmp_path: Path):
    """Fold metadata must not make a split unloadable.

    `split_replogle_nadig_fewshot__v1.json` carries `outer_fold` / `inner_fold` ints beside its
    index lists, and rejecting them raised TypeError before a single cell was read.
    """
    path = tmp_path / "split.json"
    path.write_text(
        '{"folds":[{"outer_fold":0,"inner_fold":0,'
        '"finetune":[0,1],"validation":[2],"test":[3]}]}'
    )
    fold = load_fold(path, fold=0)
    assert set(fold) == {"finetune", "validation", "test"}, "metadata keys must be dropped"
    assert fold["validation"] == [2]

    # A dict-shaped value is still an error: dropping it silently could hide a real index list
    # behind a schema change.
    bad = tmp_path / "bad.json"
    bad.write_text('{"folds":[{"finetune":{"nested":[0]}}]}')
    with pytest.raises(TypeError):
        load_fold(bad, fold=0)

    # And a fold with no index lists at all is an error, not an empty result.
    empty = tmp_path / "empty.json"
    empty.write_text('{"folds":[{"outer_fold":0}]}')
    with pytest.raises(ValueError):
        load_fold(empty, fold=0)


def test_extra_tokens_widen_the_one_hot_vocabulary(dump_dir: Path):
    """The vocabulary must be able to cover perturbations absent from the given rows.

    It fixes the model's condition-input width, so it cannot grow after training: a test
    perturbation without a column makes `cf.predict` raise "Representation ... not found" and
    produces no predictions at all. On the few-shot split that is 8 of 380 test perturbations,
    never assayed in any training line.
    """
    adata = load_hooke_anndata(dump_dir)
    apply_cellflow_gene_schema(adata, max_combination_length=2)
    base = set(adata.uns["gene_embeddings"])

    unseen = ["ENSG_NEVER_SEEN_1", "ENSG_NEVER_SEEN_2"]
    assert not base & set(unseen)

    wide = load_hooke_anndata(dump_dir)
    apply_cellflow_gene_schema(wide, max_combination_length=2, extra_tokens=unseen)
    emb = wide.uns["gene_embeddings"]

    assert set(emb) == base | set(unseen)
    assert all(t in emb for t in unseen), "unseen tokens must be representable"
    # One-hot must stay one-hot at the wider width, or the condition encoding is not what the
    # model thinks it is.
    dim = len(next(iter(emb.values())))
    assert dim == len(emb)
    for token, vec in emb.items():
        assert float(np.asarray(vec).sum()) == 1.0, f"{token} is not one-hot"

    # Absent extra_tokens, behaviour is byte-identical to before this parameter existed.
    again = load_hooke_anndata(dump_dir)
    apply_cellflow_gene_schema(again, max_combination_length=2, extra_tokens=())
    assert set(again.uns["gene_embeddings"]) == base


def test_dataset_pert_tokens_reads_every_split(dump_dir: Path):
    """The dataset vocabulary must not depend on which rows a split happens to select."""
    tokens = dataset_pert_tokens([dump_dir])
    adata = load_hooke_anndata(dump_dir)
    apply_cellflow_gene_schema(adata, max_combination_length=2)
    from_rows = {t for t in adata.uns["gene_embeddings"] if not t.startswith("NULL")}
    # Reading obs directly must find at least what walking all the rows finds.
    assert from_rows <= set(tokens) | {"NULL"}
    assert tokens == sorted(set(tokens)), "tokens must be sorted and deduped"
