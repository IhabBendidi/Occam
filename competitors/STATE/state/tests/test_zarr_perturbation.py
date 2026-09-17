"""Scale-oriented tests for the native-zarr STATE loader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from state.tx.data.zarr_perturbation import (
    CONTROL_SENTINEL,
    ZarrPerturbationDataModule,
    _derive_canonical,
    encode_obs_pert,
    load_external_perturbation_map,
)


def test_encode_mol_pert_matches_arc_drugname_drugconc():
    """Dump stores nM; Arc cell-load uses µM in the obs string."""
    from state.tx.data.zarr_perturbation import encode_mol_pert

    assert encode_mol_pert([_mol("CCO", "Allopurinol", nm="50.0")]) == "[('Allopurinol', 0.05, 'uM')]"
    assert encode_mol_pert([_mol("CCO", "DMSO_TF", nm="0.0")]) == "[('DMSO_TF', 0.0, 'uM')]"


def _mol(smiles: str, name: str = "drug", nm: str = "500.0") -> list:
    return ["compound", smiles, nm, name]


def _gene(ens: str) -> list:
    return ["genetic", ens]


def test_derive_canonical_matches_python_encoder():
    rows = [
        {"cell_type": "A", "batch_center": "b0", "negative_control": False, "gene_pert": [], "mol_pert": [_mol("CCO", "Ethanol")]},
        {"cell_type": "A", "batch_center": "b0", "negative_control": True, "gene_pert": [], "mol_pert": []},
        {"cell_type": "B", "batch_center": "b1", "negative_control": False, "gene_pert": [_gene("ENSG2"), _gene("ENSG1")], "mol_pert": []},
        {"cell_type": "B", "batch_center": "b1", "negative_control": False, "gene_pert": [_gene("ENSG1")], "mol_pert": []},
        {"cell_type": "B", "batch_center": "b1", "negative_control": False, "gene_pert": [], "mol_pert": [_mol("CCN"), _mol("CCO")]},
    ]
    obs = pl.DataFrame(rows)
    pert_str, is_control, cell_type, batch = _derive_canonical(
        obs, batch_col="batch_center", cell_type_key="cell_type", control_pert="empty"
    )
    expected = [
        encode_obs_pert(r["gene_pert"], r["mol_pert"], negative_control=r["negative_control"])
        for r in rows
    ]
    assert pert_str.tolist() == expected
    assert is_control.tolist() == [False, True, False, False, False]
    assert cell_type.tolist() == ["A", "A", "B", "B", "B"]
    assert batch.tolist() == ["b0", "b0", "b1", "b1", "b1"]


def test_derive_canonical_rejects_mixed_gene_mol():
    obs = pl.DataFrame(
        [
            {
                "cell_type": "A",
                "batch_center": "b0",
                "negative_control": False,
                "gene_pert": [_gene("ENSG1")],
                "mol_pert": [_mol("CCO")],
            }
        ]
    )
    with pytest.raises(NotImplementedError, match="mixed"):
        _derive_canonical(obs, batch_col="batch_center", cell_type_key="cell_type", control_pert="empty")


def test_derive_canonical_handles_all_null_gene_pert():
    """Tahoe types ``gene_pert`` as ``List(Null)``; perts must come from ``mol_pert``."""
    obs = pl.DataFrame(
        {
            "cell_type": ["A", "A"],
            "batch_center": ["b0", "b0"],
            "negative_control": [False, True],
            "gene_pert": pl.Series([None, None], dtype=pl.List(pl.Null)),
            "mol_pert": [[_mol("CCO")], []],
        }
    )
    assert obs.schema["gene_pert"] == pl.List(pl.Null)
    pert_str, is_control, _, _ = _derive_canonical(
        obs, batch_col="batch_center", cell_type_key="cell_type", control_pert="empty"
    )
    assert pert_str.tolist() == ["[('drug', 0.5, 'uM')]", "empty"]
    assert is_control.tolist() == [False, True]


def _write_source(dirpath: Path, *, obs: pl.DataFrame, x: np.ndarray) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    var = pl.DataFrame(
        {
            "ensembl_gene_id": [f"ENSG{i}" for i in range(x.shape[1])],
            "gene_symbol": [f"G{i}" for i in range(x.shape[1])],
        }
    )
    obs.write_parquet(dirpath / "obs.parquet")
    var.write_parquet(dirpath / "var.parquet")
    np.save(dirpath / "X.npy", x)


def test_setup_attaches_controls_per_cell_type(tmp_path: Path):
    # 0-1 A perts, 2 A NC, 3 B pert, 4 B NC
    obs = pl.DataFrame(
        [
            {"cell_type": "A", "batch_center": "p1", "negative_control": False, "gene_pert": [], "mol_pert": [_mol("CCO")]},
            {"cell_type": "A", "batch_center": "p1", "negative_control": False, "gene_pert": [], "mol_pert": [_mol("CCN")]},
            {"cell_type": "A", "batch_center": "p1", "negative_control": True, "gene_pert": [], "mol_pert": []},
            {"cell_type": "B", "batch_center": "p1", "negative_control": False, "gene_pert": [], "mol_pert": [_mol("CCC")]},
            {"cell_type": "B", "batch_center": "p1", "negative_control": True, "gene_pert": [], "mol_pert": []},
        ]
    )
    src = tmp_path / "src"
    _write_source(src, obs=obs, x=np.arange(15, dtype=np.float32).reshape(5, 3))
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "folds": [
                    {
                        "finetune": [0, 3, 4],
                        "validation": [1],
                        "test": [],
                    }
                ]
            }
        )
    )
    toml = tmp_path / "cfg.toml"
    toml.write_text(
        "\n".join(
            [
                "[datasets]",
                f'src = "{src}"',
                "[training]",
                'src = "train"',
                "[splits]",
                f'src = "{split}"',
                "[split_fold]",
                "src = 0",
            ]
        )
    )
    dm = ZarrPerturbationDataModule(
        toml_config_path=str(toml),
        feature_loading="npy",
        num_workers=0,
        cell_sentence_len=1,
        batch_size=1,
        basal_mapping_strategy="batch",
        control_pert=CONTROL_SENTINEL,
        selected_ensembl_gene_ids_path=None,
        target_sum=None,
        log1p=False,
    )
    dm.setup()
    train_rows = {row for _, row in dm.train_datasets[0].index_map}
    val_rows = {row for _, row in dm.val_datasets[0].index_map}
    # A val pert (row 1) pulls A control (row 2) into val; B has no val pert so B NC stays out.
    assert val_rows == {1, 2}
    # Train perts 0 (A) and 3 (B) pull both cell-type NCs (2 and 4).
    assert train_rows == {0, 3, 2, 4}
    assert dm.test_datasets == []
    item = dm.train_datasets[0][0]
    assert set(item) >= {
        "pert_cell_emb",
        "ctrl_cell_emb",
        "pert_emb",
        "pert_name",
        "cell_type",
        "batch_name",
        "dataset_name",
    }


def test_train_only_source_without_splits(tmp_path: Path):
    obs = pl.DataFrame(
        [
            {"cell_type": "A", "batch_center": "p1", "negative_control": False, "gene_pert": [_gene("ENSG1")], "mol_pert": []},
            {"cell_type": "A", "batch_center": "p1", "negative_control": True, "gene_pert": [], "mol_pert": []},
        ]
    )
    src = tmp_path / "hepg2"
    _write_source(src, obs=obs, x=np.ones((2, 2), dtype=np.float32))
    toml = tmp_path / "cfg.toml"
    toml.write_text(
        "\n".join(
            [
                "[datasets]",
                f'hepg2 = "{src}"',
                "[training]",
                'hepg2 = "train"',
            ]
        )
    )
    dm = ZarrPerturbationDataModule(
        toml_config_path=str(toml),
        feature_loading="npy",
        num_workers=0,
        cell_sentence_len=1,
        batch_size=1,
        control_pert=CONTROL_SENTINEL,
        target_sum=None,
        log1p=False,
    )
    dm.setup()
    assert {row for _, row in dm.train_datasets[0].index_map} == {0, 1}
    assert dm.train_datasets[0][0]["pert_name"] in {"ENSG1", "empty"}
    assert dm.get_var_dims()["pert_dim"] == 2


def test_external_perturbation_features_preserve_batch_contract(tmp_path: Path):
    obs = pl.DataFrame(
        [
            {
                "cell_type": "A",
                "batch_center": "p1",
                "negative_control": False,
                "gene_pert": [_gene("ENSG1")],
                "mol_pert": [],
            },
            {
                "cell_type": "A",
                "batch_center": "p1",
                "negative_control": True,
                "gene_pert": [],
                "mol_pert": [],
            },
        ]
    )
    src = tmp_path / "source"
    _write_source(src, obs=obs, x=np.ones((2, 2), dtype=np.float32))
    toml = tmp_path / "config.toml"
    toml.write_text("\n".join(["[datasets]", f'source = "{src}"', "[training]", 'source = "train"']))
    feature_path = tmp_path / "pert_features.pt"
    features = {
        "ENSG1": torch.tensor([0.25, 0.5, 0.75]),
        CONTROL_SENTINEL: torch.zeros(3),
    }
    torch.save(features, feature_path)

    dm = ZarrPerturbationDataModule(
        toml_config_path=str(toml),
        feature_loading="npy",
        num_workers=0,
        cell_sentence_len=1,
        batch_size=1,
        control_pert=CONTROL_SENTINEL,
        target_sum=None,
        log1p=False,
        pert_feature_mode="external",
        pert_feature_path=str(feature_path),
    )
    dm.setup()

    dataset = dm.train_datasets[0]
    actual = {dataset[i]["pert_name"]: dataset[i]["pert_emb"] for i in range(len(dataset))}
    assert dm.get_var_dims()["pert_dim"] == 3
    assert torch.equal(actual["ENSG1"], features["ENSG1"])
    assert torch.equal(actual[CONTROL_SENTINEL], features[CONTROL_SENTINEL])


def test_external_perturbation_features_validate_coverage_and_dimension(tmp_path: Path):
    feature_path = tmp_path / "pert_features.pt"
    torch.save({"pert-a": [1.0, 2.0]}, feature_path)
    with pytest.raises(TypeError, match="must be a tensor"):
        load_external_perturbation_map(feature_path, {"pert-a"})

    torch.save({"pert-a": torch.ones(2)}, feature_path)
    with pytest.raises(ValueError, match="missing 1 IDs"):
        load_external_perturbation_map(feature_path, {"pert-a", "pert-b"})

    torch.save({"pert-a": torch.ones(2), "pert-b": torch.ones(3)}, feature_path)
    with pytest.raises(ValueError, match="dimension 3, expected 2"):
        load_external_perturbation_map(feature_path, {"pert-a", "pert-b"})
