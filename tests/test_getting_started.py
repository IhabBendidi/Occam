"""Custom-data training and selective downloads, without release data or network."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from occam.config.schema import flatten_stage
from occam.data.datasets import _read_gene_names_h5py
from occam.decoder_train.trainer import DecoderTrainer

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_dry_run_never_opens_hub_or_creates_files(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    destination = tmp_path / "downloads"
    assert script("download_release").main([
        "encoder-decoder", "--root", str(destination), "--dry-run",
    ]) == 0
    assert not destination.exists()


def test_download_pins_all_selected_files_to_one_commit(tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")

    downloaded = []
    cached = tmp_path / "cache"
    cached.write_bytes(b"synthetic checkpoint")

    class FakeApi:
        def model_info(self, repo_id, revision):
            assert repo_id == "bendidiihab/Occam"
            assert revision == "main"
            return SimpleNamespace(sha="pinned-commit")

    def fake_download(repo_id, filename, revision):
        assert repo_id == "bendidiihab/Occam"
        assert revision == "pinned-commit"
        downloaded.append(filename)
        return str(cached)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    destination = tmp_path / "checkout"
    downloader = script("download_release")
    assert downloader.main([
        "data", "--benchmark", "rhj", "--root", str(destination),
    ]) == 0
    plan = downloader.download_plan("data", "rhj")
    assert downloaded == [source for source, _ in plan]
    assert len(downloaded) == 4
    for _, target in plan:
        assert (destination / target).read_bytes() == cached.read_bytes()


def test_checkpoint_choices_are_distinct_and_select_only_one_seed():
    downloader = script("download_release")
    for benchmark, expected in (
        ("rhj", "checkpoints/occam_seeds/rhj_fewshot/all_sources/seed_44/best_model.pth"),
        ("norman", "checkpoints/vcell/norman_combo_NF32ALLA/seed_44/best_model.pth"),
    ):
        assert downloader.download_plan("virtual-cell", benchmark, 44) == [
            (expected, f"checkpoints/vcell/{benchmark}/seed_44/best_model.pth"),
        ]
    assert len(downloader.download_plan("encoder-decoder")) == 1
    assert not any("12k" in source for source, _ in downloader.download_plan("data", "representation"))
    with pytest.raises(ValueError, match="requires --benchmark"):
        downloader.download_plan("virtual-cell")


def test_download_preserves_existing_different_files(tmp_path):
    downloader = script("download_release")
    cached, target = tmp_path / "cached", tmp_path / "target"
    cached.write_bytes(b"new")
    target.write_bytes(b"old")
    with pytest.raises(FileExistsError):
        downloader.install_file(cached, target)
    assert target.read_bytes() == b"old"
    target.write_bytes(b"new")
    downloader.install_file(cached, target)
    assert target.read_bytes() == b"new"


def test_representation_plan_needs_no_local_paths_or_benchmarks(tmp_path):
    training = script("train_representation")
    output = tmp_path / "new-run"
    argv = ["--data", str(tmp_path / "cells.h5ad"), "--output", str(output)]
    assert training.main(argv + ["--dry-run"]) == 0
    assert not output.exists()
    cfg = training.make_config(training.parser().parse_args(argv))
    assert cfg.raw["data"]["train_gene_col"] == "index"
    assert cfg.raw["run"]["wandb"]["enabled"] is False
    decoder, _ = flatten_stage("decoder", cfg.raw["stages"]["decoder"])
    assert decoder["loss_type"] == "nb"
    assert decoder["decoder_output_activation"] == "softplus"
    assert decoder["freeze_encoder"] is True
    assert decoder["eval_data"] == []
    assert decoder["bmdb_eval_freq"] == 0


def test_custom_validation_and_gene_column_reach_all_stages(tmp_path):
    training = script("train_representation")
    args = training.parser().parse_args([
        "--data", str(tmp_path / "train.h5ad"), "--validation", str(tmp_path / "val.h5ad"),
        "--output", str(tmp_path / "run"), "--gene-col", "ensembl", "--epochs", "3", "4", "5",
    ])
    cfg = training.make_config(args)
    assert cfg.raw["data"]["val"] == str(tmp_path / "val.h5ad")
    assert cfg.raw["data"]["val_gene_col"] == "ensembl"
    stages = cfg.raw["stages"]
    assert [stages[name]["optim"]["epochs"] for name in ("pretrain", "finetune", "decoder")] == [3, 4, 5]
    assert stages["decoder"]["data"]["train_gene_col"] == "ensembl"


def test_explicit_var_index_is_not_replaced_by_gene_symbols(tmp_path):
    path = tmp_path / "genes.h5ad"
    ad.AnnData(
        np.ones((2, 3), dtype=np.float32),
        var=pd.DataFrame({"gene_symbol": ["A", "B", "C"]}, index=["ENSG1", "ENSG2", "ENSG3"]),
    ).write_h5ad(path)
    with h5py.File(path) as handle:
        assert _read_gene_names_h5py(handle, "index") == ["ENSG1", "ENSG2", "ENSG3"]
        assert _read_gene_names_h5py(handle, "gene_symbol") == ["A", "B", "C"]


def test_decoder_reserves_disjoint_reproducible_validation_cells():
    trainer = object.__new__(DecoderTrainer)
    trainer.config = SimpleNamespace(val_split=0.1, batch_size=16, data_mode="backed")
    dataset = torch.arange(100)
    train, val = trainer._split_cell_loaders(dataset)
    assert len(train.dataset) == 90 and len(val.dataset) == 10
    assert set(train.dataset.indices).isdisjoint(val.dataset.indices)
    assert set(train.dataset.indices) | set(val.dataset.indices) == set(range(100))
    assert train.num_workers == val.num_workers == 0
    torch.manual_seed(999)
    _, second_val = trainer._split_cell_loaders(dataset)
    assert val.dataset.indices == second_val.dataset.indices
    with pytest.raises(ValueError, match="Not enough cells"):
        trainer._split_cell_loaders(torch.arange(8))


def test_virtual_cell_template_exposes_custom_splits():
    raw = yaml.safe_load((ROOT / "configs/virtual_cell.yaml").read_text())
    raw["stages"]["vcell"]["split"].update(json="custom.json", holdout_cell_line="H1", seed=9)
    flat, _ = flatten_stage("vcell", raw["stages"]["vcell"])
    assert flat["split_json"] == "custom.json"
    assert flat["holdout_cell_line"] == "H1"
    assert flat["split_seed"] == 9


def test_scib_cli_passes_selected_datasets_to_scorer(tmp_path, monkeypatch):
    from occam import paths
    from occam.cli.scib import main
    from occam.inference import loader

    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.touch()
    data = tmp_path / "pancreas.h5ad"
    data.touch()
    chosen = [("pancreas_scib", str(data), "celltype", "batch")]

    def select(names):
        assert names == ["pancreas_scib"]
        return chosen

    calls = []
    fake_metrics = ModuleType("occam.evaluation.scib_metrics")
    fake_metrics.run_scib_full = lambda model, datasets: calls.append(datasets) or {}
    monkeypatch.setitem(sys.modules, fake_metrics.__name__, fake_metrics)
    monkeypatch.setattr(paths, "scib_datasets", select)
    monkeypatch.setattr(loader, "load_encoder_wrapper", lambda *a, **kw: SimpleNamespace(config=SimpleNamespace()))
    assert main(["--checkpoint", str(checkpoint), "--datasets", "pancreas_scib"]) == 0
    assert calls == [chosen]
