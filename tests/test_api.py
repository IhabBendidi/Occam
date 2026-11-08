"""The public API: gene alignment, preprocessing, and the three loaders.

WHY THIS FILE EXISTS
    `occam.api` is the surface other people build on, so its contract has to be pinned
    rather than inspected once. The two things it exists to prevent are both SILENT:
    a matrix aligned on the wrong axis (a wrong `gene_col` gives a near-empty input that
    predicts happily), and a matrix preprocessed differently from training (log1p data
    into a raw-count encoder). Both are checked here on synthetic data, so the tests run
    in a second and do not need the 2.2 GB checkpoints.

    The loaders themselves are covered by their error paths -- what they REFUSE is the
    part that protects a user. Loading a real checkpoint is exercised by
    scripts/repro/occam_predict.py and by the panel smoke test.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from occam.api import (  # noqa: E402
    DEFAULT_MAX_ZERO_FILL, GeneSpace, load_decoder, load_encoder, load_virtual_cell,
    to_counts,
)

VOCAB = ["ENSG1", "ENSG2", "ENSG3", "ENSG4"]


class _Ad:
    """The two attributes `prepare` reads off an AnnData, without the dependency."""

    def __init__(self, X, genes, col="gene_id"):
        import pandas as pd
        self.X = np.asarray(X, dtype=np.float32)
        self.var = pd.DataFrame({col: list(genes)}, index=[str(g) for g in genes])
        self.var_names = [str(g) for g in genes]


def test_alignment_scatters_into_vocabulary_order_not_input_order():
    gs = GeneSpace(VOCAB, target_sum=10.0)
    # input lists the genes in a different order and omits ENSG3
    ad = _Ad([[1.0, 2.0, 7.0]], ["ENSG4", "ENSG1", "ENSG2"])
    X, lib, rep = gs.prepare(ad, gene_col="gene_id")
    assert X.shape == (1, 4)
    assert rep.n_matched == 3 and rep.n_zero_filled == 1
    assert X[0, 2] == 0.0, "the missing gene must be zero-filled"
    # library size is over the MATCHED columns, before the scatter
    assert lib[0, 0] == pytest.approx(10.0)
    # ENSG1 carried value 2.0 of a total 10.0 -> normalised to 2.0, then log1p
    assert X[0, 0] == pytest.approx(np.log1p(2.0), rel=1e-6)
    assert X[0, 3] == pytest.approx(np.log1p(1.0), rel=1e-6)


def test_library_size_ignores_genes_the_vocabulary_does_not_have():
    """Normalising after the scatter would divide by a sum over zero-filled columns."""
    gs = GeneSpace(VOCAB, target_sum=100.0)
    ad = _Ad([[5.0, 5.0, 90.0]], ["ENSG1", "ENSG2", "NOT_IN_VOCAB"])
    X, lib, rep = gs.prepare(ad, gene_col="gene_id")
    assert rep.n_input_dropped == 1
    assert lib[0, 0] == pytest.approx(10.0), "the dropped gene must not count"
    assert X[0, 0] == pytest.approx(np.log1p(50.0), rel=1e-6)


def test_prepare_refuses_an_input_read_on_the_wrong_axis():
    gs = GeneSpace([f"ENSG{i}" for i in range(200)])
    ad = _Ad(np.ones((2, 3)), ["SYMBOL_A", "SYMBOL_B", "SYMBOL_C"])
    with pytest.raises(ValueError) as e:
        gs.prepare(ad, gene_col="gene_id")
    assert "not one input gene" in str(e.value)


def test_prepare_refuses_an_implausible_zero_fill_fraction():
    gs = GeneSpace([f"ENSG{i}" for i in range(1000)])
    ad = _Ad(np.ones((2, 2)), ["ENSG0", "ENSG1"])
    with pytest.raises(ValueError) as e:
        gs.prepare(ad, gene_col="gene_id")
    msg = str(e.value)
    assert "zero-filled" in msg and "gene_col" in msg
    # ...but a legitimately narrow panel must still be allowed explicitly
    X, _, rep = gs.prepare(ad, gene_col="gene_id", max_zero_fill=0)
    assert X.shape == (2, 1000) and rep.n_matched == 2


def test_the_skip_flags_are_honoured():
    raw = np.array([[1.0, 3.0, 0.0, 0.0]], dtype=np.float32)
    assert GeneSpace(VOCAB, 8.0).preprocess(raw)[0, 1] == pytest.approx(np.log1p(6.0))
    assert GeneSpace(VOCAB, 8.0, skip_normalize=True).preprocess(raw)[0, 1] == \
        pytest.approx(np.log1p(3.0))
    assert GeneSpace(VOCAB, 8.0, skip_log1p=True).preprocess(raw)[0, 1] == \
        pytest.approx(6.0)
    out = GeneSpace(VOCAB, 8.0, skip_normalize=True, skip_log1p=True).preprocess(raw)
    assert out[0, 1] == pytest.approx(3.0)


def test_a_bare_array_must_already_be_in_the_checkpoint_space():
    gs = GeneSpace(VOCAB)
    X, lib, rep = gs.prepare(np.ones((3, 4), dtype=np.float32))
    assert X.shape == (3, 4) and rep.n_zero_filled == 0
    with pytest.raises(ValueError) as e:
        gs.prepare(np.ones((3, 2), dtype=np.float32))
    assert "already be in the checkpoint's space" in str(e.value)


def test_to_counts_restores_the_library_size():
    gs = GeneSpace(VOCAB, target_sum=1000.0)
    raw = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
    X, lib, _ = gs.prepare(raw)
    back = to_counts(X, lib)
    assert back.sum(axis=1)[0] == pytest.approx(100.0, rel=1e-4)
    assert np.allclose(back, raw, rtol=1e-3, atol=1e-3)


def test_a_duplicated_vocabulary_is_refused_at_construction():
    with pytest.raises(ValueError) as e:
        GeneSpace(["A", "B", "A"])
    assert "duplicates" in str(e.value)


def test_two_input_genes_mapping_to_one_column_is_refused():
    gs = GeneSpace(VOCAB)
    ad = _Ad(np.ones((1, 2)), ["ENSG1", "ENSG1"])
    with pytest.raises(ValueError) as e:
        gs.prepare(ad, gene_col="gene_id")
    assert "one vocabulary column" in str(e.value)


def test_from_checkpoint_reads_the_space_out_of_the_checkpoint():
    gs = GeneSpace.from_checkpoint({
        "gene_names": VOCAB, "target_sum": 823.0,
        "training_config": {"skip_normalize": True, "skip_log1p": False}})
    assert gs.target_sum == 823.0 and gs.skip_normalize and not gs.skip_log1p
    with pytest.raises(ValueError) as e:
        GeneSpace.from_checkpoint({"target_sum": 2000.0})
    assert "gene_names" in str(e.value)


def test_a_wrong_gene_col_is_named_rather_than_guessed():
    gs = GeneSpace(VOCAB)
    ad = _Ad(np.ones((1, 4)), VOCAB)
    with pytest.raises(ValueError) as e:
        gs.prepare(ad, gene_col="nope")
    assert "nope" in str(e.value) and "not a column" in str(e.value)


def test_the_loaders_refuse_a_missing_or_wrong_kind_of_checkpoint(tmp_path):
    import torch
    with pytest.raises(FileNotFoundError):
        load_encoder(str(tmp_path / "nothing.pth"), device="cpu")

    empty_dir = tmp_path / "emptydir"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError) as e:
        load_encoder(str(empty_dir), device="cpu")
    assert "no recognised checkpoint" in str(e.value)

    # a plain MAE checkpoint has no external decoder
    mae = tmp_path / "mae.pth"
    torch.save({"gene_names": VOCAB, "target_sum": 2000.0,
                "model_state_dict": {}}, mae)
    with pytest.raises(ValueError) as e:
        load_decoder(str(mae), device="cpu", with_encoder=False)
    assert "decoder_state_dict" in str(e.value)

    # and a non-vcell checkpoint is not a virtual cell
    with pytest.raises(ValueError) as e:
        load_virtual_cell(str(mae), device="cpu")
    assert "is_virtual_cell_checkpoint" in str(e.value)


def test_the_api_surface_is_what_the_readme_documents():
    """The README names these; renaming one silently breaks every downstream user."""
    import occam.api as api
    for name in ("GeneSpace", "AlignmentReport", "OccamEncoder", "OccamDecoder",
                 "OccamVirtualCell", "load_encoder", "load_decoder",
                 "load_virtual_cell", "to_counts"):
        assert name in api.__all__, f"{name} is not exported"
        assert hasattr(api, name), f"{name} is not defined"
    assert 0.0 < DEFAULT_MAX_ZERO_FILL <= 1.0


def test_the_readme_and_api_doc_document_the_real_surface():
    """A README that names a function that does not exist is worse than no README."""
    import re

    import occam.api as api

    readme = open(os.path.join(REPO, "README.md")).read()
    apidoc = open(os.path.join(REPO, "docs/api.md")).read()
    for name in ("load_encoder", "load_decoder", "load_virtual_cell"):
        assert name in readme, f"README does not document {name}"
        assert name in apidoc, f"docs/api.md does not document {name}"
    # Every `enc.x(` / `dec.x(` / `vc.x(` call the README shows must exist.
    shown = set(re.findall(r"\b(?:enc|dec|vc)\.([a-z_]+)\(", readme))
    for meth in shown:
        assert any(hasattr(cls, meth) for cls in
                   (api.OccamEncoder, api.OccamDecoder, api.OccamVirtualCell)), \
            f"README calls .{meth}() but no API object has it"
    # And the attributes it shows without parentheses.
    for attr in ("perturbations", "compose_modes"):
        assert hasattr(api.OccamVirtualCell, attr), attr
    assert "occam-embed" in readme and "occam_predict.py" in readme


def test_the_readme_is_a_step_by_step_guide():
    readme = open(os.path.join(REPO, "README.md")).read()
    for step in ("## 1. Install", "## 2. Get the weights and the data",
                 "## 3. Use it: the Python API",
                 "## 4. Rebuild the benchmarks", "## 5. Train", "## 6. Evaluate",
                 "## 7. The results"):
        assert step in readme, f"missing section: {step}"
    # Current official five-seed results and sum-of-singles reference.
    assert "`ctrlmean`" in readme and "`ctrlcell`" in readme
    assert "official_comparisons/official_tables.md" in readme
    for anchor in ("0.3999", "0.8955", "0.8013", "0.8822", "0.7749"):
        assert anchor in readme, f"README quotes no anchor {anchor}"
    # Installing the scoring environment is a separate step people miss.
    assert "cell-eval==0.6.0" in readme
