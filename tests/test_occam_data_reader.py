"""`competitors/shared/occam_data.py` -- the one reader every competitor imports.

⭐ THE GATE (`test_..._is_bit_identical_to_the_panel`): at `target_sum=2000.0` this reader
must reproduce `splithalf/panel.npz` **bit for bit**. That is what proves competitors see
exactly the data Occam and the five baselines were measured on, rather than something
approximately like it. Measured: max |diff| 0.0 over 42,402 x 6,983 float32.

It is a real constraint, not a formality: the equivalence depends on reproducing
`build_splithalf_panel.py::_load_axis_restricted` DTYPE FOR DTYPE -- float32 values, a
float64 row-sum accumulation over the FULL gene panel, cast back to float32 before the
divide. Any change to that sequence breaks the gate while still "working".
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "competitors", "shared"))
sys.path.insert(0, os.path.join(REPO, "src"))

BENCH = os.path.join(REPO, "data", "benchmarks", "rhj_fewshot")
PANEL = os.path.join(BENCH, "splithalf", "panel.npz")
HEPG2 = os.path.join(REPO, "data", "eval", "pert", "nadig_2024_hepg2.h5ad")

_needs_data = pytest.mark.skipif(
    not (os.path.isfile(PANEL) and os.path.isfile(HEPG2)),
    reason="eval bundle / panel not populated")


def _od():
    import occam_data
    return occam_data


# Loading all three lines is 656,343 x 6,983 float32 = 18.3 GB and ~180 s. Three tests need
# it, so it is a module-scoped fixture rather than three loads -- that is ~6 min off every
# run of the suite.
@pytest.fixture(scope="module")
def full_adata():
    if not (os.path.isfile(PANEL) and os.path.isfile(HEPG2)):
        pytest.skip("eval bundle / panel not populated")
    od = _od()
    b = od.load_occam_benchmark(BENCH, verify_sources=False)
    return b, od.load_occam_adata(b, target_sum=2000.0)


# ---------------------------------------------------------------------------
# the artifact reader
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.isdir(BENCH), reason="benchmark not built")
def test_the_benchmark_reads_and_reports_what_the_artifact_declares():
    od = _od()
    b = od.load_occam_benchmark(BENCH, verify_sources=False)
    assert b.n_genes == 6983
    assert len(b.axis_ensembl) == len(b.axis_symbols) == 6983
    assert b.axis_ensembl[0] == "ENSG00000000419" and b.axis_symbols[0] == "DPM1"
    assert len(b.val_perts) == 103
    assert len(b.test_perts) == 621
    # the axis is sorted by Ensembl id, and that order IS the scored gene order
    assert b.axis_ensembl == sorted(b.axis_ensembl)
    # symbols must be unique: discrimination_score_l1 excludes the target gene BY SYMBOL
    assert len(set(b.axis_symbols)) == 6983
    # only HepG2 excludes anything -- Jurkat and RPE1 train on everything
    assert set(b.excluded_by_file) == {"nadig_2024_hepg2.h5ad"}
    assert len(b.excluded_by_file["nadig_2024_hepg2.h5ad"]) == 724
    assert sorted(b.files) == ["HEPG2", "RPE1", "jurkat"]


@pytest.mark.skipif(not os.path.isdir(BENCH), reason="benchmark not built")
def test_an_edited_artifact_is_refused(tmp_path):
    """The artifact is frozen. Editing it must raise, not warn."""
    import shutil
    od = _od()
    dst = os.path.join(str(tmp_path), "rhj_fewshot")
    shutil.copytree(BENCH, dst, ignore=shutil.ignore_patterns("splithalf"))
    with open(os.path.join(dst, "splits.json")) as fh:
        sp = json.load(fh)
    sp["val"] = sp["val"][:-1]                       # drop one perturbation
    with open(os.path.join(dst, "splits.json"), "w") as fh:
        json.dump(sp, fh)
    with pytest.raises(od.OccamDataError) as exc:
        od.load_occam_benchmark(dst, verify_sources=False)
    assert "edited" in str(exc.value)


# ---------------------------------------------------------------------------
# ⭐ the gate
# ---------------------------------------------------------------------------

@_needs_data
def test_the_reader_at_target_sum_2000_is_bit_identical_to_the_panel():
    """⭐ Competitors must see EXACTLY the data the baselines were measured on."""
    od = _od()
    b = od.load_occam_benchmark(BENCH, verify_sources=False)
    a = od.load_occam_adata(b, cell_lines=("HEPG2",), target_sum=2000.0, log1p=True)

    z = np.load(PANEL, allow_pickle=True)
    mine = a.X[z["source_rows"]]
    assert mine.shape == z["X"].shape
    assert np.array_equal(mine, z["X"]), (
        f"max |diff| {np.abs(mine - z['X']).max():.3e}; the dtype sequence in _read_one no "
        f"longer matches build_splithalf_panel.py::_load_axis_restricted")

    # and the labels line up, modulo our control-label translation
    got = np.asarray(a.obs["target_gene"])[z["source_rows"]]
    want = np.where(z["labels"] == od.CONTROL_LABEL, "empty", z["labels"])
    assert np.array_equal(got, want)


@_needs_data
def test_the_10k_to_2k_map_lands_within_one_float32_ulp_of_the_panel():
    """The shipping path: competitors get 10k, we score at 2k.

    Not bit-exact here, and that is expected: the two matrices are computed independently
    from counts, so each carries its own float32 rounding. One ULP at the max value 5.0 is
    4.7683716e-07 -- seven orders of magnitude below the +/-10% library-size tolerance.
    """
    od = _od()
    from emission_space import TRAIN_TARGET_SUM, assert_same_space, to_eval_space

    b = od.load_occam_benchmark(BENCH, verify_sources=False)
    a = od.load_occam_adata(b, cell_lines=("HEPG2",), target_sum=TRAIN_TARGET_SUM,
                            log1p=True)
    z = np.load(PANEL, allow_pickle=True)
    back = to_eval_space(a.X[z["source_rows"]], from_target_sum=TRAIN_TARGET_SUM,
                         from_log1p=True)

    diff = np.abs(back - z["X"]).max()
    assert diff <= np.spacing(np.float32(6.0)), f"max |diff| {diff:.3e} exceeds one ULP"
    assert assert_same_space(
        back, z["X"], pred_var=list(a.var.index),
        target_var=[str(s) for s in z["axis_symbols"]])["verdict"] == "same space"


# ---------------------------------------------------------------------------
# the obs/var contract each arm relies on
# ---------------------------------------------------------------------------

@_needs_data
def test_the_emitted_schema_is_what_their_loaders_expect():
    od = _od()
    b = od.load_occam_benchmark(BENCH, verify_sources=False)
    a = od.load_occam_adata(b, cell_lines=("HEPG2",), max_cells_per_line=3000)

    for col in ("gene_pert", "pert", "target_gene", "gene", "negative_control",
                "is_control", "cell_type", "cell_line", "batch_center",
                "dataset_batch_num", "dataset_name"):
        assert col in a.obs.columns, col
    assert a.obs["negative_control"].dtype == bool
    assert (a.obs["cell_type"] == "HEPG2").all()          # exact casing
    # var index is SYMBOLS, and both ids are carried
    assert list(a.var.index[:3]) == ["DPM1", "GCLC", "NIPAL3"]
    assert list(a.var["ensembl_gene_id"][:1]) == ["ENSG00000000419"]
    # controls are renamed for their stack, and consistently
    ctrl = a.obs["is_control"].to_numpy()
    assert set(a.obs["gene_pert"].to_numpy()[ctrl]) == {"empty"}
    assert od.CONTROL_LABEL not in set(a.obs["gene_pert"])
    # the space is recorded on the object, so a downstream arm can assert it
    assert a.uns["occam"]["target_sum"] == od.DEFAULT_TARGET_SUM
    assert a.uns["occam"]["gene_axis_sha256"] == b.axis_sha256


@_needs_data
def test_the_control_label_is_switchable():
    od = _od()
    b = od.load_occam_benchmark(BENCH, verify_sources=False)
    a = od.load_occam_adata(b, cell_lines=("HEPG2",), max_cells_per_line=2000,
                            control_label=od.CONTROL_LABEL)
    ctrl = a.obs["is_control"].to_numpy()
    assert set(a.obs["target_gene"].to_numpy()[ctrl]) == {od.CONTROL_LABEL}


# ---------------------------------------------------------------------------
# the splits -- the one error that would invalidate everything silently
# ---------------------------------------------------------------------------

def test_the_splits_reproduce_our_artifact_exactly(full_adata):
    od = _od()
    b, a = full_adata
    idx = od.occam_split_indices(b, a)

    # every row in exactly one split
    seen = np.concatenate([idx["finetune"], idx["validation"], idx["test"]])
    assert len(np.unique(seen)) == len(seen) == a.n_obs

    perts = np.asarray(a.obs["target_gene"], dtype=str)
    assert set(perts[idx["validation"]]) == b.val_perts
    assert set(perts[idx["test"]]) == b.test_perts

    # all controls in finetune; none in val/test
    ctrl = np.asarray(a.obs["is_control"], dtype=bool)
    assert ctrl[idx["finetune"]].sum() == ctrl.sum() == 28474
    assert not ctrl[idx["validation"]].any() and not ctrl[idx["test"]].any()

    # the withheld-cell count matches exclusions.json
    with open(os.path.join(BENCH, "exclusions.json")) as fh:
        ex = json.load(fh)
    assert (len(idx["validation"]) + len(idx["test"])
            == ex["counts"]["nadig_2024_hepg2.h5ad"]["cells"] == 44221)

    # ⭐ the positive half: Jurkat and RPE1 are WHOLLY training
    line = np.asarray(a.obs["cell_line"], dtype=str)
    for other in ("jurkat", "RPE1"):
        rows = np.flatnonzero(line == other)
        assert rows.size and np.isin(rows, idx["finetune"]).all()

    # and val/test are HepG2 only
    assert (line[idx["validation"]] == "HEPG2").all()
    assert (line[idx["test"]] == "HEPG2").all()


def test_the_name_level_split_keeps_held_out_symbols_in_train(full_adata):
    """TxPert splits by NAME. The 724 held-out symbols legitimately train elsewhere."""
    od = _od()
    b, a = full_adata
    names = od.occam_split_perturbations(b, a)

    assert set(names["validation"]) == b.val_perts
    assert set(names["test"]) == b.test_perts
    # deliberate: they train from Jurkat and RPE1
    assert b.val_perts <= set(names["train"])
    assert b.test_perts <= set(names["train"])
    assert "empty" not in names["train"]


@_needs_data
def test_an_unknown_cell_line_is_refused():
    od = _od()
    b = od.load_occam_benchmark(BENCH, verify_sources=False)
    with pytest.raises(od.OccamDataError) as exc:
        od.load_occam_adata(b, cell_lines=("K562",))
    assert "K562" in str(exc.value)
