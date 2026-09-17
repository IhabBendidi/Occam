"""The adjacency-map sweep TSV: the declared product, the excluded cell, and the fixtures.

Two failures this guards, both silent. A confounded cell slipping back in would produce an
arm that measures neither of its two axes while still printing a number. And an arm whose
launcher does not actually set what its row declares is a row that lies.
"""

from __future__ import annotations

import itertools
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TSV = os.path.join(REPO, "configs", "rhj_adjmap_sweep.tsv")
SBATCH = os.path.join(REPO, "scripts", "slurm", "rhj_adjmap_sweep.sbatch")

LRS = ("3e-4", "1e-3", "3e-3")
DLRS = ("1e-4", "3e-4", "1e-3")
COSINES = ("0.0", "0.25", "0.5")
DPOS = ("0", "1")
MODES = ("full", "multiplicative_only", "additive_only")
FHIDS = ("0", "256")
FMFS = ("0", "1")

pytestmark = pytest.mark.skipif(not os.path.isfile(TSV), reason="sweep not built")


def _rows():
    rows, cols = [], None
    for line in open(TSV):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if cols is None:
            cols = parts
            continue
        rows.append(dict(zip(cols, parts)))
    return rows


def test_the_sweep_is_the_declared_product_minus_the_confounded_cell():
    rows = _rows()
    want = {(lr, d, c, p, m, f, mf)
            for lr, d, c, p, m, f, mf in itertools.product(
                LRS, DLRS, COSINES, DPOS, MODES, FHIDS, FMFS)
            if not (p == "1" and mf == "1")}
    got = {(r["lr"], r["decoder_lr"], r["cosine_loss_weight"], r["data_perts_only"],
            r["film_mode"], r["film_hidden_dim"], r["film_mean_free"]) for r in rows}
    assert got == want, f"missing {sorted(want - got)[:3]}, extra {sorted(got - want)[:3]}"
    assert len(rows) == 486, len(rows)
    assert len(rows) == len(got), "duplicate rows"


def test_the_confounded_cell_is_absent_and_the_reason_is_in_the_file():
    """⭐ data_perts_only=1 x film_mean_free=1 measures neither axis.

    film_mean_free subtracts interaction(z, ebar) with ebar averaged over the WHOLE graph
    node table; data_perts_only restricts that table. Together they silently change the
    reference being subtracted.
    """
    bad = [r for r in _rows()
           if r["data_perts_only"] == "1" and r["film_mean_free"] == "1"]
    assert not bad, f"{len(bad)} confounded arm(s) present, e.g. {bad[:1]}"
    head = "".join(l for l in open(TSV) if l.startswith("#"))
    assert "confounded" in head
    assert "data_perts_only" in head and "film_mean_free" in head
    # both axes are still fully measured via the other three combinations
    pairs = {(r["data_perts_only"], r["film_mean_free"]) for r in _rows()}
    assert pairs == {("0", "0"), ("0", "1"), ("1", "0")}


def test_batch_size_is_fixed_not_swept_and_the_file_says_why():
    """It changes the effective learning rate, so crossing it with lr reads neither."""
    rows = _rows()
    assert "batch_size" not in rows[0], "batch_size must not be a sweep column"
    head = "".join(l for l in open(TSV) if l.startswith("#"))
    assert "batch_size=2048" in head.replace(" ", "")
    assert "effective learning rate" in head
    body = open(SBATCH).read()
    assert "stages.vcell.optim.batch_size=2048" in body


def test_every_declared_axis_is_actually_set_by_the_launcher():
    """A row that the launcher does not honour is a row that lies."""
    body = open(SBATCH).read()
    for key in ("optim.lr=${LR}", "optim.decoder_lr=${DLR}",
                "loss.cosine_loss_weight=${COS}",
                "graph.data_perts_only=${DPO_BOOL}",
                "interaction.film_mode=${FMODE}",
                "interaction.film_hidden_dim=${FHID}",
                "interaction.film_mean_free=${FMF_BOOL}"):
        assert f"stages.vcell.{key}" in body, key
    # the fixtures too
    assert "stages.vcell.graph.pert_model_type=adjacency_map" in body
    assert "stages.vcell.loss.loss_type=smooth_l1" in body
    # and the booleans are converted, not passed as 0/1
    assert 'DPO_BOOL=$([ "$DPO" = "1" ] && echo true || echo false)' in body
    assert 'FMF_BOOL=$([ "$FMF" = "1" ] && echo true || echo false)' in body


def test_arm_ids_are_ordered_and_run_dirs_fit_a_filesystem_component():
    rows = _rows()
    assert [r["arm_id"] for r in rows] == [f"s{i:03d}" for i in range(len(rows))]
    tags = [r["tag"] for r in rows]
    assert len(set(tags)) == len(tags), "duplicate tag -- two arms would share a run dir"
    for r in rows:
        name = f"rhj_adjmap_{r['arm_id']}__{r['tag']}"
        assert len(name) <= 120, f"{name} is {len(name)} chars"


def test_the_launcher_tells_the_reader_to_check_the_f7_oscillation_first():
    """No arm ordering is interpretable while the metric swings +/-0.20 between epochs."""
    body = open(SBATCH).read()
    assert "F7" in body
    assert "pearson_delta: mean=" in body, "the reader needs the exact grep"
    assert "BatchNorm1d" in body and "learning rate" in body, "both candidates named"
