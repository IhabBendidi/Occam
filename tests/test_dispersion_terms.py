"""The F-disp1 dispersion terms: reachable, gated, aligned, and bias-immune.

WHY THIS FILE EXISTS
    `delta_cos_weight` once ran INERT for three arms at three different weights and produced
    bit-identical curves (F-loss1), because a loss consumer was added without the gate that
    loads its inputs, and because a weight can be declared in the dataclass and still be
    unsettable from a launcher. These terms are the same shape of change, so the same four
    failure modes are pinned here:

      1. THREE-LAYER REACHABILITY -- dataclass + YAML + loader schema. A key missing from any
         one of them is silently unsettable and the run proceeds at the default.
      2. THE GATE -- every weight that reads the group-mean/dispersion tables must appear in
         the gate that loads them, or the term is silently zero.
      3. ALIGNMENT -- the dispersion table is placed through the group-mean INDEX, never by
         glob order, and a table that shares no key must RAISE rather than train on zeros.
      4. BIAS IMMUNITY -- the moment-matching terms measure spread around the model's own
         mean, so a constant offset must NOT satisfy them. This is the property that makes
         them dispersion terms rather than another mean-error term.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

WEIGHTS = ("dcos_conc_weight", "radial_spread_weight", "gene_var_weight")
ALL_KEYS = WEIGHTS + ("disp_ema_momentum",)


def test_every_new_key_is_reachable_through_all_three_layers():
    from occam.config.schema import VCELL_SCHEMA
    from occam.vcell.config import VirtualCellConfig

    # the repo's own loader, not yaml.safe_load: the config uses a custom `!include` tag
    from occam.config.loader import load_config
    raw = load_config(os.path.join(REPO, "configs", "rhj_fewshot.yaml")).raw
    loss_yaml = raw["stages"]["vcell"]["loss"]
    # the schema nests the loss block under "loss"
    loss_schema = VCELL_SCHEMA["loss"]
    for k in ALL_KEYS:
        assert hasattr(VirtualCellConfig, k) or k in VirtualCellConfig.__dataclass_fields__, \
            f"{k} missing from the dataclass"
        assert k in loss_yaml, f"{k} missing from configs/rhj_fewshot.yaml -- --set cannot reach it"
        assert k in loss_schema, f"{k} missing from VCELL_SCHEMA['loss'] -- --set cannot reach it"


def test_the_gate_loads_the_tables_for_every_dispersion_weight():
    """Every weight that READS the tables must appear in the gate that LOADS them."""
    src = open(os.path.join(REPO, "src", "occam", "vcell", "pipeline.py")).read()
    gate = src[src.index("_needs_group_means = ("):src.index("_snr_min = getattr(config")]
    for k in WEIGHTS:
        assert k in gate, (
            f"{k} is consumed in the loss but absent from the table-loading gate; it would be "
            f"SILENTLY ZERO (the F-loss1 failure)")
    assert "_needs_disp" in gate and "load_group_dispersion" in gate
    # and the refusal must be a refusal, not a warning
    assert "Refusing to train" in gate


def _write_disp(path, keys, n_genes=4, n_cells=10):
    rng = np.random.default_rng(0)
    G = len(keys)
    np.savez(path, keys=np.asarray(keys),
             gvar=rng.random((G, n_genes)).astype(np.float32),
             rad_mean=rng.random(G).astype(np.float32),
             rad_std=rng.random(G).astype(np.float32),
             cos_mean=rng.random(G).astype(np.float32),
             cos_std=rng.random(G).astype(np.float32),
             n_cells=np.full(G, n_cells, dtype=np.int64),
             gene_mask=np.ones(n_genes, dtype=bool))


def test_dispersion_table_is_placed_through_the_index_not_by_order(tmp_path):
    from occam.vcell.stream_v2 import load_group_dispersion
    # the means index deliberately DISAGREES with the file's key order
    index = {("src", "L1", "A"): 2, ("src", "L1", "B"): 0, ("src", "L1", "C"): 1}
    _write_disp(tmp_path / "src.npz", ["L1|A", "L1|B", "L1|C"])
    gvar, rad_m, rad_s, cos_m, cos_s, nc = load_group_dispersion(
        str(tmp_path), index, 3, torch.device("cpu"))
    z = np.load(tmp_path / "src.npz", allow_pickle=True)
    for i, k in enumerate(["L1|A", "L1|B", "L1|C"]):
        row = index[("src", k.split("|")[0], k.split("|")[1])]
        assert float(rad_m[row]) == pytest.approx(float(z["rad_mean"][i]), rel=1e-6), \
            "rows were placed positionally, not through the index"
    assert int(nc.min()) == 10


def test_a_table_that_shares_no_key_raises(tmp_path):
    from occam.vcell.stream_v2 import load_group_dispersion
    _write_disp(tmp_path / "src.npz", ["L1|A", "L1|B"])
    # the repo's minimal test runner has no `match=` on pytest.raises, so assert the message
    # explicitly -- and the message IS the contract here: it has to say which two tables
    # disagree, or the next person re-derives it.
    try:
        load_group_dispersion(str(tmp_path), {("src", "L1", "Z"): 0}, 1, torch.device("cpu"))
    except RuntimeError as e:
        assert "share NO group key" in str(e), str(e)
    else:
        raise AssertionError("a disjoint dispersion table must RAISE, not train on zeros")


def test_groups_with_one_cell_are_nan_and_gateable(tmp_path):
    from occam.vcell.stream_v2 import load_group_dispersion
    p = tmp_path / "src.npz"
    _write_disp(p, ["L1|A"], n_cells=1)
    z = dict(np.load(p, allow_pickle=True))
    z["n_cells"] = np.asarray([1], dtype=np.int64)
    np.savez(p, **z)
    *_, nc = load_group_dispersion(str(tmp_path), {("src", "L1", "A"): 0}, 1, torch.device("cpu"))
    assert int(nc[0]) == 1, "the loss gates on n_cells >= 2; a 1-cell group must stay visible as 1"


# ---------------------------------------------------------------------------------------
# The algebra of the two moment-matching terms, replicated exactly as pipeline.py computes
# it. These are the properties the terms are FOR.
# ---------------------------------------------------------------------------------------
def _var_term(pred, centre, real_std, mask):
    u = ((pred - centre) / real_std) ** 2
    return ((u * mask).sum() / mask.sum().clamp(min=1.0) - 1.0) ** 2


def test_collapsed_prediction_gives_a_term_of_one_and_matched_gives_zero():
    torch.manual_seed(0)
    n, g = 512, 32
    real_std = torch.full((n, g), 0.1)
    mask = torch.ones(n, g)
    collapsed = torch.zeros(n, g)                      # no dispersion at all
    centre = torch.zeros(n, g)
    assert float(_var_term(collapsed, centre, real_std, mask)) == pytest.approx(1.0, abs=1e-6)
    matched = torch.randn(n, g) * 0.1                  # exactly the real dispersion
    assert float(_var_term(matched, centre, real_std, mask)) < 0.02


def test_a_constant_bias_does_not_satisfy_the_variance_term():
    """The centre is the model's OWN mean, so an offset cancels: bias cannot fake dispersion.

    This is why pipeline.py tracks an EMA of the model's per-group mean instead of using the
    real mean. With the real mean as centre, a prediction that is merely SHIFTED by one real
    std would score as perfectly dispersed while having no dispersion whatsoever.
    """
    n, g = 512, 32
    real_std = torch.full((n, g), 0.1)
    mask = torch.ones(n, g)
    shifted = torch.full((n, g), 0.1)                  # constant, offset by exactly 1 real std
    # WRONG centre (the real mean): the bias alone makes the term look perfect
    assert float(_var_term(shifted, torch.zeros(n, g), real_std, mask)) == pytest.approx(0.0, abs=1e-6)
    # RIGHT centre (the model's own mean): the term correctly reports total collapse
    own_mean = shifted.mean(dim=0, keepdim=True).expand_as(shifted)
    assert float(_var_term(shifted, own_mean, real_std, mask)) == pytest.approx(1.0, abs=1e-6)


def test_the_radial_term_is_scale_free_in_the_real_std():
    """(m2 - 1)^2 with m2 normalised by the REAL std: the same relative under-dispersion
    scores the same whatever the absolute scale, so one weight fits every group."""
    torch.manual_seed(1)
    r = torch.randn(1024) * 0.3
    for real_std in (0.3, 3.0, 30.0):
        u = ((r * (real_std / 0.3) * 0.5 - 0.0) / real_std) ** 2   # half the real spread
        assert float(((u.mean() - 1.0) ** 2)) == pytest.approx(0.5625, rel=0.15)
