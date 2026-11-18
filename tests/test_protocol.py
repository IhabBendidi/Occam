"""The split-half protocol: the reference arms, the B-control substitution, and the probe.

Three things here are load-bearing and all three would fail silently:

* the `extra` arms must never be able to satisfy the zero-signal anchor requirement -- an
  `experimental` arm named `ctrlmean` would hand the report a "floor" full of real signal;
* the PREDICTION file must carry the **B** controls while the target keeps the **A** ones.
  That single asymmetry is the entire protocol; if it inverts, every arm is scored against
  a reference that shares its own control noise and `pearson_delta` is structurally inflated;
* `ctrlmean`'s `pearson_delta` is NaN here by construction, and NaN drops out of a mean
  without complaint.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

CONTROL = "non-targeting"
N_PERT, PER_PERT, N_CTRL, N_GENES = 6, 11, 40, 7


def _corpus(seed=0):
    rng = np.random.default_rng(seed)
    labels = [f"P{i:02d}" for i in range(N_PERT) for _ in range(PER_PERT)]
    labels += [f"TRAIN{i}" for i in range(3) for _ in range(5)]
    labels += [CONTROL] * N_CTRL
    labels = np.asarray(labels, dtype=str)
    batches = np.asarray([f"b{i % 4}" for i in range(len(labels))], dtype=str)
    X = rng.normal(5, 1, size=(len(labels), N_GENES)).astype("float32")
    return X, labels, batches


def _base_panel(X, labels):
    from occam.vcell.panel.panel_io import Panel
    return Panel(real_X_model=X, real_labels=labels, real_counts=None,
                 ctrl_X_model=X[:1], ctrl_counts=None, ctrl_batches=None,
                 enc_cols=np.arange(X.shape[1]), panel_cols=np.arange(X.shape[1]),
                 panel_genes=[f"SYM{i}" for i in range(X.shape[1])], n_enc=X.shape[1])


# ---------------------------------------------------------------------------
# arms.py: the reference arms
# ---------------------------------------------------------------------------

def _arm_inputs():
    from occam.vcell.panel.panel_io import Panel
    rng = np.random.default_rng(1)
    pred = rng.normal(1, 1, size=(12, 5)).astype("float32")
    labels = np.asarray([f"P{i//2:02d}" for i in range(12)], dtype=str)
    ctrl_used = rng.normal(1, 1, size=(12, 5)).astype("float32")
    panel = Panel(real_X_model=pred, real_labels=labels, real_counts=None,
                  ctrl_X_model=rng.normal(1, 1, (9, 5)).astype("float32"),
                  ctrl_counts=None, ctrl_batches=None, enc_cols=np.arange(5),
                  panel_cols=np.arange(5), panel_genes=[f"S{i}" for i in range(5)], n_enc=5)
    return pred, labels, ctrl_used, panel


def test_reference_arms_are_emitted_around_the_model_arms_in_a_fixed_order():
    """Anchors, then `experimental`, then the model, then `meanpert`.

    The reference arm comes BEFORE the model so a job killed at the wall clock still has
    both the floor and the ceiling and the paired gap is computable from a partial run.
    """
    from occam.vcell.panel.arms import build_arms, split_arms

    pred, labels, ctrl_used, panel = _arm_inputs()
    extra = {"experimental": pred * 2, "meanpert": pred * 3}
    arms = build_arms(pred, labels, ctrl_used, panel, calibrate="none", extra=extra)
    tags = [t for t, _, _ in arms]
    assert tags[:2] == ["ctrlmean", "ctrlcell"]
    assert tags[2] == "experimental"
    assert tags[-1] == "meanpert"
    assert "model_nocal" in tags

    anchor, ref, model = split_arms(arms)
    assert anchor == ["ctrlmean", "ctrlcell"]
    assert ref == ["experimental", "meanpert"]
    assert model == ["model_nocal"]
    # the two-way helper keeps working and must NOT call a reference arm a model arm
    from occam.vcell.panel.arms import split_anchor_and_model
    assert split_anchor_and_model(arms) == (anchor, model)


def test_an_extra_arm_may_not_be_named_like_an_anchor():
    """Otherwise the anchor requirement is satisfiable with a signal-carrying arm."""
    from occam.vcell.panel.arms import build_arms

    pred, labels, ctrl_used, panel = _arm_inputs()
    with pytest.raises(ValueError) as exc:
        build_arms(pred, labels, ctrl_used, panel, extra={"ctrlmean": pred})
    assert "anchor" in str(exc.value).lower()

    # and the anchors still cannot be turned off
    with pytest.raises(ValueError):
        build_arms(pred, labels, ctrl_used, panel, anchors=(),
                   extra={"experimental": pred})


def test_unknown_and_misshaped_extras_raise():
    from occam.vcell.panel.arms import build_arms

    pred, labels, ctrl_used, panel = _arm_inputs()
    with pytest.raises(ValueError) as exc:
        build_arms(pred, labels, ctrl_used, panel, extra={"whatever": pred})
    assert "whatever" in str(exc.value)
    with pytest.raises(ValueError) as exc:
        build_arms(pred, labels, ctrl_used, panel, extra={"experimental": pred[:3]})
    assert "same rows in the same order" in str(exc.value)


def test_shrink_is_not_applied_to_the_reference_arms():
    """Shrink interpolates towards `ctrl_used`, which is meaningless for a real B cell."""
    from occam.vcell.panel.arms import build_arms

    pred, labels, ctrl_used, panel = _arm_inputs()
    exp = pred * 2
    arms = dict((t, m) for t, m, _ in build_arms(
        pred, labels, ctrl_used, panel, calibrate="none", shrinks=(0.5,),
        extra={"experimental": exp}))
    assert np.array_equal(arms["experimental"], exp)


# ---------------------------------------------------------------------------
# the B-control substitution
# ---------------------------------------------------------------------------

def test_the_prediction_file_carries_the_override_controls_and_the_target_keeps_its_own(tmp_path):
    ad = pytest.importorskip("anndata")
    from occam.vcell.panel.celleval_v1 import write_celleval_v1

    pred, labels, _, panel = _arm_inputs()
    panel.real_labels = np.concatenate([labels, np.array([CONTROL] * 4)])
    panel.real_X_model = np.concatenate([pred, np.zeros((4, 5), "float32")])
    override = np.full((3, 5), 9.0, dtype="float32")

    ppath, tpath = write_celleval_v1(pred, labels, panel, str(tmp_path), "ov",
                                     pred_ctrl=override)
    p, t = ad.read_h5ad(ppath), ad.read_h5ad(tpath)
    pc = np.asarray(p.X)[np.asarray(p.obs["target_gene"]) == CONTROL]
    tc = np.asarray(t.X)[np.asarray(t.obs["target_gene"]) == CONTROL]
    assert pc.shape == (3, 5) and np.allclose(pc, 9.0), "pred side must be the override"
    assert tc.shape == (4, 5) and np.allclose(tc, 0.0), "target side keeps the panel's own"

    # default = the panel's own controls, unchanged from every existing caller
    ppath2, _ = write_celleval_v1(pred, labels, panel, str(tmp_path), "def")
    p2 = ad.read_h5ad(ppath2)
    assert int((np.asarray(p2.obs["target_gene"]) == CONTROL).sum()) == 4

    with pytest.raises(ValueError):
        write_celleval_v1(pred, labels, panel, str(tmp_path), "bad",
                          pred_ctrl=np.zeros((3, 4), "float32"))


# ---------------------------------------------------------------------------
# run_split, end to end with cell-eval stubbed
# ---------------------------------------------------------------------------

def _tiny_model(torch, nn, n_genes, n_perts):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(n_perts + 1, n_genes)

        def forward(self, ctrl, pert_indices):
            idx = torch.tensor([p[0] if p[0] >= 0 else n_perts for p in pert_indices],
                               device=ctrl.device)
            return {"prediction": (ctrl + self.emb(idx)).clamp(min=0.0),
                    "delta_z": ctrl * 0, "z_hat": ctrl, "z_ctrl": ctrl}
    return M().eval()


def test_run_split_scores_every_arm_and_records_ctrlmean_as_undefined(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    ad = pytest.importorskip("anndata")

    from occam.vcell.panel import run as run_mod
    from occam.vcell.panel.protocol import UNDEFINED, run_split
    from occam.vcell.panel.splithalf import draw_splits, freeze_cohort

    X, labels, batches = _corpus()
    perts = [f"P{i:02d}" for i in range(N_PERT)]
    cohort = freeze_cohort(labels, eligible=perts, min_cells=10)
    assert cohort["n_perts"] == N_PERT
    sp = draw_splits(labels, batches, cohort, n_splits=1, base_seed=7)[0]
    base = _base_panel(X, labels)
    pert2id = {p: i for i, p in enumerate(perts)}

    seen = {}

    def _stub(pred_path, target_path, out_dir, **kw):
        os.makedirs(out_dir, exist_ok=True)
        p, t = ad.read_h5ad(pred_path), ad.read_h5ad(target_path)
        tag = os.path.basename(out_dir).replace("celleval_", "")
        seen[tag] = {
            "pred_ctrl": np.asarray(p.X)[np.asarray(p.obs["target_gene"]) == CONTROL],
            "real_ctrl": np.asarray(t.X)[np.asarray(t.obs["target_gene"]) == CONTROL],
            "n_pred_rows": int((np.asarray(p.obs["target_gene"]) != CONTROL).sum()),
        }
        # ctrlmean's prediction delta is exactly zero here, so pearsonr is NaN
        pdv = float("nan") if tag == "ctrlmean" else 0.4
        return {"pearson_delta": pdv, "mae": 0.1, "overlap_at_N": 0.2}

    monkeypatch.setattr(run_mod, "run_celleval_v1", _stub)

    rec = run_split(
        base, labels, sp, cohort, str(tmp_path),
        model=_tiny_model(torch, nn, X.shape[1], N_PERT), pert2id=pert2id, device="cpu",
        train_perts=[f"TRAIN{i}" for i in range(3)], calibrate="none",
        batch_size=16, seed=7)

    assert rec["anchor_arms"] == ["ctrlmean", "ctrlcell"]
    assert rec["reference_arms"] == ["experimental", "meanpert"]
    assert rec["model_arms"] == ["model_nocal"]
    assert rec["probe"] is False

    # ⭐ the protocol: B controls on every prediction file, A controls on the target
    for tag, got in seen.items():
        assert np.array_equal(got["pred_ctrl"], X[sp.ctrl_B]), f"{tag}: pred side is not B"
        assert np.array_equal(got["real_ctrl"], X[sp.ctrl_A]), f"{tag}: target side is not A"
        assert got["n_pred_rows"] == len(sp.pert_A)
    assert len(seen) == 5

    # ctrlmean's NaN pearson_delta is removed and explained, not silently averaged in
    assert "pearson_delta" not in rec["arms"]["ctrlmean"]
    assert rec["notes"]["ctrlmean.pearson_delta"] == UNDEFINED
    assert rec["arms"]["experimental"]["pearson_delta"] == 0.4

    written = json.load(open(os.path.join(tmp_path, "metrics.json")))
    assert written["cohort_sha256"] == cohort["cohort_sha256"]
    assert written["n_ctrl_A"] == written["n_ctrl_B"] == len(sp.ctrl_A)
    assert set(written["half_size"]) == set(perts)
    assert written["meanpert"]["n_perts"] == 3


def test_the_timing_probe_scores_one_arm_and_cannot_be_read_as_a_result(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    from occam.vcell.panel import run as run_mod
    from occam.vcell.panel.protocol import ProtocolError, run_split
    from occam.vcell.panel.splithalf import draw_splits, freeze_cohort

    X, labels, batches = _corpus()
    perts = [f"P{i:02d}" for i in range(N_PERT)]
    cohort = freeze_cohort(labels, eligible=perts, min_cells=10)
    sp = draw_splits(labels, batches, cohort, n_splits=1, base_seed=3)[0]
    base = _base_panel(X, labels)
    pert2id = {p: i for i, p in enumerate(perts)}
    model = _tiny_model(torch, nn, X.shape[1], N_PERT)

    calls = []

    def _stub(pred_path, target_path, out_dir, **kw):
        os.makedirs(out_dir, exist_ok=True)
        calls.append(os.path.basename(out_dir))
        return {"pearson_delta": 0.1}

    monkeypatch.setattr(run_mod, "run_celleval_v1", _stub)

    rec = run_split(base, labels, sp, cohort, str(tmp_path), model=model,
                    pert2id=pert2id, device="cpu", calibrate="none", batch_size=16,
                    score_only=["ctrlcell"], probe=True)
    assert calls == ["celleval_ctrlcell"]
    assert rec["probe"] is True
    assert os.path.isfile(os.path.join(tmp_path, "metrics_probe.json"))
    assert not os.path.isfile(os.path.join(tmp_path, "metrics.json"))
    assert rec["arms"]["ctrlmean"]["status"] == "not scored (timing probe)"

    with pytest.raises(ProtocolError):
        run_split(base, labels, sp, cohort, str(tmp_path / "x"), model=model,
                  pert2id=pert2id, device="cpu", calibrate="none",
                  score_only=["not_an_arm"], probe=True)


def test_restricting_the_arm_set_does_not_make_a_run_a_probe(tmp_path, monkeypatch):
    """⭐ The reference pass uses `score_only` to skip the checkpoint-dependent arm.

    These were one flag until 2026-09-17, so four completed reference splits were stamped
    `probe: true` and written to metrics_probe.json, where paired_report.py refuses them.
    Restricting the arm set and being a throwaway cost measurement are different things.
    """
    pytest.importorskip("torch")

    from occam.vcell.panel import run as run_mod
    from occam.vcell.panel.protocol import run_split
    from occam.vcell.panel.splithalf import draw_splits, freeze_cohort

    X, labels, batches = _corpus()
    perts = [f"P{i:02d}" for i in range(N_PERT)]
    cohort = freeze_cohort(labels, eligible=perts, min_cells=10)
    sp = draw_splits(labels, batches, cohort, n_splits=1, base_seed=11)[0]

    def _stub(pred_path, target_path, out_dir, **kw):
        os.makedirs(out_dir, exist_ok=True)
        return {"pearson_delta": 0.2}

    monkeypatch.setattr(run_mod, "run_celleval_v1", _stub)

    common = dict(model=None, pert2id={p: 0 for p in perts}, device="cpu",
                  predict_fn=lambda c, p: c, calibrate="none", batch_size=16,
                  score_only=["ctrlcell", "ctrlmean"])

    rec = run_split(_base_panel(X, labels), labels, sp, cohort,
                    str(tmp_path / "ref"), **common)
    assert rec["probe"] is False, "a restricted arm set is not a probe"
    assert os.path.isfile(os.path.join(tmp_path, "ref", "metrics.json"))
    assert not os.path.isfile(os.path.join(tmp_path, "ref", "metrics_probe.json"))
    assert rec["arms"]["experimental"]["status"] == "not scored in this pass"

    rec2 = run_split(_base_panel(X, labels), labels, sp, cohort,
                     str(tmp_path / "probe"), probe=True, **common)
    assert rec2["probe"] is True
    assert os.path.isfile(os.path.join(tmp_path, "probe", "metrics_probe.json"))
    assert not os.path.isfile(os.path.join(tmp_path, "probe", "metrics.json"))
    assert rec2["arms"]["experimental"]["status"] == "not scored (timing probe)"


def test_a_cohort_perturbation_missing_from_pert2id_refuses_before_predicting(tmp_path):
    """`predict_panel` drops it silently, which rescales discrimination and BH-FDR."""
    from occam.vcell.panel.protocol import ProtocolError, run_split
    from occam.vcell.panel.splithalf import draw_splits, freeze_cohort

    pytest.importorskip("torch")
    X, labels, batches = _corpus()
    perts = [f"P{i:02d}" for i in range(N_PERT)]
    cohort = freeze_cohort(labels, eligible=perts, min_cells=10)
    sp = draw_splits(labels, batches, cohort, n_splits=1, base_seed=5)[0]

    partial = {p: i for i, p in enumerate(perts[:-1])}
    with pytest.raises(ProtocolError) as exc:
        run_split(_base_panel(X, labels), labels, sp, cohort, str(tmp_path),
                  model=None, pert2id=partial, device="cpu")
    assert "pert2id" in str(exc.value) and perts[-1] in str(exc.value)


def test_run_splithalf_protocol_records_a_failed_split_rather_than_losing_the_rest(tmp_path):
    from occam.vcell.panel.protocol import run_splithalf_protocol
    from occam.vcell.panel.splithalf import draw_splits, freeze_cohort

    pytest.importorskip("torch")
    X, labels, batches = _corpus()
    perts = [f"P{i:02d}" for i in range(N_PERT)]
    cohort = freeze_cohort(labels, eligible=perts, min_cells=10)
    splits = draw_splits(labels, batches, cohort, n_splits=2, base_seed=5)

    # an incomplete pert2id makes every split fail; the point is that all of them are RECORDED
    recs = run_splithalf_protocol(
        _base_panel(X, labels), labels, splits, cohort, str(tmp_path),
        model=None, pert2id={}, device="cpu")
    assert [r["status"] for r in recs] == ["FAILED", "FAILED"]
    man = json.load(open(os.path.join(tmp_path, "manifest.json")))
    assert man["n_splits_declared"] == 2 and man["n_splits_scored"] == 0
    assert len(man["splits"]) == 2


def test_a_custom_predict_fn_also_drives_the_calibration_fit(tmp_path):
    """Otherwise the quantile map corrects a distribution that was never emitted.

    Found by a test that passed `predict_fn` and left `calibrate` at its "both" default:
    `fit_control_quantile_map` fell back to `model(...)`, which with model=None is a
    TypeError and with a real model is silently the wrong reference.
    """
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    from occam.vcell.panel import run as run_mod
    from occam.vcell.panel.protocol import run_split
    from occam.vcell.panel.splithalf import draw_splits, freeze_cohort

    X, labels, batches = _corpus()
    perts = [f"P{i:02d}" for i in range(N_PERT)]
    cohort = freeze_cohort(labels, eligible=perts, min_cells=10)
    sp = draw_splits(labels, batches, cohort, n_splits=1, base_seed=2)[0]

    seen = []

    def spy(ctrl_t, pert_indices):
        seen.append(tuple(ctrl_t.shape))
        return ctrl_t

    monkey_stub = lambda *a, **k: (os.makedirs(a[2], exist_ok=True) or {"pearson_delta": 0.1})
    old_fn = run_mod.run_celleval_v1
    run_mod.run_celleval_v1 = monkey_stub
    try:
        rec = run_split(_base_panel(X, labels), labels, sp, cohort, str(tmp_path),
                        model=_tiny_model(torch, nn, X.shape[1], N_PERT),
                        pert2id={p: i for i, p in enumerate(perts)}, device="cpu",
                        predict_fn=spy, calibrate="both", batch_size=16,
                        score_only=["ctrlcell"])
    finally:
        run_mod.run_celleval_v1 = old_fn

    # the spy was called by BOTH the prediction pass and the calibration fit
    assert len(seen) >= 2, seen
    assert "model_cal" in rec["model_arms"] and "model_nocal" in rec["model_arms"]


def test_every_calibrate_value_that_build_arms_accepts_also_works_end_to_end(tmp_path,
                                                                             monkeypatch):
    """⭐ `calibrate="quantile"` must actually fit the map.

    The gate in run_split read `calibrate in ("fit", "both")`. "fit" was never a valid
    value, so `quantile` silently skipped the fit, `calibrated` stayed None, and build_arms
    refused -- after loading the checkpoint and predicting 18,575 cells. Two launches of 135
    array tasks died on it, because every test used "none" or "both" and only the model
    launcher passes "quantile". This parametrises over all three.
    """
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    from occam.vcell.panel import run as run_mod
    from occam.vcell.panel.protocol import run_split
    from occam.vcell.panel.splithalf import draw_splits, freeze_cohort

    X, labels, batches = _corpus()
    perts = [f"P{i:02d}" for i in range(N_PERT)]
    cohort = freeze_cohort(labels, eligible=perts, min_cells=10)
    sp = draw_splits(labels, batches, cohort, n_splits=1, base_seed=3)[0]
    model = _tiny_model(torch, nn, X.shape[1], N_PERT)

    def _stub(pred_path, target_path, out_dir, **kw):
        os.makedirs(out_dir, exist_ok=True)
        return {"pearson_delta": 0.3}

    monkeypatch.setattr(run_mod, "run_celleval_v1", _stub)

    expected = {"none": {"model_nocal"},
                "quantile": {"model_cal"},
                "both": {"model_cal", "model_nocal"}}
    for mode, want in expected.items():
        rec = run_split(_base_panel(X, labels), labels, sp, cohort,
                        str(tmp_path / mode), model=model,
                        pert2id={p: i for i, p in enumerate(perts)}, device="cpu",
                        calibrate=mode, batch_size=16, score_only=["ctrlcell"])
        assert set(rec["model_arms"]) == want, (mode, rec["model_arms"])


def test_the_transfer_arm_adds_each_perturbations_own_cross_line_delta(tmp_path):
    """⭐ The arm that beat every model on validation (F5). It must be exactly
    `ctrl_used + delta[pert]`, per row, with the row's OWN perturbation's delta — not a
    pooled one. Getting this wrong makes it a second `meanpert` and hides the result.
    """
    from occam.vcell.panel.splithalf import transfer_arm

    ctrl = np.zeros((4, 3), dtype="float32")
    labels = np.asarray(["P0", "P0", "P1", "P1"], dtype=str)
    deltas = {"P0": np.array([1.0, 2.0, 3.0], "float32"),
              "P1": np.array([10.0, 20.0, 30.0], "float32")}

    pred, frac = transfer_arm(ctrl, labels, deltas)
    assert np.allclose(pred[:2], [1.0, 2.0, 3.0]), "row must use ITS OWN pert's delta"
    assert np.allclose(pred[2:], [10.0, 20.0, 30.0])
    assert frac == 0.0

    # the control cell is carried through, so control-draw noise cancels against the model
    ctrl2 = np.full((4, 3), 5.0, dtype="float32")
    pred2, _ = transfer_arm(ctrl2, labels, deltas)
    assert np.allclose(pred2[0], [6.0, 7.0, 8.0])

    # clipping is reported, never silent
    pred3, frac3 = transfer_arm(np.zeros((2, 3), "float32"),
                                np.asarray(["P0", "P0"], dtype=str),
                                {"P0": np.array([-1.0, -1.0, 1.0], "float32")})
    assert frac3 == pytest.approx(2 / 3) and np.all(pred3 >= 0)

    # a perturbation with no delta must RAISE, not silently become a control arm
    with pytest.raises(ValueError) as exc:
        transfer_arm(ctrl, np.asarray(["P0", "P0", "P9", "P9"], dtype=str), deltas)
    assert "P9" in str(exc.value)


def test_transfer_is_emitted_before_meanpert_and_counted_as_a_reference():
    """Emission order matters: `transfer` is the arm a truncated run most needs."""
    from occam.vcell.panel.arms import REFERENCE_TAGS, build_arms, split_arms

    pred, labels, ctrl_used, panel = _arm_inputs()
    arms = build_arms(pred, labels, ctrl_used, panel, calibrate="none",
                      extra={"experimental": pred * 2, "transfer": pred * 3,
                             "meanpert": pred * 4})
    tags = [t for t, _, _ in arms]
    assert tags == ["ctrlmean", "ctrlcell", "experimental", "model_nocal",
                    "transfer", "meanpert"], tags
    anchor, ref, model = split_arms(arms)
    assert ref == ["experimental", "transfer", "meanpert"]
    assert model == ["model_nocal"], "transfer must never be counted as a model arm"
    assert "transfer" in REFERENCE_TAGS

    # and it still may not masquerade as an anchor
    with pytest.raises(ValueError):
        build_arms(pred, labels, ctrl_used, panel, extra={"ctrlcell": pred})


def test_run_split_refuses_both_a_grand_mean_and_train_perts(tmp_path):
    """Two different estimands must not share one arm name."""
    pytest.importorskip("torch")
    from occam.vcell.panel.protocol import ProtocolError, run_split
    from occam.vcell.panel.splithalf import draw_splits, freeze_cohort

    X, labels, batches = _corpus()
    perts = [f"P{i:02d}" for i in range(N_PERT)]
    cohort = freeze_cohort(labels, eligible=perts, min_cells=10)
    sp = draw_splits(labels, batches, cohort, n_splits=1, base_seed=1)[0]
    with pytest.raises(ProtocolError) as exc:
        run_split(_base_panel(X, labels), labels, sp, cohort, str(tmp_path),
                  model=None, pert2id={p: 0 for p in perts}, device="cpu",
                  predict_fn=lambda c, p: c, calibrate="none",
                  train_perts=["TRAIN0"], grand_mean=np.zeros(X.shape[1], "float32"))
    assert "not both" in str(exc.value)
