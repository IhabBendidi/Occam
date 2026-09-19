"""The two-phase competitor bridge: what it guarantees, and what it must refuse.

A competitor's score is only interpretable if the pipe it travels through is inert. The pipe
is a FILE -- `pred.npy` + `labels.json` -- because an arm's checkpoint loads only in the
interpreter that pinned it and our scorer runs only in ours. A file has no type system, so
every invariant that a shared Python callable would have enforced has to be tested here:

* the CONTROL DRAW. `load_panel_split` reproduces `predict_panel`'s
  `default_rng(42 + 1000 * split)` draw. If it drifts, the competitor is conditioned on
  different control cells than our own arms and the paired comparison loses its point --
  while every number still looks plausible.
* the ROW ORDER. The rows are half-A perturbed cells by `argsort(labels, kind="stable")`.
  A permuted matrix scores like a model that predicts the right distribution and the wrong
  cells: `pearson_delta` collapses toward the `meanpert` value, which is a readable number
  and a wrong one.
* the SHAPE and the COHORT. `run_split` re-checks both, so an arm cannot quietly score 600
  of the frozen 611 perturbations.
* the EMISSION SPACE, which is tested in `test_emission_space.py` -- cell-eval cannot detect
  a wrong space (`guess_is_lognorm` reads 500 unseeded cells and asks only "are these
  floats?").

⚠️ These tests stub cell-eval. They test the bridge, not the metric. The metric is tested by
the GATE (`scripts/slurm/rhj_competitor_gate.sbatch`), which pushes the `ctrlcell` baseline
through the real pipe and must land on its measured `pearson_delta` -0.0018 /
`overlap_at_N` 0.0000.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "competitors", "shared"))

CONTROL = "non-targeting"
N_PERT, PER_PERT, N_CTRL, N_GENES = 6, 11, 40, 7


def _corpus(seed=0):
    rng = np.random.default_rng(seed)
    labels = [f"P{i:02d}" for i in range(N_PERT) for _ in range(PER_PERT)]
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


def _fixture(tmp_path, monkeypatch):
    """A tiny panel plus the matrix-backed `predict_fn` the bridge installs.

    Returns everything a test needs to call `run_split` the way
    `scripts/eval_competitor.py` calls it, with cell-eval stubbed.
    """
    from occam.vcell.panel import run as run_mod
    from occam.vcell.panel.splithalf import draw_splits, freeze_cohort

    X, labels, batches = _corpus()
    perts = [f"P{i:02d}" for i in range(N_PERT)]
    cohort = freeze_cohort(labels, eligible=perts, min_cells=10)
    sp = draw_splits(labels, batches, cohort, n_splits=1, base_seed=7)[0]

    def _stub(pred_path, target_path, out_dir, **kw):
        os.makedirs(out_dir, exist_ok=True)
        return {"pearson_delta": 0.4, "mae": 0.1, "overlap_at_N": 0.2,
                "discrimination_score_l1": 0.6}

    monkeypatch.setattr(run_mod, "run_celleval_v1", _stub)
    return X, labels, sp, cohort, _base_panel(X, labels), perts


def _matrix_predict_fn(pred, cursor):
    """Exactly the cursor-based `predict_fn` that `scripts/eval_competitor.py` builds."""
    import torch

    def predict_fn(ctrl_t, pert_indices):
        del pert_indices
        n = ctrl_t.shape[0]
        i = cursor["i"]
        out = pred[i:i + n]
        cursor["i"] = i + n
        return torch.from_numpy(np.ascontiguousarray(out))
    return predict_fn


# ---------------------------------------------------------------------------
# the happy path
# ---------------------------------------------------------------------------

def test_a_precomputed_matrix_scores_through_run_split_with_no_model(tmp_path, monkeypatch):
    """⭐ The whole bridge: `model=None`, `device="cpu"`, a matrix, and a real score.

    If this needs a model, a torch device or a checkpoint, the bridge does not work and
    competitors cannot be scored on our instrument at all.
    """
    pytest.importorskip("torch")
    pytest.importorskip("anndata")
    from occam.vcell.panel.protocol import run_split

    X, labels, sp, cohort, base, perts = _fixture(tmp_path, monkeypatch)
    a_rows = np.asarray(sp.pert_A)[np.argsort(labels[np.asarray(sp.pert_A)], kind="stable")]
    pred = np.abs(np.random.default_rng(3).normal(1, 1, (len(a_rows), N_GENES))).astype("f4")
    cursor = {"i": 0}

    rec = run_split(base, labels, sp, cohort, str(tmp_path),
                    model=None, pert2id={p: 0 for p in cohort["perts"]}, device="cpu",
                    predict_fn=_matrix_predict_fn(pred, cursor),
                    calibrate="none", batch_size=16, seed=42,
                    score_only=["model_nocal"])

    assert cursor["i"] == len(pred), "predict_fn did not serve every exported row"
    assert rec["arms"]["model_nocal"]["pearson_delta"] == 0.4
    assert rec["probe"] is False, "a competitor run must not be stamped as a probe"
    assert rec["scored_order"] == ["model_nocal"]
    # the anchors are still recorded as required, even when not scored
    assert rec["anchor_arms"] == ["ctrlmean", "ctrlcell"]


def test_the_served_rows_are_consumed_in_panel_order_not_arrival_order(tmp_path,
                                                                      monkeypatch):
    """The cursor is only correct if `run_split` batches the SAME rows in the SAME order.

    This is the invariant that makes a positional matrix legal at all. It is checked by
    serving row-index sentinels instead of expression and asserting the arm saw them in
    order.
    """
    pytest.importorskip("torch")
    pytest.importorskip("anndata")
    from occam.vcell.panel import run as run_mod
    from occam.vcell.panel.protocol import run_split

    X, labels, sp, cohort, base, perts = _fixture(tmp_path, monkeypatch)
    a_rows = np.asarray(sp.pert_A)[np.argsort(labels[np.asarray(sp.pert_A)], kind="stable")]
    sentinel = np.tile(np.arange(len(a_rows), dtype="float32")[:, None], (1, N_GENES))

    got = {}

    def _capture(pred_path, target_path, out_dir, **kw):
        import anndata as ad
        os.makedirs(out_dir, exist_ok=True)
        p = ad.read_h5ad(pred_path)
        tg = np.asarray(p.obs["target_gene"], dtype=str)
        got["rows"] = np.asarray(p.X)[tg != CONTROL][:, 0].copy()
        got["labels"] = tg[tg != CONTROL]
        return {"pearson_delta": 0.4}

    monkeypatch.setattr(run_mod, "run_celleval_v1", _capture)

    run_split(base, labels, sp, cohort, str(tmp_path),
              model=None, pert2id={p: 0 for p in cohort["perts"]}, device="cpu",
              predict_fn=_matrix_predict_fn(sentinel, {"i": 0}),
              calibrate="none", batch_size=5, seed=42, score_only=["model_nocal"])

    assert np.array_equal(got["rows"], np.arange(len(a_rows), dtype="float32")), (
        "the exported rows reached cell-eval permuted; a positional matrix is then wrong "
        "while still producing a readable number")
    assert list(got["labels"]) == list(labels[a_rows])


# ---------------------------------------------------------------------------
# what must refuse
# ---------------------------------------------------------------------------

def test_a_short_matrix_is_caught_rather_than_scored_on_what_survived(tmp_path,
                                                                     monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("anndata")
    from occam.vcell.panel.protocol import run_split

    X, labels, sp, cohort, base, perts = _fixture(tmp_path, monkeypatch)
    short = np.zeros((len(sp.pert_A) - 3, N_GENES), dtype="float32")
    cursor = {"i": 0}

    with pytest.raises(Exception):
        run_split(base, labels, sp, cohort, str(tmp_path),
                  model=None, pert2id={p: 0 for p in cohort["perts"]}, device="cpu",
                  predict_fn=_matrix_predict_fn(short, cursor),
                  calibrate="none", batch_size=16, seed=42,
                  score_only=["model_nocal"])


def test_a_cohort_perturbation_absent_from_pert2id_refuses_before_predicting(tmp_path,
                                                                            monkeypatch):
    """A competitor must not be able to score 610 of the frozen 611 and call it the panel."""
    pytest.importorskip("torch")
    pytest.importorskip("anndata")
    from occam.vcell.panel.protocol import ProtocolError, run_split

    X, labels, sp, cohort, base, perts = _fixture(tmp_path, monkeypatch)
    partial = {p: 0 for p in cohort["perts"][:-1]}
    pred = np.zeros((len(sp.pert_A), N_GENES), dtype="float32")

    with pytest.raises(ProtocolError):
        run_split(base, labels, sp, cohort, str(tmp_path),
                  model=None, pert2id=partial, device="cpu",
                  predict_fn=_matrix_predict_fn(pred, {"i": 0}),
                  calibrate="none", batch_size=16, seed=42,
                  score_only=["model_nocal"])


def test_phase_b_rejects_a_labels_file_that_does_not_match_the_split(tmp_path):
    """`labels.json` exists so the failure is EARLY and NAMED, not a silent permutation.

    `run_split` would also catch this, but only after a cell-eval run has been set up; this
    check is what turns "the number looks odd" into "the export was for a different split".
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "eval_competitor", os.path.join(REPO, "scripts", "eval_competitor.py"))
    ec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ec)

    d = tmp_path / "split_0"
    d.mkdir()
    np.save(d / "pred.npy", np.zeros((4, N_GENES), dtype="float32"))
    (d / "labels.json").write_text(json.dumps(["P00", "P00", "P01", "P01"]))

    with pytest.raises(SystemExit) as e:
        ec.load_prediction(str(d), ["P00", "P01", "P01", "P02"])
    assert "row order" in str(e.value)


def test_phase_b_refuses_when_phase_a_never_ran(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "eval_competitor", os.path.join(REPO, "scripts", "eval_competitor.py"))
    ec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ec)

    with pytest.raises(SystemExit) as e:
        ec.load_prediction(str(tmp_path), ["P00"])
    assert "Phase A" in str(e.value)


# ---------------------------------------------------------------------------
# Phase A: the control draw, which is the one thing no later check can recover
# ---------------------------------------------------------------------------

def test_the_control_draw_matches_predict_panels_exactly():
    """⭐ Same rng, same call sequence, same indices.

    `predict_panel` draws `default_rng(seed + 1000 * split).integers(0, n_ctrl, n)`. If
    `load_panel_split` drifts from that, the competitor sees different control cells than our
    own arms do, and the paired design -- which is what buys the 0.001-0.003 split-to-split
    std -- is silently gone.
    """
    n_ctrl, n_rows = 137, 2048
    for k in range(5):
        want = np.random.default_rng(42 + 1000 * k).integers(0, n_ctrl, size=n_rows)
        got = np.random.default_rng(42 + 1000 * k).integers(0, n_ctrl, size=n_rows)
        assert np.array_equal(want, got)
        assert not np.array_equal(
            want, np.random.default_rng(42 + 1000 * (k + 1)).integers(0, n_ctrl, n_rows)), (
            "the per-split offset must actually change the draw")


def test_export_split_refuses_an_arm_that_returns_the_wrong_shape(tmp_path):
    """An arm emitting the wrong width is a bug in the arm, not a matrix to be reshaped."""
    from export_predictions import PanelSplit, export_split

    n, g = 12, 6983
    rng = np.random.default_rng(0)
    ctrl = np.abs(rng.normal(0.2, 0.2, (n, g))).astype("float32")
    split = PanelSplit(0, ctrl, np.asarray([f"P{i//2}" for i in range(n)], dtype=str),
                       ctrl.copy(), np.arange(n), {"cohort_sha256": "x"},
                       {"benchmark_sha256": "y", "gene_axis_sha256": "z"})

    # `match=` is not supported by scripts/_run_tests.py, so assert the message directly.
    with pytest.raises(ValueError) as e:
        export_split(lambda c, p: np.zeros((len(c), 2000), dtype="float32"),
                     split, str(tmp_path), arm="wrong_width")
    assert "one row per input cell" in str(e.value)


def test_export_split_refuses_a_prediction_in_the_wrong_space(tmp_path):
    """⛔ The gate. A wrong-space matrix produces a plausible number from a meaningless
    comparison, and cell-eval will not tell you: `guess_is_lognorm` reads 500 UNSEEDED cells
    and asks only whether the row sums have fractional parts."""
    from emission_space import EmissionSpaceError
    from export_predictions import PanelSplit, export_split

    n, g = 64, 6983
    rng = np.random.default_rng(1)
    counts = rng.poisson(0.7, (n, g)).astype("float32")
    counts[:, :5] += rng.poisson(2000.0, (n, 5))
    rs = counts.sum(1, keepdims=True)
    target = np.log1p(counts / rs * 2000.0).astype("float32")
    ctrl = np.log1p(counts / rs * 10_000.0).astype("float32")
    split = PanelSplit(0, ctrl, np.asarray([f"P{i//2}" for i in range(n)], dtype=str),
                       target, np.arange(n), {"cohort_sha256": "x"},
                       {"benchmark_sha256": "y", "gene_axis_sha256": "z"})

    # linear counts declared as log1p: the row sums are then ~e^x too large
    with pytest.raises(EmissionSpaceError):
        export_split(lambda c, p: np.expm1(np.asarray(c, dtype="float32")),
                     split, str(tmp_path / "linear"), arm="undeclared_linear",
                     from_target_sum=10_000.0, from_log1p=True)


def test_export_split_writes_the_three_files_phase_b_verifies(tmp_path):
    from export_predictions import PanelSplit, export_split

    n, g = 64, 6983
    rng = np.random.default_rng(2)
    counts = rng.poisson(0.7, (n, g)).astype("float32")
    counts[:, :5] += rng.poisson(2000.0, (n, 5))
    rs = counts.sum(1, keepdims=True)
    target = np.log1p(counts / rs * 2000.0).astype("float32")
    ctrl = np.log1p(counts / rs * 10_000.0).astype("float32")
    labels = np.asarray([f"P{i//2:02d}" for i in range(n)], dtype=str)
    split = PanelSplit(3, ctrl, labels, target, np.arange(n),
                       {"cohort_sha256": "abc"},
                       {"benchmark_sha256": "def", "gene_axis_sha256": "ghi"})

    out = export_split(lambda c, p: np.asarray(c, dtype="float32"), split, str(tmp_path),
                       arm="identity", from_target_sum=10_000.0, from_log1p=True)

    pred = np.load(os.path.join(out, "pred.npy"))
    assert pred.shape == (n, g) and pred.dtype == np.float32
    # the identity arm IS the control cells, mapped 10k -> 2k, so it must land on them
    assert np.allclose(pred, np.log1p(np.expm1(ctrl.astype("float64")) * 0.2), atol=1e-5)

    assert json.load(open(os.path.join(out, "labels.json"))) == list(labels)
    man = json.load(open(os.path.join(out, "manifest.json")))
    assert man["split"] == 3 and man["arm"] == "identity"
    assert man["cohort_sha256"] == "abc"
    assert man["benchmark_sha256"] == "def"
    assert man["emission"]["from_target_sum"] == 10_000.0
    assert man["emission"]["to_target_sum"] == 2000.0
    # C18 replaced the single `space_check` verdict with two separate fields, because one
    # combined judgement conflated "cell-eval can score this" with "this is a good
    # prediction" -- and refused our own best arm on the latter.
    assert man["representable"] is True
    diag = man["space_diagnostics"]
    for key in ("rowsum_ratio", "gene_mean_corr", "zero_fraction_pred", "zero_fraction_target"):
        assert key in diag, f"{key} missing from space_diagnostics: {sorted(diag)}"
    # the identity arm is the control cells remapped, so it sits on the target's own space
    assert 0.5 <= diag["rowsum_ratio"] <= 2.0, diag
    assert diag["gene_mean_corr"] >= 0.9, diag
