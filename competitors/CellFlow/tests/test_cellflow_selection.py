"""Selection-callback tests, with emphasis on the branch a lucky run never exercises.

`_restore_best` decides whether the single saved pickle holds the VALIDATION-SELECTED state or
merely the final one. A smoke run only exercises it when an early validation happens to win, so it
is tested directly here instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cellflow_selection import (  # noqa: E402
    SELECTION_SCORE_COMPONENTS,
    SelectionCallback,
    _condition_label,
)


class _FakeState:
    """Stand-in for a flax TrainState: only `.params` and `.replace` are used."""

    def __init__(self, params):
        self.params = params

    def replace(self, *, params):
        return _FakeState(params)


class _FakeSolver:
    def __init__(self, params):
        self.vf_state = _FakeState(params)
        self.vf_state_inference = _FakeState(params)


def _callback(tmp_path: Path, **kw) -> SelectionCallback:
    n_genes, k = 6, 3
    return SelectionCallback(
        pcs=np.zeros((n_genes, k), dtype=np.float32),
        means=np.zeros(n_genes, dtype=np.float32),
        genes=[f"g{i}" for i in range(n_genes)],
        control_x=np.zeros((2, n_genes), dtype=np.float32),
        cell_type="HepG2",
        output_root=tmp_path,
        scorer_py="/bin/true",
        scorer="/bin/true",
        hooke_tx_dir="/tmp",
        **kw,
    )


def _feed(cb: SelectionCallback, solver, score: float, *, final: bool, monkeypatch):
    """Drive one scoring round with a chosen score, bypassing the real scorer."""
    both = dict.fromkeys(SELECTION_SCORE_COMPONENTS, score)
    monkeypatch.setattr(cb, "_score", lambda *a, **k: dict(both))
    hook = cb.on_train_end if final else cb.on_log_iteration
    return hook({}, {}, {}, solver)


def test_restores_the_best_state_when_it_is_not_the_final_one(tmp_path, monkeypatch):
    cb = _callback(tmp_path)
    solver = _FakeSolver({"w": np.array([1.0])})

    # Best round: snapshot must capture THESE params, not later ones.
    _feed(cb, solver, 0.90, final=False, monkeypatch=monkeypatch)
    best_params = solver.vf_state_inference.params

    # Then the model degrades and finishes worse.
    solver.vf_state_inference = _FakeState({"w": np.array([2.0])})
    solver.vf_state = _FakeState({"w": np.array([2.0])})
    _feed(cb, solver, 0.10, final=True, monkeypatch=monkeypatch)

    assert cb.best is not None
    assert cb.best["selection_score"] == pytest.approx(0.90)
    assert not cb.best["final"], "the best state was not the final one"
    # Both states must carry the restored params, since predict reads the inference one.
    for attr in ("vf_state_inference", "vf_state"):
        assert getattr(solver, attr).params["w"] == pytest.approx(best_params["w"])
    assert solver.vf_state_inference.params["w"] == pytest.approx(1.0), "restore did not happen"


def test_keeps_the_final_state_when_it_wins(tmp_path, monkeypatch):
    cb = _callback(tmp_path)
    solver = _FakeSolver({"w": np.array([1.0])})
    _feed(cb, solver, 0.10, final=False, monkeypatch=monkeypatch)
    solver.vf_state_inference = _FakeState({"w": np.array([9.0])})
    solver.vf_state = _FakeState({"w": np.array([9.0])})
    _feed(cb, solver, 0.99, final=True, monkeypatch=monkeypatch)

    assert cb.best["final"] is True
    assert solver.vf_state_inference.params["w"] == pytest.approx(9.0), "must not roll back"


def test_a_partial_composite_never_wins(tmp_path, monkeypatch):
    """A half-computed score must not be selectable; skipping the state is the safe outcome."""
    cb = _callback(tmp_path)
    solver = _FakeSolver({"w": np.array([1.0])})

    monkeypatch.setattr(cb, "_score", lambda *a, **k: {SELECTION_SCORE_COMPONENTS[0]: 0.99})
    out = cb.on_log_iteration({}, {}, {}, solver)
    assert out == {}
    assert cb.best is None, "a missing component must not produce a best state"

    monkeypatch.setattr(
        cb, "_score", lambda *a, **k: dict.fromkeys(SELECTION_SCORE_COMPONENTS, float("nan"))
    )
    assert cb.on_log_iteration({}, {}, {}, solver) == {}
    assert cb.best is None, "a non-finite component must not produce a best state"


def test_a_scoring_failure_does_not_kill_training(tmp_path, monkeypatch):
    cb = _callback(tmp_path)
    solver = _FakeSolver({"w": np.array([1.0])})

    def boom(*a, **k):
        raise RuntimeError("scorer exploded")

    monkeypatch.setattr(cb, "_score", boom)
    assert cb.on_log_iteration({}, {}, {}, solver) == {}
    assert cb.best is None
    # And a later success still selects normally.
    _feed(cb, solver, 0.5, final=False, monkeypatch=monkeypatch)
    assert cb.best["selection_score"] == pytest.approx(0.5)


def test_history_and_summary_are_recorded(tmp_path, monkeypatch):
    cb = _callback(tmp_path)
    solver = _FakeSolver({"w": np.array([1.0])})
    for score in (0.3, 0.7, 0.5):
        _feed(cb, solver, score, final=False, monkeypatch=monkeypatch)
    summary = cb.summary()
    assert summary["n_scored_states"] == 3
    assert [h["selection_score"] for h in summary["history"]] == [0.3, 0.7, 0.5]
    assert summary["best"]["selection_score"] == pytest.approx(0.7)
    assert "pearson_delta" in summary["selection_metric"]


def test_condition_labels():
    assert _condition_label("ENSG1") == "ENSG1"
    assert _condition_label(("ENSG1", "ENSG2")) == "ENSG1+ENSG2"
