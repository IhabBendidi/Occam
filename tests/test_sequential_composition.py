"""`compose_sequential`: the rule State and PerturbDiff use for a PAIR on norman_combo.

WHY THIS FILE EXISTS
    Neither arm has a multi-perturbation mechanism of its own, so their pair predictions
    come from a rule WE apply at the export boundary. A wrong rule fails silently -- it
    returns a plausible profile and simply scores worse, which reads as "this method is
    bad at composition" rather than "our wrapper is broken". The properties are therefore
    pinned.

    The load-bearing one is the FIRST test: a single perturbation must be passed straight
    through, so a wrapped arm is bit-identical to the unwrapped arm on rhj_fewshot and the
    two benchmarks' numbers stay comparable.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "competitors"))
sys.path.insert(0, os.path.join(REPO, "src"))

from shared.export_predictions import compose_sequential  # noqa: E402

# A toy "model": each perturbation adds its own fixed vector, and the shift depends on the
# input, so the two chain orders genuinely differ (a purely additive model would make
# `ab` and `ba` identical and the test vacuous).
SHIFT = {"A": np.array([1.0, 0.0, 0.0]), "B": np.array([0.0, 2.0, 0.0]),
         "C": np.array([0.0, 0.0, 3.0]), "": np.zeros(3)}


def _model(ctrl, labels):
    out = np.empty((len(labels), ctrl.shape[1]), dtype=np.float32)
    for i, lab in enumerate(labels):
        s = SHIFT.get(str(lab), np.zeros(ctrl.shape[1]))
        # the `* (1 + ctrl[i, 0])` term makes the map input-dependent, hence non-commuting
        out[i] = ctrl[i] + s * (1.0 + ctrl[i, 0])
    return out


CTRL = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, 1.0]], dtype=np.float32)


def test_a_single_perturbation_is_passed_straight_through_unchanged():
    """The property that keeps a wrapped arm comparable on rhj_fewshot."""
    for order in ("ab", "ba", "mean"):
        wrapped = compose_sequential(_model, order=order)
        got = wrapped(CTRL, ["A", "B", "C"])
        want = _model(CTRL, ["A", "B", "C"])
        assert np.array_equal(got, want), order


def test_a_pair_is_the_two_step_chain_not_a_sum():
    wrapped = compose_sequential(_model, order="ab")
    got = wrapped(CTRL[:1], ["A+B"])
    step1 = _model(CTRL[:1], ["A"])
    want = _model(step1, ["B"])
    assert np.allclose(got, want)
    # and it is NOT the additive answer
    additive = CTRL[:1] + SHIFT["A"] + SHIFT["B"]
    assert not np.allclose(got, additive)


def test_mean_is_exactly_the_average_of_the_two_orders():
    ab = compose_sequential(_model, order="ab")(CTRL, ["A+B"] * 3)
    ba = compose_sequential(_model, order="ba")(CTRL, ["A+B"] * 3)
    mean = compose_sequential(_model, order="mean")(CTRL, ["A+B"] * 3)
    assert np.allclose(mean, 0.5 * (ab + ba))
    # the toy model is deliberately non-commuting, or this test proves nothing
    assert not np.allclose(ab, ba)


def test_a_mixed_batch_is_handled_row_by_row():
    wrapped = compose_sequential(_model, order="ab")
    got = wrapped(CTRL, ["A", "B+C", "C"])
    assert np.allclose(got[0], _model(CTRL[:1], ["A"])[0])
    assert np.allclose(got[2], _model(CTRL[2:3], ["C"])[0])
    chained = _model(_model(CTRL[1:2], ["B"]), ["C"])[0]
    assert np.allclose(got[1], chained)


def test_the_wrapper_refuses_a_model_that_changes_the_width():
    def _bad(ctrl, labels):
        return np.zeros((len(labels), ctrl.shape[1] + 1), dtype=np.float32)

    with pytest.raises(ValueError) as e:
        compose_sequential(_bad, order="ab")(CTRL[:1], ["A+B"])
    assert "feeds the output back in" in str(e.value)


def test_an_unknown_order_is_refused_at_construction():
    with pytest.raises(ValueError) as e:
        compose_sequential(_model, order="sum")
    assert "ab|ba|mean" in str(e.value)


def test_only_state_and_perturbdiff_are_meant_to_be_wrapped():
    """TxPert sums natively and CellFlow aggregates a set; wrapping either would
    measure our rule instead of theirs."""
    src = open(os.path.join(REPO, "competitors/shared/export_predictions.py")).read()
    i = src.index("def compose_sequential")
    doc = src[i:i + 3000]
    assert "State" in doc and "PerturbDiff" in doc
    assert "TxPert and CellFlow are NOT wrapped" in doc
    assert "NO published combinatorial protocol" in doc
