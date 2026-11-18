"""The split-half invariants. Every one of these has a way of going wrong silently.

The protocol's whole purpose is to stop one split, or one power-dependent DE gene set, from
determining the result. That only holds if the halves are genuinely balanced and disjoint,
the cohort is genuinely frozen, and a cohort change does not quietly reshuffle everything
else.
"""

from __future__ import annotations

import numpy as np
import pytest

CONTROL = "non-targeting"


def _corpus(n_perts=8, per_pert=11, n_ctrl=40, n_batches=4, n_genes=6, seed=0):
    """Deliberately ODD cells per perturbation, so the drop path is always exercised."""
    rng = np.random.default_rng(seed)
    labels = [f"P{i:02d}" for i in range(n_perts) for _ in range(per_pert)]
    labels += [CONTROL] * n_ctrl
    labels = np.asarray(labels, dtype=str)
    batches = np.asarray([f"b{i % n_batches}" for i in range(len(labels))], dtype=str)
    X = rng.normal(5, 1, size=(len(labels), n_genes)).astype("float32")
    return X, labels, batches


def test_halves_are_equal_sized_and_disjoint_for_every_perturbation():
    from occam.vcell.panel.splithalf import (assert_disjoint_halves, draw_splits,
                                             freeze_cohort)

    X, labels, batches = _corpus()
    cohort = freeze_cohort(labels, eligible=sorted(set(labels) - {CONTROL}), min_cells=10)
    splits = draw_splits(labels, batches, cohort, n_splits=3, base_seed=42)

    for sp in splits:
        assert_disjoint_halves(sp)                      # raises if not
        assert len(sp.pert_A) == len(sp.pert_B)
        assert len(sp.ctrl_A) == len(sp.ctrl_B)
        # per perturbation, not just in total
        for p in cohort["perts"]:
            na = int((labels[sp.pert_A] == p).sum())
            nb = int((labels[sp.pert_B] == p).sum())
            assert na == nb == cohort["half_size"][p]


def test_odd_counts_drop_exactly_one_cell_each():
    """11 cells per perturbation -> 5|5 and one dropped, never 6|5."""
    from occam.vcell.panel.splithalf import draw_splits, freeze_cohort

    X, labels, batches = _corpus(per_pert=11, n_ctrl=41)
    cohort = freeze_cohort(labels, eligible=sorted(set(labels) - {CONTROL}), min_cells=10)
    sp = draw_splits(labels, batches, cohort, n_splits=1, base_seed=7)[0]

    n_perts = len(cohort["perts"])
    assert sp.n_dropped_odd == n_perts + 1, "one per odd perturbation, plus the controls"
    assert len(np.intersect1d(sp.dropped, sp.pert_A)) == 0
    assert len(np.intersect1d(sp.dropped, sp.pert_B)) == 0


def test_the_same_seed_reproduces_the_same_split():
    from occam.vcell.panel.splithalf import draw_splits, freeze_cohort

    X, labels, batches = _corpus()
    cohort = freeze_cohort(labels, eligible=sorted(set(labels) - {CONTROL}), min_cells=10)
    a = draw_splits(labels, batches, cohort, n_splits=2, base_seed=42)
    b = draw_splits(labels, batches, cohort, n_splits=2, base_seed=42)
    for x, y in zip(a, b):
        assert np.array_equal(x.pert_A, y.pert_A)
        assert np.array_equal(x.ctrl_B, y.ctrl_B)
    # and different seeds really do differ
    c = draw_splits(labels, batches, cohort, n_splits=2, base_seed=43)
    assert not np.array_equal(a[0].pert_A, c[0].pert_A)


def test_adding_a_perturbation_does_not_reshuffle_the_others():
    """⭐ The property a single RNG stream would fail.

    With one generator consumed in a loop, changing the cohort shifts every LATER
    perturbation's halves -- so a cohort edit would silently change results everywhere
    instead of only where it should. Per-(split, perturbation) seeding prevents that.
    """
    from occam.vcell.panel.splithalf import draw_splits, freeze_cohort

    X, labels, batches = _corpus(n_perts=6)
    elig = sorted(set(labels) - {CONTROL})

    full = freeze_cohort(labels, eligible=elig, min_cells=10)
    sp_full = draw_splits(labels, batches, full, n_splits=1, base_seed=42)[0]

    # drop the FIRST perturbation from the cohort; the rest must be untouched
    smaller = freeze_cohort(labels, eligible=elig[1:], min_cells=10)
    sp_small = draw_splits(labels, batches, smaller, n_splits=1, base_seed=42)[0]

    for p in smaller["perts"]:
        a_full = np.sort(sp_full.pert_A[labels[sp_full.pert_A] == p])
        a_small = np.sort(sp_small.pert_A[labels[sp_small.pert_A] == p])
        assert np.array_equal(a_full, a_small), (
            f"{p}'s half changed when a DIFFERENT perturbation left the cohort"
        )


def test_control_halves_are_batch_balanced():
    """Unstratified halves leave a batch term that enters the two deltas with opposite signs."""
    from occam.vcell.panel.splithalf import draw_splits, freeze_cohort

    X, labels, batches = _corpus(n_ctrl=120, n_batches=4)
    cohort = freeze_cohort(labels, eligible=sorted(set(labels) - {CONTROL}), min_cells=10)
    sp = draw_splits(labels, batches, cohort, n_splits=1, base_seed=3)[0]

    for b in np.unique(batches):
        na = int((batches[sp.ctrl_A] == b).sum())
        nb = int((batches[sp.ctrl_B] == b).sum())
        assert abs(na - nb) <= 1, f"batch {b}: {na} vs {nb} controls"


def test_the_cohort_refuses_drift():
    from occam.vcell.panel.splithalf import CohortDriftError, assert_cohort, freeze_cohort

    X, labels, batches = _corpus()
    cohort = freeze_cohort(labels, eligible=sorted(set(labels) - {CONTROL}), min_cells=10)

    assert_cohort(labels, cohort)                       # the real set passes
    with pytest.raises(CohortDriftError) as exc:
        assert_cohort(np.asarray([p for p in labels if p != "P00"]), cohort)
    assert "missing" in str(exc.value)
    with pytest.raises(CohortDriftError):
        assert_cohort(np.append(labels, "NOT_IN_COHORT"), cohort)


def test_min_cells_excludes_shallow_perturbations_and_says_why():
    from occam.vcell.panel.splithalf import freeze_cohort

    X, labels, batches = _corpus(n_perts=3, per_pert=11)
    labels = np.append(labels, ["SHALLOW"] * 4)
    cohort = freeze_cohort(labels, eligible=sorted(set(labels) - {CONTROL}), min_cells=10)

    assert "SHALLOW" not in cohort["perts"]
    assert "4 cells" in cohort["dropped"]["SHALLOW"]
    assert cohort["min_cells"] == 10


def test_the_per_split_panel_puts_A_on_the_real_side_and_B_controls_on_the_model_side():
    """This mapping IS the protocol; if it inverts, every arm is scored against the wrong half."""
    pytest.importorskip("torch")
    from occam.vcell.panel.panel_io import Panel
    from occam.vcell.panel.splithalf import draw_splits, freeze_cohort, panel_for_split

    X, labels, batches = _corpus(n_perts=4, per_pert=10, n_ctrl=20, n_genes=5)
    cohort = freeze_cohort(labels, eligible=sorted(set(labels) - {CONTROL}), min_cells=10)
    sp = draw_splits(labels, batches, cohort, n_splits=1, base_seed=1)[0]

    base = Panel(real_X_model=X, real_labels=labels, real_counts=None,
                 ctrl_X_model=X[:1], ctrl_counts=None, ctrl_batches=batches,
                 enc_cols=np.arange(X.shape[1]), panel_cols=np.arange(X.shape[1]),
                 panel_genes=[f"G{i}" for i in range(X.shape[1])], n_enc=X.shape[1])

    panel, b_pert, b_labels = panel_for_split(base, labels, sp, cohort)

    # the real side is A perturbed + A controls
    assert len(panel.real_labels) == len(sp.pert_A) + len(sp.ctrl_A)
    assert int((panel.real_labels == CONTROL).sum()) == len(sp.ctrl_A)
    # the model-visible side is the B controls, and nothing else
    assert panel.ctrl_X_model.shape[0] == len(sp.ctrl_B)
    assert np.array_equal(panel.ctrl_X_model, X[sp.ctrl_B])
    # the B perturbed rows line up with the A label sequence, so arms are comparable row-wise
    a_labels = panel.real_labels[panel.real_labels != CONTROL]
    assert np.array_equal(a_labels, b_labels)
    assert b_pert.shape[0] == len(a_labels)


def test_meanpert_is_per_perturbation_not_per_cell():
    """Per-cell averaging is dominated by the deepest perturbation -- a different baseline."""
    from occam.vcell.panel.splithalf import meanpert_arm, train_pert_grand_mean

    # DEEP has 100 cells at 10.0, SHALLOW has 2 cells at 0.0
    X = np.concatenate([np.full((100, 3), 10.0), np.zeros((2, 3))]).astype("float32")
    labels = np.asarray(["DEEP"] * 100 + ["SHALLOW"] * 2, dtype=str)

    grand, info = train_pert_grand_mean(X, labels, ["DEEP", "SHALLOW"])
    assert np.allclose(grand, 5.0), "equal weight per perturbation gives 5.0, per cell ~9.8"
    assert info["weighting"] == "equal per perturbation"
    assert info["n_perts"] == 2

    ctrl_used = np.ones((4, 3), dtype="float32")
    pred, frac = meanpert_arm(ctrl_used, grand, np.zeros(3, dtype="float32"))
    assert np.allclose(pred, 6.0)            # 1 + (5 - 0)
    assert frac == 0.0
    # and the clip is reported, not silent
    pred2, frac2 = meanpert_arm(ctrl_used, np.zeros(3, "float32"),
                                np.full(3, 10.0, "float32"))
    assert frac2 == 1.0 and np.all(pred2 == 0.0)
