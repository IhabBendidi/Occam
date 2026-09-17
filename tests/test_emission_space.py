"""The one evaluation space, and the guard cell-eval cannot provide.

Every one of these failures is silent in cell-eval. `_convert_to_normlog` asks only
`guess_is_lognorm`, which is *"True if any of 500 random cells has a row sum with a
fractional part > 0.01"* -- i.e. "are these floats?" -- and then `_evaluator.py:273`
hardcodes `pdex_kwargs["is_log1p"] = True` regardless. Measured, 5 unseeded repeats: a
library-normed LINEAR matrix scores `guess_is_lognorm = True` five times out of five, sails
through, and has its fold changes computed as though it were log1p.

So these tests are the only thing standing between a competitor's emission and a wrong
number that looks right.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "competitors", "shared"))

PANEL = os.path.join(REPO, "data", "benchmarks", "rhj_fewshot", "splithalf", "panel.npz")


def _mod():
    import emission_space
    return emission_space


def _synthetic(n=200, g=6983, seed=0, target_sum=2000.0, axis_share=0.866, lam=0.7,
               n_high=5, lam_high=2000.0):
    """A matrix in the canonical space, built the way the panel was built.

    Counts -> divide by the FULL-panel row sum -> x target_sum -> restrict -> log1p. The
    axis share is what makes `expm1(x).sum(1)` land below `target_sum`, exactly as the real
    panel does (measured median 1731.8 against a nominal 2000).

    ⚠️ The defaults reproduce the REAL panel's regime on all three axes that matter, and
    each was pinned after a test failed vacuously without it:

      * `g = 6983` and `lam = 0.7` give ~0.25 expression per gene and a ~0.50 zero fraction.
        An earlier fixture used 300 genes, i.e. ~5.8 per gene -- a regime where integer
        rounding changes almost nothing, so `test_integer_re_emission_is_refused` passed for
        the wrong reason.
      * `n_high` genes at `lam_high` give the HEAVY TAIL. The real panel's max is 5.01 in
        log1p space, i.e. one gene carrying ~8.6% of the library. Without a tail every value
        is much less than 1, where `log1p(x) ~= x` and log1p is numerically almost
        indistinguishable from linear -- so `test_library_normed_linear_is_refused` also
        passed for the wrong reason. Real single-cell data has that tail; a fixture without
        it cannot exercise the check that matters.
    """
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam, size=(n, g)).astype(np.float64)
    counts[:, :n_high] = rng.poisson(lam_high, size=(n, n_high))
    full_rowsum = counts.sum(axis=1, keepdims=True) / axis_share   # pretend genes were dropped
    full_rowsum[full_rowsum == 0] = 1.0
    return np.log1p(counts / full_rowsum * target_sum).astype(np.float32)


def test_the_synthetic_fixture_matches_the_real_panels_regime():
    """If the fixture is not sparse like the real data, the tests below prove nothing."""
    m = _mod()
    r = m.describe_space(_synthetic())
    # the real panel: zero fraction 0.5147, median expm1 rowsum 1731.8, max 5.9248
    assert 0.40 < r.zero_fraction < 0.60, r.zero_fraction
    assert 1500.0 < r.median_rowsum_expm1 < 1950.0, r.median_rowsum_expm1
    assert 3.0 < r.vmax < 8.0, f"needs the heavy tail real data has, got max {r.vmax}"
    assert r.looks_log1p


# ---------------------------------------------------------------------------
# the real panel: the space is what we think it is
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.isfile(PANEL), reason="panel.npz not built")
def test_the_real_panel_is_log1p_at_2000_on_the_FULL_panel_denominator():
    """⭐ The single most important fact in the port, asserted rather than remembered.

    If `expm1(X).sum(axis=1)` were ~2000 the panel would have been normalised on the
    6,983-gene axis, and the 10k->2k conversion handed to competitors would NOT be exact.
    Measured on the real artifact: median 1731.8, i.e. 2000 x 0.866.
    """
    m = _mod()
    z = np.load(PANEL, allow_pickle=True)
    X = z["X"][:5000]

    r = m.assert_is_log1p(X, name="panel.npz", target_sum=m.EVAL_TARGET_SUM)
    assert r.looks_log1p
    assert r.vmax <= np.log1p(m.EVAL_TARGET_SUM)

    # NOT 2000 -- that is the whole point
    assert 1500.0 < r.median_rowsum_expm1 < 1950.0, r.median_rowsum_expm1
    assert abs(r.median_rowsum_expm1 / m.EVAL_TARGET_SUM - m.AXIS_UMI_SHARE) < 0.05
    assert not (1990.0 < r.median_rowsum_expm1 < 2010.0), (
        "the panel looks axis-normalised, which would break the exactness of to_eval_space")


@pytest.mark.skipif(not os.path.isfile(PANEL), reason="panel.npz not built")
def test_the_10k_to_2k_map_is_bit_exact_on_real_rows():
    """⭐ Competitors train at 10,000; we score at 2,000. The map must cost nothing.

    Exact because the per-cell full-panel denominator cancels:
        log1p(expm1(log1p(c/s * 10000)) * 0.2) == log1p(c/s * 2000)
    """
    m = _mod()
    X = np.load(PANEL, allow_pickle=True)["X"][:4000]

    x10k = np.log1p(np.expm1(X.astype(np.float64))
                    * (m.TRAIN_TARGET_SUM / m.EVAL_TARGET_SUM))
    back = m.to_eval_space(x10k, from_target_sum=m.TRAIN_TARGET_SUM, from_log1p=True)

    assert back.dtype == np.float32
    assert np.abs(back - X).max() == 0.0, np.abs(back - X).max()
    assert m.assert_same_space(back, X)["verdict"] == "same space"


# ---------------------------------------------------------------------------
# the failure modes, all of which cell-eval accepts
# ---------------------------------------------------------------------------

def test_library_normed_linear_is_refused_although_cell_eval_accepts_it():
    """⛔ The dangerous one. `guess_is_lognorm` returns True for this, 5/5."""
    m = _mod()
    X = _synthetic()
    linear = np.expm1(X.astype(np.float64))     # same data, no log1p

    with pytest.raises(m.EmissionSpaceError) as exc:
        m.assert_is_log1p(linear, name="linear")
    msg = str(exc.value)
    assert "log1p" in msg and "exceeds" in msg

    with pytest.raises(m.EmissionSpaceError):
        m.assert_same_space(linear, X)


def test_log1p_at_the_wrong_target_sum_is_refused():
    """Handing back the 10k matrix unconverted. Scale-invariant metrics would hide it."""
    m = _mod()
    X = _synthetic()
    x10k = np.log1p(np.expm1(X.astype(np.float64)) * 5.0)

    with pytest.raises(m.EmissionSpaceError) as exc:
        m.assert_same_space(x10k, X)
    assert "library size" in str(exc.value)
    # and the reason it matters is stated, not just the number
    assert "scale-invariant" in str(exc.value)


def test_integer_re_emission_is_refused():
    """The repo measured this exact transform costing 0.034 pearson_delta."""
    m = _mod()
    X = _synthetic()
    ints = np.log1p(np.round(np.expm1(X.astype(np.float64))))

    with pytest.raises(m.EmissionSpaceError):
        m.assert_same_space(ints, X)


def test_a_constant_prediction_is_refused():
    """A dead emission head. 83.6% of genes constant is a real defect this repo has hit."""
    m = _mod()
    X = _synthetic()
    with pytest.raises(m.EmissionSpaceError):
        m.assert_same_space(np.full_like(X, 0.3), X)


def test_negatives_and_nonfinite_are_refused():
    m = _mod()
    X = _synthetic()
    with pytest.raises(m.EmissionSpaceError) as exc:
        m.assert_is_log1p(X - 0.05, name="neg")
    assert "negative" in str(exc.value)

    bad = X.astype(np.float64).copy()
    bad[0, 0] = np.nan
    bad[1, 1] = np.inf
    with pytest.raises(m.EmissionSpaceError) as exc:
        m.assert_is_log1p(bad, name="nonfinite")
    assert "non-finite" in str(exc.value)


def test_gene_order_is_part_of_the_contract():
    """`discrimination_score_l1` excludes each perturbation's own gene BY SYMBOL."""
    m = _mod()
    X = _synthetic(g=50)
    syms = [f"GENE{i}" for i in range(X.shape[1])]

    assert m.assert_same_space(X, X, pred_var=syms, target_var=syms)["verdict"] == "same space"
    with pytest.raises(m.EmissionSpaceError) as exc:
        m.assert_same_space(X, X, pred_var=syms[::-1], target_var=syms)
    assert "ORDER" in str(exc.value) or "order" in str(exc.value)


def test_a_width_mismatch_is_refused_before_anything_else():
    m = _mod()
    X = _synthetic(g=60)
    with pytest.raises(m.EmissionSpaceError) as exc:
        m.assert_same_space(X[:, :50], X)
    assert "6,983" in str(exc.value) or "width" in str(exc.value)


# ---------------------------------------------------------------------------
# to_eval_space's own contract
# ---------------------------------------------------------------------------

def test_to_eval_space_is_identity_when_the_target_sums_match():
    m = _mod()
    X = _synthetic()
    out = m.to_eval_space(X, from_target_sum=m.EVAL_TARGET_SUM, from_log1p=True)
    assert np.abs(out - X).max() < 1e-6


def test_to_eval_space_handles_a_linear_emitter():
    """An arm run with `log1p=False` emits linear; it is rescaled then log1p'd once."""
    m = _mod()
    X = _synthetic()
    linear_10k = np.expm1(X.astype(np.float64)) * 5.0          # linear at 10k
    out = m.to_eval_space(linear_10k, from_target_sum=m.TRAIN_TARGET_SUM, from_log1p=False)
    assert np.abs(out - X).max() < 1e-5
    assert m.assert_same_space(out, X)["verdict"] == "same space"


def test_to_eval_space_refuses_negative_linear_input():
    m = _mod()
    with pytest.raises(m.EmissionSpaceError) as exc:
        m.to_eval_space(np.full((4, 5), -1.0), from_target_sum=m.TRAIN_TARGET_SUM,
                        from_log1p=False)
    assert "negative" in str(exc.value)
