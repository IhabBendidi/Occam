"""The nested seed/split test: the SEED is the unit of replication, not the split.

WHY THIS FILE EXISTS
    `norman_combo` is a 5-seed x 5-split design, and the two levels are not
    interchangeable. Splits re-slice the same cells from the SAME trained checkpoint, so
    their spread measures panel sampling, not model variance. Pooling all 25 values
    inflates n by 5x and shrinks the standard error by ~2.2x -- measured on this
    benchmark, per-split SEMs are 0.0003-0.0034 against effects of 0.02-0.06, so a
    split-level test calls essentially everything significant.

    These tests pin the thing that prevents that: averaging within seed FIRST, and
    refusing to test at all when only one seed exists.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _mod():
    spec = importlib.util.spec_from_file_location(
        "_sst", os.path.join(REPO, "scripts/seed_split_test.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_splits_are_averaged_within_seed_before_anything_else():
    m = _mod()
    cell = {(42, 0): 0.10, (42, 1): 0.20, (42, 2): 0.30,
            (43, 0): 1.00, (43, 1): 3.00}
    means, counts = m.seed_means(cell)
    assert means[42] == pytest.approx(0.20)
    assert means[43] == pytest.approx(2.00)
    assert counts == {42: 3, 43: 2}, "the split count per seed must be reported"


def test_the_unit_of_replication_is_the_seed_not_the_split():
    """n in the test must be the SEED count, never the pooled observation count."""
    m = _mod()
    cell = {(s, sp): 0.5 for s in (42, 43, 44) for sp in range(5)}
    means, _ = m.seed_means(cell)
    assert len(means) == 3, "3 seeds x 5 splits must reduce to 3, not 15"


def test_it_refuses_to_test_with_one_seed():
    """The situation today. A p-value from n=1 would be fabricated."""
    src = open(os.path.join(REPO, "scripts/seed_split_test.py")).read()
    assert "only {n} shared seed(s): no test is possible" in src
    assert "report it as a single observation" in src


def test_the_wilcoxon_floor_is_stated_rather_than_hidden():
    """At n=5 the exact two-sided p cannot reach 0.05; saying so is the point."""
    m = _mod()
    p, _w = m.wilcoxon_p([0.1, 0.2, 0.3, 0.4, 0.5])   # all same sign, n=5
    assert p == pytest.approx(2 / 32), f"exact floor at n=5 is 0.0625, got {p}"
    src = open(os.path.join(REPO, "scripts/seed_split_test.py")).read()
    assert "cannot reach 0.05" in src


def test_lower_is_better_metrics_flip_the_sign_test():
    m = _mod()
    assert "mae" in m.LOWER_IS_BETTER
    src = open(os.path.join(REPO, "scripts/seed_split_test.py")).read()
    assert "x < 0 if lower else x > 0" in src, \
        "the sign test must respect the metric's direction"


def test_the_t_distribution_helper_matches_known_values():
    """No scipy in this env, so the p-value maths is ours and must be checked."""
    m = _mod()
    # two-sided p at |t| = 2.776, df = 4 is 0.05 by construction of the critical value
    assert m._t_sf(2.776, 4) == pytest.approx(0.05, abs=5e-4)
    assert m._t_sf(4.604, 4) == pytest.approx(0.01, abs=5e-4)
    assert m._t_sf(0.0, 4) == pytest.approx(1.0, abs=1e-6)


def test_underpowered_is_said_out_loud():
    src = open(os.path.join(REPO, "scripts/seed_split_test.py")).read()
    assert "UNDERPOWERED" in src and "d_z" in src
    assert "sign result" in src, "the sign result must be offered as the quotable one"
