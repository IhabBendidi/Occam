"""The three places a subtle error would enter the split-half headline.

1. `std` must use `ddof=1` — numpy's default understates a 5-sample std by 11.8%.
2. The paired gap is the mean of the per-split DIFFERENCES. `mean(gap)` happens to equal
   the difference of means, but `std(gap)` is NOT `sqrt(std_a**2 + std_b**2)`: the splits
   are shared, which is the entire point of pairing.
3. A probe record, or splits that disagree on the cohort, must be refused rather than
   averaged into something that looks like a result.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "paired_report.py")


def _mod():
    spec = importlib.util.spec_from_file_location("paired_report", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# reference (experimental) minus model, per split, chosen so the gap is exactly 0.05 with
# a NONZERO spread that differs from the naive sqrt(std_a^2 + std_b^2)
REF = [0.60, 0.62, 0.58, 0.64, 0.56]
MODEL = [0.55, 0.59, 0.53, 0.57, 0.51]
COHORT = "a" * 64


def _write_run(root, n_splits=5, probe=False, cohort=COHORT, per_pert=True):
    os.makedirs(root, exist_ok=True)
    for i in range(n_splits):
        d = os.path.join(root, f"split_{i:02d}")
        os.makedirs(d, exist_ok=True)
        rec = {
            "protocol": "splithalf/v1", "split": i, "probe": probe,
            "cohort_sha256": cohort if i == 0 else cohort, "n_perts": 3,
            "n_rows_per_arm": 30, "n_ctrl_B": 10, "n_ctrl_A": 10, "n_dropped_odd": 0,
            "half_size": {"P0": 4, "P1": 12, "P2": 30},
            "anchor_arms": ["ctrlmean", "ctrlcell"],
            "reference_arms": ["experimental", "meanpert"],
            "model_arms": ["model_cal"],
            "scored_order": ["ctrlmean", "ctrlcell", "experimental", "model_cal",
                             "meanpert"],
            "arms": {
                "ctrlmean": {"overlap_at_N": 0.02, "mae_delta": 0.9},
                "ctrlcell": {"pearson_delta": 0.05, "overlap_at_N": 0.03,
                             "mae_delta": 0.8},
                "experimental": {"pearson_delta": REF[i], "overlap_at_N": 0.40,
                                 "mae_delta": 0.30},
                "model_cal": {"pearson_delta": MODEL[i], "overlap_at_N": 0.30,
                              "mae_delta": 0.40},
                "meanpert": {"pearson_delta": 0.20, "overlap_at_N": 0.15,
                             "mae_delta": 0.60},
            },
            "notes": {"ctrlmean.pearson_delta": "undefined (zero-variance prediction delta)"},
            "timings_s": {}, "meanpert": {},
        }
        name = "metrics_probe.json" if probe else "metrics.json"
        with open(os.path.join(d, name), "w") as fh:
            json.dump(rec, fh)

        if not per_pert:
            continue
        # per-perturbation results, with the gap SHRINKING as m grows: P0 (m=4) has a big
        # gap, P2 (m=30) almost none. A real gap must behave this way.
        for arm, off in (("experimental", 0.0), ("model_cal", 1.0), ("ctrlcell", 1.0),
                         ("ctrlmean", 1.0), ("meanpert", 1.0)):
            ad = os.path.join(d, f"celleval_{arm}")
            os.makedirs(ad, exist_ok=True)
            with open(os.path.join(ad, "results.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["target_gene", "pearson_delta"])
                for p, base, gap in (("P0", 0.50, 0.20), ("P1", 0.60, 0.05),
                                     ("P2", 0.70, 0.01)):
                    w.writerow([p, base - off * gap])


def test_std_uses_ddof_1_and_the_gap_is_the_mean_of_differences(tmp_path):
    m = _mod()
    root = str(tmp_path / "run")
    _write_run(root)
    assert m.main.__module__  # the module imported

    import sys
    argv = sys.argv
    try:
        sys.argv = ["paired_report.py", "--run-dir", root]
        assert m.main() == 0
    finally:
        sys.argv = argv

    rep = json.load(open(os.path.join(root, "report", "metrics.json")))
    assert rep["n_splits"] == 5 and rep["partial"] is False

    # per-arm std, ddof=1
    want_std = m._std1(REF)
    got = rep["summary"]["experimental"]["pearson_delta"]
    assert abs(got["mean"] - m._mean(REF)) < 1e-12
    assert abs(got["std_across_splits"] - want_std) < 1e-12
    # and it is NOT numpy's ddof=0
    ddof0 = (sum((x - m._mean(REF)) ** 2 for x in REF) / len(REF)) ** 0.5
    assert abs(want_std - ddof0) > 1e-6

    gs = {(r["arm"], r["metric"]): r for r in rep["gap_summary"]}
    g = gs[("model_cal", "pearson_delta")]
    per = [a - b for a, b in zip(REF, MODEL)]
    assert abs(g["gap_mean"] - m._mean(per)) < 1e-12
    assert abs(g["gap_mean"] - 0.05) < 1e-9
    # ⭐ the paired std is the std of the DIFFERENCES, not the independent-sum formula
    assert abs(g["gap_std_across_splits"] - m._std1(per)) < 1e-12
    naive = (m._std1(REF) ** 2 + m._std1(MODEL) ** 2) ** 0.5
    assert abs(g["gap_std_across_splits"] - naive) > 0.01
    assert g["n_splits_arm_better"] == 0


def test_lower_is_better_metrics_flip_the_gap_sign(tmp_path):
    """`mae_delta` is an ERROR. A gap must mean "reference better" in both directions."""
    m = _mod()
    root = str(tmp_path / "run")
    _write_run(root)
    import sys
    argv = sys.argv
    try:
        sys.argv = ["paired_report.py", "--run-dir", root]
        m.main()
    finally:
        sys.argv = argv

    rep = json.load(open(os.path.join(root, "report", "metrics.json")))
    gs = {(r["arm"], r["metric"]): r for r in rep["gap_summary"]}
    # experimental mae_delta 0.30 vs model 0.40: the reference is better, so POSITIVE
    assert gs[("model_cal", "mae_delta")]["gap_mean"] == pytest.approx(0.10)
    # and on pearson_delta, where higher is better, also positive
    assert gs[("model_cal", "pearson_delta")]["gap_mean"] > 0


def test_the_strata_table_shows_the_gap_shrinking_with_half_size(tmp_path):
    m = _mod()
    root = str(tmp_path / "run")
    _write_run(root)
    import sys
    argv = sys.argv
    try:
        sys.argv = ["paired_report.py", "--run-dir", root]
        m.main()
    finally:
        sys.argv = argv

    rows = list(csv.DictReader(open(os.path.join(root, "report", "strata.csv"))))
    sel = {r["stratum"]: float(r["gap_mean"]) for r in rows
           if r["arm"] == "model_cal" and r["metric"] == "pearson_delta"}
    # m>=5 keeps P1+P2, m>=10 keeps P1+P2, m>=25 keeps only P2 (gap 0.01)
    assert sel["m>=25"] < sel["m>=5"], "the gap must shrink as m grows"
    assert sel["m>=25"] == pytest.approx(0.01)


def test_a_probe_record_is_refused(tmp_path):
    m = _mod()
    root = str(tmp_path / "probe_run")
    _write_run(root, probe=True)
    with pytest.raises(SystemExit) as exc:
        m.load_records(root)
    assert "probe" in str(exc.value).lower()


def test_splits_that_disagree_on_the_cohort_are_refused(tmp_path):
    """A rank over a different cohort and a BH-FDR over a different frame are not comparable."""
    m = _mod()
    root = str(tmp_path / "run")
    _write_run(root, n_splits=2)
    p = os.path.join(root, "split_01", "metrics.json")
    rec = json.load(open(p))
    rec["cohort_sha256"] = "b" * 64
    json.dump(rec, open(p, "w"))
    with pytest.raises(SystemExit) as exc:
        m.load_records(root)
    assert "cohort" in str(exc.value).lower()


def test_a_partial_run_is_labelled_partial_not_quietly_averaged(tmp_path):
    m = _mod()
    root = str(tmp_path / "run")
    _write_run(root, n_splits=3)
    import sys
    argv = sys.argv
    try:
        sys.argv = ["paired_report.py", "--run-dir", root]
        assert m.main() == 0
    finally:
        sys.argv = argv
    rep = json.load(open(os.path.join(root, "report", "metrics.json")))
    assert rep["partial"] is True and rep["n_splits"] == 3
    body = open(os.path.join(root, "report", "report.md")).read()
    assert "PARTIAL" in body


def test_the_report_states_the_sign_convention_and_the_ceiling_caveat(tmp_path):
    """Both have been misread before; they belong in the file, not in someone's memory."""
    m = _mod()
    root = str(tmp_path / "run")
    _write_run(root)
    import sys
    argv = sys.argv
    try:
        sys.argv = ["paired_report.py", "--run-dir", root]
        m.main()
    finally:
        sys.argv = argv
    body = open(os.path.join(root, "report", "report.md")).read()
    assert "positive gap" in body.lower()
    assert "not an upper bound" in body
    assert "undefined (zero-variance prediction delta)" in body
    assert "Only within-protocol, same-split comparisons are valid." in body


def test_all_metrics_md_carries_every_arm_every_split_and_every_metric(tmp_path):
    """`report.md` promotes six metrics. This file is the full record it is checked against.

    The failure this guards is quiet: a report that silently drops the metrics nobody
    promoted cannot be reconstructed afterwards, because the per-arm cell-eval directories
    are the only other copy and they are large enough to be deleted first.
    """
    import sys

    m = _mod()
    root = str(tmp_path / "run")
    _write_run(root)
    argv = sys.argv
    try:
        sys.argv = ["paired_report.py", "--run-dir", root]
        assert m.main() == 0
    finally:
        sys.argv = argv

    body = open(os.path.join(root, "report", "ALL_METRICS.md")).read()

    # every arm appears as a table row, and every metric as its own section
    for arm in ("ctrlmean", "ctrlcell", "experimental", "meanpert", "model_cal"):
        assert f"| `{arm}` |" in body, arm
    for metric in ("pearson_delta", "overlap_at_N", "mae_delta"):
        assert f"### `{metric}`" in body, metric

    # a per-split column per split, plus mean and std
    assert "| split 0 | split 1 | split 2 | split 3 | split 4 | mean | std (ddof=1) |" in body

    # the individual per-split values are present, not only the means
    assert "0.6000" in body and "0.6400" in body        # experimental, splits 0 and 3
    assert "0.5500" in body and "0.5100" in body        # model_cal, splits 0 and 4

    # what an arm IS, and which one is the model
    assert "## What an *arm* is" in body
    assert "**the model**" in body
    assert "NOT an upper bound" in body
    assert "Chance is 0.5" in body

    # the model's own section, with the gap to every other arm
    assert "## `model_cal` in detail" in body
    for other in ("ctrlmean", "ctrlcell", "experimental", "meanpert"):
        assert f"gap vs `{other}`" in body, other

    # the long-form machine-readable copy: one row per split x arm x metric
    rows = list(csv.DictReader(open(os.path.join(root, "report", "all_metrics.csv"))))
    got = {(r["split"], r["arm"], r["metric"]) for r in rows}
    assert ("0", "experimental", "pearson_delta") in got
    assert ("4", "model_cal", "mae_delta") in got
    assert len({r["metric"] for r in rows}) >= 3
    assert len({r["split"] for r in rows}) == 5
    # mae_delta is flagged as an error metric in the machine-readable copy too
    mae = [r for r in rows if r["metric"] == "mae_delta"]
    assert mae and all(r["lower_is_better"] == "1" for r in mae)


def test_the_model_arm_is_not_confused_with_an_anchor_or_a_reference(tmp_path):
    """`--model-arm` defaults to the arm that is neither, so a renamed arm cannot hijack it."""
    import sys

    m = _mod()
    root = str(tmp_path / "run")
    _write_run(root)
    argv = sys.argv
    try:
        sys.argv = ["paired_report.py", "--run-dir", root]
        m.main()
    finally:
        sys.argv = argv
    body = open(os.path.join(root, "report", "ALL_METRICS.md")).read()
    assert "## `model_cal` in detail" in body
    for a in ("ctrlmean", "ctrlcell", "experimental", "meanpert"):
        assert f"## `{a}` in detail" not in body
