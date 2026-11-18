"""The grid file must be exactly the declared Cartesian product, with unique arms.

The TSV is checked in rather than computed in bash because index arithmetic is how an arm
silently becomes a duplicate of its neighbour, and the mapping then lives only in shell
history. These tests are what make the file trustworthy.
"""

from __future__ import annotations

import itertools
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = os.path.join(REPO, "configs", "rhj_fewshot_grid.tsv")

MODES = ["full", "multiplicative_only", "additive_only"]
FHIDS = ["0", "256"]
FMFS = ["0", "1"]
LOSSES = ["smooth_l1", "nb"]
N_GRID = len(MODES) * len(FHIDS) * len(FMFS) * len(LOSSES)      # 24
N_ANCHOR = 2
N_BASELINE = 1


def _rows():
    rows, cols = [], None
    for line in open(GRID):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if cols is None:
            cols = parts
            continue
        rows.append(dict(zip(cols, parts)))
    return rows


pytestmark = pytest.mark.skipif(not os.path.isfile(GRID), reason="grid not built")


def test_the_grid_is_exactly_the_declared_cartesian_product():
    rows = _rows()
    grid = [r for r in rows if r["kind"] == "grid"]
    assert len(grid) == N_GRID, f"expected {N_GRID} grid arms, found {len(grid)}"

    got = {(r["film_mode"], r["film_hidden_dim"], r["film_mean_free"], r["loss_type"])
           for r in grid}
    want = set(itertools.product(MODES, FHIDS, FMFS, LOSSES))
    assert got == want, f"missing {want - got}, unexpected {got - want}"


def test_the_no_film_anchors_are_present():
    """Without them the table cannot answer 'does FiLM help at all'."""
    anchors = [r for r in _rows() if r["kind"] == "anchor"]
    assert len(anchors) == N_ANCHOR
    assert all(r["interaction_type"] == "additive" for r in anchors)
    assert {r["loss_type"] for r in anchors} == set(LOSSES), (
        "one anchor per loss, or the anchor comparison is confounded by the loss"
    )


def test_the_baseline_arm_is_present_and_carries_nine_columns():
    """The non-learned val anchors. A val pearson_delta with nothing beside it is unreadable.

    ⚠️ Nine columns, not ten. `run_baselines` is driven by `kind` in the launcher because a
    tenth column added while an array was already queued would feed each pending task's
    nine-field `read` a tenth value, and `read` assigns the remainder to the LAST variable —
    every pending arm would have run with seed "42 0".
    """
    rows = _rows()
    base = [r for r in rows if r["kind"] == "baseline"]
    assert len(base) == N_BASELINE
    b = base[0]
    assert "baselines" in b["tag"], "the tag must say what makes this arm different"
    assert b["interaction_type"] == "additive", (
        "the baselines are not learned, so the arm reuses the no-FiLM configuration"
    )
    # every data row has exactly the header's column count
    header = next(l for l in open(GRID) if not l.startswith("#")).rstrip("\n").split("\t")
    assert len(header) == 9, f"the grid must stay at 9 columns, found {len(header)}"
    for line in open(GRID):
        line = line.rstrip("\n")
        if not line or line.startswith("#") or line.split("\t")[0] == "arm_id":
            continue
        assert len(line.split("\t")) == 9, f"{line.split(chr(9))[0]} has the wrong width"


def test_the_launcher_turns_the_baseline_kind_into_the_flag():
    """The TSV declares the arm; the launcher must actually set run_baselines for it."""
    sbatch = os.path.join(REPO, "scripts", "slurm", "rhj_fewshot_grid.sbatch")
    body = open(sbatch).read()
    assert 'if [ "$KIND" = "baseline" ]' in body
    assert "stages.vcell.eval.run_baselines=${BASELINES}" in body
    assert "BASELINES=false" in body, "every non-baseline arm must set it false explicitly"
    # and the array index of the baseline row is documented, since it ships separately
    rows = _rows()
    idx = next(i for i, r in enumerate(rows) if r["kind"] == "baseline")
    assert f"--array={idx}" in body, (
        f"the baseline row is index {idx}; the launcher header must say so"
    )


def test_arm_ids_and_tags_are_unique_and_ordered():
    rows = _rows()
    ids = [r["arm_id"] for r in rows]
    tags = [r["tag"] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate arm_id — two array tasks would collide"
    assert len(set(tags)) == len(tags), "duplicate tag — two arms would share a run dir"
    # zero-padded and in order, so `ls` sorts in grid order
    assert ids == [f"a{i:02d}" for i in range(len(rows))]


def test_every_tag_encodes_every_varied_axis():
    """A run dir you cannot decode from its name is a run dir you cannot attribute."""
    abbr = {"full": "full", "multiplicative_only": "mult", "additive_only": "add"}
    for r in _rows():
        if r["kind"] != "grid":
            continue
        tag = r["tag"]
        assert f"fmode-{abbr[r['film_mode']]}" in tag
        assert f"fh{r['film_hidden_dim']}" in tag
        assert f"fmf{r['film_mean_free']}" in tag
        assert f"s{r['seed']}" in tag
        assert ("nb" if r["loss_type"] == "nb" else "smoothl1") in tag


def test_run_dir_names_stay_within_a_filesystem_component():
    for r in _rows():
        name = f"rhj_fewshot_{r['arm_id']}__{r['tag']}"
        assert len(name) <= 120, f"{name} is {len(name)} chars"


def test_film_scale_is_not_an_axis():
    """Fixed at 1.0 by decision. If it appears here, the grid silently doubled."""
    assert "film_scale" not in open(GRID).read().split("\n")[6], (
        "film_scale must not be a grid column"
    )


def test_the_collector_reads_the_benchmark_hashes_from_where_the_ledger_writes_them():
    """A cross-check that silently finds nothing has not passed — it has stopped running.

    The ledger writes `benchmark_sha256` / `gene_axis_sha256` at the STAGE top level. The
    collector read them from `st["extra"]`, got None for all 27 arms, and printed
    "all completed arms share benchmark ? and axis ?" — indistinguishable from success.
    """
    import importlib.util
    import json
    import tempfile

    spec = importlib.util.spec_from_file_location(
        "collect", os.path.join(REPO, "scripts", "collect_rhj_results.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    tmp = tempfile.mkdtemp(prefix="occam_collect_")
    grid_rows = _rows()[:1]
    arm = grid_rows[0]
    cfg = os.path.join(tmp, "grid.tsv")
    with open(cfg, "w") as fh:
        fh.write("\t".join(arm.keys()) + "\n")
        fh.write("\t".join(arm.values()) + "\n")

    d = os.path.join(tmp, "runs", f"rhj_fewshot_{arm['arm_id']}__{arm['tag']}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "ledger.json"), "w") as fh:
        json.dump({"stages": {"vcell": {
            "status": "completed", "config_hash": "abc12345",
            # top level, exactly as the orchestrator writes it
            "benchmark_sha256": "d934211e5eb1e37e" + "0" * 48,
            "gene_axis_sha256": "ef64f2ba7ff7816e" + "0" * 48,
            "benchmark_id": "rhj_fewshot",
            "metrics": {"best_metric": 0.35, "epochs": 30},
        }}}, fh)

    import contextlib
    import io

    out = os.path.join(tmp, "table.md")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = mod.main(["--grid", cfg, "--runs", os.path.join(tmp, "runs"), "--out", out])
    printed = buf.getvalue()
    assert rc == 0, printed

    # ⭐ the hash must appear in the summary line. "share benchmark ? and axis ?" is the
    # dead-check signature and must not come back.
    assert "d934211e5eb1e37e" in printed, printed[-400:]
    assert "ef64f2ba7ff7816e" in printed, printed[-400:]
    assert "benchmark ? and axis ?" not in printed
    assert "UNVERIFIED" not in printed

    # and when the hashes are genuinely absent, the collector must SAY so rather than
    # printing a summary that looks like a pass
    with open(os.path.join(d, "ledger.json"), "w") as fh:
        json.dump({"stages": {"vcell": {
            "status": "completed", "config_hash": "abc12345",
            "metrics": {"best_metric": 0.35, "epochs": 30},
        }}}, fh)
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        mod.main(["--grid", cfg, "--runs", os.path.join(tmp, "runs"), "--out", out])
    assert "UNVERIFIED" in buf2.getvalue(), buf2.getvalue()[-400:]
