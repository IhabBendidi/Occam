"""The seed-replication grid: ranking, the anchor, and the two refusals.

Step 15 exists because 26 arms at one seed resolve `film_mode`-sized effects and nothing
smaller — the BatchNorm-contamination noise floor alone was ±0.007. The failure mode this
file guards is subtler than a wrong number: ranking an UNFINISHED grid silently replicates
"whatever finished first", and replicating the winners without the no-FiLM anchor produces a
table that cannot answer whether FiLM helps by more than seed noise.
"""

from __future__ import annotations

import importlib.util
import json
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "build_seed_grid.py")
HEADER = ("arm_id", "kind", "tag", "interaction_type", "film_mode", "film_hidden_dim",
          "film_mean_free", "loss_type", "seed")

ARMS = [
    ("a00", "grid", "fmode-full_fh0_fmf0_smoothl1_s42", "film", "full", "0", "0",
     "smooth_l1", 0.31),
    ("a01", "grid", "fmode-mult_fh256_fmf1_smoothl1_s42", "film", "multiplicative_only",
     "256", "1", "smooth_l1", 0.42),
    ("a02", "grid", "fmode-add_fh0_fmf0_nb_s42", "film", "additive_only", "0", "0",
     "nb", 0.37),
    ("a03", "anchor", "noflim-additive_smoothl1_s42", "additive", "full", "0", "0",
     "smooth_l1", 0.29),
]


def _mod():
    spec = importlib.util.spec_from_file_location("build_seed_grid", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fixture(tmp, statuses=None):
    """A grid TSV plus one ledger per arm, under a private repo-shaped tree."""
    statuses = statuses or {}
    cfg = os.path.join(tmp, "configs")
    runs = os.path.join(tmp, "runs")
    os.makedirs(cfg, exist_ok=True)
    grid_path = os.path.join(cfg, "grid.tsv")
    with open(grid_path, "w") as fh:
        fh.write("# a comment line the reader must skip\n")
        fh.write("\t".join(HEADER) + "\n")
        for a in ARMS:
            fh.write("\t".join(list(a[:8]) + ["42"]) + "\n")
    for a in ARMS:
        st = statuses.get(a[0], "completed")
        d = os.path.join(runs, f"rhj_fewshot_{a[0]}__{a[2]}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "ledger.json"), "w") as fh:
            json.dump({"stages": {"vcell": {"status": st, "config_hash": a[0] * 4,
                                            "metrics": {"best_metric": a[8],
                                                        "epochs": 30}}}}, fh)
    return grid_path, runs


def _read_tsv(path):
    rows, cols = [], None
    for line in open(path):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if cols is None:
            cols = parts
            continue
        rows.append(dict(zip(cols, parts)))
    return rows


def test_the_top_arms_plus_the_no_film_anchor_are_replicated_at_the_new_seeds(tmp_path):
    m = _mod()
    grid, runs = _fixture(str(tmp_path))
    out = os.path.join(str(tmp_path), "seeds.tsv")
    assert m.main(["--grid", grid, "--runs", runs, "--out", out, "--n-top", "2"]) == 0

    rows = _read_tsv(out)
    # 2 top arms + 1 anchor, x 2 additional seeds = 6 new jobs
    assert len(rows) == 6
    assert sorted({r["seed"] for r in rows}) == ["43", "44"]
    # ranked on the validation metric: a01 (0.42) then a02 (0.37); a00 (0.31) is out
    kinds = sorted({r["kind"] for r in rows})
    assert kinds == ["seed-a01", "seed-a02", "seed-a03"], kinds
    assert not any("a00" in r["kind"] for r in rows)
    # the anchor keeps its own interaction type -- replicating it as FiLM would be pointless
    anchor = [r for r in rows if r["kind"] == "seed-a03"]
    assert all(r["interaction_type"] == "additive" for r in anchor)
    # the FiLM configuration of each arm is carried verbatim; only the seed changes
    a01 = [r for r in rows if r["kind"] == "seed-a01"]
    assert all(r["film_mode"] == "multiplicative_only" and r["film_hidden_dim"] == "256"
               and r["film_mean_free"] == "1" for r in a01)
    # unique tags, and the seed is IN the tag so the run directories cannot collide
    tags = [r["tag"] for r in rows]
    assert len(set(tags)) == len(tags)
    assert all(r["tag"].endswith("_s" + r["seed"]) for r in rows)
    assert all(len(r["arm_id"]) == 3 and r["arm_id"].startswith("s") for r in rows)


def test_an_unfinished_grid_is_refused(tmp_path):
    """"top-2 of 13" is a different selection than "top-2"."""
    m = _mod()
    grid, runs = _fixture(str(tmp_path), statuses={"a02": "running"})
    out = os.path.join(str(tmp_path), "seeds.tsv")
    with pytest.raises(SystemExit) as exc:
        m.main(["--grid", grid, "--runs", runs, "--out", out])
    assert "not finished" in str(exc.value)
    assert not os.path.exists(out)

    # ...and --allow-partial proceeds, because sometimes that is the right call
    assert m.main(["--grid", grid, "--runs", runs, "--out", out, "--n-top", "1",
                   "--allow-partial"]) == 0
    assert len(_read_tsv(out)) == 4          # 1 top + 1 anchor, x 2 seeds


def test_seed_42_is_refused_because_its_run_dir_already_exists(tmp_path):
    m = _mod()
    grid, runs = _fixture(str(tmp_path))
    out = os.path.join(str(tmp_path), "seeds.tsv")
    with pytest.raises(SystemExit) as exc:
        m.main(["--grid", grid, "--runs", runs, "--out", out, "--seeds", "42,43"])
    assert "42" in str(exc.value)


def test_explicit_arms_override_the_ranking_and_skip_the_completeness_check(tmp_path):
    m = _mod()
    grid, runs = _fixture(str(tmp_path), statuses={"a01": "failed"})
    out = os.path.join(str(tmp_path), "seeds.tsv")
    assert m.main(["--grid", grid, "--runs", runs, "--out", out,
                   "--arms", "a00,a03", "--allow-partial"]) == 0
    rows = _read_tsv(out)
    assert sorted({r["kind"] for r in rows}) == ["seed-a00", "seed-a03"]
    assert len(rows) == 4

    with pytest.raises(SystemExit) as exc:
        m.main(["--grid", grid, "--runs", runs, "--out", out, "--arms", "a99",
                "--allow-partial"])
    assert "a99" in str(exc.value)


def test_the_generated_file_records_how_the_arms_were_chosen(tmp_path):
    """A TSV that does not say it was ranked on VALIDATION invites the wrong claim."""
    m = _mod()
    grid, runs = _fixture(str(tmp_path))
    out = os.path.join(str(tmp_path), "seeds.tsv")
    m.main(["--grid", grid, "--runs", runs, "--out", out, "--n-top", "2"])
    head = "".join(l for l in open(out) if l.startswith("#"))
    assert "VALIDATION" in head
    assert "NOT the benchmark result" in head
    assert "scripts/build_seed_grid.py" in head
    assert "rhj_fewshot_grid.sbatch" in head        # the launch line is in the file
