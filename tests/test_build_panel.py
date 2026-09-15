"""The panel builder's contract, especially the disjoint control pools.

If the model-visible controls and the scoring controls overlap, every
control-derived arm is scored against cells it was built from and the result
looks good for the wrong reason. It is also the one leak that cannot be
detected downstream, so it has to be prevented here.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILDER = os.path.join(REPO, "scripts", "build_panel.py")

N_PERT, PER_PERT, N_CTRL, N_GENES = 12, 60, 900, 64


@pytest.fixture(scope="module")
def corpus():
    """A small synthetic corpus, and a panel built from it."""
    np = pytest.importorskip("numpy")
    ad = pytest.importorskip("anndata")
    pd = pytest.importorskip("pandas")
    sp = pytest.importorskip("scipy.sparse")

    tmp = tempfile.mkdtemp(prefix="occam_panel_")
    src = os.path.join(tmp, "src.h5ad")
    rng = np.random.default_rng(0)
    labels = ([f"GENE{i:03d}" for i in range(N_PERT) for _ in range(PER_PERT)]
              + ["non-targeting"] * N_CTRL)
    n = len(labels)
    X = sp.csr_matrix(rng.negative_binomial(4, 0.3, size=(n, N_GENES)).astype("float32"))
    obs = pd.DataFrame({"target_gene": labels, "cell_line": ["K562"] * n},
                       index=[f"c{i}" for i in range(n)])
    var = pd.DataFrame(index=[f"G{i:04d}" for i in range(N_GENES)])
    ad.AnnData(X=X, obs=obs, var=var).write_h5ad(src)

    out = os.path.join(tmp, "panel")
    rc = subprocess.run(
        [sys.executable, BUILDER, "--source", src, "--out-dir", out,
         "--context-col", "cell_line", "--context", "K562",
         "--cells-per-pert", "40", "--n-controls", "400",
         "--thin-to-median", "60"],
        capture_output=True, text=True,
    )
    if rc.returncode != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise AssertionError(f"build_panel.py failed:\n{rc.stdout[-2000:]}\n{rc.stderr[-2000:]}")
    yield {"tmp": tmp, "src": src, "out": out}
    shutil.rmtree(tmp, ignore_errors=True)


def test_emits_the_full_file_contract(corpus):
    expected = {
        "panel_real.h5ad", "panel_controls_model.h5ad", "genes.csv", "perts.csv",
        "control_split.json", "panel_manifest.json", "seen_split.json",
    }
    assert expected <= set(os.listdir(corpus["out"]))


def test_shapes_are_what_was_asked_for(corpus):
    ad = pytest.importorskip("anndata")
    real = ad.read_h5ad(os.path.join(corpus["out"], "panel_real.h5ad"))
    ctrl = ad.read_h5ad(os.path.join(corpus["out"], "panel_controls_model.h5ad"))

    is_ctrl = (real.obs.target_gene == "non-targeting").values
    assert int(is_ctrl.sum()) == 400, "scoring control pool is the requested size"
    assert int((~is_ctrl).sum()) == N_PERT * 40, "perturbed cells = perts x cells_per_pert"
    assert ctrl.n_obs == 400, "model-visible pool is the requested size"
    assert list(real.var_names) == list(ctrl.var_names), "both files share a gene axis"


def test_control_pools_are_disjoint_by_provenance(corpus):
    ad = pytest.importorskip("anndata")
    real = ad.read_h5ad(os.path.join(corpus["out"], "panel_real.h5ad"))
    ctrl = ad.read_h5ad(os.path.join(corpus["out"], "panel_controls_model.h5ad"))

    m = (real.obs.target_gene == "non-targeting").values
    scoring = set(zip(real.obs.source_file[m].astype(str), real.obs.source_row[m]))
    model = set(zip(ctrl.obs.source_file.astype(str), ctrl.obs.source_row))
    assert len(scoring) == 400 and len(model) == 400
    assert not (scoring & model), (
        f"{len(scoring & model)} source cells appear in BOTH control pools"
    )


def test_control_pools_are_disjoint_at_the_byte_level(corpus):
    """Independent of provenance bookkeeping: no control row is duplicated."""
    ad = pytest.importorskip("anndata")
    real = ad.read_h5ad(os.path.join(corpus["out"], "panel_real.h5ad"))
    ctrl = ad.read_h5ad(os.path.join(corpus["out"], "panel_controls_model.h5ad"))

    m = (real.obs.target_gene == "non-targeting").values
    a = {r.tobytes() for r in real.X[m].toarray()}
    b = {r.tobytes() for r in ctrl.X.toarray()}
    assert not (a & b), f"{len(a & b)} control rows are byte-identical across the pools"


def test_thinning_happens_once_and_is_recorded(corpus):
    with open(os.path.join(corpus["out"], "panel_manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    thin = manifest["thinning"]
    assert thin["mode"] == "fixed_median"
    assert thin["median_umi_after"] == pytest.approx(60, abs=5)
    assert thin["median_umi_before"] > thin["median_umi_after"], "thinning never up-samples"


def test_manifest_records_provenance(corpus):
    with open(os.path.join(corpus["out"], "panel_manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert manifest["controls"]["disjoint"] is True
    assert manifest["controls"]["per_pool"] == 400
    assert manifest["gene_axis"]["n"] == N_GENES
    assert manifest["sources"][0]["sha256"], "source checksum recorded"
    assert "command" in manifest and "build_panel.py" in manifest["command"]


def test_refuses_an_infeasible_control_split(corpus):
    """Half the available controls is a hard ceiling, and the error says so."""
    rc = subprocess.run(
        [sys.executable, BUILDER, "--source", corpus["src"],
         "--out-dir", os.path.join(corpus["tmp"], "bad"),
         "--context-col", "cell_line", "--context", "K562",
         "--cells-per-pert", "40", "--n-controls", str(N_CTRL // 2 + 50), "--dry-run"],
        capture_output=True, text=True,
    )
    assert rc.returncode != 0, "should refuse rather than silently shrink a pool"
    assert "disjoint" in rc.stderr.lower()
    assert f"at most {N_CTRL // 2}" in rc.stderr, "error states the actual ceiling"


def test_loader_accepts_the_built_panel(corpus):
    """The builder's output must satisfy the loader's disjointness assertion."""
    pytest.importorskip("anndata")
    import anndata as ad

    from occam.vcell.panel.panel_io import assert_disjoint_controls

    real = ad.read_h5ad(os.path.join(corpus["out"], "panel_real.h5ad"))
    ctrl = ad.read_h5ad(os.path.join(corpus["out"], "panel_controls_model.h5ad"))
    real_obs = {c: real.obs[c].astype(str).tolist() for c in
                ("target_gene", "source_file")}
    real_obs["source_row"] = real.obs.source_row.tolist()
    real_obs["_index"] = list(real.obs_names)
    ctrl_obs = {c: ctrl.obs[c].astype(str).tolist() for c in
                ("target_gene", "source_file")}
    ctrl_obs["source_row"] = ctrl.obs.source_row.tolist()
    ctrl_obs["_index"] = list(ctrl.obs_names)

    method = assert_disjoint_controls(real_obs, ctrl_obs, real.X, ctrl.X)
    assert method == "provenance"
