"""Execute the panel path end to end, so "it works" is checked rather than asserted.

The four broken call sites survived because nothing ever ran this path: the only two callers
were wrong in the same way, and a bare `except Exception` printed the TypeErrors. This builds
a tiny panel, loads it, predicts, calibrates and writes the cell-eval pair -- everything up to
the subprocess boundary. cell-eval itself is not invoked (it takes hours and lives in another
pyenv); the pair it would consume is inspected directly instead.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILDER = os.path.join(REPO, "scripts", "build_panel.py")

N_PERT, PER_PERT, N_CTRL, N_GENES = 10, 40, 500, 48


@pytest.fixture(scope="module")
def panel_dir():
    np = pytest.importorskip("numpy")
    ad = pytest.importorskip("anndata")
    pd = pytest.importorskip("pandas")
    sp = pytest.importorskip("scipy.sparse")

    tmp = tempfile.mkdtemp(prefix="occam_panel_smoke_")
    src = os.path.join(tmp, "src.h5ad")
    rng = np.random.default_rng(0)
    labels = ([f"GENE{i:03d}" for i in range(N_PERT) for _ in range(PER_PERT)]
              + ["non-targeting"] * N_CTRL)
    n = len(labels)
    X = sp.csr_matrix(rng.negative_binomial(4, 0.3, size=(n, N_GENES)).astype("float32"))
    obs = pd.DataFrame({"target_gene": pd.Categorical(labels),
                        "cell_line": pd.Categorical(["K562"] * n)},
                       index=[f"c{i}" for i in range(n)])
    var = pd.DataFrame(index=[f"SYM{j}" for j in range(N_GENES)])
    ad.AnnData(X=X, obs=obs, var=var).write_h5ad(src)

    out = os.path.join(tmp, "panel")
    rc = subprocess.run(
        [sys.executable, BUILDER, "--source", src, "--out-dir", out,
         "--context-col", "cell_line", "--context", "K562",
         "--pert-col", "target_gene",
         "--cells-per-pert", "20", "--n-controls", "200", "--thin-to-median", "60"],
        capture_output=True, text=True)
    if rc.returncode != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise AssertionError(f"build_panel.py failed:\n{rc.stdout[-1500:]}\n{rc.stderr[-1500:]}")
    yield out
    shutil.rmtree(tmp, ignore_errors=True)


def _tiny_model(torch, nn, n_genes, hidden=8):
    """A stand-in with the interface predict_panel uses: model(ctrl, pert_idx)['prediction']."""
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Linear(n_genes, hidden)
            self.dec = nn.Linear(hidden, n_genes)
            self.emb = nn.Embedding(N_PERT + 1, hidden)

        def forward(self, ctrl, pert_indices):
            z = ctrl if ctrl.shape[1] == hidden else self.enc(ctrl)
            idx = torch.tensor([p[0] if p[0] >= 0 else N_PERT for p in pert_indices],
                               device=ctrl.device)
            out = self.dec(z + self.emb(idx))
            return {"prediction": out.clamp(min=0.0), "delta_z": z * 0, "z_hat": z,
                    "z_ctrl": z}
    return M().eval()


def test_the_panel_path_runs_end_to_end(panel_dir):
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    np = pytest.importorskip("numpy")

    from occam.vcell.config import VirtualCellConfig
    from occam.vcell.panel.calibrate import apply_quantile_map, fit_control_quantile_map
    from occam.vcell.panel.arms import build_arms
    from occam.vcell.panel.celleval_v1 import write_celleval_v1
    from occam.vcell.panel.panel_io import load_panel
    from occam.vcell.panel.predict import predict_panel

    cfg = VirtualCellConfig()
    cfg.panel_dir = panel_dir
    cfg.panel_symbol_map = ""          # panel genes are already encoder-vocabulary symbols

    genes = [f"SYM{j}" for j in range(N_GENES)]

    # BUG 1's fix: all four arguments.
    panel = load_panel(cfg, genes, skip_normalize=False, skip_log1p=False)
    assert panel.n_keep > 0 and panel.n_enc == len(genes)

    model = _tiny_model(torch, nn, panel.n_enc)
    pert2id = {f"GENE{i:03d}": i for i in range(N_PERT)}

    # BUG 2's fix: keywords, and FOUR return values.
    pred, labels, ctrl_used, cin_idx = predict_panel(
        model, panel, pert2id, "cpu", batch_size=64, seed=0)
    assert pred.shape[1] == panel.n_keep
    assert len(labels) == pred.shape[0] == len(cin_idx)
    assert np.isfinite(pred).all()
    assert predict_panel.last_nonfinite_rows == 0

    # BUG 3's fix: no predict_fn argument at all.
    src_q, ref_q = fit_control_quantile_map(
        model, panel, pert2id, "cpu", max_cells=200, seed=0)
    calibrated = apply_quantile_map(pred, src_q, ref_q, "cpu")
    assert calibrated.shape == pred.shape and np.isfinite(calibrated).all()

    # the arms, anchors first
    arms = build_arms(pred, labels, ctrl_used, panel, calibrate="both",
                      calibrated=calibrated)
    tags = [a[0] for a in arms]
    assert tags[:2] == ["ctrlmean", "ctrlcell"], "anchors must be emitted first"
    assert any(t.startswith("model_cal") for t in tags)

    # and the cell-eval pair, written but not scored
    out = os.path.join(panel_dir, "_smoke")
    ppath, tpath = write_celleval_v1(arms[-1][1], arms[-1][2], panel, out, "smoke")
    assert os.path.isfile(ppath) and os.path.isfile(tpath)

    ad = pytest.importorskip("anndata")
    p, t = ad.read_h5ad(ppath), ad.read_h5ad(tpath)
    # cell-eval's three hard requirements
    assert p.shape[1] == t.shape[1], "gene axis must match"
    assert list(p.var.index) == list(t.var.index), "gene ORDER must match elementwise"
    assert set(p.obs["target_gene"]) == set(t.obs["target_gene"]), "label sets must match"
    # var.index must be SYMBOLS, or discrimination's self-gene exclusion silently does nothing
    assert not all(str(g).isdigit() for g in p.var.index)
    assert "non-targeting" in set(p.obs["target_gene"]), "controls must be on the pred side"


def test_an_absent_panel_directory_raises_with_a_build_hint(panel_dir):
    pytest.importorskip("torch")
    from occam.vcell.config import VirtualCellConfig
    from occam.vcell.panel.panel_io import load_panel

    cfg = VirtualCellConfig()
    cfg.panel_dir = os.path.join(panel_dir, "does_not_exist")
    with pytest.raises(FileNotFoundError) as exc:
        load_panel(cfg, [f"SYM{j}" for j in range(N_GENES)], False, False)
    assert "build_panel" in str(exc.value).lower() or "panel" in str(exc.value).lower()
