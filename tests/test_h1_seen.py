"""The OOD-context path (h1_seen): rhj_fewshot checkpoints scored on a cell line they never saw.

Pinned here:
  * `--ood-context` only works on a panel that DECLARES itself an OOD context, and only for
    a checkpoint trained on the axis the panel was derived from;
  * the same-benchmark gate still refuses such a panel without the flag;
  * the competitor export scatters the panel's axis subset onto the model axis and gathers
    it back, with the missing genes zero at the input and absent from the output.
"""
import hashlib
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _eval_mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_eval_sh_ood", os.path.join(REPO, "scripts/eval_splithalf.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AXIS = ["E1", "E2", "E3", "E4"]
AXIS_SHA = hashlib.sha256("\n".join(AXIS).encode()).hexdigest()


def _ckpt():
    return {"benchmark": {"id": "rhj_fewshot", "gene_axis_sha256": AXIS_SHA,
                          "n_axis_genes": 4, "excluded_perts": {"f.h5ad": ["B"]}},
            "split_info": {"train": ["A", "B"]}}


def _ood_manifest(**kw):
    m = {"benchmark_id": "h1_seen", "cell_line": "H1", "ood_context": True,
         "gene_axis_sha256": "x", "n_axis_genes": 3,
         "seen_from": {"benchmark_id": "rhj_fewshot", "gene_axis_sha256": AXIS_SHA}}
    m.update(kw)
    return m


def _refuses(fn, needle):
    try:
        fn()
    except SystemExit as e:
        assert needle in str(e), str(e)
        return
    raise AssertionError(f"expected a SystemExit mentioning {needle!r}")


def _panel_dir(tmp_path, perts):
    d = tmp_path / "panel"
    d.mkdir()
    (d / "cohort.json").write_text(json.dumps({"cells_per_pert": {p: 20 for p in perts}}))
    return str(d)


def test_ood_gate_accepts_a_declared_ood_panel_and_records_the_shift(tmp_path):
    mod = _eval_mod()
    out = tmp_path / "out"
    out.mkdir()
    rec = mod.assert_ood_context("c.pth", _ood_manifest(), ckpt=_ckpt(),
                                 panel_dir=_panel_dir(tmp_path, ["A", "B"]),
                                 out_dir=str(out))
    assert rec["n_cohort"] == 2 and rec["n_in_split_info_train"] == 2
    assert json.load(open(out / "context_shift.json"))["panel_cell_line"] == "H1"


def test_ood_gate_refuses_an_undeclared_panel(tmp_path):
    mod = _eval_mod()
    _refuses(lambda: mod.assert_ood_context(
        "c.pth", _ood_manifest(ood_context=False), ckpt=_ckpt(),
        panel_dir=_panel_dir(tmp_path, ["A"])), "ood_context")


def test_ood_gate_refuses_a_checkpoint_from_another_axis(tmp_path):
    mod = _eval_mod()
    m = _ood_manifest(seen_from={"benchmark_id": "rhj_fewshot", "gene_axis_sha256": "other"})
    _refuses(lambda: mod.assert_ood_context(
        "c.pth", m, ckpt=_ckpt(), panel_dir=_panel_dir(tmp_path, ["A"])), "not the axis")


def test_same_benchmark_gate_still_refuses_the_ood_panel(tmp_path):
    mod = _eval_mod()
    _refuses(lambda: mod.assert_checkpoint_matches(
        "c.pth", _ood_manifest(), ckpt=_ckpt(), panel_dir=_panel_dir(tmp_path, ["A"])),
        "not trained on this benchmark")


def test_competitor_export_scatters_and_gathers_the_axis_subset(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.join(REPO, "competitors", "shared"))
    import export_predictions as ep

    # a fake model-axis benchmark under a temporary repo root
    root = tmp_path / "repo"
    (root / "data" / "benchmarks" / "fakebench").mkdir(parents=True)
    (root / "data" / "benchmarks" / "fakebench" / "gene_axis.csv").write_text(
        "ensembl_gene_id,gene_name\n" + "".join(f"{e},{e}\n" for e in AXIS))
    monkeypatch.setattr(ep, "_HERE", str(root / "competitors" / "shared"))

    panel_axis = ["E3", "E1", "E4"]                     # subset, reordered
    labels = np.array(["P", "P", "non-targeting", "non-targeting"])
    X = np.log1p(np.arange(12, dtype=np.float32).reshape(4, 3))
    pdir = tmp_path / "panel"
    pdir.mkdir()
    np.savez(pdir / "panel.npz", X=X, labels=labels, batches=np.array(["b"] * 4),
             axis_ensembl=np.array(panel_axis, dtype=object),
             axis_symbols=np.array(panel_axis, dtype=object))
    np.savez(pdir / "splits.npz", pert_A_0=np.array([0]), pert_B_0=np.array([1]),
             ctrl_A_0=np.array([2]), ctrl_B_0=np.array([3]), dropped_0=np.array([], int))
    (pdir / "cohort.json").write_text(json.dumps({"cohort_sha256": "c"}))
    (pdir / "splits_manifest.json").write_text(json.dumps(
        {"cohort_sha256": "c", "n_splits": 1, "ood_context": True, "cell_line": "H1",
         "benchmark_sha256": "b", "gene_axis_sha256": "g",
         "seen_from": {"benchmark_id": "fakebench", "gene_axis_sha256": AXIS_SHA}}))

    split = ep.load_panel_split(str(pdir), 0, target_sum=2000.0)
    assert split.ctrl.shape == (1, 4)
    assert split.ctrl[0, AXIS.index("E2")] == 0.0          # the gene the panel lacks
    np.testing.assert_allclose(split.ctrl[0, split.model_cols], X[3])
    assert list(split.model_cols) == [AXIS.index(g) for g in panel_axis]

    got = {}

    def ident(ctrl, names):
        got["w"] = ctrl.shape[1]
        return ctrl                                         # emission = input
    monkeypatch.setattr(ep, "assert_representable", lambda *a, **k: None)
    monkeypatch.setattr(ep, "space_diagnostics", lambda *a, **k: {
        "rowsum_ratio": 1.0, "gene_mean_corr": 1.0, "zero_fraction_pred": 0.0,
        "zero_fraction_target": 0.0})
    out = ep.export_split(ident, split, str(tmp_path / "exp"), arm="t",
                          from_target_sum=2000.0)
    pred = np.load(os.path.join(out, "pred.npy"))
    assert got["w"] == 4 and pred.shape == (1, 3)           # model axis in, panel axis out
    np.testing.assert_allclose(pred[0], X[3], rtol=1e-5)
    assert json.load(open(os.path.join(out, "manifest.json")))["ood_context"] is True
