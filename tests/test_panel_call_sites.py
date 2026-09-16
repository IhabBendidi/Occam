"""Every panel call site must bind against the real signature.

Four calls in this path were wrong for the whole life of the port. `check_call_arity.py`
now catches three of the four classes statically. The fourth cannot be caught by arity --
`predict_panel(model, panel, pert2id, device, config)` is a *legal* five-positional call
that silently binds `config` to the `predict_fn` parameter, which is then invoked as
`config(ct, pidx)`. Only a type checker or a test sees that, so this file asserts the
calls use KEYWORDS past the fourth positional argument, which makes the mistake
unexpressible.
"""

from __future__ import annotations

import ast
import inspect
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITES = (
    os.path.join(REPO, "src", "occam", "vcell", "pipeline.py"),
    os.path.join(REPO, "scripts", "eval_panel.py"),
)


def test_predict_panel_is_never_called_with_a_fifth_positional_argument():
    """The fifth positional slot is `predict_fn`. Passing a config there is legal and wrong."""
    for path in SITES:
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "predict_panel"):
                n_pos = len(node.args)
                assert n_pos <= 4, (
                    f"{os.path.relpath(path, REPO)}:{node.lineno}: predict_panel called "
                    f"with {n_pos} positional args; the 5th is `predict_fn`, so anything "
                    f"passed there is later invoked as predict_fn(ct, pidx)"
                )


def test_fit_control_quantile_map_is_never_handed_predict_panel():
    """`predict_fn` is invoked as predict_fn(ct, pidx); predict_panel takes (model, panel)."""
    for path in SITES:
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "fit_control_quantile_map"):
                names = [a.id for a in node.args if isinstance(a, ast.Name)]
                kw = {k.arg: k.value for k in node.keywords}
                assert "predict_panel" not in names, (
                    f"{os.path.relpath(path, REPO)}:{node.lineno}: predict_panel passed as "
                    f"predict_fn; the None default builds the right closure"
                )
                pf = kw.get("predict_fn")
                if isinstance(pf, ast.Name):
                    assert pf.id != "predict_panel"


def test_the_real_signatures_bind_the_arguments_the_sites_pass():
    """Bind the actual call shapes against the live signatures."""
    pytest.importorskip("torch")
    from occam.vcell.panel.calibrate import fit_control_quantile_map
    from occam.vcell.panel.celleval_v1 import run_celleval_v1
    from occam.vcell.panel.panel_io import load_panel
    from occam.vcell.panel.predict import predict_panel

    # the shapes the fixed sites use
    inspect.signature(load_panel).bind("cfg", ["g"], False, False)
    inspect.signature(predict_panel).bind(
        "model", "panel", {}, "cpu", eval_control="single", batch_size=512, seed=42,
        ctrl_avg_k=1)
    inspect.signature(fit_control_quantile_map).bind(
        "model", "panel", {}, "cpu", max_cells=20000, seed=42, ctrl_avg_k=1)
    # and the kwarg whose absence made the DE reuse dead
    inspect.signature(run_celleval_v1).bind(
        "pred.h5ad", "target.h5ad", "out", num_threads=16, timeout=100,
        de_batch_size=2000, de_real="real_de.csv", allow_discrete=True)

    # the ORIGINAL broken shapes must not bind
    with pytest.raises(TypeError):
        inspect.signature(load_panel).bind("cfg", ["g"])
    with pytest.raises(TypeError):
        inspect.signature(run_celleval_v1).bind("p", "t", "o", de_real_LEGACY="x")


def test_run_celleval_v1_delegates_rather_than_duplicating_the_runner():
    """Two subprocess wrappers is how `de_real` existed in one and not the other."""
    src = open(os.path.join(REPO, "src", "occam", "vcell", "panel",
                            "celleval_v1.py"), encoding="utf-8").read()
    assert "from occam.decoder_train.metrics import run_cell_eval" in src
    assert "subprocess.run" not in src, "the panel must not shell out on its own any more"
