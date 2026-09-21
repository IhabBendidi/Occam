"""`from_vcell_checkpoint` returns a TUPLE. Every call site must unpack it.

This killed 135 array tasks. `VirtualCellTrainer.from_vcell_checkpoint` returns
`(trainer, ckpt)`; binding it to a single name yields a tuple that looks like a trainer
until the first attribute access, so the job gets all the way through loading 898 MB of
weights and printing "Loaded vcell checkpoint: epoch=29" before dying on
`trainer.current_epoch`. `scripts/eval_panel.py` had the identical bug, unnoticed.

`scripts/check_call_arity.py` cannot catch this: it resolves module-level `def`s, and this
is a classmethod on an imported class. Hence a targeted test.
"""

from __future__ import annotations

import ast
import inspect
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALLERS = ("scripts/eval_splithalf.py", "scripts/eval_panel.py")


def test_from_vcell_checkpoint_really_returns_two_values():
    pytest.importorskip("torch")
    from occam.vcell.pipeline import VirtualCellTrainer

    src = inspect.getsource(VirtualCellTrainer.from_vcell_checkpoint)
    tree = ast.parse("if 1:\n" + "\n".join("    " + l for l in src.splitlines()))
    returns = [n for n in ast.walk(tree)
               if isinstance(n, ast.Return) and n.value is not None]
    assert returns, "no return statement found"
    tuples = [n for n in returns if isinstance(n.value, ast.Tuple)]
    assert tuples, "expected a tuple return; if this changed, fix the call sites"
    assert all(len(n.value.elts) == 2 for n in tuples), (
        "the contract is (trainer, ckpt); a different arity needs every caller updated")


def test_every_load_frozen_call_site_unpacks_EIGHT_values():
    """`load_frozen` returns 8 values and every caller must unpack 8.

    It returned 7 until `sum_of_singles` (the norman_combo additivity baseline) was added.
    Unpacking 7 from an 8-tuple raises, which is the harmless case; the dangerous one is a
    call site that unpacks a DIFFERENT 8 and silently shifts `cohort` into `grand`. Both are
    caught by counting the names.
    """
    import ast
    import glob
    import os

    expected = 8
    sites = 0
    for path in glob.glob(os.path.join(REPO, "scripts", "*.py")):
        tree = ast.parse(open(path).read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            fn = node.value.func
            name = getattr(fn, "attr", None) or getattr(fn, "id", None)
            if name != "load_frozen":
                continue
            tgt = node.targets[0]
            assert isinstance(tgt, ast.Tuple), \
                f"{os.path.basename(path)}: load_frozen result is not unpacked as a tuple"
            got = len(tgt.elts)
            assert got == expected, (
                f"{os.path.basename(path)}: load_frozen unpacked into {got} names, "
                f"expected {expected}. Adding a return value means updating EVERY caller.")
            sites += 1
    assert sites >= 2, f"found only {sites} load_frozen call site(s); the scan is broken"


def test_every_call_site_unpacks_both_values():
    for rel in CALLERS:
        path = os.path.join(REPO, rel)
        if not os.path.isfile(path):
            continue
        tree = ast.parse(open(path).read())
        found = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            call = node.value
            if not isinstance(call, ast.Call):
                continue
            fn = call.func
            if not (isinstance(fn, ast.Attribute)
                    and fn.attr == "from_vcell_checkpoint"):
                continue
            found += 1
            target = node.targets[0]
            assert isinstance(target, ast.Tuple), (
                f"{rel}:{node.lineno} binds from_vcell_checkpoint to a single name. "
                f"It returns (trainer, ckpt), so this is a tuple pretending to be a "
                f"trainer until the first attribute access.")
            assert len(target.elts) == 2, f"{rel}:{node.lineno} unpacks {len(target.elts)}"
        assert found, f"{rel} no longer calls from_vcell_checkpoint; drop it from CALLERS"
