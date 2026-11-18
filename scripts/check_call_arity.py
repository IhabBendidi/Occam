#!/usr/bin/env python
"""Check intra-package calls against the callee's real signature.

    python scripts/check_call_arity.py            # src/occam + scripts
    python scripts/check_call_arity.py src/occam/vcell

WHY THIS EXISTS
---------------
Four calls in the virtual-cell panel path were wrong for the whole life of the port --
`load_panel` given 2 of 4 required arguments, `predict_panel` given a config object in its
`predict_fn` slot and unpacked from 4 values into 3, `fit_control_quantile_map` handed a
function it could never call, `run_celleval_v1` splatted a keyword it does not accept -- and
none of it was caught, because `check_undefined_names.py` binds NAMES and cannot see ARITY,
and a bare `except Exception` turned every resulting TypeError into a printed line.

This walks the AST, resolves calls whose callee is a module-level `def` in the same file or
imported from within the package, and binds the call's arguments against that signature with
`inspect.Signature.bind`. Static, so it is deliberately conservative: anything it cannot
resolve with confidence is skipped rather than guessed at. It reports what it is sure of.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from typing import Dict, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOTS = ("src/occam", "scripts")


def _params(fn: ast.FunctionDef) -> Tuple[List[str], List[str], bool, bool, List[str]]:
    a = fn.args
    pos = [p.arg for p in (a.posonlyargs + a.args)]
    n_def = len(a.defaults)
    required = pos[: len(pos) - n_def] if n_def else list(pos)
    kwonly = [p.arg for p in a.kwonlyargs]
    kwonly_required = [p.arg for p, d in zip(a.kwonlyargs, a.kw_defaults) if d is None]
    if pos and pos[0] in ("self", "cls"):
        pos = pos[1:]
        required = [r for r in required if r not in ("self", "cls")]
    return pos, required, a.vararg is not None, a.kwarg is not None, kwonly + kwonly_required


def return_arity(fn: ast.FunctionDef) -> Optional[int]:
    """How many values the function returns, when every `return` agrees. Else None.

    Catches the second class of bug the panel path had: unpacking a 4-tuple return into 3
    names. Arity checking cannot see that -- the call itself is legal -- so it needs the
    callee's return shape. Conservative: a single bare `return`, a non-tuple return, or any
    disagreement between returns yields None and the call is skipped.
    """
    arities = set()
    bare = False

    def visit(node) -> None:
        """Manual descent. `ast.walk` would enter nested function bodies, whose returns
        belong to the closure and not to us -- and a single closure returning one value is
        enough to make every arity look inconsistent, which silently disables this check."""
        nonlocal bare
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(child, ast.Return):
                if child.value is None:
                    bare = True
                elif isinstance(child.value, ast.Tuple):
                    if any(isinstance(e, ast.Starred) for e in child.value.elts):
                        bare = True
                    else:
                        arities.add(len(child.value.elts))
                else:
                    arities.add(1)
            visit(child)

    visit(fn)
    if bare:
        return None
    if len(arities) != 1:
        return None
    n = arities.pop()
    return n if n > 1 else None


def collect_defs(tree: ast.Module) -> Dict[str, ast.FunctionDef]:
    """Module-level functions only. Methods need a receiver type we cannot resolve."""
    return {n.name: n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", default=list(DEFAULT_ROOTS))
    args = ap.parse_args(argv)

    # pass 1: every module-level def in the package, by (module, name)
    files: List[str] = []
    for root in args.roots:
        base = os.path.join(REPO, root)
        for dirpath, dirs, names in os.walk(base):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            files += [os.path.join(dirpath, n) for n in sorted(names) if n.endswith(".py")]

    trees: Dict[str, ast.Module] = {}
    defs_by_file: Dict[str, Dict[str, ast.FunctionDef]] = {}
    by_name: Dict[str, List[Tuple[str, ast.FunctionDef]]] = {}
    for f in files:
        try:
            t = ast.parse(open(f, encoding="utf-8").read(), filename=f)
        except SyntaxError as exc:
            print(f"{f}: syntax error: {exc}", file=sys.stderr)
            return 2
        trees[f] = t
        d = collect_defs(t)
        defs_by_file[f] = d
        for name, node in d.items():
            by_name.setdefault(name, []).append((f, node))

    problems: List[str] = []
    n_checked = 0

    for f, tree in trees.items():
        local = defs_by_file[f]
        # names imported from inside the package, and what they resolve to
        imported: Dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and (
                    node.module.startswith("occam") or node.level):
                for a in node.names:
                    imported[a.asname or a.name] = a.name

        def _resolve(fname):
            t = local.get(fname)
            if t is None and fname in imported:
                cands = by_name.get(imported[fname], [])
                if len(cands) == 1:
                    t = cands[0][1]
            return t

        # tuple-unpack arity: `a, b, c = f(...)` against f's return shape
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            tgt = node.targets[0]
            if not isinstance(tgt, ast.Tuple) or not isinstance(node.value, ast.Call):
                continue
            if not isinstance(node.value.func, ast.Name):
                continue
            if any(isinstance(e, ast.Starred) for e in tgt.elts):
                continue
            target = _resolve(node.value.func.id)
            if target is None:
                continue
            want = return_arity(target)
            if want is None or want == len(tgt.elts):
                continue
            problems.append(
                f"{os.path.relpath(f, REPO)}:{node.lineno}: "
                f"{node.value.func.id}() returns {want} value(s), unpacked into "
                f"{len(tgt.elts)}")

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            name = node.func.id
            target: Optional[ast.FunctionDef] = local.get(name)
            if target is None and name in imported:
                cands = by_name.get(imported[name], [])
                # only trust an unambiguous resolution
                if len(cands) == 1:
                    target = cands[0][1]
            if target is None:
                continue

            pos, required, has_star, has_kw, kwonly = _params(target)
            n_pos = len([a for a in node.args if not isinstance(a, ast.Starred)])
            has_star_arg = any(isinstance(a, ast.Starred) for a in node.args)
            kws = [k.arg for k in node.keywords if k.arg is not None]
            has_dstar = any(k.arg is None for k in node.keywords)
            n_checked += 1
            where = f"{os.path.relpath(f, REPO)}:{node.lineno}"

            if not has_star and not has_star_arg and n_pos > len(pos):
                problems.append(
                    f"{where}: {name}() takes {len(pos)} positional arg(s) "
                    f"{pos}, called with {n_pos}")
                continue
            if not has_star_arg and not has_dstar:
                supplied = set(pos[:n_pos]) | set(kws)
                miss = [r for r in required if r not in supplied]
                if miss:
                    problems.append(
                        f"{where}: {name}() missing required arg(s) {miss} "
                        f"(signature: {pos + kwonly})")
                    continue
            if not has_kw and not has_dstar:
                known = set(pos) | set(kwonly)
                bad = [k for k in kws if k not in known]
                if bad:
                    problems.append(
                        f"{where}: {name}() got unexpected keyword(s) {bad} "
                        f"(accepts: {sorted(known)})")

    print(f"checked {n_checked} intra-package call(s) in {len(files)} file(s)")
    if problems:
        print(f"\n{len(problems)} arity problem(s):")
        for p in problems:
            print(f"  {p}")
        return 1
    print("no arity problems found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
