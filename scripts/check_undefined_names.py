#!/usr/bin/env python
"""Report global names that are used but never bound in a module.

A deliberately simple static check, and a valuable one for this repo: splitting
two 3,000-line modules into ten moved plenty of helpers across file boundaries,
and a name that is only referenced deep inside a rarely-taken branch will pass
every import test and then fail at runtime. This found seven such breakages.

    python scripts/check_undefined_names.py [src/occam]

Not a type checker and not a linter. It binds imports, assignments, defs,
classes, arguments, `global`, `except ... as`, comprehension targets and `with
... as`, then reports Load-context names that were never bound. False positives
are possible (star-imports, some conditional definitions); treat output as a
list to inspect, not a verdict.
"""

from __future__ import annotations

import ast
import builtins
import os
import sys


def _ensure_src_on_path() -> None:
    """Make `occam` importable from a source checkout without PYTHONPATH."""
    import os as _os
    _src = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src")
    if _os.path.isdir(_src) and _src not in sys.path:
        sys.path.insert(0, _src)


_ensure_src_on_path()


def bound_names(tree: ast.AST) -> set:
    bound = set(dir(builtins)) | {
        "__name__", "__file__", "__doc__", "__all__", "__package__", "__spec__",
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.Global):
            bound.update(node.names)
        elif isinstance(node, ast.Nonlocal):
            bound.update(node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    return bound


def scan(path: str):
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    bound = bound_names(tree)
    seen = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in bound:
                seen.setdefault(node.id, node.lineno)
    return sorted(seen.items(), key=lambda kv: kv[1])


def main() -> int:
    roots = sys.argv[1:] or ["src/occam"]
    problems = []
    for root in roots:
        if os.path.isfile(root):
            targets = [root]
        else:
            targets = []
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d != "__pycache__"]
                targets += [os.path.join(dirpath, f)
                            for f in sorted(filenames) if f.endswith(".py")]
        for path in targets:
            for name, lineno in scan(path):
                problems.append(f"{path}:{lineno}: undefined name {name!r}")

    if problems:
        print(f"{len(problems)} potentially undefined name(s):")
        for p in problems:
            print(" ", p)
        return 1
    print("no undefined global names found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
