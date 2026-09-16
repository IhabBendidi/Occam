#!/usr/bin/env python
"""Minimal test runner for environments without pytest.

Collects ``Test*`` classes and ``test_*`` methods from ``tests/test_*.py`` and
runs them, providing the small subset of the pytest API the suite uses
(``pytest.approx``, ``pytest.raises``, ``pytest.mark.skipif``). Prefer real
pytest where it is available; this exists so the suite can run under the
reference interpreter, which is where correctness actually matters.
"""

from __future__ import annotations

import importlib.util
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
import traceback


class _Approx:
    def __init__(self, value, rel=None, abs=None):  # noqa: A002 -- pytest's own name
        self.value = value
        self.rel = 1e-6 if rel is None else rel
        self.abs = 1e-9 if abs is None else abs

    def __eq__(self, other):
        return abs(other - self.value) <= max(self.abs, self.rel * abs(self.value))

    def __repr__(self):
        return f"approx({self.value!r})"


class _Raises:
    """``pytest.raises``, including ``.value`` -- tests assert on the message.

    A refusal is only worth testing if the message tells the reader what to do
    instead, so the caught exception has to be reachable. ``match=`` is not
    supported: nothing uses it, and a shim that silently accepts an argument it
    ignores is worse than one that raises TypeError.
    """

    def __init__(self, exc):
        self.exc = exc
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"DID NOT RAISE {self.exc}")
        if not issubclass(et, self.exc):
            return False
        self.value = ev
        return True


class _Mark:
    @staticmethod
    def skipif(condition, reason=""):
        def deco(fn):
            fn.__skip__ = (bool(condition), reason)
            return fn
        return deco

    @staticmethod
    def slow(fn):
        return fn


class _Skipped(Exception):
    pass


def _fixture(*args, **kwargs):
    """Minimal @pytest.fixture. Supports scope="module" and generator teardown."""
    def deco(fn):
        fn.__fixture__ = kwargs.get("scope", "function")
        return fn
    # Bare @pytest.fixture (no parentheses)
    if args and callable(args[0]) and not kwargs:
        return deco(args[0])
    return deco


class _Pytest:
    approx = staticmethod(_Approx)
    raises = staticmethod(_Raises)
    mark = _Mark
    fixture = staticmethod(_fixture)

    @staticmethod
    def skip(reason=""):
        raise _Skipped(reason)

    @staticmethod
    def importorskip(name, reason=""):
        import importlib
        try:
            return importlib.import_module(name)
        except ImportError:
            raise _Skipped(reason or f"{name} not installed")

    @staticmethod
    def fail(reason=""):
        raise AssertionError(reason)


class _MonkeyPatch:
    """``monkeypatch.setattr`` / ``setitem`` / ``setenv``, with a working undo.

    The undo is the reason this exists rather than a bare ``setattr`` in the test: a stub
    for ``run_celleval_v1`` that survived into the next test file would make every later
    cell-eval assertion meaningless while still printing PASS.
    """

    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value):
        had = hasattr(target, name)
        old = getattr(target, name, None)
        setattr(target, name, value)
        self._undo.append(
            (lambda: setattr(target, name, old)) if had
            else (lambda: delattr(target, name)))

    def delattr(self, target, name):
        old = getattr(target, name)
        delattr(target, name)
        self._undo.append(lambda: setattr(target, name, old))

    def setitem(self, mapping, key, value):
        had = key in mapping
        old = mapping.get(key)
        mapping[key] = value
        self._undo.append((lambda: mapping.__setitem__(key, old)) if had
                          else (lambda: mapping.pop(key, None)))

    def setenv(self, name, value):
        self.setitem(os.environ, name, str(value))

    def undo(self):
        while self._undo:
            self._undo.pop()()


def _builtin_fixtures(label):
    """``tmp_path`` and ``monkeypatch``, fresh per test and torn down after it.

    Not routed through the module fixture cache: ``tmp_path`` must be a new directory for
    every test, and ``monkeypatch`` must be undone between tests.
    """
    import pathlib
    import tempfile

    state = {}

    def make(name):
        if name == "tmp_path":
            d = tempfile.mkdtemp(prefix="occam_test_")
            state["tmp_path"] = d
            return pathlib.Path(d)
        if name == "monkeypatch":
            mp = _MonkeyPatch()
            state["monkeypatch"] = mp
            return mp
        raise KeyError(name)

    def teardown():
        mp = state.get("monkeypatch")
        if mp is not None:
            mp.undo()
        d = state.get("tmp_path")
        if d is not None:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    return make, teardown


_BUILTIN_FIXTURES = ("tmp_path", "monkeypatch")


def main() -> int:
    sys.modules.setdefault("pytest", _Pytest())

    # Static checks first: they are seconds, they need no fixtures, and one of them covers a
    # class of bug (call arity) that four broken panel call sites proved the test suite
    # cannot see on its own.
    import subprocess as _sp
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for _checker in ("check_undefined_names.py", "check_call_arity.py"):
        _rc = _sp.call([sys.executable, os.path.join(_repo, "scripts", _checker)])
        if _rc:
            print(f"\n{_checker} FAILED")
            return _rc

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tests_dir = os.path.join(repo, "tests")
    selected = sys.argv[1:]

    files = sorted(
        f for f in os.listdir(tests_dir)
        if f.startswith("test_") and f.endswith(".py")
        and (not selected or any(s in f for s in selected))
    )
    if not files:
        print("no test files matched")
        return 1

    passed = failed = skipped = 0
    failures = []

    for fname in files:
        path = os.path.join(tests_dir, fname)
        spec = importlib.util.spec_from_file_location(fname[:-3], path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        print(f"\n{fname}")
        try:
            spec.loader.exec_module(mod)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            failures.append((fname, "<import>", traceback.format_exc()))
            print(f"  FAIL <import>: {type(exc).__name__}: {exc}")
            continue

        fixtures = {n: getattr(mod, n) for n in dir(mod)
                    if hasattr(getattr(mod, n), "__fixture__")}
        cache, finalizers = {}, []

        def resolve(name):
            """Instantiate a fixture, memoised, honouring generator teardown."""
            if name in cache:
                return cache[name]
            fn = fixtures[name]
            import inspect
            kwargs = {p: resolve(p) for p in inspect.signature(fn).parameters
                      if p in fixtures}
            value = fn(**kwargs)
            if inspect.isgenerator(value):
                gen, value = value, next(value)
                finalizers.append(gen)
            cache[name] = value
            return value

        def bind(fn, label=""):
            """Wrap a test so its fixture parameters are supplied."""
            import inspect
            try:
                names = list(inspect.signature(fn).parameters)
            except (TypeError, ValueError):
                names = []
            params = [p for p in names if p in fixtures]
            builtins_needed = [p for p in names
                               if p in _BUILTIN_FIXTURES and p not in fixtures]
            if not params and not builtins_needed:
                return fn

            def run():
                make, teardown = _builtin_fixtures(label)
                kwargs = {p: resolve(p) for p in params}
                kwargs.update({p: make(p) for p in builtins_needed})
                try:
                    return fn(**kwargs)
                finally:
                    teardown()
            return run

        cases = []
        for name in dir(mod):
            obj = getattr(mod, name)
            if hasattr(obj, "__fixture__"):
                continue
            if name.startswith("Test") and isinstance(obj, type):
                inst = obj()
                for m in sorted(dir(inst)):
                    if m.startswith("test_"):
                        cases.append((f"{name}.{m}", bind(getattr(inst, m), f"{name}.{m}")))
            elif name.startswith("test_") and callable(obj):
                cases.append((name, bind(obj, name)))

        for label, fn in cases:
            skip = getattr(fn, "__skip__", None)
            if skip and skip[0]:
                skipped += 1
                print(f"  SKIP {label}  ({skip[1]})")
                continue
            try:
                fn()
                passed += 1
                print(f"  PASS {label}")
            except _Skipped as exc:
                skipped += 1
                print(f"  SKIP {label}  ({exc})")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                failures.append((fname, label, traceback.format_exc()))
                print(f"  FAIL {label}: {type(exc).__name__}: {exc}")

        for gen in finalizers:
            try:
                next(gen)
            except StopIteration:
                pass
            except Exception as exc:  # noqa: BLE001
                print(f"  WARN fixture teardown: {type(exc).__name__}: {exc}")

    print(f"\n{'=' * 70}")
    print(f"{passed} passed, {failed} failed, {skipped} skipped")
    for fname, label, tb in failures:
        print(f"\n--- {fname}::{label} ---\n{tb}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
