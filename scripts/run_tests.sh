#!/usr/bin/env bash
# Run the test suite against the interpreter that produced the released weights.
#
# The reference pyenv (Python 3.8.19 / torch 2.4.1) has no pytest, and installing
# one into it is not worth perturbing the reference environment, so this uses a
# minimal shim for the handful of pytest features the tests rely on. If you have
# pytest available for that interpreter, `python -m pytest tests/` works too.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
exec python "$HERE/scripts/_run_tests.py" "$@"
