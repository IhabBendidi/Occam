"""Verify a released checkpoint against the golden fixtures.

    occam-verify
    occam-verify --checkpoint path/to/other.pth

Thin wrapper around ``scripts/verify_checkpoint.py`` so the check is available
from an installed wheel, not only from a source checkout.
"""

from __future__ import annotations

import os
import runpy
import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    script = os.path.join(repo, "scripts", "verify_checkpoint.py")
    if not os.path.isfile(script):
        print(
            "occam-verify needs scripts/verify_checkpoint.py and the golden/ "
            "fixtures, which ship with the source checkout rather than the "
            "wheel.\nClone the repo and run:\n"
            "    python scripts/verify_checkpoint.py",
            file=sys.stderr,
        )
        return 2

    sys.argv = [script] + argv
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
