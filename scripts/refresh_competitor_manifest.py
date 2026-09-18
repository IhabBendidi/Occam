#!/usr/bin/env python
"""Refresh `competitors/MANIFEST.json` to the CURRENT state of the vendored trees.

WHY THIS EXISTS, AND WHY IT IS NOT THE COPIER
----------------------------------------------
`scripts/copy_competitor_code.py` was the one-shot port tool and now REFUSES to run: upstream
is no longer the source of truth for these arms. But the manifest still has a job -- it is the
port's integrity baseline, and `tests/test_competitor_manifest.py` asserts every vendored file
still matches its recorded sha256. So an intended edit to vendored code needs a way to update
the record WITHOUT re-copying anything.

⚠️ RUN THIS ONLY AFTER AN EDIT YOU MEANT. The test exists to catch accidental modification;
refreshing the manifest is how you say "that change was deliberate". Refreshing reflexively
whenever the test complains defeats the check entirely.

    python scripts/refresh_competitor_manifest.py --dry-run
    python scripts/refresh_competitor_manifest.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPET = os.path.join(REPO, "competitors")
MANIFEST = os.path.join(COMPET, "MANIFEST.json")

# Run products and environments: never part of the port's integrity baseline.
DENY_DIRS = {".venv", "__pycache__", "outputs", "wandb", "lightning_logs", "indices_cache",
             "ckpt", "valscore", "runs", ".pytest_cache", "site-packages"}
DENY_EXTS = {".pyc", ".pyo", ".so"}


def _sha256(p: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _walk(root: str):
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in DENY_DIRS]
        for fn in sorted(fns):
            if os.path.splitext(fn)[1] in DENY_EXTS:
                continue
            ap = os.path.join(dp, fn)
            if os.path.islink(ap):
                continue
            yield ap, os.path.relpath(ap, root).replace(os.sep, "/")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if not os.path.isfile(MANIFEST):
        raise SystemExit(f"no manifest at {MANIFEST}; the port has not been done")
    man = json.load(open(MANIFEST))

    added, changed, removed, total = [], [], [], 0
    for arm, rec in man.get("arms", {}).items():
        root = os.path.join(COMPET, arm)
        was = dict(rec.get("files") or {})
        now = {}
        for ap, rel in _walk(root):
            now[rel] = {"sha256": _sha256(ap), "bytes": os.path.getsize(ap)}
            old = was.get(rel)
            if old is None:
                added.append(f"{arm}/{rel}")
            elif old.get("sha256") != now[rel]["sha256"]:
                changed.append(f"{arm}/{rel}")
        removed += [f"{arm}/{r}" for r in was if r not in now]
        rec["files"] = now
        rec["n_files"] = len(now)
        rec["bytes"] = sum(v["bytes"] for v in now.values())
        total += len(now)

    print(f"[manifest] {total} file(s) across {len(man.get('arms', {}))} arm(s)")
    for label, xs in (("changed", changed), ("added", added), ("removed", removed)):
        if xs:
            print(f"[manifest] {len(xs)} {label}: {xs[:8]}")
    if not (changed or added or removed):
        print("[manifest] already up to date")
        return 0
    if args.dry_run:
        print("[manifest] --dry-run: nothing written")
        return 0

    man["schema"] = 2
    with open(MANIFEST, "w") as fh:
        json.dump(man, fh, indent=2, sort_keys=True)
    print(f"[manifest] wrote {os.path.relpath(MANIFEST, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
