#!/usr/bin/env python
"""Copy competitor CODE (never data) from hooke-tx, and record exactly what was copied.

WHY A SCRIPT AND NOT `cp`
-------------------------
The upstream tree is **141 GB**, of which ~92 GB is untracked *and* unignored there
(`STATE/state/processed_data*/`, `test_norman*/`, `test_replogle*/`, `debugging/`,
`cell-eval-outdir/`). A stray `cp -r` brings that in. This script instead:

* copies only an explicit per-arm allowlist,
* **refuses** any path matching the payload denylist, even if the allowlist names it,
* records every copied file's sha256 plus its upstream path in `competitors/MANIFEST.json`,
  so the port is auditable and `tests/test_competitor_manifest.py` can verify it later,
* prints the size it added, so the gate in the plan (`du -sh competitors/`) is trivial.

The competitors read OUR data through `competitors/shared/occam_data.py`, so nothing
dataset-shaped is copied at all -- not their zarr dumps, not their caches, not their
checkpoints.

    python scripts/copy_competitor_code.py --dry-run
    python scripts/copy_competitor_code.py
    python scripts/copy_competitor_code.py --arm TxPert
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import date
from typing import Dict, List, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = "/mnt/ps/home/CORP/ihab.bendidi/sandbox/bc-hooke/src/hooke-tx/competitors"
DEST = os.path.join(REPO, "competitors")

# Per arm: the exact paths to copy, relative to that arm's upstream directory. Code, configs,
# env specs, launchers, tests and the design docs that explain the knobs -- nothing else.
ALLOWLIST: Dict[str, List[str]] = {
    "STATE": [
        "state/src/state",            # the package, 1.8 M
        "state/pyproject.toml", "state/uv.lock", "state/.python-version",
        # `pyproject.toml` has `readme = "README.md"`; without it `uv sync` fails the
        # editable build with `OSError: Readme file does not exist`. MEASURED 2026-09-17.
        "state/README.md",
        "state/train_state_fewshot.sh", "state/predict_state_fewshot.sh",
        "state/tests/test_zarr_perturbation.py",
        "hepg2.toml", "hepg2_noval.toml", "hepg2_noval_fixedbatch.toml",
        "state/ihabxcell_hepg2_fewshot_zarr.toml",
    ],
    "TxPert": [
        "gspp", "main.py", "configs", "launch", "data", "tests",
        "pyproject.toml", "uv.lock", ".python-version",
        "README_HOOKE_TX.md", "CLAUDE.md",
    ],
    "CellFlow": [
        "hooke_bridge", "train_cellflow.py", "predict_cellflow.py",
        "cellflow_selection.py", "configs", "tests",
        "train_replogle_nadig_fewshot.sh", "predict_replogle_nadig_fewshot.sh",
        "README.md",
        ".python-version",            # at the ARM root, not under cellflow/
        "cellflow/src", "cellflow/pyproject.toml",
        # `cellflow/pyproject.toml` declares `readme = "README.md"` and a
        # PolyForm-Noncommercial licence file, both resolved relative to that
        # directory. Without them `uv sync` fails the editable build with
        # `OSError: Readme file does not exist`. MEASURED 2026-09-17, same class of
        # failure as STATE's.
        "cellflow/README.md", "cellflow/LICENSE",
        # ⚠️ Present on disk despite the upstream PR note calling CellFlow's environment
        # unreconstructible because this file "was gitignored and never committed". Copying
        # it is what makes the arm reproducible at all, so it is deliberately in the list.
        "cellflow/uv.lock",
    ],
    "PerturbDiff": [
        "src", "configs", "launch", "tools",
        "pyproject.toml", "uv.lock", ".python-version",
        "ZARR_BENCHMARK.md", "CLAUDE.md",
    ],
}

# Refused even if an allowlist entry would pull them in. Mirrors the .gitignore rules.
DENY_DIRS = {
    ".venv", "venv", "outputs", "outputs_sweep", "wandb", "lightning_logs", "log", "logs",
    "cache", "checkpoints", "debugging", "cell-eval-outdir", "celleval", "artifacts",
    "indices_cache", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".git",
    "results_real", "results_decoded", "results_generated", "docs", "hub",
}
DENY_DIR_GLOBS = ("processed_data*", "test_norman*", "test_replogle*", "*.egg-info")
DENY_FILE_GLOBS = (
    "*.ckpt", "*.pkl", "*.pt", "*.pth", "*.h5ad", "*.h5", "*.npy", "*.npz", "*.zarr",
    "*.out", "*.err", "*.log", "*.sif", "*.parquet", "*.arrow", "*.feather",
    "*.pyc", "*.pyo", "*.so", "*_agg_results.csv", "*_de.csv",
)

# A copied tree that exceeds this is a sign the denylist missed something.
MAX_ARM_BYTES = 40 * 1024 * 1024


def _denied_dir(name: str) -> bool:
    return name in DENY_DIRS or any(fnmatch.fnmatch(name, g) for g in DENY_DIR_GLOBS)


def _denied_file(name: str) -> bool:
    return any(fnmatch.fnmatch(name, g) for g in DENY_FILE_GLOBS)


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _walk(src: str) -> List[Tuple[str, str]]:
    """(abs_src, rel) for every file under `src` that survives the denylist."""
    if os.path.isfile(src):
        return [] if _denied_file(os.path.basename(src)) else [(src, os.path.basename(src))]
    out: List[Tuple[str, str]] = []
    for dirpath, dirs, names in os.walk(src):
        dirs[:] = sorted(d for d in dirs if not _denied_dir(d))
        for n in sorted(names):
            if _denied_file(n):
                continue
            ap = os.path.join(dirpath, n)
            if os.path.islink(ap):          # never follow symlinks into datasets
                continue
            out.append((ap, os.path.relpath(ap, src)))
    return out


def _upstream_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", UPSTREAM, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", choices=sorted(ALLOWLIST),
                    help="copy only these arms (repeatable); default all")
    ap.add_argument("--upstream", default=UPSTREAM)
    ap.add_argument("--dest", default=DEST)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    arms = args.arm or sorted(ALLOWLIST)
    if not os.path.isdir(args.upstream):
        raise SystemExit(f"upstream not found: {args.upstream}")

    man_path = os.path.join(args.dest, "MANIFEST.json")
    manifest = json.load(open(man_path)) if os.path.isfile(man_path) else {"schema": 1}
    manifest.setdefault("arms", {})
    commit = _upstream_commit()
    print(f"[copy] upstream {args.upstream}\n[copy] commit   {commit}\n")

    grand_files = grand_bytes = 0
    for arm in arms:
        src_root = os.path.join(args.upstream, arm)
        dst_root = os.path.join(args.dest, arm)
        entries: Dict[str, Dict[str, object]] = {}
        n_files = n_bytes = 0
        missing: List[str] = []

        for rel in ALLOWLIST[arm]:
            src = os.path.join(src_root, rel)
            if not os.path.exists(src):
                missing.append(rel)
                continue
            for ap, sub in _walk(src):
                rel_out = rel if os.path.isfile(src) else os.path.join(rel, sub)
                dst = os.path.join(dst_root, rel_out)
                size = os.path.getsize(ap)
                if not args.dry_run:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(ap, dst)
                entries[rel_out] = {
                    "sha256": _sha256(ap) if not args.dry_run else None,
                    "bytes": size,
                    "upstream": os.path.relpath(ap, args.upstream),
                }
                n_files += 1
                n_bytes += size

        flag = ""
        if n_bytes > MAX_ARM_BYTES:
            flag = "  ⚠️ OVER BUDGET -- the denylist probably missed something"
        print(f"[copy] {arm:12s} {n_files:5d} file(s)  {n_bytes / 1e6:8.2f} MB{flag}")
        if missing:
            print(f"       ⚠️ {len(missing)} allowlist path(s) absent upstream: "
                  f"{missing[:4]}")
        if n_bytes > MAX_ARM_BYTES:
            raise SystemExit(
                f"{arm} would copy {n_bytes / 1e6:.1f} MB, over the {MAX_ARM_BYTES / 1e6:.0f} "
                f"MB budget. Refusing: this script exists to stop payload entering the repo.")

        manifest["arms"][arm] = {
            "upstream_dir": os.path.relpath(src_root, args.upstream),
            "upstream_commit": commit,
            "copied": str(date.today()),
            "n_files": n_files,
            "bytes": n_bytes,
            "allowlist": ALLOWLIST[arm],
            "missing_upstream": missing,
            "files": entries,
        }
        grand_files += n_files
        grand_bytes += n_bytes

    manifest["denylist"] = {
        "dirs": sorted(DENY_DIRS), "dir_globs": list(DENY_DIR_GLOBS),
        "file_globs": list(DENY_FILE_GLOBS),
        "note": "refused even when an allowlist entry would pull them in",
    }
    print(f"\n[copy] TOTAL {grand_files} file(s), {grand_bytes / 1e6:.2f} MB")
    if args.dry_run:
        print("[copy] --dry-run: nothing written")
        return 0

    with open(man_path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"[copy] wrote {os.path.relpath(man_path, REPO)}")
    print("[copy] now verify nothing large slipped in:")
    print("    du -sh competitors/ && git status --porcelain -uall competitors/ | wc -l")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
