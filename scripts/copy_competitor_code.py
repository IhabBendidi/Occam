#!/usr/bin/env python
"""Copy competitor CODE (never data) from the upstream benchmark, and record exactly what was copied.

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
# ⚠️ AN EXTERNAL FILESYSTEM PATH. It keeps the upstream tree's real directory names
# because that is where the code actually lives -- the 2026-09-18 name sweep rewrote
# this string and broke the copier until it was restored. Renaming our own references
# is a docs decision; renaming someone else's directory on disk is not ours to make.
UPSTREAM = ("/mnt/ps/home/CORP/ihab.bendidi/sandbox/bc-hooke/src/"
            "hooke-tx/competitors")   # noqa: legacy-name
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
    ],
    "TxPert": [
        "gspp", "main.py", "configs", "data", "tests",
        "pyproject.toml", "uv.lock", ".python-version",
    ],
    "CellFlow": [
        # ⚠️ `occam_bridge/`, `train_cellflow.py`, `predict_cellflow.py`,
        # `cellflow_selection.py`, `configs/` and `tests/` are NOT listed. They are OURS now:
        # the bridge was renamed from `hooke_bridge` and carries our `occam.py`, the selection
        # callback scores through our cell-eval, and the only config we keep is ours. Listing
        # them re-copies the upstream originals OVER the port -- measured twice on 2026-09-18,
        # each time restoring a tree of files this rename had deleted. The arm's own code is
        # tracked in git; upstream is no longer its source of truth.
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
        "src", "configs", "tools",
        "pyproject.toml", "uv.lock", ".python-version",
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
# Upstream reference material for a benchmark we do not compare against (decision
# 2026-09-18: "we never care about being comparable to the upstream benchmark"). Their launchers, their
# zarr manifests and their design docs described that benchmark; none is reachable from an
# occam config, and together they carried ~200 of the 537 name references. Refused rather
# than dropped from the allowlist alone, so re-adding a tree cannot quietly bring them back.
DENY_FILE_GLOBS = (
    # ⚠️ THESE GLOBS MUST KEEP THE UPSTREAM TREE'S OWN SPELLING. They match FILENAMES on
    # disk in someone else's repo, so they are data, not prose. The 2026-09-18 name sweep
    # rewrote "*hooke*" to "*upstream*" here, which silently stopped denying anything and let
    # a tree of reference files come back on the next copy.
    "*HOOKE*", "*hooke*", "ZARR_BENCHMARK.md",     # noqa: legacy-name
    "*_fewshot.sh", "*_zarr_*.sh", "*_fewshot_*.sh",
    "ihabxcell_*.toml", "hepg2*.toml",
    "*.ckpt", "*.pkl", "*.pt", "*.pth", "*.h5ad", "*.h5", "*.npy", "*.npz", "*.zarr",
    "*.out", "*.err", "*.log", "*.sif", "*.parquet", "*.arrow", "*.feather",
    "*.pyc", "*.pyo", "*.so", "*_agg_results.csv", "*_de.csv",
)

# A copied tree that exceeds this is a sign the denylist missed something.
MAX_ARM_BYTES = 40 * 1024 * 1024

# ⛔ NEVER OVERWRITE THESE. They are upstream files we deliberately ADAPTED so each arm reads
# our data; re-copying them silently reverts the whole port to the pristine upstream version.
#
# MEASURED 2026-09-17: a plain `python scripts/copy_competitor_code.py` reverted all NINE
# wirings in one command, and `tests/test_competitor_manifest.py` then PASSED -- because every
# file matched its upstream sha256 again. The test cannot catch this; only this list can.
#
# Paths are relative to the arm directory. Add a path here the moment it is adapted.
PROTECTED = {
    "TxPert": (
        "gspp/data/datamodule.py",
    ),
    "STATE": (
        "state/src/state/tx/data/zarr_perturbation.py",
    ),
    # CellFlow's adapted files are no longer in its ALLOWLIST at all, so there is nothing for
    # the copier to overwrite and nothing to protect. Empty deliberately, with this note, so
    # the absence reads as a decision rather than an omission.
    "CellFlow": (),
    "PerturbDiff": (
        "src/data/metadata_cache.py",
        "src/data/dataset/dataset_core.py",
        "src/data/data_module/data_module.py",
        "src/data/data_module/data_module_setup.py",
        "src/data/data_module/__init__.py",
        "src/apps/training/training_datamodule_builder.py",
    ),
}


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
    ap.add_argument("--force", action="store_true",
                    help="re-copy anyway. ⛔ This OVERWRITES the port with upstream originals "
                         "for every file not in PROTECTED.")
    args = ap.parse_args(argv)

    arms = args.arm or sorted(ALLOWLIST)
    if not os.path.isdir(args.upstream):
        raise SystemExit(f"upstream not found: {args.upstream}")

    # ⛔ THIS IS A ONE-SHOT PORT TOOL. Upstream is no longer the source of truth for these
    # arms: their code lives in our git, has been adapted in nine files, and had every
    # reference to the upstream benchmark's name removed (2026-09-18). Re-copying restores the
    # originals for everything not in PROTECTED -- which happened twice that day, each time
    # undoing the rename across a tree of files and silently reintroducing names the user had
    # asked to be gone.
    #
    # PROTECTED cannot express "this whole tree is ours now", because the allowlist copies
    # DIRECTORIES (`src`, `configs`, `gspp`). So the refusal is the guard.
    man_exists = os.path.isfile(os.path.join(args.dest, "MANIFEST.json"))
    if man_exists and not (args.force or args.dry_run):
        print(f"[copy] ⛔ REFUSING: {os.path.relpath(os.path.join(args.dest, 'MANIFEST.json'), REPO)} "
              f"already exists, so the port has been done.")
        print(f"[copy] Upstream is no longer the source of truth for these arms -- their code "
              f"is in our git and has been adapted and renamed.")
        print(f"[copy] Re-copying OVERWRITES the port for every file not in PROTECTED.")
        print(f"[copy] Pass --force if that is genuinely what you want, or --dry-run to look.")
        return 0

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
        protected: List[str] = []

        for rel in ALLOWLIST[arm]:
            src = os.path.join(src_root, rel)
            if not os.path.exists(src):
                missing.append(rel)
                continue
            for ap, sub in _walk(src):
                rel_out = rel if os.path.isfile(src) else os.path.join(rel, sub)
                dst = os.path.join(dst_root, rel_out)
                size = os.path.getsize(ap)
                if rel_out in PROTECTED.get(arm, ()):
                    # Adapted on purpose. Skipped, and recorded as skipped, so the manifest
                    # does not claim it matches upstream when it deliberately does not.
                    protected.append(rel_out)
                    entries[rel_out] = {
                        "sha256": None,
                        "bytes": os.path.getsize(dst) if os.path.isfile(dst) else size,
                        "upstream": os.path.relpath(ap, args.upstream),
                        "adapted": True,
                        "note": "adapted for the occam port; NOT overwritten by this script",
                    }
                    continue
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
        if protected:
            print(f"       🔒 {len(protected)} adapted file(s) PRESERVED, not overwritten: "
                  f"{protected}")
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
            "protected_adapted": protected,
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
