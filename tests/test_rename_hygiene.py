"""The old names must not survive anywhere, and the repo must be self-contained.

Two properties, both cheap and both easy to regress:

1. No banned token (the research tree's two project names) appears outside a
   small, explicitly enumerated allowlist.
2. Nothing under ``src/`` or ``configs/`` reaches back into the source repo, and
   no module imports outside the package.
"""

from __future__ import annotations

import ast
import os
import re

BANNED = re.compile(r"ihabnet|scmae", re.IGNORECASE)

# Retired claims and dropped subsystems that must not reappear.
RETIRED = {
    # The XCell datasets are dropped; the new panel builder is corpus-agnostic.
    "xcell": re.compile(r"\bxcell\b", re.IGNORECASE),
    # cell-eval2 is deliberately absent; Occam scores with cell-eval v1 only.
    "cell-eval2": re.compile(r"cell-?eval2|celleval2", re.IGNORECASE),
    # Contradicted headline: FINDINGS F41 measures that quadrant at chance.
    "0.5818": re.compile(r"\b0\.5818\b"),
}

ALLOW_MARKER = "noqa: legacy-name"

# The ONLY files permitted to contain a banned token without a line marker.
# _compat.py needs the literal strings to do its job; the provenance doc is
# where the archaeology lives on purpose.
FILE_ALLOWLIST = {
    "src/occam/_compat.py",
    "docs/reproducing_chain_a.md",
}

SEARCH_ROOTS = ("src", "configs", "tests", "scripts", "docs")

SKIP_DIRS = ()

# Files whose job is to READ the tree Occam replaces, so naming it is inherent.
# Distinct from FILE_ALLOWLIST below, which is about *shipped* code.
SKIP_FILES = (
    "tests/test_rename_hygiene.py",   # necessarily contains the patterns it searches for
    "scripts/check_upload_gate.py",   # ditto: it greps the release for them
    "scripts/bootstrap_assets.sh",    # copies the assets out of the source tree
)
SEARCH_FILES = ("README.md", "MODEL_CARD.md", "pyproject.toml")
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".md", ".sh", ".cfg", ".txt", ".json"}


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _iter_text_files():
    root = _repo_root()
    for rel in SEARCH_FILES:
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            yield rel, p
    for sub in SEARCH_ROOTS:
        base = os.path.join(root, sub)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames
                           if d not in {"__pycache__", ".git", ".ruff_cache"}]
            for fn in filenames:
                if os.path.splitext(fn)[1] not in TEXT_SUFFIXES:
                    continue
                p = os.path.join(dirpath, fn)
                rel = os.path.relpath(p, root).replace(os.sep, "/")
                if rel in SKIP_FILES or rel.startswith(SKIP_DIRS):
                    continue
                yield rel, p


def test_no_banned_tokens_outside_allowlist():
    offenders = []
    for rel, path in _iter_text_files():
        if rel.replace(os.sep, "/") in FILE_ALLOWLIST:
            continue
        with open(path, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh, 1):
                if ALLOW_MARKER in line:
                    continue
                if BANNED.search(line):
                    offenders.append(f"{rel}:{i}: {line.strip()[:100]}")
    assert not offenders, "banned project names survive:\n  " + "\n  ".join(offenders)


def test_allowlist_is_exactly_what_we_expect():
    """Guard the guard: an allowlist that silently grows defeats the test."""
    root = _repo_root()
    present = {rel for rel in FILE_ALLOWLIST
               if os.path.exists(os.path.join(root, rel))}
    # Every allowlisted file that exists must actually still need the exemption.
    for rel in present:
        with open(os.path.join(root, rel), encoding="utf-8", errors="replace") as fh:
            assert BANNED.search(fh.read()), (
                f"{rel} is allowlisted but contains no banned token; "
                f"remove it from FILE_ALLOWLIST"
            )
    assert len(FILE_ALLOWLIST) == 2, (
        "the allowlist should stay at two entries: the compat shim and the "
        "provenance doc"
    )


def test_retired_subsystems_and_claims_are_absent():
    """Retired subsystems must not come back in CODE or CONFIG.

    Prose is exempt on purpose. "What is not here" sections and migration notes
    have to name what was dropped in order to be useful, and a reader who
    wonders where the XCell modes went is better served by an answer than by
    silence. What must never reappear is the code, the config surface, or the
    retired numeric claim as a live result.
    """
    offenders = []
    for rel, path in _iter_text_files():
        rel_posix = rel.replace(os.sep, "/")
        if rel_posix in FILE_ALLOWLIST:
            continue
        if not rel_posix.startswith(("src/", "configs/")):
            continue
        with open(path, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh, 1):
                if ALLOW_MARKER in line:
                    continue
                for label, pattern in RETIRED.items():
                    if pattern.search(line):
                        offenders.append(f"{rel}:{i}: [{label}] {line.strip()[:90]}")
    assert not offenders, "retired subsystems/claims reappeared:\n  " + "\n  ".join(offenders)


def test_no_paths_back_into_the_source_repo():
    offenders = []
    root = _repo_root()
    for rel, path in _iter_text_files():
        # The bootstrap script and the provenance doc are the two intended
        # touchpoints back into the research tree.
        if rel.replace(os.sep, "/") in {
            "scripts/bootstrap_assets.sh",
            "scripts/port_file.py", "docs/reproducing_chain_a.md",
        }:
            continue
        if not rel.startswith(("src", "configs")):
            continue
        with open(path, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh, 1):
                if "scMAE-main" in line:  # noqa: legacy-name
                    offenders.append(f"{rel}:{i}: {line.strip()[:100]}")
    assert not offenders, "src/ or configs/ reaches into the source repo:\n  " + "\n  ".join(offenders)


def test_package_imports_stay_inside_the_package():
    """No module may import a sibling top-level module from the old flat layout."""
    root = os.path.join(_repo_root(), "src")
    # Anything importable that is neither stdlib, third-party, nor occam.*
    forbidden_roots = {
        "training_pipeline", "decoder_training_pipeline", "inference_pipeline",
        "reconstruction_pipeline", "external_decoder", "decoder_discriminator",
        "optimized_chunked", "output_manager", "evaluation_utils", "gpl_loss",
        "virtual_cell", "datasets", "transforms", "losses", "model", "util",
    }
    offenders = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, _repo_root())
            with open(path, encoding="utf-8") as fh:
                try:
                    tree = ast.parse(fh.read(), filename=path)
                except SyntaxError as exc:
                    offenders.append(f"{rel}: syntax error: {exc}")
                    continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level:  # relative import, fine
                        continue
                    names = [(node.module or "").split(".")[0]]
                else:
                    continue
                for n in names:
                    if n in forbidden_roots:
                        offenders.append(f"{rel}:{node.lineno}: imports '{n}'")
    assert not offenders, ("modules import the old flat layout:\n  "
                           + "\n  ".join(offenders))
