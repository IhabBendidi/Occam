"""Machine-local dataset registry.

Every path that the research tree hardcoded lives here instead, resolved from
``configs/paths.yaml`` (see ``configs/paths.example.yaml``). This is what makes
the repo runnable on a machine other than the one it was written on.

Four separate ``_get_bmdb_dir()`` implementations existed in the source tree
(in the MAE trainer, the decoder trainer, the benchmark runner, and twice in the
inference pipeline). They collapse to :func:`bmdb_dir` here.

Resolution order for a path key:

1. environment variable ``OCCAM_<KEY>`` (e.g. ``OCCAM_TRAIN_CORPUS``)
2. the loaded registry (``configs/paths.yaml``, or whatever ``load_registry`` was given)
3. the packaged default, where one exists (only ``bmdb_dir``)

Keys marked optional in :data:`_OPTIONAL` resolve to ``None`` instead of raising
when unset or missing on disk, and their dependent features degrade with a
warning rather than failing.

No path in this repository is absolute. Registry values are relative by
convention and resolved against the repo root, so a checkout behaves the same
on any machine; an absolute value is accepted if a user sets one.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = [
    "load_registry",
    "registry",
    "resolve",
    "get",
    "require",
    "bmdb_dir",
    "eval_datasets",
    "scib_datasets",
    "kg_graph_dir",
    "PathRegistryError",
]


class PathRegistryError(RuntimeError):
    """A required path is unset, or set but absent from disk."""


# Keys that may legitimately be unset; dependent features must degrade.
_OPTIONAL = frozenset({"bmdb_dir", "kg_graph_dir"})

# Filenames inside ``eval_dir`` / ``scib_dir``. These match the released
# evaluation bundle written by ``scripts/export_eval_data.py``; the registry
# defaults point at ``data/eval/...`` inside the repo, so downloading the bundle
# into place is the whole of the setup.
#
# (name, filename)
_BENCHMARK_FILES = (
    ("nadig_2024_hepg2", "nadig_2024_hepg2.h5ad"),
    ("nadig_2024_jurkat", "nadig_2024_jurkat.h5ad"),
    ("replogle_2022_rpe1_e", "replogle_2022_rpe1_e.h5ad"),
)

# (name, filename, cell_type_col, batch_col)
_SCIB_FILES = (
    ("pancreas_scib", "pancreas_scib.h5ad", "celltype", "batch"),
    ("pbmc_12k", "pbmc_12k.h5ad", "labels", "batch"),
    ("pbmc_68k", "pbmc_68k.h5ad", "celltype", "batch"),
)

_REGISTRY: Optional[Dict[str, Any]] = None
_REGISTRY_SOURCE: Optional[str] = None


def _repo_root() -> Path:
    """Repo root when running from a source checkout, else the package dir."""
    here = Path(__file__).resolve()
    # src/occam/paths.py -> src/occam -> src -> <root>
    candidate = here.parent.parent.parent
    if (candidate / "pyproject.toml").is_file():
        return candidate
    return here.parent


def _default_registry_path() -> Path:
    return _repo_root() / "configs" / "paths.yaml"


def load_registry(path: Optional[str | os.PathLike] = None, *, required: bool = False) -> Dict[str, Any]:
    """Load and cache the path registry.

    With ``path=None`` this reads ``configs/paths.yaml`` if present. A missing
    registry is not an error unless ``required`` -- environment overrides alone
    are enough for some workflows, and ``bmdb_dir`` has a packaged default.
    """
    global _REGISTRY, _REGISTRY_SOURCE

    target = Path(path) if path is not None else _default_registry_path()
    if not target.is_file():
        if required:
            raise PathRegistryError(
                f"path registry not found: {target}\n"
                f"Copy configs/paths.example.yaml to {target} and edit it for this machine."
            )
        _REGISTRY, _REGISTRY_SOURCE = {}, None
        return _REGISTRY

    import yaml  # deferred: keeps `import occam` free of a yaml dependency at import time

    with open(target) as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise PathRegistryError(f"{target} must contain a mapping, got {type(data).__name__}")

    _REGISTRY, _REGISTRY_SOURCE = data, str(target)
    return _REGISTRY


def registry() -> Dict[str, Any]:
    """The loaded registry, loading ``configs/paths.yaml`` on first access."""
    if _REGISTRY is None:
        load_registry()
    assert _REGISTRY is not None
    return _REGISTRY


def _env_key(key: str) -> str:
    return "OCCAM_" + key.upper()


def resolve(value: Optional[str]) -> Optional[str]:
    """Make a registry value absolute, relative to the REPO ROOT.

    Registry paths are deliberately allowed to be relative -- the shipped
    defaults point at ``data/eval/...`` inside the repo, so a user who
    downloads the released evaluation bundle into place needs no configuration
    at all. Resolving against the repo root rather than the working directory
    means it does not matter where the command is run from.
    """
    if value in (None, ""):
        return None
    value = os.path.expanduser(str(value))
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(str(_repo_root()), value))


def get(key: str, default: Optional[str] = None) -> Optional[str]:
    """Resolve ``key`` without checking that it exists on disk."""
    env = os.environ.get(_env_key(key))
    if env:
        return resolve(env)
    value = registry().get(key)
    if value in (None, ""):
        return resolve(default) if default else default
    return resolve(value)


def require(key: str) -> str:
    """Resolve ``key`` and assert it exists on disk.

    Raises :class:`PathRegistryError` naming both the registry key and the
    environment variable that can override it, so the fix is obvious from the
    message alone.
    """
    value = get(key)
    if value is None:
        source = _REGISTRY_SOURCE or str(_default_registry_path()) + " (not found)"
        raise PathRegistryError(
            f"required path '{key}' is not set.\n"
            f"  registry: {source}\n"
            f"  override:  {_env_key(key)}=<path>\n"
            f"See configs/paths.example.yaml."
        )
    if not os.path.exists(value):
        raise PathRegistryError(
            f"path '{key}' is set but does not exist on disk:\n"
            f"  {value}\n"
            f"  registry: {_REGISTRY_SOURCE or 'environment only'}\n"
            f"  override:  {_env_key(key)}=<path>"
        )
    return value


def optional(key: str) -> Optional[str]:
    """Resolve an optional path, returning ``None`` (with a warning) if unusable."""
    value = get(key)
    if value is None:
        return None
    if not os.path.exists(value):
        warnings.warn(
            f"optional path '{key}' is set but missing on disk ({value}); "
            f"the dependent feature will be skipped.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return value


def bmdb_dir() -> str:
    """Directory holding the BMDB gene-relationship databases.

    Order: ``OCCAM_BMDB_DIR`` -> registry ``bmdb_dir`` -> the copy packaged with
    the wheel -> ``assets/bmdb_database`` in a source checkout. The packaged
    default is why the BMDB benchmark works on a bare ``pip install``.
    """
    from occam._compat import legacy_env  # deferred to avoid a cycle at import

    explicit = os.environ.get("OCCAM_BMDB_DIR") or legacy_env("BMDB_DIR")
    if explicit and os.path.isdir(explicit):
        return explicit

    from_registry = resolve(registry().get("bmdb_dir"))
    if from_registry and os.path.isdir(from_registry):
        return from_registry

    for candidate in (
        Path(__file__).resolve().parent / "assets" / "bmdb_database",  # inside the wheel
        _repo_root() / "assets" / "bmdb_database",                     # source checkout
    ):
        if candidate.is_dir():
            return str(candidate)

    raise PathRegistryError(
        "BMDB database directory not found. It ships with the wheel at "
        "occam/assets/bmdb_database; in a source checkout it is assets/bmdb_database. "
        "Override with OCCAM_BMDB_DIR=<dir> or the 'bmdb_dir' registry key."
    )


def eval_datasets(names: Optional[list[str]] = None) -> list[tuple[str, str]]:
    """``(name, path)`` for the encoder benchmark datasets, under ``eval_dir``."""
    root = require("eval_dir")
    wanted = set(names) if names else None
    out = []
    for name, filename in _BENCHMARK_FILES:
        if wanted is not None and name not in wanted:
            continue
        out.append((name, os.path.join(root, filename)))
    if wanted is not None:
        missing = wanted - {n for n, _ in out}
        if missing:
            raise PathRegistryError(f"unknown benchmark dataset(s): {sorted(missing)}")
    return out


def scib_datasets(names: Optional[list[str]] = None) -> list[tuple[str, str, str, str]]:
    """``(name, path, cell_type_col, batch_col)`` for the scIB datasets."""
    root = require("scib_dir")
    wanted = set(names) if names else None
    out = []
    for name, filename, ct_col, batch_col in _SCIB_FILES:
        if wanted is not None and name not in wanted:
            continue
        out.append((name, os.path.join(root, filename), ct_col, batch_col))
    if wanted is not None:
        missing = wanted - {n for n, *_ in out}
        if missing:
            raise PathRegistryError(f"unknown scib dataset(s): {sorted(missing)}")
    return out


def kg_graph_dir() -> str:
    """Perturbation knowledge-graph cache directory."""
    return require("kg_graph_dir")
