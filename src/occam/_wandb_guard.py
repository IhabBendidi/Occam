"""Guard against a local ``wandb/`` directory shadowing the installed package.

The research tree ran this at **import time** in ``training_pipeline.py``, which
means importing the training module renamed a directory on disk. That is not
acceptable in a library, so it lives here behind an explicit call:
:func:`ensure_no_shadow`, invoked from ``occam.train`` and the CLI entry points.

In Occam this is effectively a no-op -- ``wandb/`` is gitignored and ``WANDB_DIR``
points into ``runs/<name>/`` -- but it is kept because the failure it prevents
(``import wandb`` picking up a run directory) is silent and confusing.

Also carries the wandb 0.21.x ``working_set()`` workaround: upstream catches
only ``KeyError`` and ``UnicodeDecodeError``, and a package with ``None``
metadata (corrupt dist-info on GPFS) raises ``TypeError``.
"""

from __future__ import annotations

import os
import warnings

__all__ = ["ensure_no_shadow", "wandb", "WANDB_AVAILABLE"]

_ENSURED = False


def ensure_no_shadow(*extra_dirs: str) -> None:
    """Rename any ``wandb/`` directory that would shadow the package. Idempotent."""
    global _ENSURED
    if _ENSURED:
        return
    _ENSURED = True

    roots = [os.getcwd(), *extra_dirs]
    for root in roots:
        shadow = os.path.join(root, "wandb")
        if not os.path.isdir(shadow):
            continue
        renamed = os.path.join(root, "wandb_runs")
        if os.path.isdir(renamed):
            continue
        try:
            os.rename(shadow, renamed)
            print(f"Renamed '{shadow}' to '{renamed}' to avoid import shadowing")
        except OSError as exc:
            warnings.warn(f"could not rename {shadow}: {exc}", RuntimeWarning)

    os.environ.setdefault("WANDB_DIR", os.path.join(os.getcwd(), "wandb_runs"))


# Wandb import with fallback + rich diagnostics
try:
    import wandb
    WANDB_AVAILABLE = True
    # Fix wandb 0.21.x bug: working_set() doesn't catch TypeError when a
    # package has None metadata (corrupt dist-info on GPFS).  Upstream only
    # catches KeyError and UnicodeDecodeError — adding TypeError here.
    import wandb.util as _wandb_util
    def _safe_working_set():
        from importlib.metadata import distributions
        for d in distributions():
            try:
                yield _wandb_util.InstalledDistribution(key=d.metadata["Name"], version=d.version)
            except (KeyError, UnicodeDecodeError, TypeError):
                pass
    _wandb_util.working_set = _safe_working_set
except ImportError as e:
    WANDB_AVAILABLE = False
    try:
        import sys, site, importlib.util
        print("==== Wandb Import Diagnostics ====")
        print(f"ImportError: {e}")
        print(f"Python executable: {sys.executable}")
        print(f"CWD: {os.getcwd()}")
        print(f"sys.path[0:10]: {sys.path[:10]}")
        try:
            print(f"ENABLE_USER_SITE: {getattr(site, 'ENABLE_USER_SITE', None)}")
            print(f"USER_SITE: {site.getusersitepackages()}")
        except Exception as se:
            print(f"USER_SITE lookup failed: {se}")
        try:
            print(f"SITE_PACKAGES: {site.getsitepackages()}")
        except Exception as se:
            print(f"SITE_PACKAGES lookup failed: {se}")
        spec = importlib.util.find_spec('wandb')
        print(f"importlib.util.find_spec('wandb'): {spec}")
        print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
        print(f"RANK={os.environ.get('RANK')} LOCAL_RANK={os.environ.get('LOCAL_RANK')} WORLD_SIZE={os.environ.get('WORLD_SIZE')}")
        print("==================================")
    except Exception:
        pass
    warnings.warn(f"wandb somehow not available: {e}. Install with: pip install wandb")
    print(f"wandb somehow not available: {e}. Install with: pip install wandb")

