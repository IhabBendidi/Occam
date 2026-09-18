"""Expression-matrix handles for the the upstream benchmark canonical zarr format.

Counterpart to :class:`src.data.file_handle.H5Store`, exposing the same three
methods the dataset uses (``dataset_file`` / ``close_all`` / ``__getstate__``) so
:class:`~src.data.dataset.dataset_core.H5adSentenceDataset` needs no structural
change.

``feature_loading`` mirrors ``TaskDataset.setup_details`` in the upstream benchmark and
``_Source._ensure_open`` in STATE's zarr loader:

* ``ram``  -- whole matrix resident (``X.npy`` if present, else ``X.zarr``); fastest
* ``npy``  -- ``np.load(..., mmap_mode="r")``; requires ``X.npy``
* ``zarr`` -- lazy ``zarr.open``; per-row chunk decompression, slowest

Two properties matter for correctness and memory:

1. **Materialized arrays live in a module-level cache keyed by (path, mode),
   not on the instance.** One `H5adSentenceDataset` is built per split
   (train / validation / test) over the same files, so per-instance storage
   would hold three copies of every matrix. hepg2 alone is 96616 x 2000 float32
   ~= 0.77 GB, so the four benchmark cell lines would cost ~9 GB instead of ~3.
2. **The cache is never pickled.** Under the default fork start method a
   DataLoader worker inherits the already-populated cache and shares the pages
   copy-on-write; nothing is re-read and nothing is serialized per worker.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)

_VALID_MODES = ("ram", "npy", "zarr")

# (resolved_path, feature_loading) -> materialized array / lazy zarr handle.
# Process-global on purpose; see module docstring.
_ARRAY_CACHE: Dict[tuple, object] = {}
_ARRAY_CACHE_LOCK = threading.RLock()


def _materialize(data_dir: str, feature_loading: str):
    """Return the expression matrix for ``data_dir``, loading it at most once."""
    key = (str(data_dir), feature_loading)
    with _ARRAY_CACHE_LOCK:
        cached = _ARRAY_CACHE.get(key)
        if cached is not None:
            return cached

        base = Path(data_dir)
        zarr_path = base / "X.zarr"
        npy_path = base / "X.npy"

        if feature_loading == "npy":
            if not npy_path.exists():
                raise FileNotFoundError(
                    f"feature_loading='npy' requires {npy_path} to exist."
                )
            logger.info("[%s] mmap X.npy", base.name)
            arr = np.load(npy_path, mmap_mode="r")
        elif feature_loading == "ram":
            if npy_path.exists():
                logger.info("[%s] loading X.npy into RAM", base.name)
                arr = np.load(npy_path)
            else:
                import zarr

                logger.info("[%s] loading X.zarr into RAM", base.name)
                arr = np.asarray(zarr.open(str(zarr_path), mode="r"))
        elif feature_loading == "zarr":
            import zarr

            logger.info("[%s] opening X.zarr lazily", base.name)
            arr = zarr.open(str(zarr_path), mode="r")
        else:
            raise ValueError(
                f"Invalid feature_loading={feature_loading!r}; must be one of {_VALID_MODES}."
            )

        _ARRAY_CACHE[key] = arr
        return arr


class ZarrStore:
    """Path -> expression-matrix handle, with the H5Store method contract."""

    def __init__(
        self,
        dataset_path_map: Dict[str, str],
        feature_loading: str = "ram",
        max_open: int = 16,  # accepted for parity with H5Store; zarr needs no LRU
    ):
        if feature_loading not in _VALID_MODES:
            raise ValueError(
                f"Invalid feature_loading={feature_loading!r}; must be one of {_VALID_MODES}."
            )
        self.dataset_path_map = dict(dataset_path_map)
        self.feature_loading = feature_loading
        self.max_open = int(max_open)
        self._pid = os.getpid()

    def dataset_file(self, name: str):
        """Return the row-indexable expression matrix for dataset ``name``."""
        return _materialize(self.dataset_path_map[name], self.feature_loading)

    def preload(self):
        """Materialize every dataset in the *current* process.

        Call this from the main process before DataLoader workers fork, so the
        matrices are paged in once and shared copy-on-write rather than loaded
        independently by each worker. A no-op for ``feature_loading='zarr'``,
        which is lazy by design.
        """
        if self.feature_loading == "zarr":
            return
        for name in self.dataset_path_map:
            self.dataset_file(name)

    def close_all(self):
        """No-op: arrays are shared process-wide and outlive any one store."""
        return

    def __getstate__(self):
        # Nothing heavy is stored on the instance, but be explicit: the arrays
        # must never be pickled into worker processes.
        state = self.__dict__.copy()
        state["_pid"] = os.getpid()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._pid = os.getpid()


def clear_array_cache():
    """Drop all materialized matrices (tests / long-lived interactive sessions)."""
    with _ARRAY_CACHE_LOCK:
        _ARRAY_CACHE.clear()


__all__ = ["ZarrStore", "clear_array_cache"]
