"""Dataset readers and the chunked HDF5 loader."""

from __future__ import annotations

__all__ = ["OccamAdataDataset", "OccamChunkedDataset", "OccamChunkedReader"]


def __getattr__(name):  # lazy: keeps `import occam.data` cheap
    if name in ("OccamAdataDataset", "OccamChunkedDataset"):
        from occam.data import datasets as _d
        return getattr(_d, name)
    if name == "OccamChunkedReader":
        from occam.data.chunked import OptimizedChunkedDataset
        return OptimizedChunkedDataset
    raise AttributeError(f"module 'occam.data' has no attribute {name!r}")
