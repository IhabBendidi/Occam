"""Expression transforms matching Upstream-Tx experiment knobs."""

from __future__ import annotations

import numpy as np
from anndata import AnnData


def apply_normalize_log1p(
    adata: AnnData,
    *,
    target_sum: float | None = None,
    log1p: bool = False,
    inplace: bool = True,
) -> AnnData:
    """Library-size normalize and/or log1p ``adata.X`` (dense float32).

    Mirrors Upstream ``data.target_sum`` / ``data.log1p`` for public Norman/Replogle
    dumps that store raw (or un-logged) counts in ``X.zarr``.
    """
    out = adata if inplace else adata.copy()
    x = np.asarray(out.X, dtype=np.float32)
    if target_sum is not None:
        if target_sum <= 0:
            raise ValueError(f"target_sum must be > 0, got {target_sum}")
        row_sums = x.sum(axis=1, keepdims=True)
        # Avoid div-by-zero on empty cells; leave them as zeros.
        scale = np.divide(target_sum, row_sums, out=np.zeros_like(row_sums), where=row_sums > 0)
        x = x * scale
    if log1p:
        x = np.log1p(x)
    out.X = np.asarray(x, dtype=np.float32)
    return out
