"""Read h5ad `obs`/`var` columns with h5py, resolving anndata's categorical encoding.

Deliberately not anndata: the eval bundle under `data/eval/pert/` was written by a newer
anndata than the training environment can read, so every builder and audit script in this
repo goes through h5py. That reader was written twice before this module existed, which is
how `var['gene_name']`-is-a-categorical-group became a bug in one copy and not the other.
"""

from __future__ import annotations

from typing import List, Optional

__all__ = ["decode", "read_column"]


def decode(values) -> List[str]:
    """Bytes or numpy scalars -> a list of `str`."""
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def read_column(grp, name: str) -> Optional[List[str]]:
    """One `obs`/`var` column as strings, or None if the column is absent.

    Categoricals appear two ways depending on the writer's anndata version: as a subgroup
    holding `codes` + `categories`, or as a plain dataset of codes with the categories in a
    sibling `__categories` group. Both are handled.
    """
    if name not in grp:
        return None
    node = grp[name]
    if isinstance(node, type(grp)):                       # modern categorical subgroup
        codes = node["codes"][:]
        cats = decode(node["categories"][:])
        return [cats[c] if 0 <= c < len(cats) else "" for c in codes]
    if "__categories" in grp and name in grp["__categories"]:   # legacy encoding
        cats = decode(grp["__categories"][name][:])
        return [cats[c] if 0 <= c < len(cats) else "" for c in node[:]]
    return decode(node[:])
