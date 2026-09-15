"""Ensembl-ID <-> gene-symbol translation, for gene-relationship benchmarks only.

The released vocabulary is 44,560 Ensembl IDs. The BMDB gene-relationship
databases are keyed by gene SYMBOLS. Without a translation between them a
benchmark over those databases matches almost nothing: measured on the released
checkpoint, CORUM's 3,505 symbols are covered by **4** raw-vocabulary entries
and by **3,500** mapped ones.

⚠️ Scope. This translation belongs to the benchmark and nowhere else. A
decoder's output rows are indexed by the vocabulary it was trained on, so
rewriting that vocabulary anywhere in the training, inference or
reconstruction path would silently decouple the weights from their gene axis.
:func:`to_symbols` therefore returns the kept row indices alongside the
symbols, so the caller subsets the weights with them.

The map ships as an asset (``assets/gene_ids/ensembl_to_symbol.tsv``, built by
``scripts/build_gene_id_map.py`` with its own ``PROVENANCE.json``), so the
benchmark works on a bare ``pip install`` with no configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_CACHE: Optional[Dict[str, str]] = None


def map_path() -> Optional[str]:
    """Locate the mapping asset: env override, then wheel, then source checkout."""
    explicit = os.environ.get("OCCAM_GENE_ID_MAP")
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    here = Path(__file__).resolve().parent
    for candidate in (
        here / "assets" / "gene_ids" / "ensembl_to_symbol.tsv",            # wheel
        here.parent.parent / "assets" / "gene_ids" / "ensembl_to_symbol.tsv",  # checkout
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def load_ensembl_to_symbol(path: Optional[str] = None) -> Dict[str, str]:
    """The Ensembl -> symbol map, or ``{}`` when the asset is absent.

    Absence is not an error: a checkpoint whose vocabulary is already symbols
    needs no map, and the benchmark says so and carries on.
    """
    global _CACHE
    if path is None:
        if _CACHE is not None:
            return _CACHE
        path = map_path()
        if path is None:
            _CACHE = {}
            return _CACHE
    mapping: Dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        header = fh.readline()
        if "\t" not in header:                       # no header; it was a data row
            fh.seek(0)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0] and parts[1]:
                mapping[parts[0]] = parts[1]
    if os.environ.get("OCCAM_GENE_ID_MAP") is None and path == map_path():
        _CACHE = mapping
    return mapping


def detect_id_type(
    gene_names: Sequence[str], mapping: Optional[Dict[str, str]] = None,
) -> Tuple[str, int, int]:
    """Is this vocabulary Ensembl IDs or symbols? Decide from the data, not the shape.

    Counts how many names match the map's Ensembl *keys* versus its symbol
    *values*, so it works for any naming convention or species rather than
    pattern-matching on ``ENSG``. Returns ``(kind, ensembl_hits, symbol_hits)``
    with ``kind`` in ``{"ensembl", "symbol", "unknown"}``; ``"unknown"`` means
    no map was available.
    """
    mapping = load_ensembl_to_symbol() if mapping is None else mapping
    if not mapping:
        return "unknown", 0, 0
    names = [str(g) for g in gene_names]
    ensembl_hits = sum(1 for g in names if g in mapping)
    symbol_values = set(mapping.values())
    symbol_hits = sum(1 for g in names if g in symbol_values)
    if ensembl_hits > symbol_hits and ensembl_hits > 0:
        return "ensembl", ensembl_hits, symbol_hits
    return "symbol", ensembl_hits, symbol_hits


def to_symbols(
    gene_names: Sequence[str], mapping: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], List[int]]:
    """Translate to symbols, returning the symbols and the row indices kept.

    Unmapped names are dropped, so the caller MUST subset its weight matrix
    with the returned indices; the two are only aligned together.
    """
    mapping = load_ensembl_to_symbol() if mapping is None else mapping
    symbols: List[str] = []
    indices: List[int] = []
    for i, name in enumerate(gene_names):
        symbol = mapping.get(str(name))
        if symbol is not None:
            symbols.append(symbol)
            indices.append(i)
    return symbols, indices
