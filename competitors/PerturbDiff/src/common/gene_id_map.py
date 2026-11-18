"""Ensembl id <-> gene symbol translation.

PerturbDiff's pretrained embedding tables are keyed by **gene symbol**:

* ``cov_encoding.replogle_gene_embedding_path`` (genept perturbation embeddings)
* ``cov_encoding.gene_embedding_path`` (per-gene embeddings, when
  ``model.use_gene_embedding=true``)

The the upstream benchmark zarr benchmark identifies perturbations by **Ensembl gene id**, so
those lookups miss entirely without a translation. The datasets' own
``var.parquet`` is not enough: it covers the 2000 *measured* genes, which overlap
the ~2023 *perturbed* genes by only 20%. This module reads the genome-wide table
built by ``tools/build_gene_id_map.py`` from mygene.info (53,653 pairs; 99.9% of
the benchmark's perturbed ids resolve).

Two tables are chained, earlier wins:

1. ``configs/metadata/vocab_id2name.csv`` — upstream PerturbDiff's own id->name
   table (25,133 rows, columns ``id,name``), shipped with its data release.
2. ``configs/metadata/gene_id_map.csv`` — the mygene.info table (53,653 rows),
   used for anything upstream's table lacks.

Upstream's goes first because the embedding tables were built against it, and
symbol naming has moved on since. Measured against the real genept pickle on the
benchmark's 2023 perturbations:

===========================  ==================
mapping                      resolves in genept
===========================  ==================
mygene only                  1965  (97.1%)
upstream vocab only          1979  (97.8%)
**chained (this module)**    **1980  (97.9%)**
===========================  ==================

mygene returns current HGNC symbols (``TIMCC``, ``PRP4K``, ``VMA22``) where
genept was built with the older ones (``FAM136A``, ``PRPF4B``, ``CCDC115``);
conversely mygene alone resolves ``CASP8AP2``, which upstream's table lacks.

Ambiguity is resolved deterministically so a run is reproducible: when one
Ensembl id carries several symbols the alphabetically first is used, and when one
symbol carries several Ensembl ids the lowest-numbered id is used. Neither case
occurs for the benchmark's perturbation set, but 3835 of 45893 symbols are
many-to-one in the reverse direction genome-wide.
"""

import csv
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_METADATA_DIR = Path(__file__).resolve().parents[2] / "configs" / "metadata"
# Ordered by precedence: upstream's table first, mygene as the fallback.
DEFAULT_MAP_PATHS = (
    _METADATA_DIR / "vocab_id2name.csv",
    _METADATA_DIR / "gene_id_map.csv",
)
DEFAULT_MAP_PATH = _METADATA_DIR / "gene_id_map.csv"  # kept for callers that name it


def _ensembl_sort_key(gene_id: str):
    """Sort ENSG ids by their numeric part so the choice is stable."""
    digits = "".join(ch for ch in gene_id if ch.isdigit())
    return (int(digits) if digits else 1 << 62, gene_id)


def _read_table(csv_path: Path) -> List[Tuple[str, str]]:
    """Read one mapping CSV as (symbol, ensembl_id) pairs.

    Accepts both schemas in use: ``gene_symbol,ensembl_gene_id`` (ours) and
    upstream's ``id,name``. Upstream's file is kept verbatim so its provenance
    is obvious.
    """
    with open(csv_path, newline="") as fin:
        reader = csv.DictReader(fin)
        fields = set(reader.fieldnames or ())
        if {"gene_symbol", "ensembl_gene_id"} <= fields:
            return [(r["gene_symbol"], r["ensembl_gene_id"]) for r in reader]
        if {"id", "name"} <= fields:
            return [(r["name"], r["id"]) for r in reader]
        raise ValueError(
            f"{csv_path} has unrecognised columns {sorted(fields)}; expected "
            f"gene_symbol/ensembl_gene_id or id/name."
        )


@lru_cache(maxsize=4)
def load_gene_id_map(path: Optional[str] = None) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return ``(ensembl_to_symbol, symbol_to_ensembl)``.

    :param path: a single CSV to use instead of the chained defaults.

    Cached: the tables are ~80k rows combined and several call sites want them.
    """
    if path:
        paths: Sequence[Path] = (Path(path),)
    else:
        paths = tuple(p for p in DEFAULT_MAP_PATHS if p.exists())
        if not paths:
            raise FileNotFoundError(
                f"no gene id map found under {_METADATA_DIR}. Regenerate with "
                f"`uv sync --extra assets && .venv/bin/python tools/build_gene_id_map.py`."
            )
    for p in paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"gene id map not found at {p}.")

    ensembl_to_symbol: Dict[str, str] = {}
    symbol_to_ensembl: Dict[str, str] = {}
    for csv_path in paths:
        ens_to_syms: Dict[str, List[str]] = {}
        sym_to_ens: Dict[str, List[str]] = {}
        for symbol, gene_id in _read_table(Path(csv_path)):
            ens_to_syms.setdefault(gene_id, []).append(symbol)
            sym_to_ens.setdefault(symbol, []).append(gene_id)
        added = 0
        for k, v in ens_to_syms.items():
            if k not in ensembl_to_symbol:  # earlier table wins
                ensembl_to_symbol[k] = sorted(v)[0]
                added += 1
        for k, v in sym_to_ens.items():
            if k not in symbol_to_ensembl:
                symbol_to_ensembl[k] = sorted(v, key=_ensembl_sort_key)[0]
        logger.info(
            "gene id map: +%d ensembl ids from %s (total %d)",
            added, Path(csv_path).name, len(ensembl_to_symbol),
        )
    return ensembl_to_symbol, symbol_to_ensembl


def to_symbol(gene_id: str, path: Optional[str] = None) -> Optional[str]:
    """Ensembl id -> symbol, or None. Composite ids ("A+B") are not handled here."""
    ensembl_to_symbol, _ = load_gene_id_map(path)
    return ensembl_to_symbol.get(str(gene_id))


def pert_to_symbol(pert: str, path: Optional[str] = None) -> Optional[str]:
    """Translate a perturbation string as produced by the zarr metadata cache.

    Handles the ``"A+B"`` multi-gene form by translating each id and rejoining in
    the original order. Returns None if *any* component is unmappable, so callers
    can count a partial translation as a miss rather than silently using half a
    perturbation.
    """
    parts = str(pert).split("+")
    symbols = [to_symbol(p, path) for p in parts]
    if any(s is None for s in symbols):
        return None
    return "+".join(symbols)


def translate_perts(
    perts: Iterable[str], path: Optional[str] = None
) -> Tuple[Dict[str, str], List[str]]:
    """Translate many perturbations at once.

    :returns: ``(mapping, unmapped)`` -- ``mapping`` covers only what resolved.
    """
    mapping, unmapped = {}, []
    for pert in perts:
        symbol = pert_to_symbol(pert, path)
        if symbol is None:
            unmapped.append(pert)
        else:
            mapping[pert] = symbol
    return mapping, unmapped


__all__ = [
    "DEFAULT_MAP_PATH",
    "load_gene_id_map",
    "to_symbol",
    "pert_to_symbol",
    "translate_perts",
]
