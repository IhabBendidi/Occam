"""The BMDB benchmark needs Ensembl IDs translated to symbols, or it measures nothing.

The databases are keyed by gene symbols (`CORUM.txt` starts `SUB1,GTF3C3`);
the released vocabulary is 44,560 Ensembl IDs. With no translation the
benchmark reported `valid_pairs: 1`. These tests pin the translation, its
direction detection, and the index alignment the weight matrix depends on.
"""

from __future__ import annotations

import csv
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET = os.path.join(REPO, "assets", "gene_ids", "ensembl_to_symbol.tsv")
CORUM = os.path.join(REPO, "assets", "bmdb_database", "CORUM.txt")

_reason = "gene-ID map not built; run scripts/build_gene_id_map.py"


def _fake_map():
    return {"ENSG1": "AAA", "ENSG2": "BBB", "ENSG3": "CCC"}


def test_to_symbols_returns_indices_aligned_with_the_kept_rows():
    from occam.gene_ids import to_symbols

    symbols, keep = to_symbols(["ENSG1", "nope", "ENSG3"], _fake_map())
    assert symbols == ["AAA", "CCC"]
    # The caller subsets its weight matrix with these, so they must be the
    # positions in the ORIGINAL vocabulary, not a renumbering.
    assert keep == [0, 2]


def test_direction_is_detected_from_the_data_not_from_the_name_shape():
    from occam.gene_ids import detect_id_type

    assert detect_id_type(["ENSG1", "ENSG2"], _fake_map())[0] == "ensembl"
    assert detect_id_type(["AAA", "BBB"], _fake_map())[0] == "symbol"
    # No map available: say so rather than guessing a direction.
    assert detect_id_type(["ENSG1"], {})[0] == "unknown"


def test_a_symbol_vocabulary_is_left_alone():
    """An HVG checkpoint's vocabulary is already symbols; translating would break it."""
    from occam.gene_ids import detect_id_type

    kind, ens_hits, sym_hits = detect_id_type(["AAA", "CCC", "unknown"], _fake_map())
    assert kind == "symbol"
    assert (ens_hits, sym_hits) == (0, 2)


@pytest.mark.skipif(not os.path.isfile(ASSET), reason=_reason)
def test_the_shipped_map_covers_the_released_vocabulary():
    from occam.gene_ids import load_ensembl_to_symbol, map_path

    assert map_path() is not None
    mapping = load_ensembl_to_symbol()
    # 43,825 of 44,728 at build time. A floor, not the exact figure, so
    # rebuilding from a newer source is not a test failure.
    assert len(mapping) > 40000
    assert all(k.startswith("ENS") for k in list(mapping)[:50])


@pytest.mark.skipif(not (os.path.isfile(ASSET) and os.path.isfile(CORUM)), reason=_reason)
def test_translation_is_what_makes_bmdb_pairs_resolvable():
    """The measured 4-vs-3,500 gap is the whole reason this module exists."""
    from occam.gene_ids import load_ensembl_to_symbol

    mapping = load_ensembl_to_symbol()
    with open(CORUM, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    corum = {r["entity1"] for r in rows} | {r["entity2"] for r in rows}

    raw_hits = len(corum & set(mapping.keys()))          # Ensembl IDs vs symbols
    mapped_hits = len(corum & set(mapping.values()))     # symbols vs symbols
    assert raw_hits < 50, "CORUM should not match Ensembl IDs"
    assert mapped_hits > 3000
    assert mapped_hits > 50 * max(raw_hits, 1)
