"""Build a genome-wide human gene symbol <-> Ensembl id map from mygene.info.

Why this exists
---------------
PerturbDiff's pretrained perturbation embeddings (`genept`) and its cell-type
embeddings are keyed by **gene symbol**, while the the upstream benchmark zarr benchmark
identifies perturbations by **Ensembl gene id**. Translating between them needs a
genome-wide table: the datasets' own ``var.parquet`` only covers the 2000
*measured* genes, which overlap the ~2023 *perturbed* genes by only 20%.

Output
------
``configs/metadata/gene_id_map.csv`` with columns::

    gene_symbol,ensembl_gene_id

One row per (symbol, ensembl id) pair. A symbol can map to several Ensembl ids
(and vice versa) -- both are kept, and the consumer decides how to resolve. See
``src/common/gene_id_map.py`` for the loader, which prefers the lowest-numbered
Ensembl id when a symbol is ambiguous, so the mapping is deterministic.

Usage
-----
    uv sync --extra assets
    .venv/bin/python tools/build_gene_id_map.py [--out PATH]

Requires network access to mygene.info.
"""

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "configs" / "metadata" / "gene_id_map.csv"


def fetch_all_human_genes():
    """Every human gene mygene.info knows about, with symbol + Ensembl id."""
    import mygene

    mg = mygene.MyGeneInfo()
    # '__all__' + fetch_all uses the scroll API to page through the whole
    # species index rather than the 10k-hit query cap.
    cursor = mg.query(
        "__all__",
        species="human",
        fields="symbol,ensembl.gene,type_of_gene",
        fetch_all=True,
    )
    return cursor


def ensembl_ids(hit):
    """A hit's Ensembl gene ids; the field is a dict or a list of dicts."""
    ens = hit.get("ensembl")
    if ens is None:
        return []
    if isinstance(ens, dict):
        ens = [ens]
    out = []
    for entry in ens:
        gene = entry.get("gene")
        if isinstance(gene, str):
            out.append(gene)
        elif isinstance(gene, list):
            out.extend(g for g in gene if isinstance(g, str))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("querying mygene.info for all human genes ...", flush=True)
    pairs = set()
    n_hits = n_no_symbol = n_no_ensembl = 0
    types = {}
    for hit in fetch_all_human_genes():
        n_hits += 1
        symbol = hit.get("symbol")
        if not symbol:
            n_no_symbol += 1
            continue
        ids = ensembl_ids(hit)
        if not ids:
            n_no_ensembl += 1
            continue
        types[hit.get("type_of_gene", "unknown")] = (
            types.get(hit.get("type_of_gene", "unknown"), 0) + 1
        )
        for gid in ids:
            pairs.add((str(symbol), str(gid)))

    rows = sorted(pairs)
    with open(args.out, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["gene_symbol", "ensembl_gene_id"])
        writer.writerows(rows)

    symbols = {s for s, _ in rows}
    ens = {e for _, e in rows}
    print(f"\nhits returned          : {n_hits:,}")
    print(f"  skipped, no symbol   : {n_no_symbol:,}")
    print(f"  skipped, no ensembl  : {n_no_ensembl:,}")
    print(f"rows written           : {len(rows):,}  -> {args.out}")
    print(f"  unique symbols       : {len(symbols):,}")
    print(f"  unique ensembl ids   : {len(ens):,}")
    top = sorted(types.items(), key=lambda kv: -kv[1])[:6]
    print("  gene types           : " + ", ".join(f"{k}={v:,}" for k, v in top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
