#!/usr/bin/env python
"""Build the Ensembl-ID -> gene-symbol map that the BMDB benchmark needs.

    python scripts/build_gene_id_map.py --source <h5ad with var['gene_symbol']>

WHY THIS EXISTS
---------------
The released vocabulary is 44,560 Ensembl IDs. The BMDB gene-relationship
databases are keyed by gene SYMBOLS -- `CORUM.txt` starts `SUB1,GTF3C3`. With
no translation between them the benchmark matches almost nothing: measured on
the released checkpoint, CORUM's 3,505 symbols were covered by **4** entries of
the raw vocabulary and by **3,500** once mapped. The benchmark was reporting
`valid_pairs: 1`.

WHAT IT WRITES
--------------
`assets/gene_ids/ensembl_to_symbol.tsv` -- two columns, restricted to the
vocabulary of the released checkpoint, so the asset is small and its coverage
is a measured property of the model we ship rather than of some upstream file.
Plus `PROVENANCE.json`: source path and sha256, source shape, coverage, and the
date, because this asset is DERIVED from someone else's data file and a reader
needs to know from what.

The mapping is used at BENCHMARK time only. It never rewrites the vocabulary a
checkpoint was trained on -- the decoder's output rows are indexed by that
vocabulary, and changing it anywhere else would decouple the two.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import date

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO, "assets", "gene_ids")
DEFAULT_CKPT = os.path.join(
    REPO, "checkpoints", "release", "occam_small_encoder_decoder.pth")


def _sha256(path: str, limit: int = 256 * 1024 * 1024) -> str:
    """sha256 of the first `limit` bytes -- enough to pin a 100 MB source file."""
    h = hashlib.sha256()
    read = 0
    with open(path, "rb") as fh:
        while read < limit:
            chunk = fh.read(min(8 * 1024 * 1024, limit - read))
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True,
                    help="h5ad whose var_names are Ensembl IDs and whose "
                         "var['gene_symbol'] holds the symbols")
    ap.add_argument("--symbol-col", default="gene_symbol")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT,
                    help="restrict the map to this checkpoint's gene_names")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    if not os.path.isfile(args.source):
        ap.error(f"source not found: {args.source}")
    if not os.path.isfile(args.checkpoint):
        ap.error(f"checkpoint not found: {args.checkpoint}")

    import anndata as ad
    import torch

    adata = ad.read_h5ad(args.source, backed="r")
    if args.symbol_col not in adata.var.columns:
        print(f"'{args.symbol_col}' not in var (have {list(adata.var.columns)})",
              file=sys.stderr)
        return 2
    full = {}
    for ens, sym in zip(adata.var_names, adata.var[args.symbol_col]):
        ens, sym = str(ens), str(sym)
        if ens and sym and sym != "nan":
            full[ens] = sym
    source_shape = tuple(adata.shape)
    if getattr(adata, "file", None) is not None:
        adata.file.close()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocab = [str(g) for g in ck["gene_names"]]
    del ck

    pairs = [(g, full[g]) for g in vocab if g in full]
    if not pairs:
        print("no vocabulary entry matched the source's Ensembl IDs; wrong source?",
              file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    tsv = os.path.join(args.out_dir, "ensembl_to_symbol.tsv")
    with open(tsv, "w", encoding="utf-8") as fh:
        fh.write("ensembl_id\tgene_symbol\n")
        for ens, sym in pairs:
            fh.write(f"{ens}\t{sym}\n")

    symbols = [s for _, s in pairs]
    provenance = {
        "built": date.today().isoformat(),
        "builder": "scripts/build_gene_id_map.py",
        "source": os.path.abspath(args.source),
        "source_sha256_first_256mb": _sha256(args.source),
        "source_shape": list(source_shape),
        "source_entries": len(full),
        "checkpoint": os.path.relpath(args.checkpoint, REPO),
        "vocabulary_size": len(vocab),
        "mapped": len(pairs),
        "unmapped": len(vocab) - len(pairs),
        "coverage": round(len(pairs) / len(vocab), 6),
        "distinct_symbols": len(set(symbols)),
        "duplicate_symbols": len(symbols) - len(set(symbols)),
        "used_for": "the BMDB gene-relationship benchmark only; never to "
                    "rewrite a checkpoint's training vocabulary",
    }
    with open(os.path.join(args.out_dir, "PROVENANCE.json"), "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2, sort_keys=True)

    print(f"wrote {tsv}")
    print(f"  {len(pairs):,}/{len(vocab):,} of the released vocabulary mapped "
          f"({100 * len(pairs) / len(vocab):.1f}%)")
    print(f"  {provenance['distinct_symbols']:,} distinct symbols, "
          f"{provenance['duplicate_symbols']:,} duplicated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
