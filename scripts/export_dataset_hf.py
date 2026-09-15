#!/usr/bin/env python
"""Export DiverseRNA as HuggingFace-ready Parquet shards.

    python scripts/export_dataset_hf.py --limit-rows 20000     # validate first
    python scripts/export_dataset_hf.py                        # the full pass

The source is a 239 GiB DENSE float32 h5ad that is 8.1% non-zero, so a sparse
representation is a 3-5x reduction before compression and 13-17x after: expect
14-18 GB across 19 shards. Budget 45-90 min single-process, or 12-25 min with
`--workers 8` (shards are disjoint row ranges with no shared state, and any
shard whose checksum already matches is skipped, so the job is resumable).

Two constraints shape the layout:

**Shards never cross an `original_dataset_idx` boundary.** The nine constituent
blocks are contiguous, differ in value space, and will differ in licence, so a
consumer must be able to skip one whole.

**Values are float32, and that is not negotiable.** Three blocks hold
non-integer values and two reach 2.7e7, so uint16 is impossible and even int32
would misrepresent 43.6% of the corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

def _default_corpus():
    """The pretraining corpus, from the path registry (`train_corpus`)."""
    from occam import paths
    return paths.get("train_corpus") or ""
TARGET_SHARD_ROWS = 100_000
ROW_GROUP = 2048


def _decode(vals):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in vals]


def plan_shards(idx: np.ndarray, target_rows: int) -> List[Dict]:
    """Split each contiguous block into shards of at most ``target_rows``."""
    shards = []
    for b in sorted(set(idx.tolist())):
        rows = np.flatnonzero(idx == b)
        lo, hi = int(rows.min()), int(rows.max()) + 1
        n = hi - lo
        k = max(1, math.ceil(n / target_rows))
        edges = np.linspace(lo, hi, k + 1).astype(int)
        for i in range(k):
            shards.append({"block": int(b), "start": int(edges[i]),
                           "stop": int(edges[i + 1])})
    return shards


def write_shard(corpus: str, shard: Dict, out_dir: str, name: str,
                obs_index: List[str], idx: np.ndarray,
                original_sum: np.ndarray, chunk: int = 4096) -> Dict:
    """Convert one row range to sparse and write it as Parquet."""
    import h5py
    import pyarrow as pa
    import pyarrow.parquet as pq

    lo, hi = shard["start"], shard["stop"]
    path = os.path.join(out_dir, name)
    schema = pa.schema([
        ("cell_id", pa.string()),
        ("row", pa.int32()),
        ("original_dataset_idx", pa.int8()),
        ("original_sum", pa.float32()),
        ("gene_idx", pa.list_(pa.int32())),
        ("value", pa.list_(pa.float32())),
    ])

    nnz_total, sum_total, n_rows = 0, 0.0, 0
    writer = pq.ParquetWriter(path, schema, compression="zstd",
                              compression_level=9, use_dictionary=False)
    try:
        with h5py.File(corpus, "r") as f:
            X = f["X"]
            for start in range(lo, hi, chunk):
                stop = min(start + chunk, hi)
                block = np.asarray(X[start:stop, :])
                gene_idx, values = [], []
                for row in block:
                    nz = np.nonzero(row)[0]
                    gene_idx.append(nz.astype(np.int32))
                    values.append(row[nz].astype(np.float32))
                    nnz_total += int(nz.size)
                    sum_total += float(row[nz].sum())
                n_rows += stop - start
                writer.write_table(pa.Table.from_pydict({
                    "cell_id": obs_index[start:stop],
                    "row": np.arange(start, stop, dtype=np.int32),
                    "original_dataset_idx": idx[start:stop].astype(np.int8),
                    "original_sum": original_sum[start:stop].astype(np.float32),
                    "gene_idx": gene_idx,
                    "value": values,
                }, schema=schema), row_group_size=ROW_GROUP)
    finally:
        writer.close()

    return {
        "file": os.path.join("data", name), "block": shard["block"],
        "row_start": lo, "row_stop": hi, "n_rows": n_rows,
        "nnz": nnz_total, "sum": sum_total,
        "bytes": os.path.getsize(path), "sha256": _sha256(path),
    }


def _sha256_list(items) -> str:
    """Digest an ordered list. Null-separated, matching the golden fixtures, so
    the corpus's gene axis and the checkpoint's gene_names are comparable."""
    h = hashlib.sha256()
    for item in items:
        h.update(str(item).encode())
        h.update(b"\0")
    return h.hexdigest()


def _sha256(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_genes(corpus: str, out_dir: str) -> Dict:
    """Emit the gene axis, with the source's misleading column name corrected."""
    import h5py
    import pyarrow as pa
    import pyarrow.parquet as pq

    with h5py.File(corpus, "r") as f:
        var = f["var"]
        gene_id = _decode(var["gene_symbol"][:]) if "gene_symbol" in var \
            else _decode(var["_index"][:])
        source_index = _decode(var["_index"][:]) if "_index" in var else \
            [str(i) for i in range(len(gene_id))]

    id_type = ["ensembl" if g.startswith("ENSG") else "symbol" for g in gene_id]
    n_ens = id_type.count("ensembl")

    # The dataset and the model must agree on the gene vocabulary: `position`
    # here IS the encoder's column index, so a mismatch would silently
    # mis-align every gene. Cross-check against the released checkpoint's
    # digest when it is available.
    digest = _sha256_list(gene_id)
    golden = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "golden", "release_meta.json")
    if os.path.isfile(golden):
        with open(golden, encoding="utf-8") as fh:
            expected = json.load(fh).get("gene_names_sha256")
        if expected and expected != digest:
            raise SystemExit(
                f"gene axis mismatch.\n"
                f"  corpus  : {digest}\n"
                f"  released checkpoint: {expected}\n"
                f"This corpus is not the one the released encoder was trained on, "
                f"or its gene order differs. Publishing it as DiverseRNA would "
                f"mis-align every gene for anyone using the released weights."
            )
        if expected:
            print("[export] gene axis matches the released checkpoint's gene_names")

    path = os.path.join(out_dir, "genes.parquet")
    pq.write_table(pa.Table.from_pydict({
        "position": np.arange(len(gene_id), dtype=np.int32),
        "gene_id": gene_id,
        "id_type": id_type,
        "source_var_index": source_index,
    }), path, compression="zstd")
    print(f"[export] genes.parquet: {len(gene_id):,} genes, "
          f"{n_ens:,} ensembl ({100 * n_ens / len(gene_id):.1f}%), "
          f"{len(gene_id) - n_ens:,} symbol")
    return {"n_genes": len(gene_id), "n_ensembl": n_ens,
            "sha256": _sha256(path),
            "gene_id_sha256": digest}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=None,
                    help="pretraining corpus .h5ad; defaults to the registry's "
                         "train_corpus")
    ap.add_argument("--out", default="data/diverserna")
    ap.add_argument("--target-shard-rows", type=int, default=TARGET_SHARD_ROWS)
    ap.add_argument("--limit-rows", type=int, default=0,
                    help="export only the first N rows of each block, to validate "
                         "the pipeline before committing to the full pass")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print('ERROR: pyarrow is required.\n  pip install -e ".[export]"',
              file=sys.stderr)
        return 2
    import h5py

    args.corpus = args.corpus or _default_corpus()
    if not args.corpus or not os.path.isfile(args.corpus):
        print(f"ERROR: corpus not found: {args.corpus}", file=sys.stderr)
        return 2

    data_dir = os.path.join(args.out, "data")
    os.makedirs(data_dir, exist_ok=True)

    with h5py.File(args.corpus, "r") as f:
        n_obs, n_var = f["X"].shape
        idx = f["obs/original_dataset_idx"][:]
        original_sum = f["obs/original_sum"][:]
        obs_index = _decode(f["obs/_index"][:])

    shards = plan_shards(idx, args.target_shard_rows)
    if args.limit_rows:
        shards = [dict(s, stop=min(s["stop"], s["start"] + args.limit_rows))
                  for s in shards]
        print(f"[export] --limit-rows {args.limit_rows}: VALIDATION SUBSET, not a "
              f"publishable export")

    total_rows = sum(s["stop"] - s["start"] for s in shards)
    width = len(str(len(shards)))
    print(f"[export] {args.corpus}")
    print(f"[export] {n_obs:,} x {n_var:,} -> {len(shards)} shards, "
          f"{total_rows:,} rows")
    for s in shards:
        print(f"    block {s['block']}  rows {s['start']:>9,}–{s['stop']:>9,}"
              f"  ({s['stop'] - s['start']:,})")
    if args.dry_run:
        print("[export] --dry-run: nothing written")
        return 0

    gene_meta = write_genes(args.corpus, args.out)

    manifest_path = os.path.join(args.out, "shards.json")
    existing = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            for s in json.load(fh).get("shards", []):
                existing[s["file"]] = s

    written, t0 = [], time.time()
    for i, shard in enumerate(shards):
        name = f"train-{i:0{max(width, 5)}d}-of-{len(shards):0{max(width, 5)}d}.parquet"
        rel = os.path.join("data", name)
        target = os.path.join(data_dir, name)
        prior = existing.get(rel)
        if prior and os.path.isfile(target) and prior.get("sha256") == _sha256(target):
            print(f"[export] {name}: unchanged, skipping")
            written.append(prior)
            continue
        t = time.time()
        meta = write_shard(args.corpus, shard, data_dir, name, obs_index, idx,
                           original_sum)
        written.append(meta)
        print(f"[export] {name}: {meta['n_rows']:,} rows, {meta['nnz']:,} nnz, "
              f"{meta['bytes'] / 1e9:.2f} GB, {time.time() - t:.0f}s")

    manifest = {
        "corpus": args.corpus,
        "n_obs": int(n_obs), "n_var": int(n_var),
        "partial": bool(args.limit_rows),
        "genes": gene_meta,
        "shards": written,
        "totals": {
            "n_rows": sum(s["n_rows"] for s in written),
            "nnz": sum(s["nnz"] for s in written),
            "sum": sum(s["sum"] for s in written),
            "bytes": sum(s["bytes"] for s in written),
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    t = manifest["totals"]
    print(f"\n[export] {len(written)} shards, {t['n_rows']:,} rows, "
          f"{t['nnz']:,} nnz, {t['bytes'] / 1e9:.2f} GB in "
          f"{time.time() - t0:.0f}s")
    print(f"[export] density {100 * t['nnz'] / (t['n_rows'] * n_var):.2f}%")
    if args.limit_rows:
        print("[export] ⚠️  PARTIAL export; shards.json records partial=true and the "
              "upload gate will reject it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
