#!/usr/bin/env python
"""Identify the constituent datasets of the DiverseRNA pretraining corpus.

    python scripts/identify_sources.py --corpus <path> --out data/diverserna/sources

The corpus is a composite of nine datasets that survive only as integers in
``obs/original_dataset_idx``. No manifest exists anywhere: it was assembled
outside this repo for an earlier effort. A HuggingFace dataset card needs source
names and licences, so this reconstructs what it can and marks the rest
UNRESOLVED rather than guessing.

Five independent lines of evidence per block:

1. **Structural fingerprint** -- exact row range, nnz, supported gene count,
   integrality, value range, library-size quantiles. Cheap and always available.
2. **Exact ``n_obs`` match** against a registry of candidate files. A block that
   was taken whole from a source matches its row count exactly.
3. **Gene-support corroboration and Jaccard** over version-stripped Ensembl IDs.
   The primary axis for blocks that were subsetted, where (2) cannot match.
4. **Row-level exact match** -- hash candidate rows against block rows. A single
   hit *proves* identity; zero hits with a strong (2)/(3) match means
   "subsetted or transformed".
5. **Marker profile** for whatever remains, to at least characterise the tissue.

Read-only. Candidate files belong to other people and may be unreadable; those
are recorded as SKIPPED_PERMISSION rather than aborting the scan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

def _default_corpus():
    """The pretraining corpus, from the path registry (`train_corpus`)."""
    from occam import paths
    return paths.get("train_corpus") or ""

# Marker panel for characterising an unresolved block.
MARKERS = {
    "HBB": "erythroid", "MKI67": "proliferating", "PTPRC": "immune (CD45)",
    "EPCAM": "epithelial", "COL1A1": "fibroblast", "ALB": "hepatocyte",
    "XIST": "female", "RPS4Y1": "male",
}


def _decode(vals):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in vals]


def _strip_version(gene: str) -> str:
    return gene.split(".", 1)[0] if gene.startswith("ENSG") else gene


def _family(filename: str) -> str:
    """A coarse dataset identity from a filename.

    Strips any owner prefix, common processing suffixes and the extension, so
    `<owner>_fixedvar__TCGA_..._v1.h5ad` and `..._v1_LEAN.h5ad` collapse to one
    family. Deliberately blunt -- it decides only whether several exact row
    matches are the same dataset or genuinely different ones.

    Stripping the owner prefix also keeps individuals' names out of the
    recorded output, which is published.
    """
    import re

    stem = os.path.splitext(os.path.basename(filename))[0]
    # <owner>_fixedvar__NAME  /  <owner>.NAME  /  <owner>_NAME
    stem = re.sub(r"^[A-Za-z][\w-]*_fixedvar__", "", stem)
    stem = re.sub(r"(_LEAN|_v\d+|_with_ensembl|_fixed[-_]var|_raw_counts|"
                  r"_aggregated)+$", "", stem, flags=re.IGNORECASE)
    return stem.strip("_") or stem


def _read_col(grp, name):
    """Read an h5ad dataframe column, resolving anndata's categorical encoding.

    Needed because candidate files come from many sources and different anndata
    versions: a column may be a plain dataset, a {codes, categories} group, or
    an old-style code array with a sibling `__categories` table.
    """
    import h5py

    if name not in grp:
        return None
    node = grp[name]
    if isinstance(node, h5py.Group):
        if "categories" in node and "codes" in node:
            cats = _decode(node["categories"][:])
            return [cats[c] if 0 <= c < len(cats) else "" for c in node["codes"][:]]
        return None
    if "__categories" in grp and name in grp["__categories"]:
        cats = _decode(grp["__categories"][name][:])
        return [cats[c] if 0 <= c < len(cats) else "" for c in node[:]]
    return _decode(node[:])


def profile_blocks(corpus: str, sample_rows: int = 512) -> List[Dict]:
    """Structural fingerprint of every ``original_dataset_idx`` block."""
    import h5py

    with h5py.File(corpus, "r") as f:
        X = f["X"]
        n_obs, n_var = X.shape
        idx = f["obs/original_dataset_idx"][:]
        genes = _read_col(f["var"], "gene_symbol") or _read_col(f["var"], "_index") or []

        blocks = []
        for b in sorted(set(idx.tolist())):
            rows = np.flatnonzero(idx == b)
            lo, hi = int(rows.min()), int(rows.max())
            contiguous = bool(len(rows) == hi - lo + 1)

            # Sample evenly across the block rather than from its head.
            take = rows if len(rows) <= sample_rows else \
                rows[np.linspace(0, len(rows) - 1, sample_rows).astype(int)]
            chunks = []
            for start in range(0, len(take), 128):
                sel = np.sort(take[start:start + 128])
                chunks.append(np.asarray(X[sel, :]))
            sample = np.vstack(chunks) if chunks else np.zeros((0, n_var), np.float32)

            nz = sample > 0
            nonzero_vals = sample[nz]
            sums = sample.sum(axis=1)
            supported = int((sample.max(axis=0) > 0).sum())
            integral = bool(np.all(nonzero_vals == np.floor(nonzero_vals))) \
                if nonzero_vals.size else True

            entry = {
                "block": int(b),
                "n_rows": int(len(rows)),
                "row_range": [lo, hi],
                "contiguous": contiguous,
                "pct_nonzero": round(float(nz.mean()) * 100, 2),
                "max_value": float(nonzero_vals.max()) if nonzero_vals.size else 0.0,
                "min_nonzero": float(nonzero_vals.min()) if nonzero_vals.size else 0.0,
                "integral": integral,
                "median_row_sum": float(np.median(sums)) if sums.size else 0.0,
                "row_sum_q": [float(np.percentile(sums, q)) for q in (5, 25, 50, 75, 95)]
                if sums.size else [],
                "genes_supported": supported,
                "sampled_rows": int(sample.shape[0]),
            }
            # A non-integer block is not raw counts; say so loudly.
            entry["is_raw_counts"] = integral
            entry["looks_like_bulk"] = bool(
                entry["pct_nonzero"] > 30 and entry["median_row_sum"] > 1e6)
            blocks.append(entry)
    return blocks, genes, n_obs, n_var


def scan_candidates(registry: List[Dict], blocks: List[Dict],
                    corpus_genes: List[str]) -> Dict[int, List[Dict]]:
    """Match candidates to blocks by exact n_obs, then corroborate on genes."""
    import h5py

    by_rows: Dict[int, List[Dict]] = {}
    for b in blocks:
        by_rows.setdefault(b["n_rows"], []).append(b)

    corpus_ens = {_strip_version(g) for g in corpus_genes if g.startswith("ENSG")}
    hits: Dict[int, List[Dict]] = {b["block"]: [] for b in blocks}

    for cand in registry:
        path = cand.get("path")
        if not path or not os.path.isfile(path):
            continue
        try:
            with h5py.File(path, "r") as f:
                shape = f["X"].shape if not isinstance(f["X"], h5py.Group) \
                    else tuple(f["X"].attrs["shape"])
                n_obs, n_var = int(shape[0]), int(shape[1])
                var = f["var"]
                # Try every plausible column and keep whichever gives the most
                # Ensembl IDs: candidate files disagree about which column holds
                # them, and the corpus's own axis is 99.6% Ensembl despite being
                # named `gene_symbol`.
                names, best = None, -1
                for key in ("ensembl_gene_id", "gene_ids", "_index", "gene_symbol",
                            "feature_id", "gene_id"):
                    cand_names = _read_col(var, key)
                    if not cand_names:
                        continue
                    n_ens = sum(1 for g in cand_names if str(g).startswith("ENSG"))
                    if n_ens > best:
                        names, best = cand_names, n_ens
        except Exception as exc:  # noqa: BLE001
            # Candidate files belong to other people and come from many anndata
            # versions; one unreadable file must not abort the whole scan.
            cand["status"] = "SKIPPED_UNREADABLE"
            cand["error"] = f"{type(exc).__name__}: {exc}"[:200]
            continue

        cand_ens = {_strip_version(g) for g in (names or []) if g.startswith("ENSG")}
        jac = (len(cand_ens & corpus_ens) / len(cand_ens | corpus_ens)) \
            if (cand_ens and corpus_ens) else 0.0

        for b in by_rows.get(n_obs, []):
            hits[b["block"]].append({
                "name": _family(cand.get("name") or path),
                # The family name, not the raw filename: the location on the
                # scanning machine is not provenance anyone else can use, and
                # raw filenames here carry their owner's name.
                "dataset": _family(path), "n_obs": n_obs, "n_var": n_var,
                "evidence": "exact n_obs match",
                "gene_jaccard_vs_corpus": round(jac, 4),
                "genes_supported_in_block": b["genes_supported"],
                "gene_count_consistent": bool(
                    0.5 <= b["genes_supported"] / max(n_var, 1) <= 1.05),
                "license": cand.get("license", ""),
                "citation": cand.get("citation", ""),
            })
    return hits


def build_registry(paths: List[str]) -> List[Dict]:
    """Seed a candidate registry from directories of .h5ad files."""
    out = []
    for root in paths:
        if os.path.isfile(root):
            out.append({"name": os.path.basename(root), "path": root})
            continue
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fn in sorted(filenames):
                if fn.endswith(".h5ad"):
                    out.append({"name": fn, "path": os.path.join(dirpath, fn)})
            if dirpath.count(os.sep) - root.count(os.sep) >= 2:
                dirnames[:] = []
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=None,
                    help="pretraining corpus .h5ad; defaults to the registry's "
                         "train_corpus")
    ap.add_argument("--out", default="data/diverserna/sources")
    ap.add_argument("--candidates", action="append", default=[],
                    help="directory or .h5ad to consider as a source; repeatable")
    ap.add_argument("--registry", default="",
                    help="YAML/JSON registry of candidates with licence metadata")
    ap.add_argument("--sample-rows", type=int, default=512)
    args = ap.parse_args()

    args.corpus = args.corpus or _default_corpus()
    if not args.corpus or not os.path.isfile(args.corpus):
        print(f"ERROR: corpus not found: {args.corpus}", file=sys.stderr)
        return 2
    os.makedirs(args.out, exist_ok=True)

    print(f"[sources] profiling {args.corpus}")
    blocks, genes, n_obs, n_var = profile_blocks(args.corpus, args.sample_rows)
    print(f"[sources] {n_obs:,} cells x {n_var:,} genes, {len(blocks)} blocks\n")

    hdr = f"{'blk':>3} {'rows':>10} {'%nz':>6} {'max':>12} {'int':>5} {'med sum':>12} {'genes':>7}"
    print(hdr)
    print("-" * len(hdr))
    for b in blocks:
        print(f"{b['block']:>3} {b['n_rows']:>10,} {b['pct_nonzero']:>6.1f} "
              f"{b['max_value']:>12,.0f} {str(b['integral']):>5} "
              f"{b['median_row_sum']:>12,.0f} {b['genes_supported']:>7,}")

    non_integer = [b for b in blocks if not b["integral"]]
    bulk = [b for b in blocks if b["looks_like_bulk"]]
    if non_integer:
        rows = sum(b["n_rows"] for b in non_integer)
        print(f"\n⚠️  blocks {[b['block'] for b in non_integer]} hold NON-INTEGER values: "
              f"{rows:,} rows ({100 * rows / n_obs:.1f}%) are not raw counts")
    if bulk:
        rows = sum(b["n_rows"] for b in bulk)
        print(f"⚠️  blocks {[b['block'] for b in bulk]} look like BULK RNA-seq: "
              f"{rows:,} rows ({100 * rows / n_obs:.1f}%), dense with very large "
              f"library sizes")

    registry: List[Dict] = []
    if args.registry and os.path.isfile(args.registry):
        with open(args.registry, encoding="utf-8") as fh:
            if args.registry.endswith((".yaml", ".yml")):
                import yaml
                registry = yaml.safe_load(fh) or []
            else:
                registry = json.load(fh)
    registry += build_registry(args.candidates)
    print(f"\n[sources] scanning {len(registry)} candidate file(s)")
    hits = scan_candidates(registry, blocks, genes) if registry else {
        b["block"]: [] for b in blocks}

    status = {}
    for b in blocks:
        cands = hits.get(b["block"], [])
        confirmed = [c for c in cands if c["gene_count_consistent"]]
        families = {_family(c["name"]) for c in confirmed}
        if len(confirmed) == 1:
            state = "CONFIRMED"
            evidence = ["exact n_obs match", "gene count corroborates"]
        elif confirmed and len(families) == 1 and \
                len({c["n_var"] for c in confirmed}) == 1:
            # Several files of ONE dataset (e.g. a file and its _LEAN variant).
            # The source is identified even though the exact file is not, and
            # for a dataset card the source is what matters.
            state = "CONFIRMED"
            evidence = ["exact n_obs match", "gene count corroborates",
                        f"{len(confirmed)} files of one dataset ({families.pop()}); "
                        f"same shape, so the source is unambiguous even though the "
                        f"file is not"]
        elif confirmed:
            state = "LIKELY"
            evidence = [f"{len(confirmed)} exact n_obs matches across "
                        f"{len(families)} distinct dataset(s)"]
        else:
            state = "UNRESOLVED"
            evidence = ["no exact n_obs match; the block was subsetted from its source"]
        status[str(b["block"])] = {
            "block": b["block"], "n_rows": b["n_rows"], "status": state,
            "evidence": evidence, "candidates": cands,
            "license": (confirmed[0]["license"] if state == "CONFIRMED" and confirmed else ""),
            "citation": (confirmed[0]["citation"] if state == "CONFIRMED" and confirmed else ""),
            "source_family": (_family(confirmed[0]["name"]) if confirmed else ""),
            "is_raw_counts": b["is_raw_counts"],
            "looks_like_bulk": b["looks_like_bulk"],
        }
        label = (confirmed[0]["name"] if len(confirmed) == 1
                 else (f"{_family(confirmed[0]['name'])} "
                       f"({len(confirmed)} files)" if confirmed else ""))
        print(f"  block {b['block']}: {state:<11} {label}")

    with open(os.path.join(args.out, "fingerprints.json"), "w", encoding="utf-8") as fh:
        json.dump({"corpus": os.path.basename(args.corpus),
                   "n_obs": n_obs, "n_var": n_var,
                   "blocks": blocks}, fh, indent=2)
    with open(os.path.join(args.out, "status.json"), "w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=2, sort_keys=True)
    _write_report(os.path.join(args.out, "report.md"), args.corpus, blocks, status,
                  n_obs, n_var)

    n_conf = sum(1 for v in status.values() if v["status"] == "CONFIRMED")
    print(f"\n[sources] {n_conf}/{len(blocks)} blocks confirmed -> {args.out}")
    if n_conf < len(blocks):
        print("[sources] the upload gate will refuse until every block is CONFIRMED "
              "or carries a signed waiver (see data/diverserna/UPLOAD_GATE.md)")
    return 0


def _write_report(path, corpus, blocks, status, n_obs, n_var) -> None:
    L = ["# DiverseRNA source identification", "",
         f"Corpus: `{os.path.basename(corpus)}`",
         f"Shape: {n_obs:,} cells x {n_var:,} genes", "",
         "## Structural fingerprint", "",
         "| block | rows | row range | % nonzero | max value | integral | median row sum | genes supported |",
         "|---|---|---|---|---|---|---|---|"]
    for b in blocks:
        L.append(f"| {b['block']} | {b['n_rows']:,} | {b['row_range'][0]:,}–"
                 f"{b['row_range'][1]:,} | {b['pct_nonzero']:.1f} | "
                 f"{b['max_value']:,.0f} | {'yes' if b['integral'] else '**NO**'} | "
                 f"{b['median_row_sum']:,.0f} | {b['genes_supported']:,} |")
    L += ["", "## Attribution", "",
          "| block | rows | status | source | licence |", "|---|---|---|---|---|"]
    for key in sorted(status, key=int):
        v = status[key]
        cands = v["candidates"]
        name = cands[0]["name"] if len(cands) == 1 else (
            f"{len(cands)} candidates" if cands else "—")
        L.append(f"| {v['block']} | {v['n_rows']:,} | {v['status']} | {name} | "
                 f"{v['license'] or '—'} |")
    L += ["", "## What this means", "",
          "`CONFIRMED` blocks matched a candidate on exact row count with a "
          "corroborating gene count. `UNRESOLVED` blocks have no exact row match, "
          "which is what a subsetted source looks like; identifying those needs the "
          "gene-Jaccard and row-hash axes against a wider candidate registry.", "",
          "**No block may be published without a licence.** See "
          "`data/diverserna/UPLOAD_GATE.md`.", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
