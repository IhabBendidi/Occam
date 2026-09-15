#!/usr/bin/env python
"""Generate data/eval/DATASET_CARD.md from the export manifest.

Generated rather than written by hand so the shapes, densities and column lists
cannot drift from the files they describe.
"""

from __future__ import annotations

import argparse
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GROUPS = {
    "pert": ("Perturbation benchmark", "`occam-benchmark`",
             "Six tasks: gene-relationship recall against five BMDB sources, perturbation "
             "consistency, iLISI, KNN separability, linear separability, linear "
             "invertibility."),
    "scib": ("Batch integration (scIB)", "`occam-scib`",
             "Biological conservation against batch mixing, combined as "
             "`0.6 * bio + 0.4 * batch`."),
}

KEYS = {
    "pert": "eval_dir", "scib": "scib_dir",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=os.path.join(REPO, "data", "eval", "manifest.json"))
    ap.add_argument("--out", default=os.path.join(REPO, "data", "eval", "DATASET_CARD.md"))
    args = ap.parse_args()

    if not os.path.isfile(args.manifest):
        print(f"ERROR: {args.manifest} not found. Run scripts/export_eval_data.py first.")
        return 2
    with open(args.manifest, encoding="utf-8") as fh:
        man = json.load(fh)

    ds = man["datasets"]
    tot = man["totals"]
    by_group = {}
    for d in ds:
        by_group.setdefault(d["group"], []).append(d)

    ratio = tot["source_bytes"] / max(tot["bytes"], 1)
    L = [
        "---",
        "pretty_name: Occam evaluation bundle",
        "license: other",
        "license_name: see-dataset-card",
        "size_categories:",
        "  - 100K<n<1M",
        "task_categories:",
        "  - feature-extraction",
        "tags:",
        "  - single-cell",
        "  - transcriptomics",
        "  - benchmark",
        "---",
        "",
        "# Occam evaluation bundle",
        "",
        f"The {len(ds)} datasets Occam's benchmarks read, repacked into "
        f"**{tot['bytes'] / 1e9:.1f} GB** from **{tot['source_bytes'] / 1e9:,.0f} GB** "
        f"of source files ({ratio:.0f}x smaller).",
        "",
        "## Install",
        "",
        "Download into `data/eval/` inside the repo and everything resolves with no",
        "configuration — the shipped `configs/paths.example.yaml` uses **relative**",
        "paths that resolve against the repo root:",
        "",
        "```",
        "data/eval/",
        "├── pert/           nadig_2024_hepg2.h5ad  nadig_2024_jurkat.h5ad  replogle_2022_rpe1_e.h5ad",
        "├── scib/           pancreas_scib.h5ad  pbmc_12k.h5ad  pbmc_68k.h5ad",
        "└── manifest.json",
        "```",
        "",
        "```bash",
        "cp configs/paths.example.yaml configs/paths.yaml     # relative defaults, no edit needed",
        "occam-benchmark --checkpoint checkpoints/release/occam_small_encoder_decoder.pth",
        "occam-scib      --checkpoint checkpoints/release/occam_small_encoder_decoder.pth",
        "```",
        "",
        "Put them elsewhere by editing `configs/paths.yaml`, or per-run with",
        "`OCCAM_EVAL_DIR` / `OCCAM_SCIB_DIR`.",
        "",
        "## Contents",
        "",
    ]

    for group, items in sorted(by_group.items(), key=lambda kv: list(GROUPS).index(kv[0])):
        title, cmd, what = GROUPS.get(group, (group, "", ""))
        L += [f"### {title}", "",
              f"Used by {cmd}. {what}", "",
              f"Registry key: `{KEYS.get(group, group)}`", "",
              "| dataset | cells | genes | nonzero | size | obs cols | var cols |",
              "|---|---|---|---|---|---|---|"]
        for d in items:
            L.append(
                f"| `{d['name']}` | {d['shape'][0]:,} | {d['shape'][1]:,} | "
                f"{d['density_pct']:.1f}% | {d['bytes'] / 1e9:.2f} GB | "
                f"{len(d['obs_columns'])} | {len(d['var_columns'])} |")
        L.append("")

    L += [
        "## What was changed, and what was not",
        "",
        "**Kept in full**: `X`, every `obs` column, every `var` column, both indices",
        "and their names. Ensembl IDs are preserved — in several files they are the",
        "`var` index rather than a column, so the index name is carried too.",
        "",
        "**Dropped**: `obsm` only.",
        "",
        "The three perturbation files each carried 28 `obsm` arrays of precomputed",
        "baseline embeddings (`X_ica`, `X_nmf`, `X_ridge_pca`, `X_factor_analysis`, …),",
        "about 2.3 GB per file. Those are other models' outputs rather than evaluation",
        "inputs, and their provenance is not recorded anywhere, so shipping them would",
        "invite comparison against numbers nobody can trace.",
        "",
        "**Rewritten**: dense → CSR, gzip-compressed.",
        "",
        "## Why the size difference is so large",
        "",
        "Two independent factors, and neither involves discarding data:",
        "",
        "| dataset | source | logical content | exported |",
        "|---|---|---|---|",
    ]
    for d in ds:
        dense = d["shape"][0] * d["shape"][1] * 4 / 1e9
        L.append(f"| `{d['name']}` | {d['source_bytes'] / 1e9:,.1f} GB | "
                 f"~{dense:.1f} GB dense | {d['bytes'] / 1e9:.2f} GB |")
    L += [
        "",
        "1. **HDF5 fragmentation.** `nadig_2024_hepg2` holds 9.79 GB of logical data in",
        "   a 247 GB file — free space from repeated rewrites, never reclaimed. Nothing",
        "   was lost; the file was simply never repacked.",
        "2. **Dense storage of sparse data.** These matrices are 1.6–43% non-zero, so CSR",
        "   plus gzip does the rest.",
        "",
        "Verified per dataset on export: shape, both indices, every column's values, and",
        "the matrix itself compared against the source. `manifest.json` records shape,",
        "non-zero count, column lists, sha256 and the original path for each file.",
        "",
        "## Provenance and licensing",
        "",
        "⚠️ **Not cleared for redistribution.** These are third-party datasets",
        "redistributed here for convenience; each carries its own terms, and none has",
        "been reviewed. `scripts/check_upload_gate.py` refuses to publish until every",
        "dataset has a source, a licence and a citation recorded, and nothing in this",
        "repository can upload anything.",
        "",
        "| dataset | source file | licence | citation |",
        "|---|---|---|---|",
    ]
    for d in ds:
        name = d.get("source_filename") or os.path.basename(d.get("source", ""))
        L.append(f"| `{d['name']}` | `{name}` | **TBD** | **TBD** |")
    L += [
        "",
        "The filenames above are what each dataset was exported from, not an upstream",
        "reference. Resolving the upstream publication, licence and citation for each is",
        "the first step toward being able to release this bundle.",
        "",
    ]

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"wrote {args.out} ({len(L)} lines, {len(ds)} datasets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
