---
pretty_name: Occam evaluation bundle
license: other
license_name: see-dataset-card
size_categories:
  - 100K<n<1M
task_categories:
  - feature-extraction
tags:
  - single-cell
  - transcriptomics
  - benchmark
---

# Occam evaluation bundle

The 6 datasets Occam's benchmarks read, repacked into **5.6 GB** from **1,031 GB** of source files (184x smaller).

## Install

Download into `data/eval/` inside the repo and everything resolves with no
configuration — the shipped `configs/paths.example.yaml` uses **relative**
paths that resolve against the repo root:

```
data/eval/
├── pert/           nadig_2024_hepg2.h5ad  nadig_2024_jurkat.h5ad  replogle_2022_rpe1_e.h5ad
├── scib/           pancreas_scib.h5ad  pbmc_12k.h5ad  pbmc_68k.h5ad
└── manifest.json
```

```bash
cp configs/paths.example.yaml configs/paths.yaml     # relative defaults, no edit needed
occam-benchmark --checkpoint checkpoints/release/occam_small_encoder_decoder.pth
occam-scib      --checkpoint checkpoints/release/occam_small_encoder_decoder.pth
```

Put them elsewhere by editing `configs/paths.yaml`, or per-run with
`OCCAM_EVAL_DIR` / `OCCAM_SCIB_DIR`.

## Contents

### Perturbation benchmark

Used by `occam-benchmark`. Six tasks: gene-relationship recall against five BMDB sources, perturbation consistency, iLISI, KNN separability, linear separability, linear invertibility.

Registry key: `eval_dir`

| dataset | cells | genes | nonzero | size | obs cols | var cols |
|---|---|---|---|---|---|---|
| `nadig_2024_hepg2` | 145,473 | 9,624 | 43.2% | 1.35 GB | 26 | 14 |
| `nadig_2024_jurkat` | 262,956 | 8,882 | 39.4% | 2.05 GB | 25 | 14 |
| `replogle_2022_rpe1_e` | 247,914 | 8,749 | 40.0% | 1.94 GB | 27 | 14 |

### Batch integration (scIB)

Used by `occam-scib`. Biological conservation against batch mixing, combined as `0.6 * bio + 0.4 * batch`.

Registry key: `scib_dir`

| dataset | cells | genes | nonzero | size | obs cols | var cols |
|---|---|---|---|---|---|---|
| `pancreas_scib` | 16,382 | 19,093 | 17.8% | 0.16 GB | 5 | 2 |
| `pbmc_12k` | 11,990 | 3,346 | 13.7% | 0.01 GB | 5 | 5 |
| `pbmc_68k` | 68,579 | 32,738 | 1.7% | 0.10 GB | 4 | 2 |

## What was changed, and what was not

**Kept in full**: `X`, every `obs` column, every `var` column, both indices
and their names. Ensembl IDs are preserved — in several files they are the
`var` index rather than a column, so the index name is carried too.

**Dropped**: `obsm` only.

The three perturbation files each carried 28 `obsm` arrays of precomputed
baseline embeddings (`X_ica`, `X_nmf`, `X_ridge_pca`, `X_factor_analysis`, …),
about 2.3 GB per file. Those are other models' outputs rather than evaluation
inputs, and their provenance is not recorded anywhere, so shipping them would
invite comparison against numbers nobody can trace.

**Rewritten**: dense → CSR, gzip-compressed.

## Why the size difference is so large

Two independent factors, and neither involves discarding data:

| dataset | source | logical content | exported |
|---|---|---|---|
| `nadig_2024_hepg2` | 247.2 GB | ~5.6 GB dense | 1.35 GB |
| `nadig_2024_jurkat` | 398.0 GB | ~9.3 GB dense | 2.05 GB |
| `replogle_2022_rpe1_e` | 375.1 GB | ~8.7 GB dense | 1.94 GB |
| `pancreas_scib` | 1.3 GB | ~1.3 GB dense | 0.16 GB |
| `pbmc_12k` | 0.2 GB | ~0.2 GB dense | 0.01 GB |
| `pbmc_68k` | 9.0 GB | ~9.0 GB dense | 0.10 GB |

1. **HDF5 fragmentation.** `nadig_2024_hepg2` holds 9.79 GB of logical data in
   a 247 GB file — free space from repeated rewrites, never reclaimed. Nothing
   was lost; the file was simply never repacked.
2. **Dense storage of sparse data.** These matrices are 1.6–43% non-zero, so CSR
   plus gzip does the rest.

Verified per dataset on export: shape, both indices, every column's values, and
the matrix itself compared against the source. `manifest.json` records shape,
non-zero count, column lists, sha256 and the original path for each file.

## Provenance and licensing

⚠️ **Not cleared for redistribution.** These are third-party datasets
redistributed here for convenience; each carries its own terms, and none has
been reviewed. `scripts/check_upload_gate.py` refuses to publish until every
dataset has a source, a licence and a citation recorded, and nothing in this
repository can upload anything.

| dataset | source file | licence | citation |
|---|---|---|---|
| `nadig_2024_hepg2` | `nadig_2024_hepg2.h5ad` | **TBD** | **TBD** |
| `nadig_2024_jurkat` | `nadig_2024_jurkat.h5ad` | **TBD** | **TBD** |
| `replogle_2022_rpe1_e` | `replogle_2022_rpe1_e.h5ad` | **TBD** | **TBD** |
| `pancreas_scib` | `pancreas_scib.ensembl.h5ad` | **TBD** | **TBD** |
| `pbmc_12k` | `pbmc_12k.ensembl.h5ad` | **TBD** | **TBD** |
| `pbmc_68k` | `pbmc_68k.ensembl.h5ad` | **TBD** | **TBD** |

The filenames above are what each dataset was exported from, not an upstream
reference. Resolving the upstream publication, licence and citation for each is
the first step toward being able to release this bundle.
