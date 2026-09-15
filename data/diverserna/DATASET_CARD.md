---
pretty_name: DiverseRNA
license: other
license_name: unresolved-see-dataset-card
size_categories:
  - 1M<n<10M
task_categories:
  - feature-extraction
tags:
  - single-cell
  - transcriptomics
  - rna-seq
configs:
  - config_name: default
    data_files: "data/train-*.parquet"
---

# DiverseRNA

The pretraining corpus of the [Occam](../../README.md) encoder:
**1,434,299 rows x 44,728 genes**.

> ## ⚠️ Read this before using or redistributing
>
> 1. **It is not uniformly raw counts, despite the source filename.**
>    624,870 rows (43.6%) hold non-integer values.
> 2. **It is not uniformly single-cell.** 34,259 rows
>    (2.4%) are bulk RNA-seq.
> 3. **Provenance is incomplete.** 2 of 9 constituent blocks are
>    identified. The corpus was assembled outside the Occam repository and no
>    manifest of its sources exists.
> 4. **It is therefore NOT cleared for redistribution.** See
>    [UPLOAD_GATE.md](UPLOAD_GATE.md). The publication script refuses to run
>    until every block has a source and a licence.

## What it is

A composite assembled for an earlier modelling effort, in which the constituent
datasets survive only as integers in `obs/original_dataset_idx`. Occam's encoder
was pretrained on it; the checkpoint's 44,728-entry `gene_names` is byte-identical
to this corpus's gene axis, which is how the two were tied together after the
fact.

## Composition

| block | rows | row range | % nonzero | max value | integral | median row sum | genes supported | status | source | licence |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 502,080 | 0–502,079 | 7.0% | 977 | yes | 11,033 | 8,248 | UNRESOLVED | — | **TBD** |
| 1 | 102,205 | 502,080–604,284 | 8.7% | 3,908 | yes | 16,528 | 15,432 | UNRESOLVED | — | **TBD** |
| 2 | 99,300 | 604,285–703,584 | 6.8% | 5,801 | yes | 5,978 | 18,287 | UNRESOLVED | — | **TBD** |
| 3 | 71,585 | 703,585–775,169 | 6.8% | 404,058 | yes | 10,514 | 17,572 | UNRESOLVED | — | **TBD** |
| 4 | 88,399 | 775,170–863,568 | 7.4% | 8 | **NO** | 496 | 16,403 | UNRESOLVED | — | **TBD** |
| 5 | 31,542 | 863,569–895,110 | 7.3% | 9 | **NO** | 3,004 | 20,693 | UNRESOLVED | — | **TBD** |
| 6 | 504,929 | 895,111–1,400,039 | 7.3% | 7 | **NO** | 2,054 | 20,267 | UNRESOLVED | — | **TBD** |
| 7 | 10,526 | 1,400,040–1,410,565 | 50.2% | 26,309,592 | yes | 60,686,904 | 35,708 | CONFIRMED | GTEx_homosapiens_bulk_with_ensembl.h5ad | **TBD** |
| 8 | 23,733 | 1,410,566–1,434,298 | 38.3% | 27,590,498 | yes | 40,963,360 | 19,556 | CONFIRMED | TCGA_bulk_rna_gene_expression | **TBD** |

Regenerate with `python scripts/identify_sources.py`; the evidence for each row is
in [`sources/report.md`](sources/report.md).

### The value space is not uniform

Blocks **4, 5, 6** hold non-integer
values. They are not counts, and they are not a simple per-cell rescaling of
counts either -- dividing each row by its smallest non-zero entry leaves only
about half the values integral, so whatever transform was applied was not a
scalar. It is not recorded anywhere.

Consequence: **do not assume a count model.** Anything that calls `expm1` and
treats the result as counts -- a negative-binomial or Poisson likelihood, a
depth-thinning step, an emission model -- is on unstated ground for 43.6% of
this corpus.

### Bulk contamination

Blocks **7, 8** are bulk RNA-seq, not single
cells: 38–50% dense with library sizes of 40–60 million, against a single-cell
median in the thousands. They are 34,259 rows (2.4%)
of a corpus used to pretrain a single-cell model.

That may well be a deliberate and defensible design choice -- bulk profiles are
a reasonable source of expression covariance. It had simply never been written
down, and a model card that says "pretrained on single-cell data" would be
wrong.

## Schema

One row per profile.

| column | type | notes |
|---|---|---|
| `cell_id` | string | the source `obs` index verbatim. ⚠️ These are positional strings `'0'`…`'1434298'` and carry no meaning. |
| `row` | int32 | position in the source file |
| `original_dataset_idx` | int8 | constituent block, 0–8 |
| `original_sum` | float32 | ⚠️ equals `X.sum(axis=1)` exactly, verified. A cached row sum of whatever is in `X`, **not** a record of pre-normalisation depth. |
| `gene_idx` | list&lt;int32&gt; | ascending column positions of the non-zeros |
| `value` | list&lt;float32&gt; | ⚠️ float32 is required: three blocks are non-integer and two reach 2.7e7, so no integer type is safe |

`genes.parquet` carries the gene axis:

| column | notes |
|---|---|
| `position` | 0…44727. **This is the encoder's column index.** |
| `gene_id` | the source's gene identifier |
| `id_type` | `ensembl` or `symbol` |
| `source_var_index` | the source `var` index — positional strings, never usable for gene matching |

⚠️ The source stores the gene axis in a column named `gene_symbol`, but 44,560 of
44,728 entries (99.6%) are Ensembl gene IDs. The release calls the column
`gene_id` and records `id_type` per gene; the misleading name is not carried over.

## How Occam used it

* Vocabulary: `genes.parquet.gene_id`, in `position` order, byte-identical to the
  released checkpoint's `gene_names`.
* Preprocessing: library-size normalise to `target_sum = 2000`, then `log1p`.
* Split: 90/10 chunk-wise, chunk size 8192, seed 42.

See `configs/occam_small.yaml`.

## Sharding

19 Parquet shards, zstd-9, **never crossing a block boundary** — the blocks are
contiguous, differ in value space, and will differ in licence, so a consumer must
be able to skip one whole. `shards.json` maps each shard to its block, row range,
non-zero count and checksum.

Build with `python scripts/export_dataset_hf.py`; reassemble any subset into a
CSR `.h5ad` with `python scripts/shards_to_h5ad.py`.

## Known issues

* 7 of 9 blocks are UNRESOLVED. They have no exact row-count
  match against the candidate registry, which is what a subsetted source looks
  like.
* No licence is recorded for any block yet.
* The two identified blocks are GTEx and TCGA, **both of which carry
  redistribution conditions**. TCGA gene-level expression derived from
  controlled-access sequence is subject to NIH/dbGaP terms.
* The path recorded inside the released checkpoints no longer exists; the
  surviving copy was identified by matching `gene_names` against the corpus's
  gene axis.

## Citation

Pending. Every constituent dataset must be cited independently; see the
`citation` field per block in `sources/status.json` once resolved.
