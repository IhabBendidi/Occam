# The HuggingFace release — what is published, where, and why it is private

## Objective

Occam's paper needs three artifacts to be fetchable by one person on one machine with no
configuration: the trained model, the evaluation data it is scored on, and the corpus it was
pretrained on. Everything in the repo already uses **relative paths only**, so the contract is
"download these folders into these locations and evaluation runs" — nothing to edit.

## What this document answers

Which artifacts are published, which are deliberately not, why the repo is private, and how to
re-run or extend the upload. Every size here is measured, not estimated.

## Where it lives

**`https://huggingface.co/bendidiihab/Occam` — one repo, `repo_type=model`, PRIVATE.**

One repo rather than three so that a consumer resolves one name. `repo_type=model` because the
model is the primary artifact; the two datasets ride along as folders.

```
checkpoints/occam_small_encoder_decoder.pth    the pretrained encoder + decoder
checkpoints/MANIFEST.json                      checksums, param counts, verified recipe
checkpoints/MODEL_CARD.md
checkpoints/vcell/rhj_fewshot_Y06/            the rhj_fewshot main checkpoint + CARD.md
checkpoints/vcell/norman_combo_NQ02/          the norman_combo main checkpoint + CARD.md
benchmarks/norman_combo/                       axis, cohort, 5 splits, baselines
benchmarks/h1_seen/                            H1 (VCC 2025) local copy h1_seen_raw.h5ad + panel
metrics/metrics_long.csv  metrics/*.csv        EVERY cell-eval metric, per arm, per split
metrics/per_perturbation/                      cell-eval's per-perturbation rows
repro/{rhj,norman}_{train,eval}.sh             the reproduction path
repro/occam_predict.py  repro/README.md        inference on your own control cells
eval/pert/*.h5ad  eval/scib/*.h5ad             the evaluation bundle
eval/manifest.json  eval/DATASET_CARD.md
diverserna/data/train-*.parquet                the pretraining corpus, sharded
diverserna/genes.parquet  diverserna/DATASET_CARD.md
```

Download targets in a clone of the repo, which is what `configs/paths.example.yaml` already
points at:

| repo folder | local destination |
|---|---|
| `eval/pert/` | `data/eval/pert/` |
| `eval/scib/` | `data/eval/scib/` |
| `diverserna/` | `data/diverserna/` |
| `checkpoints/*.pth`, `checkpoints/*.json`, `checkpoints/*.md` | `checkpoints/release/` |
| `checkpoints/vcell/` | anywhere — pass the path to `--checkpoint` |
| `benchmarks/norman_combo/` | `data/benchmarks/norman_combo/` |
| `benchmarks/h1_seen/` | `data/benchmarks/h1_seen/` — rebuild with `scripts/repro/build_benchmark_h1.sh` |
| `metrics/` | `data/release/metrics/` (regenerable) |
| `repro/` | `scripts/repro/` (also in git) |

## ⚠️ Why it is PRIVATE, and what would have to change

`scripts/check_upload_gate.py` gates **public** release and **currently refuses**, correctly:

- **DiverseRNA: 7 of the 9 constituent blocks are unattributed.** The two identified are GTEx
  (block 7) and TCGA (block 8), and both carry redistribution conditions — gene-level TCGA
  expression derived from controlled-access sequence is subject to NIH/dbGaP terms.
- **The evaluation bundle: no licence or citation is recorded for any of the six datasets.**
  They are third-party data.

Private storage for people who already have access to these files is a different act from
publishing them, which is why `scripts/publish_hf.py` exists *and* the gate still says no.
`publish_hf.py` passes `private=True` on create and **refuses to upload to a repo that reports
`private=False`** — so the gate cannot be bypassed by flipping the repo's visibility.

Making this public is a licensing decision with a named human approver, not an engineering task.

## Checkpoints: one is published, two are not

### The two trained vcell checkpoints ARE published (added 2026-09-22)

| checkpoint | size | sha256 (first 16) | result |
|---|---|---|---|
| `checkpoints/vcell/rhj_fewshot_Y06/best_model.pth` | 2.23 GB | `fe93eeac08d345f3` | `pearson_delta` **0.4043** on 611 held-out HepG2 perturbations (bar: `transfer` 0.2323) |
| `checkpoints/vcell/norman_combo_NQ02/best_model.pth` | 2.23 GB | `91aed55e3db04afb` | **0.8169** additive / **0.7380** non-additive on double perturbations (bar: `sum_of_singles` 0.8824 / 0.7757) |

Both are **self-contained**: encoder (frozen, 94.2 M) + decoder (128.5 M) + perturbation
model (334.2 M) + FiLM (0.33 M) = **557,191,681 parameters**, plus `gene_names` (44,728,
ordered), `pert2id` (18,216), `target_sum` and the benchmark's own sha256. Nothing needs the
absolute `training_config["encoder_checkpoint"]` path they record.

⚠️ `publish_hf.py` **verifies both sha256 values before uploading anything** and refuses
on a mismatch. The path that selects them (`runs/<arm>__<tag>/...`) is overwritten by a
re-run of the same arm, so the path alone does not tie a file to a published number.

### The MAE stage checkpoints are still NOT published

| checkpoint | size | contents | published |
|---|---|---|---|
| `checkpoints/release/occam_small_encoder_decoder.pth` | 0.93 GB | unified: finetuned encoder (13 tensors, 94,279,864 params) + external decoder (99 tensors, 137,393,859 params) | **yes** |
| `checkpoints/reference/occam_small_stage2_finetune.pth` | 2.51 GB | full MAE: encoder + MAE decoder + 44,728-way mask predictor | no |
| `checkpoints/reference/occam_small_stage1_pretrain.pth` | 2.51 GB | same shape, before finetuning | no |

The two reference files are **stage outputs, not deliverables**, and the finetuned encoder they
carry is already inside the release as `encoder_state_dict`. Publishing the release alone costs
a consumer exactly one thing: the ability to restart the chain from stage 1 or stage 2 instead
of from the corpus. Anyone reproducing the chain runs `occam train` from
`configs/occam_small.yaml`, which rebuilds both.

## Measured sizes

| artifact | files | size |
|---|---|---|
| released checkpoint + manifest + card | 3 | 0.93 GB |
| evaluation bundle (3 perturbation + 3 scIB `.h5ad`) | 8 | 5.60 GB |
| DiverseRNA shards + `genes.parquet` + card + `sources/` | 27 | 8.83 GB |
| **total** | 39 | **15.36 GB** |

DiverseRNA is **29× smaller than its source** (239.1 GiB dense `.h5ad` → ~8.8 GB Parquet+zstd
over 5.20 G nonzeros; true density 8.11%). A tuned encoding was measured at ~6.6 GB (39×, a
further 24%) via `gene_idx` int32→uint16, delta-encoded indices at zstd-22, dropping the
redundant `cell_id` column, and per-block value dtypes. **The current encoding was chosen**
deliberately: uint16/delta coding costs a consumer a decode step, and the artifact is read by
`scripts/shards_to_h5ad.py` and by people with pandas, not only by our loader.

## How to run it

```bash
python scripts/publish_hf.py --repo bendidiihab/Occam --dry-run   # sizes, no side effects
python scripts/publish_hf.py --repo bendidiihab/Occam
```

**The upload is incremental and resumable**, so it is safe to run repeatedly — that is how the
corpus was added after the checkpoint and eval bundle. Mechanics: `upload_large_folder` gives
per-file retry and parallelism but has no `path_in_repo`, so the script first builds a
**hardlink mirror** of the intended repo layout at `../occam_hf_stage` (same GPFS fileset, so
no copy and no extra space) and uploads that. `--only <repo path>` restricts what gets staged;
pruning is always checked against the full artifact list, so a restricted run never un-stages
the rest.

The staging directory is disposable — delete it any time; it is rebuilt from hardlinks in
seconds.

## Verified state

| when | what | result |
|---|---|---|
| 2026-09-15 | repo created | `private: True` |
| 2026-09-15 | pass 1: checkpoint + eval bundle | 11 files, **6.53 GB**, re-read from the Hub with `files_metadata=True` and every size matched |
| 2026-09-15 | pass 2: DiverseRNA | 27 files, **8.83 GB**, all 20 shards present |
| 2026-09-15 | whole repo re-read from the Hub | **39 files, 15.36 GB, `private: True`**, every size matched |

| 2026-09-22 | pass 3: both vcell checkpoints + cards + `metrics/` | sha256 of BOTH checkpoints verified before upload; re-read from the Hub |
| 2026-09-22 | pass 4: `graphs/`, `benchmarks/rhj_fewshot/`, `metrics_all/`, `repro/` | **332 files, 21.235 GB, `private: True`** |
| 2026-09-23 | pass 5: `benchmarks/h1_seen/` (99 perts, `h1_seen_raw.h5ad` 8.85 GB, sha256 `3cf2642f…`), `repro/` | SUPERSEDED by pass 6 (still in the Hub's file history) |
| 2026-09-23 | pass 6 (SUPERSEDED by pass 7: uncompressed copy): `benchmarks/h1_seen/` = the **80-pert** panel, `h1_seen_raw.h5ad` 7.37 GB (80 perts + 38,176 controls), `benchmarks/shared/h1_k562_perts.json`, `repro/` | re-read from the Hub: raw copy LFS sha256 `ea72830a…` = local; `private: True` |
| 2026-09-23 | pass 7: raw copies rewritten int32 + gzip-4 like the other published h5ad (content identical, rebuild MATCH verified): `h1_seen_raw.h5ad` 2.04 GB `73ea8369…`, `k562_seen_raw.h5ad` 0.68 GB `59f2ea90…`; `benchmarks/k562_seen/`, `benchmarks/k562_seen_h1set/` added; `metrics/`, `metrics_all/` regenerated from the live tree and **42 files of archived arms deleted** from the Hub | re-read from the Hub: both LFS sha256 = local; `private: True` |

Re-read from the Hub after pass 4, with `files_metadata=True`:

| folder | files | size |
|---|---|---|
| `diverserna/` | 27 | 8.832 GB |
| `eval/` | 10 | 6.247 GB |
| `checkpoints/` | 7 | 5.354 GB |
| `graphs/` | 5 | 0.399 GB |
| `benchmarks/` | 34 | 0.343 GB |
| `metrics_all/` | 119 | 0.038 GB |
| `metrics/` | 119 | 0.022 GB |
| `repro/` | 10 | <0.001 GB |
| **total** | **332** | **21.235 GB** |

The corpus export is self-consistent against the source, checked after the fact from
`shards.json` rather than trusted from the exporter's own exit code:

| check | value |
|---|---|
| `partial` | `False` — the export ran to completion |
| rows | 1,434,299 — **exactly** the source `n_obs` |
| genes | 44,728, of which 44,560 are Ensembl IDs |
| nonzeros | 5,172,820,437 (8.07% density) |
| rows per `original_dataset_idx` block | 502,080 / 102,205 / 99,300 / 71,585 / 88,399 / 31,542 / 504,929 / 10,526 / 23,733 — matches the nine-block table in the dataset card exactly |
| shard boundaries | 20 shards, **no shard crosses a block boundary**, so a consumer can skip a whole block on licensing grounds |

Each shard carries its own `sha256`, `nnz` and `sum` in `shards.json`, so a download can be
verified without the source.

## What was added on 2026-09-22

| artifact | size | why |
|---|---|---|
| `checkpoints/vcell/rhj_fewshot_Y06/` | 2.23 GB | the `rhj_fewshot` main checkpoint + card |
| `checkpoints/vcell/norman_combo_NQ02/` | 2.23 GB | the `norman_combo` main checkpoint + card |
| `metrics/` | 28 MB | **all 35** cell-eval metrics x 422 eval roots x 5 splits x panel, plus per-perturbation rows for the published and reference arms |
| `repro/` | 40 KB | train / eval / predict, four commands per benchmark |

New total **20.62 GB**; ~4.5 GB of it is new, and the upload is incremental.

**Why the full metric dump.** Reports quote four metrics and their 5-split mean. cell-eval
0.6.0 computes 35, and a mean of five numbers hides whether an effect is consistent across
splits or carried by one. `metrics/metrics_long.csv` is tidy
(benchmark, arm, compose_mode, config_tag, scored_arm, split, panel, metric, value), and
every published number re-derives from it exactly — verified for all four headline metrics
on both checkpoints.

⚠️ `metrics/` is **regenerable** (`python scripts/export_split_metrics.py`) and is
published for convenience, not as a primary artifact. The primary records are the
`metrics.json` and `results.csv` files inside each run directory, which are not published
(they are ~1.8 TB with their `.h5ad` intermediates).

## 2026-09-24: DiverseRNA removed from the Hub (user request)

`diverserna/` (27 files, 8.83 GB: 20 Parquet shards, `genes.parquet`, cards, `sources/`) deleted
from `bendidiihab/Occam` in one commit, **df21826d47bf17cb1137877031c3d5d42bb3613f**. Verified by
re-listing: 495 files remain, 0 match `diverserna`. `scripts/publish_hf.py` no longer stages it.
The corpus stays local at `data/diverserna/`. History then SQUASHED (user request): 23 commits → 1,
**3e001faa05667ea738fe71b0a0a536e6a070828b**, so DiverseRNA is in no revision. Only branch `main`,
no tags. Earlier commit hashes quoted in this doc (e.g. df21826d) no longer exist on the Hub.

## 2026-09-25: per-gene embedding tables added under `assets/`

Commit **3aa085acea47f0b881aafc0cd30e335b09e34973** uploads `data/assets/` → `assets/` (386 MB): the
Occam GenePT full-vocabulary table (3072-d, 17,843 genes) and ESM2 table (1280-d, 18,149 genes)
read by the ALL-sources default and the embedding arms, the two PerturbDiff GenePT tables, each
`.pkl.json` sidecar, and `assets/README.md`. Re-listed from the Hub: all four `.pkl` LFS sha256
equal the local files. Download: `hf download bendidiihab/Occam --include 'assets/*' --local-dir
/tmp/dl && mkdir -p data/assets && mv /tmp/dl/assets/* data/assets/`. The loader now refuses a
table whose sha256 differs from its sidecar, and a missing table's error prints that command.
`scripts/publish_hf.py` stages `data/assets`.

## 2026-09-25: audit — competitor / baseline results on the Hub are INCOMPLETE and STALE

| what | local (current) | on the Hub |
|---|---|---|
| every cell-eval metric per split (`metrics_all/metrics_long.csv`) | 244,733 rows, 5 benchmarks | 97,677 rows, **rhj + norman only** (no h1_seen / k562_seen / k562_seen_h1set) |
| one row per (benchmark, panel, model, seed, split) (`data/release/final/`) | yes | **absent** |
| per-split cell-eval outputs (`metrics.json`, `results.csv`, `agg_results.csv`) | 600 splits (100 competitor + 20 baseline roots) , ~0.12 GB | per-perturbation CSVs for 35 roots (norman/rhj, mostly seed 42) |
| `pred_de.csv` per split | ~249 GB | absent |
| prediction + target `.h5ad` per split | ~1.6 TB | absent |

**Fixed the same day (user: scores + cell-eval CSVs).** Uploaded, each folder mirrored (stale Hub
files removed): `metrics_all/` (commit 017983ac; 244,733 rows, all 5 benchmarks — re-read from the
Hub), `metrics/` (375df697), `final/` (24b721dc; 8,063 rows: 6 competitor arms, 6 baselines, all
Occam arms, per seed and split), `ablation/` (7035488a; 23 arms, per seed and split + summary),
`eval_outputs/` (90378285; 125 roots — 100 competitor + 25 baseline — × 5 splits × 5 benchmarks,
2,875 files, 142 MB: metrics.json + results.csv + agg_results.csv; staged by
`scripts/stage_eval_outputs.py`). `pred_de.csv` and `.h5ad` NOT pushed (user decision).

## Competitor checkpoints — PUBLIC repo `bendidiihab/Occam-competitor-checkpoints` (2026-09-25)

User decision: public, checkpoints only (no README, no cards). Layout
`<competitor>/<benchmark>/seed_<N>/` for txpert, state, perturbdiff, cellflow × rhj_fewshot,
norman_combo × seeds 42–46 (40 folders; h1_seen / k562_seen / k562_seen_h1set reuse the
rhj_fewshot checkpoints). Each folder = the validation-selected checkpoint that produced our
scores (path read from the export manifest) + the files its loader reads (TxPert config; STATE
config + one-hot maps + var dims + data module; PerturbDiff sampling config + manifest + GenePT
table; CellFlow whole run dir: CellFlow.pkl, pca/, genes.txt, config.yaml, run_meta.json — the
pickle embeds the training AnnData, published on the user's instruction). STATE's upstream
licences (code CC BY-NC-SA 4.0; weights Arc State Model Non-Commercial License + Acceptable Use
Policy; `competitors/STATE/licenses/`) sit at the top of `state/`. Built by
`scripts/stage_competitor_ckpts.py --include-cellflow` (136.2 GB); archived / defective runs are
excluded by construction. TxPert uploaded and checked (102.9 MB rhj s42 on the Hub); the rest
in progress, verified file by file (size + sha256) — log `runs/slurm/hf_competitor_ckpts_upload.log`.

**Public repo completed + reload tested (2026-09-25).** First pass left CellFlow's `pca/` out in all
10 folders (the stager symlinked the directory; the uploader follows file links only) — uploaded
directly (commit 87635635, 30/30 files size+sha256 verified); the stager now links files inside
directories. Final: 204 files, every staged file matches the Hub. Reload test: downloaded
`txpert/rhj_fewshot/seed_42/` from the public repo and ran `competitors/TxPert/export_occam.py
--checkpoint <downloaded>/checkpoint.ckpt --seed 42 --split 0` (gspp env, rhj_fewshot panel) — it
loaded and was predicting (1,012 / 7,178 batches, no errors) when stopped on the user's instruction.

## Occam 5-seed checkpoints — PRIVATE `bendidiihab/Occam`, `checkpoints/occam_seeds/` (2026-09-25)

`checkpoints/occam_seeds/rhj_fewshot/{gostring,all_sources}/seed_<42..46>/` = `best_model.pth`
(the checkpoint the split-half scorer uses) + `resolved_config.yaml`. `gostring` = AA03 schedule +
smooth_l1 + delta-cos, GO+STRING adjacency (`ABL_A_DC`, 2.23 GB each); `all_sources` = the same
with GO+STRING adjacency ‖ GenePT ‖ ESM2 (`ABL_N_ALL`, 2.55 GB each; tables stored in the
checkpoint, reloads without the files). Upload log `runs/slurm/hf_occam_seeds_upload.log`.
Norman (user: GO+STRING = NF32DCA only; all-sources = both weights):
`checkpoints/occam_seeds/norman_combo/gostring/seed_<N>/` = NF32DCA (NF32 recipe, smooth_l1 +
delta-cos 0.0255, GO+STRING); `.../all_sources_dcos0.32977/` = NF32ALL; `.../all_sources_dcos0.0255/`
= NF32ALLA (both ALL-sources concatenated). Same two files per folder. Log
`runs/slurm/hf_occam_norman_upload.log` (waits for all 15 trainings, then uploads and verifies).
