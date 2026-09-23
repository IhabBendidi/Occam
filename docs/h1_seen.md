# h1_seen — RHJ checkpoints on H1 hESC (seen perturbation, UNSEEN cell line)

## Context

**Standing objective.** Predict a perturbation's effect in a context where it was never
measured, from a single control cell.

**This experiment's question.** Models trained on `rhj_fewshot` (HepG2 + Jurkat + RPE1) have
never seen H1 hESC (VCC 2025): not its perturbations, not its controls. For perturbations they
HAVE seen in training, how much of their RHJ score survives the change of cell line?

**Varied:** the evaluated cell line (HepG2 on RHJ → H1 here). **Fixed:** the checkpoints (no
training), the split-half protocol (5 splits, halves A/B from the H1 cells only, model input =
one half-B control cell, cell-eval 0.6.0, nocal), the 4 headline metrics.

**What each outcome means.**
- model `pearson_delta` near or above `transfer`, well above `ctrlcell` → what the models
  learned about a perturbation carries over to a new cell line;
- near `ctrlcell` / `meanpert` → it does not leave the training cell lines.

## Panel (built 2026-09-23, `scripts/build_h1_seen_panel.py`, job 18365133, 13 min)

| item | value |
|---|---|
| source | VCC 2025 H1: `adata_{Training,Validation,Test}.h5ad` (paths + sha256 in `benchmark.json` `sources`) |
| local copy | `data/benchmarks/h1_seen/h1_seen_raw.h5ad` — raw counts, 126,546 cells × 18,080 genes (seen perts + controls) |
| perturbations | **99** = H1 perts eligible in rhj_fewshot. Strata by RHJ split: **70 `train`** (trained in HepG2, Jurkat, RPE1), **27 `test` + 2 `val`** (withheld from HepG2, trained only in Jurkat/RPE1) |
| from which H1 file | 50 Training · 18 Validation · 31 Test |
| controls | 38,176 (identical in all 3 files, read once) |
| cells per half | median m **395** (RHJ: 22); 97/99 perts have m ≥ 25 |
| per split | 44,157 perturbed + 19,088 control cells per half |
| gene axis | **6,535** of the 6,983 RHJ axis genes (448 absent in H1 → zero at model input, not scored) |
| normalisation | log1p(count / rowsum × 2000), rowsum over the 8,910 H1 genes also in the RHJ HepG2 file |
| depth | median **53,674 UMI** (not thinned; part of the shift) |
| `meanpert` | RHJ HepG2 training grand-mean DELTA (1,657 perts) placed on the H1 control mean |
| `transfer` | cell-weighted mean delta of the same pert over the RHJ lines where it was TRAINED (median 209 cells) |

⚠️ m ≈ 395 cells per half means the `experimental` arm is much closer to a true ceiling than on
RHJ (m = 22). The two panels' absolute numbers are therefore not comparable; compare arms
WITHIN this panel.

## Rebuilding it (for other users)

`scripts/repro/build_benchmark_h1.sh` rebuilds the whole panel from the published local copy
`benchmarks/h1_seen/h1_seen_raw.h5ad` (HF, sha256 `3cf2642f…`) plus `benchmarks/rhj_fewshot/`
and the three RHJ `eval/pert/` files, then compares it with the published panel.

**Verified 2026-09-23 (job 18365148, 12 min):** rebuilt into a scratch dir → **MATCH** on
axis, cohort, every cell of `X`, labels, batches, all 5 A/B splits, and the `meanpert` /
`transfer` baselines.

## Arms (launched 2026-09-23, `scripts/repro/h1_seen.sh`)

| arm | what | job |
|---|---|---|
| ctrlcell, ctrlmean, experimental, meanpert, transfer | reference arms | 18365137 |
| Occam Y06 / Q02 / M06 / M03 | `--ood-context`, grid `configs/h1_seen_occam.tsv` | 18365138 |
| TxPert (single-cell ctrl) | s42, `best-ep49.ckpt` | export 18365139 → score 18365140 |
| STATE | s42, `best_celleval.ckpt` | 18365141 → 18365142 |
| CellFlow | s42, `occam_s42` | 18365143 → 18365144 |
| PerturbDiff genept | s42 | 18365145 → 18365146 |

**Deviations for competitors (recorded in each export manifest as `context_proxy`):** all four
look up the cell type in their training vocabulary, and H1 is not in it. They are run with
cell type **HEPG2** (the RHJ evaluated line). STATE and PerturbDiff also need a batch covariate;
each cell gets a seeded uniform draw from the model's HepG2 batches (a single constant batch was
measured to distort STATE by 0.37). Occam has no cell-line input and needs no proxy.

## Results

⏳ Running. Filled in as each split lands, before being reported.
