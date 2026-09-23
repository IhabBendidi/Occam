> ⛔ ARCHIVED 2026-09-23 — the 99-perturbation h1_seen panel and every number measured on it.
> SUPERSEDED by the 80-perturbation panel (docs/h1_seen.md). Do not quote these numbers.
> Panel: archive/data/benchmarks/h1_seen_99 · scores: archive/runs/h1_seen_99 · exports:
> archive/runs/competitors/*__h1_seen_99. Kept only as the record of what was measured when.

# h1_seen — RHJ checkpoints on H1 hESC (seen perturbation, UNSEEN assay, UNSEEN cell line)

> ⛔ **2026-09-23 (user decision): `h1_seen` is now the 80-perturbation panel** — the 99 below
> intersected with the K562 essential screen (≥ 10 cells each), frozen in
> `data/benchmarks/shared/h1_k562_perts.json`, so H1 and its K562 twin (`k562_seen_h1set`) are
> scored on the same set. The 99-perturbation panel and every number measured on it (below,
> kept for the record) are **SUPERSEDED**; they were moved to `data/benchmarks/h1_seen_99/` and
> `runs/h1_seen_99/` and all arms are re-scored on the 80.

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

## Gene axis — identical for every arm (checked 2026-09-23)

| arm | model input | scored output |
|---|---|---|
| Occam | the 6,535 panel genes at their encoder columns; the 448 RHJ-axis genes absent in H1 and all 37,745 non-axis columns are 0 (as in RHJ training) | 6,535 |
| TxPert / STATE / CellFlow / PerturbDiff | 6,983-wide RHJ axis, the same 448 genes set to 0; emission gathered back to the 6,535 (`PanelSplit.model_cols`) | 6,535 (export manifests: `n_genes 6535`) |
| baselines | the panel itself | 6,535 |

So input and scoring genes are the same 6,535 for every arm: the RHJ axis minus the 448 genes H1
does not measure. They cannot be added back — H1 has no counts for them.

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

### Occam: COMPLETE, 20/20 splits (2026-09-23). ⏳ baselines and competitors pending

nocal, `decode_seq_mean` (no-op on singles), mean ± SD over splits. `mae` is an ERROR (↓).
Strata = per-perturbation `pearson_delta` mean over the 70 perts trained in HepG2 too
(`train`) vs the 29 withheld from HepG2 (trained only in Jurkat/RPE1).

| arm | splits | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ (chance 0.5) | pΔ, 70 `train` | pΔ, 29 held from HepG2 |
|---|---|---|---|---|---|---|---|
| Y06 (published) | 5 | 0.0977 ± 0.0019 | 0.1378 | 0.3195 | **0.5995** | 0.1010 | 0.0898 |
| Q02 | 5 | **0.0990** ± 0.0019 | 0.1420 | **0.3244** | 0.5638 | 0.1028 | 0.0896 |
| M06 | 5 | 0.0979 ± 0.0019 | 0.1419 | 0.3225 | 0.5624 | 0.1017 | 0.0889 |
| M03 | 5 | 0.0872 ± 0.0019 | 0.1387 | 0.3205 | 0.5509 | 0.0898 | 0.0809 |

For reference, the same checkpoint (Y06) on RHJ: pΔ 0.4043.

### Anchors on this panel — COMPLETE, 5/5 splits, seed 42 (job 18365351)

| arm | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ | pΔ, 70 `train` | pΔ, 29 held from HepG2 |
|---|---|---|---|---|---|---|
| ctrlcell (floor) | 0.0016 | 0.0142 | 0.0000 | 0.5080 | 0.0016 | 0.0017 |
| ctrlmean | 0.0036 | 0.0133 | 0.1725 | 0.5068 | 0.0028 | 0.0057 |
| meanpert (HepG2 grand delta) | 0.0849 | 0.0158 | 0.1911 | 0.5078 | 0.0911 | 0.0701 |
| **transfer** (RHJ delta of the same pert) | **0.2239** | 0.0171 | 0.2531 | 0.5685 | 0.2296 | 0.2102 |
| experimental (replicate half) | 0.8897 | 0.0063 | 0.6071 | 0.9892 | 0.8898 | 0.8896 |

### F-h1-1. Occam does NOT carry the seen perturbation into H1: pΔ ≈ 0.10, between `meanpert` (0.085) and `transfer` (0.224)

Every Occam checkpoint sits at 0.087–0.099 pΔ: +0.013 over `meanpert`, **−0.125 under
`transfer`**. Copying the RHJ delta of the same perturbation onto an H1 control beats the model
by more than 2×. `disc_l1` 0.55–0.60 is barely above chance (0.5); `transfer` 0.57.

⚠️ **The output LEVEL is off, not only the delta.** Occam `mae` is **0.138–0.142**, ~10× every
baseline on the SAME panel (0.013–0.017), including `ctrlcell`, which is just the input cell.
(Correction: an earlier line here attributed the 0.14 to H1's depth — wrong; the baselines see
the same depth.) So the model's reconstruction of an H1 cell is far from the cell. Because the
scored control rows are the REAL half-B cells (`pred_ctrl = ctrl_B`), any reconstruction offset
enters the predicted delta directly, which would depress pΔ on its own. Whether the perturbation
effect itself transfers is therefore not yet separated from the reconstruction failure.

`overlap_at_N` 0.32 exceeds `transfer` (0.25): a single high number next to a low pΔ and a
10× mae is consistent with the known emission artifact of `overlap_at_N`, and is not read as a win.

### Baselines at 5 control-draw seeds (launched 2026-09-23)
Seed = the half-B control-cell draw (`predict_panel`: rng = seed + 1000·split). It moves
`ctrlcell`, `meanpert`, `transfer`; `ctrlmean` and `experimental` do not depend on it.
seed 42: job 18365351 (`runs/h1_seen/reference`); seeds 43–46: jobs 18365479–82
(`runs/h1_seen/reference_s4N`). Same jobs for rhj_fewshot (18365483–86,
`runs/rhj_splithalf/reference_s4N`) and norman_combo (18365487–90, `runs/norman_combo/reference_s4N`).


### Competitors on the 99-pert panel (SUPERSEDED by the 80-pert panel), seed 42, 5/5 splits

Proxy cell type HEPG2; STATE/PerturbDiff with a per-cell HepG2 batch draw. `mae` ↓.

| arm | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ | pΔ, 70 `train` | pΔ, 29 held |
|---|---|---|---|---|---|---|
| TxPert (single-cell ctrl) | 0.0945 ± 0.0019 | 0.1478 | 0.3348 | 0.5116 | 0.0975 | 0.0872 |
| PerturbDiff genept | 0.0868 ± 0.0018 | 0.1420 | 0.3251 | 0.5241 | 0.0898 | 0.0795 |
| CellFlow | 0.0488 ± 0.0014 | 0.1630 | 0.2727 | 0.5069 | 0.0457 | 0.0561 |
| STATE | 0.0486 ± 0.0015 | 0.2330 | 0.3001 | 0.5066 | 0.0485 | 0.0491 |

Against Occam on the same panel (0.087–0.099) and the anchors (meanpert 0.0849, transfer
0.2239): every model, ours and theirs, sits at or below `meanpert` + 0.015 and all are ≥ 2×
under `transfer`; every model's `mae` is ~10× the baselines' (0.14–0.23 vs 0.013–0.017).

## ⏳ h1_seen — 80-perturbation panel (built 2026-09-23, job 18367618) and its re-scoring

Built from the 99-pert local copy with `--eligible file:data/benchmarks/shared/h1_k562_perts.json
--write-raw-subset`: **80 perts** (56 RHJ-`train`, 22 `test`, 2 `val`) + 38,176 controls =
105,338 cells; axis 6,535; 33,559 perturbed + 19,088 control cells per half; 10/10 self-checks.
New local copy `data/benchmarks/h1_seen/h1_seen_raw.h5ad` sha256 `ea72830a…` (replaces `3cf2642f…`
on HF). Rebuild-from-copy test: job 18368097.

All arms re-scored on it (`BENCH=h1_seen bash scripts/repro/ood_seen.sh`):
baselines seeds 42–46 jobs 18368098–102 · Occam Y06/Q02/M06/M03 18368103 · TxPert
18368104→05 · STATE 18368106→07 · CellFlow 18368108→09 · PerturbDiff genept 18368110→11.
