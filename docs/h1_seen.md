# h1_seen — RHJ checkpoints on H1 hESC (seen perturbation, UNSEEN assay, UNSEEN cell line)

> The live panel has **80 perturbations** (since 2026-09-23). The earlier 99-perturbation panel
> is superseded; its results are not part of the current release.

## Context

**Standing objective.** Predict a perturbation's effect in a context where it was never
measured, from a single control cell.

**This benchmark's question.** Models trained on `rhj_fewshot` (HepG2 + Jurkat + RPE1) have
never seen H1 hESC (VCC 2025): not its perturbations, not its controls, not its assay. For
perturbations they HAVE seen in training, how much of their RHJ score survives?

**Read it together with `k562_seen_h1set`** (docs/k562_seen.md): the SAME 80 perturbations in
K562, in a seen assay. The K562 → H1 gap is the assay shift; K562 → RHJ is the cell line.

**Varied:** the evaluated cell line and assay. **Fixed:** the checkpoints (no training), the
split-half protocol (5 splits, halves A/B from the H1 cells only, model input = one half-B
control cell, cell-eval 0.6.0, nocal), the 4 headline metrics: `pearson_delta` ↑, `mae` ↓
(error), `overlap_at_N` ↑, `discrimination_score_l1` ↑ (chance 0.5).

**What each outcome means.**
- model `pearson_delta` near or above `transfer`, well above `ctrlcell` → what the models
  learned about a perturbation carries over to a new cell line and assay;
- near `ctrlcell` / `meanpert` → it does not leave the training cell lines.

## Panel (built 2026-09-23, job 18367618)

| item | value |
|---|---|
| source | VCC 2025 H1: `adata_{Training,Validation,Test}.h5ad` (paths + sha256 in `benchmark.json`) |
| local copy (on HF) | `data/benchmarks/h1_seen/h1_seen_raw.h5ad` — raw counts, 105,338 cells × 18,080 genes = the 80 perts + 38,176 controls; sha256 `ea72830a…` |
| perturbations | **80** = `data/benchmarks/shared/h1_k562_perts.json`: H1 perts eligible in rhj_fewshot with ≥ 10 cells in H1 AND in the K562 essential screen. 56 RHJ-`train` (trained in HepG2, Jurkat, RPE1), 22 `test` + 2 `val` (withheld from HepG2) |
| controls | 38,176 (identical in all 3 VCC files, read once) |
| per split | 33,559 perturbed + 19,088 control cells per half |
| gene axis | **6,535** of the 6,983 RHJ axis genes (448 absent in H1 → 0 at input, not scored). ⛔ The 448 carry 21.6 % of HepG2's axis mass (MT-CO1, EEF1A1, RPL/RPS…): zero there is a fake strong down-regulation |
| normalisation | log1p(count / rowsum × 2000), rowsum over the 8,910 H1 genes also in the RHJ HepG2 file. ⛔ **Not the training rule** (audit 2026-09-23): training uses the full source-file panel; H1 ∩ HepG2 misses the heavy hitters (72 % of HepG2 counts), so H1 inputs are inflated ~1.39×. See docs/audits_2026-09-23.md |
| depth | median 53,674 UMI (not thinned; part of the shift) |
| `meanpert` | RHJ HepG2 training grand-mean DELTA placed on the H1 control mean |
| `transfer` | cell-weighted mean delta of the same pert over the RHJ lines where it was TRAINED |

## Gene axis — identical for every arm

| arm | model input | scored output |
|---|---|---|
| Occam | the 6,535 panel genes at their encoder columns; the 448 absent genes and all non-axis columns are 0 (as in RHJ training) | 6,535 |
| TxPert / STATE / CellFlow / PerturbDiff | 6,983-wide RHJ axis with the same 448 at 0; emission gathered back to the 6,535 | 6,535 |
| baselines | the panel | 6,535 |

Competitors with a cell-type label are given **HEPG2**; STATE / PerturbDiff also get a seeded
per-cell draw of HepG2 batches (a constant batch distorts STATE by 0.37). Recorded in each
export manifest as `context_proxy`. Occam has no cell-line input.

## Rebuilding

`OUT=/tmp/h1_check bash scripts/repro/build_benchmark_h1.sh` — checks the local copy's sha256,
rebuilds cohort, splits and baselines, compares every quantity with the published panel.
**Verified 2026-09-23 (job 18368097):** rebuilt from the 80-pert HF copy → **MATCH** on axis, cohort, every cell of `X`, labels, batches, all 5 A/B splits, `meanpert`, `transfer`.

## Arms (launched 2026-09-23, `BENCH=h1_seen bash scripts/repro/ood_seen.sh`)

| arm | job |
|---|---|
| baselines, control-draw seeds 42–46 | 18368098–102 |
| Occam Y06 / Q02 / M06 / M03 | 18368103 |
| TxPert (single-cell ctrl) | export 18368104 → score 18368105 |
| STATE | 18368106 → 18368107 |
| CellFlow | 18368108 → 18368109 |
| PerturbDiff genept | 18368110 → 18368111 |

## Results (80 perts) — Occam and baselines COMPLETE; ⏳ competitors relaunched

Occam: 5 splits, seed 42. Baselines: mean over control-draw seeds 42–46 × 5 splits (24 of 25
split-seeds; seed 46 split 4 still running). `mae` ↓. Strata: pΔ over the 56 perts also trained
in HepG2 (`train`) vs the 24 withheld from HepG2 (trained only in Jurkat/RPE1).

| arm | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ (chance 0.5) | pΔ, 56 `train` | pΔ, 24 held from HepG2 |
|---|---|---|---|---|---|---|
| ctrlcell (floor) | 0.0009 | 0.0147 | 0.0000 | 0.5090 | 0.0004 | 0.0020 |
| ctrlmean | 0.0037 | 0.0137 | 0.1700 | 0.5082 | 0.0026 | 0.0063 |
| M03 | 0.0818 ± 0.0021 | 0.1388 | 0.3227 | 0.5591 | 0.0854 | 0.0735 |
| meanpert | 0.0848 | 0.0162 | 0.1882 | 0.5096 | 0.0922 | 0.0673 |
| Y06 (published) | 0.0925 ± 0.0020 | 0.1377 | 0.3212 | **0.6032** | 0.0970 | 0.0821 |
| M06 | 0.0928 ± 0.0021 | 0.1419 | 0.3245 | 0.5737 | 0.0979 | 0.0810 |
| Q02 | 0.0944 ± 0.0021 | 0.1420 | 0.3264 | 0.5718 | 0.0996 | 0.0824 |
| **transfer** | **0.2325** | 0.0175 | 0.2542 | 0.5695 | 0.2442 | 0.2052 |
| experimental | 0.8882 | 0.0065 | 0.6054 | 0.9869 | 0.8908 | 0.8821 |

### F-h1-1 (80-pert panel). Occam does not carry a seen perturbation into H1

Every Occam checkpoint sits at pΔ 0.082–0.094: at most +0.010 over `meanpert` (0.0848) and
**−0.138 under `transfer`** (copying the RHJ delta of the same perturbation onto an H1
control). The ordering holds in both strata. Occam's `mae` is ~9× every baseline on the same
panel (0.138–0.142 vs 0.014–0.018, including `ctrlcell`, which is the input cell itself): the
model's reconstruction of an H1 cell is far from the cell, and because the scored control rows
are the real half-B cells that offset enters the predicted delta directly. Whether the effect
itself fails to transfer or only the reconstruction does is not separated yet (the
`delta_sum_realctrl` emission test, which adds the model's delta to the real control, is proposed,
not run). Y06 has the best `disc_l1` (0.6032). Same conclusion as on the superseded 99-pert panel.

## 2026-09-23: STATE at 5 seeds (seeds 43/44/46 scored with the library-size waiver)

`h1_seen` (80 perts), 5 seeds × 5 splits, nocal; mean ± sd across seeds of the per-seed split-mean.
STATE seeds 43/44/46 emit 8.3x the target's library size; scored as is per the user
(`libsize_check_waived` in the export manifests).

| model | pΔ ↑ | mae ↓ (error) | overlap_at_N ↑ | disc_l1 ↑ | seeds |
|---|---|---|---|---|---|
| Y06 (Occam) | **0.0950 (0.0026)** | **0.1397** (0.0017) | 0.3226 (0.0012) | **0.5952** (0.0101) | 4 (s43 scoring) |
| PerturbDiff genept | 0.0879 (0.0025) | 0.1455 (0.0025) | 0.3257 (0.0015) | 0.5237 (0.0011) | 5 |
| TxPert single-cell | 0.0868 (0.0015) | 0.1489 (0.0045) | **0.3385** (0.0069) | 0.5155 (0.0094) | 5 |
| STATE | 0.0473 (0.0102) | 0.1996 (0.0270) | 0.3188 (0.0126) | 0.5073 (0.0044) | 5 |
| CellFlow | 0.0445 (0.0114) | 0.1732 (0.0112) | 0.2826 (0.0089) | 0.5044 (0.0032) | 5 |

STATE per seed pΔ: 0.0455 / 0.0383 / 0.0553 / 0.0602 / 0.0372 (s42–46).
