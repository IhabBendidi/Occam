# k562_seen — RHJ checkpoints on K562 (seen perturbation, SEEN assay, UNSEEN cell line)

## Context

**Standing objective.** Predict a perturbation's effect in a context where it was never
measured, from a single control cell.

**Where this sits.** The same `rhj_fewshot` checkpoints (trained on HepG2 + Jurkat + RPE1),
scored without training on three shifts:

| benchmark | perturbation | assay | cell line | perturbations |
|---|---|---|---|---|
| `rhj_fewshot` | few-shot (withheld from HepG2, seen in Jurkat/RPE1) | seen | seen (HepG2) | 611 |
| `k562_seen` | seen | seen | **unseen** (K562) | 528 RHJ test perts |
| `k562_seen_h1set` | seen | seen | **unseen** (K562) | 80 (= `h1_seen`) |
| `h1_seen` | seen | **unseen** | **unseen** (H1) | 80 |

**Questions.**
1. `k562_seen` vs `rhj_fewshot` (same test perturbations, 528 of 611): what does changing ONLY
   the cell line cost, with the assay held fixed?
2. `k562_seen_h1set` vs `h1_seen` (identical 80 perturbations): what does changing ONLY the
   assay cost? If K562 ≫ H1, the H1 collapse (F-h1-1) is mostly the assay/depth shift; if
   K562 ≈ H1, it is the cell line.

**Varied:** the evaluated cell line / assay. **Fixed:** checkpoints, split-half protocol
(5 splits, halves A/B from the panel's own cells, one half-B control cell as input, cell-eval
0.6.0, nocal), the 4 headline metrics, `pearson_delta` ↑, `mae` ↓ (error), `overlap_at_N` ↑,
`discrimination_score_l1` ↑ (chance 0.5).

## Source and panels

| item | value |
|---|---|
| source | Replogle 2022 K562 **essential** CRISPRi screen, `replogle_2022_k562_e.h5ad` (310,385 × 8,563, raw counts, 48 gem groups) — same screen family as the RHJ `replogle_2022_rpe1_e` file, hence "seen assay" |
| local copy | `data/benchmarks/k562_seen/k562_seen_raw.h5ad`: the 529 RHJ test perts ∪ the 80 shared perts (587) + all 10,691 controls, raw counts, all genes. Both K562 panels rebuild from it |
| `k562_seen` cohort | RHJ `test` perts with ≥ 10 K562 cells: **528** (median 116 cells) |
| `k562_seen_h1set` cohort | the 80 in `data/benchmarks/shared/h1_k562_perts.json` (56 RHJ-`train`, 22 `test`, 2 `val`) |
| gene axis | the **6,640** RHJ axis genes K562 measures (343 absent → 0 at input, not scored) |
| normalisation | log1p(count / rowsum × 2000), rowsum over K562 genes also in the RHJ HepG2 file (same rule as `h1_seen`) |
| `meanpert` | RHJ HepG2 training grand-mean delta, placed on the K562 control mean |
| `transfer` | cell-weighted mean delta of the same pert over the RHJ lines where it was trained (Jurkat + RPE1 for `test`; all three for `train`) |

Competitors with a cell-type label get **HEPG2** (STATE / PerturbDiff also a seeded per-cell
draw of HepG2 batches), recorded in each export manifest as `context_proxy`. Occam has no
cell-line input.

## Rebuilding

`scripts/repro/build_benchmark_k562.sh` (from the HF local copy, sha256-checked, verified
against the published panels). Scored with `BENCH=k562_seen bash scripts/repro/ood_seen.sh`.

## Built (2026-09-23)

| | `k562_seen` (job 18367850) | `k562_seen_h1set` (job 18367851) |
|---|---|---|
| cohort | 528 RHJ test perts (1 of the 529 present has < 10 cells) | 80 (56 RHJ-`train`, 22 `test`, 2 `val`) |
| per split | 34,696 perturbed + 5,345 control cells per half | 5,980 perturbed + 5,345 control cells per half |
| self-checks | 10/10 PASS | 10/10 PASS; perts.json identical to the h1_seen 80-list |

Local copy `k562_seen_raw.h5ad`: 88,966 cells (78,275 perturbed from 587 perts + 10,691
controls) × 8,563 genes, 2.57 GB, sha256 `3bcdc909…`. Axis 6,640 genes; normalisation universe
7,598 K562 genes also in the RHJ HepG2 file; median library size 13,164 UMI (H1: 53,674).

## Jobs (2026-09-23, `scripts/repro/ood_seen.sh`)

| arm | `k562_seen` | `k562_seen_h1set` (after build 18367851) |
|---|---|---|
| baselines seeds 42–46 | 18368120–24 | 18368134–38 |
| Occam Y06/Q02/M06/M03 | 18368125 | 18368139 |
| TxPert export → score | 18368126 → 27 | 18368140 → 41 |
| STATE | 18368128 → 29 | 18368142 → 43 |
| CellFlow | 18368130 → 31 | 18368144 → 45 |
| PerturbDiff genept | 18368132 → 33 | 18368146 → 47 |

## Results

⏳ Pending.
