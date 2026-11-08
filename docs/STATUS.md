# STATUS — what is current. Read this first.

Last updated 2026-11-08. If something is not listed here, it is not current. Stale material
lives only in `archive/` (see `archive/README.md`) and must not be quoted or built on.

## Official Occam virtual-cell architecture (current)

For RHJ, H1, and K562, the official Occam reference is **`ABL_N_ALL`**: AA03 schedule,
concatenated GO+STRING, GenePT, and ESM2 sources, and `smooth_l1 + 0.5 * delta-cos`.
The 0.5 delta-cos weight is not loss-balanced. The older adjacency-only `ABL_A_DC`
is a representation ablation. For Norman, the official setup is **`NF32ALLA`**:
concatenated sources, NF32 schedule, `smooth_l1 + 0.0255 * delta-cos`, and
`decode_seq_mean`.

## Benchmarks (all frozen; `data/benchmarks/<name>/`, on HF under `benchmarks/<name>/`)

| benchmark | perturbation | assay | cell line | scored perts | doc | rebuild |
|---|---|---|---|---|---|---|
| `rhj_fewshot` | few-shot (withheld from HepG2, seen in Jurkat/RPE1) | seen | seen (HepG2) | 611 | `rhj_splithalf.md` | `build_benchmark_rhj.sh` |
| `k562_seen` | seen | seen | **unseen** (K562) | 528 RHJ test perts | `k562_seen.md` | `build_benchmark_k562.sh` |
| `k562_seen_h1set` | seen | seen | **unseen** (K562) | 80 (same as `h1_seen`) | `k562_seen.md` | `build_benchmark_k562.sh` |
| `h1_seen` | seen | **unseen** | **unseen** (H1 hESC) | 80 | `h1_seen.md` | `build_benchmark_h1.sh` |
| `norman_combo` | unseen PAIRS from seen singles | seen | K562 (Norman 2019) | 111 pairs (scored: `additive` / `nonadditive`, 36 each) | `norman_benchmark.md`, `norman_results.md` | `build_benchmark_norman.sh` |

The 80 shared by `h1_seen` and `k562_seen_h1set`: `data/benchmarks/shared/h1_k562_perts.json`.
Protocol for all: split-half, 5 seeded splits, cell-eval 0.6.0, **nocal only**, headline
metrics `pearson_delta` ↑, `mae` ↓ (error), `overlap_at_N` ↑, `discrimination_score_l1` ↑.

## Published checkpoints (HF `checkpoints/vcell/`)

| checkpoint | benchmark | result | card |
|---|---|---|---|
| `rhj_fewshot_Y06` | rhj_fewshot | `pearson_delta` **0.4043** (anchors: meanpert 0.1324 · transfer 0.2323 · experimental 0.3722) | `checkpoints/cards/rhj_fewshot_Y06_CARD.md` |
| `norman_combo_NF32ALLA` (seeds 42–46) | norman_combo | 5-seed Pearson delta 0.8955 ± 0.0032 / 0.8013 ± 0.0044 additive / nonadditive; bars: sum_of_singles 0.8822 / 0.7749 | `checkpoints/cards/norman_combo_NF32ALLA_CARD.md` |

## Current recipes and historical configs

| | recipe | doc |
|---|---|---|
| `scripts/grids/ablation/ABL_N_ALL.tsv` with `configs/rhj_fewshot_pertemb.yaml` | **ABL_N_ALL**: concatenated sources, smooth_l1 + 0.5 delta-cos, AA03 schedule | `ABLATION_PLAN.md` |
| `configs/rhj_fewshot.yaml` | historical Y06 Q02 reproduction config | `default_recipe.md` |
| `configs/norman_combo_official.yaml` | **NF32ALLA**: GO+STRING + GenePT + ESM2, smooth_l1 + 0.0255 delta-cos, 150 ep, bs 32, eta_min 3e-4, `decode_seq_mean` | `default_recipe.md` |

`configs/norman_combo.yaml` and `configs/norman_combo_pertemb.yaml` remain for historical
reproduction. NQ02, NF32, NF32DC, NF32DCA, and NF32ALL are Norman ablations, not official
checkpoints. The current RHJ/H1/K562 all-source arm is `ABL_N_ALL`; `ABL_A_DC` remains
an adjacency-only ablation.

## Current five-seed comparisons

`data/release/official_comparisons/official_tables.md` is the current comparison
entrypoint: all five benchmarks, all four headline metrics, competitors, and
`ctrlmean`/`ctrlcell` anchors. Its 11 panel tables contain 110 rows, each with
five seeds × five splits. Rebuild with `python scripts/official_score_tables.py`.
It combines the frozen `data/release/final/` scores with the Norman NF32ALLA
supplement in `data/release/norman_allsource/`; check both coverage ledgers.
The frozen final source has not been overwritten with a merged release.
See `docs/release_audit.md` for the remote audit and verification limits.

## Final table campaign (historical progress, 2026-09-23)

5 seeds × 5 splits for every model and baseline: `docs/FINAL_TABLES_PLAN.md`; completeness by
`scripts/score_ledger.py`; the table itself `data/release/final/scores_by_seed_split.csv`
(`scripts/final_table.py`).

## Ablation grid (in progress, 2026-09-23)

Y06 with one part changed per arm (perturbation prior, loss terms, encoder), 5 seeds, scored on
rhj + 3 OOD panels: `docs/ABLATION_PLAN.md`; driver `scripts/ablation/ablate.py` (arms in
`scripts/ablation/arms.tsv`); table `data/release/ablation/`.

## Where each current number lives

| topic | doc |
|---|---|
| current official arms, competitors and anchors (5 seeds × 5 splits) | `data/release/official_comparisons/official_tables.md` |
| rhj_fewshot: historical arms, baselines, competitors | `rhj_scoreboard.md` |
| rhj_fewshot: findings log (chronological; corrections inline) | `rhj_splithalf.md` |
| rhj_fewshot: baselines | `rhj_baseline_results.md` |
| rhj_fewshot: competitors (headline TxPert single-cell 0.3463, PerturbDiff genept 0.2449, STATE 0.2362, CellFlow 0.2154) and their deviations | `rhj_competitors_results.md`, `rhj_competitors.md` |
| norman_combo: arms, sweeps, competitors | `norman_results.md`, `norman_competitors_seeds.md` |
| h1_seen / k562_seen | `h1_seen.md`, `k562_seen.md` |
| every cell-eval metric, per split | `data/release/metrics/` (published arms), `data/release/metrics_all/` (all live arms) — `python scripts/export_split_metrics.py` |

## Experiment families (historical status recorded 2026-09-23)

| family | what | status |
|---|---|---|
| rhj Y01–06, AA01–05 | the schedule sweep that produced the default | done |
| rhj RB032–RB512 | Y06 at batch 32…512 | running |
| rhj Q02 / M06 / M03 | the three loss bases (30 ep) | done; used on h1/k562 |
| norman NA–NF | recipe sweep on norman_combo (batch, eta_min, epochs, loss scale, lr) | NF32, NF33–43 running |
| competitors | `txpert_single`, `state`, `cellflow`, `perturbdiff_genept` (+ onehot) × seeds 42–46, rhj and norman | seed replicates running |

## Archived (do not use)

`archive/` — pre-recipe sweep checkpoints and scores, the WITHDRAWN TxPert 0.3077 arm and its
diagnostics, the superseded 99-perturbation H1 panel, broken/aborted runs, smoke tests, and the
docs that tabulated them. Every move is logged in `archive/MANIFEST.tsv`.
