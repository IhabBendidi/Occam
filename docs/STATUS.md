# STATUS — what is current. Read this first.

Last updated 2026-09-23. If something is not listed here, it is not current. Stale material
lives only in `archive/` (see `archive/README.md`) and must not be quoted or built on.

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
| `norman_combo_NQ02` | norman_combo | 0.8169 / 0.7380 additive / nonadditive (bar: sum_of_singles 0.8824 / 0.7757) | `checkpoints/cards/norman_combo_NQ02_CARD.md` |

## Defaults (configs)

| | recipe | doc |
|---|---|---|
| `configs/rhj_fewshot.yaml` | Q02 loss base, 150 ep, eta_min 3e-4, bs 2048, lr 1e-3 (= Y06) | `default_recipe.md` |
| `configs/norman_combo.yaml` | Q02 Norman weights, 150 ep, **bs 256, eta_min 1e-3** (= NB10, adopted 2026-09-23), `decode_seq_mean` | `default_recipe.md` |

## Where each current number lives

| topic | doc |
|---|---|
| rhj_fewshot: live arms, baselines, competitors, one table | `rhj_scoreboard.md` |
| rhj_fewshot: findings log (chronological; corrections inline) | `rhj_splithalf.md` |
| rhj_fewshot: baselines | `rhj_baseline_results.md` |
| rhj_fewshot: competitors (headline TxPert single-cell 0.3463, PerturbDiff genept 0.2449, STATE 0.2362, CellFlow 0.2154) and their deviations | `rhj_competitors_results.md`, `rhj_competitors.md` |
| norman_combo: arms, sweeps, competitors | `norman_results.md`, `norman_competitors_seeds.md` |
| h1_seen / k562_seen | `h1_seen.md`, `k562_seen.md` |
| every cell-eval metric, per split | `data/release/metrics/` (published arms), `data/release/metrics_all/` (all live arms) — `python scripts/export_split_metrics.py` |

## Live experiment families (training runs in `runs/`)

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
