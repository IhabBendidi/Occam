# Occam ablation grid (rhj_fewshot, Y06 base, 5 seeds)

**Status 2026-09-23:** 7 arms (A1–A3, B0–B3) × 5 seeds launched, pending in the queue (jobs
18387759–765). A4/A5 (GenePT, ESM2) and C (encoder) are waiting on code, being prepared now.
No ablation results yet.

## 1. Objective and question

- **Overall objective.** Best model on the rhj_fewshot benchmark (perturbation withheld from
  HepG2, seen in Jurkat/RPE1). It also transfers to unseen cell lines (h1_seen, k562_seen,
  k562_seen_h1set). Headline metric: cell-eval `pearson_delta` (↑), with `mae` (↓, error),
  `overlap_at_N` (↑) and `discrimination_score_l1` (↑).
- **Question.** Which parts of the official model (Y06) carry its performance? Every arm is Y06
  with exactly one thing changed. The Y06 runs themselves (seed 42 is the published run, seeds
  43–46 are S4N_Y06) are the reference for every axis.
- **Fixed in every arm.**
  - Training data: rhj_fewshot.
  - Recipe: 150 ep, eta_min 3e-4, bs 2048, lr 1e-3.
  - Model: adjacency_map + FiLM.
  - Decoder: the released decoder, fine-tuned.
  - Seeds: 42–46.
  - Scoring: 5 split-halves on 4 panels, nocal.
- **Reading a result.**
  - The unit of replication is the seed.
  - Measured Y06 seed spread: `pearson_delta` 0.3953 ± 0.0077 over seeds 42/44/45/46.
  - An arm whose 5-seed mean sits within about 2 sd of Y06 has not been shown to matter.
  - An arm clearly below Y06 means the removed part carries performance on that panel.
  - The four panels can disagree. For example, a part may matter on rhj but not on the OOD cell
    lines.

Anchors (rhj_fewshot `pearson_delta`): meanpert 0.1324 · transfer 0.2323 · experimental 0.3722 ·
Y06 0.4043 (seed 42) / 0.3953 ± 0.0077 (4 seeds).

## 2. Arms (source of truth: `scripts/ablation/arms.tsv`)

### Axis A: how a perturbation is represented
| arm | change from Y06 | status |
|---|---|---|
| ref: Y06 | adjacency map over GO + STRING | done |
| `ABL_GO` (A1) | GO graph only | launched, 18387759 |
| `ABL_STR` (A2) | STRING graph only | launched, 18387760 |
| `ABL_LRN` (A3) | no prior: learnable embedding per gene (`no_gnn`) | launched, 18387761 |
| `ABL_GPT` (A4) | frozen GenePT text embedding per gene → same trainable module | code in progress |
| `ABL_ESM` (A5) | frozen ESM2 protein embedding per gene → same trainable module | code + embedding build in progress |

### Axis B: loss terms, added one at a time
| arm | loss | status |
|---|---|---|
| `ABL_L0` (B0) | smooth_l1 only | launched, 18387762 |
| `ABL_L1` (B1) | + profile cosine 0.5 | launched, 18387763 |
| `ABL_L2` (B2) | + delta-cosine 0.5 | launched, 18387764 |
| `ABL_L3` (B3) | + delta-magnitude 0.01363 | launched, 18387765 |
| ref: Y06 (B4) | + latent MSE 0.0044 (delta_z) | done |

### Axis C: the encoder (non-stream loader; the stream_v2 path is left untouched)
Y06 trains through `stream_v2`, which precomputes control embeddings and therefore only supports a
frozen encoder. Per the user (2026-09-23), the encoder arms use the other (non-stream) loader
rather than changing stream_v2. C0P is the same-path control: compare C1–C3 against **C0P**, not
against Y06, so that the loader change is not counted as an encoder effect.

| arm | encoder | status |
|---|---|---|
| `ABL_C0P` | pretrained, frozen (same as Y06, non-stream loader) | path check in progress |
| `ABL_C1` | pretrained, fine-tuned | path check in progress |
| `ABL_C2` | random init, trained from scratch | path check in progress |
| `ABL_C3` | random init, frozen | path check in progress |

## 3. How to run it (reusable)

    python scripts/ablation/ablate.py grids                     # scripts/grids/ablation/<arm>.tsv
    python scripts/ablation/ablate.py launch --arms ABL_GO --dry-run
    python scripts/ablation/ablate.py launch --arms ABL_GO      # sbatch, logged to runs/_ledger/ablation_launches.tsv
    python scripts/ablation/ablate.py register                  # score units -> configs/score_registry.tsv
    python scripts/ablation/ablate.py status                    # per arm x seed: job, trained, splits scored per panel
    python scripts/export_split_metrics.py --scope all --out data/release/metrics_all
    python scripts/final_table.py && python scripts/ablation/ablate.py table

- The arm spec lives in `scripts/ablation/arms.tsv`. Each row holds:
  - grid-column overrides of the Y06 row;
  - `COS_W`;
  - `EXTRA_SETS` config overrides;
  - walltime;
  - status.
- Launching goes through `scripts/slurm/rhj_loss_sweep.sbatch`. That script gained two opt-in
  hooks: `COS_W` and `EXTRA_SETS`. Both are absent from every earlier launch.
- The grids are generated from `scripts/grids/rhj_seeds.tsv:S43_Y06`, so they are Y06
  byte-for-byte except for the arm's own change.
- Scoring is automatic: the score ledger daemon picks up the registered units once
  `final_model.pth` exists.
- To add an arm: add one row to `arms.tsv`, then run `grids`, `launch` and `register`.

## 4. Outputs

- `data/release/ablation/scores_by_seed_split.csv` has one row per (panel, arm, seed, split).
  It contains every cell-eval metric, with columns `axis` and `change`. Y06 is included as `ref`.
- `data/release/ablation/summary.csv` gives, per panel × arm and for the 4 headline metrics:
  - the mean over seeds of the per-seed split-mean;
  - the sd across seeds;
  - n seeds and n rows.

## 5. Results

None yet.
