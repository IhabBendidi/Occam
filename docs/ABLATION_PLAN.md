# Occam ablation grid (rhj_fewshot, Y06 base, 5 seeds)

**Status 2026-09-23:** all 13 arms × 5 seeds launched (65 trainings): A1/A2 18387759/760 FAILED (loader list-coercion bug) → relaunched 18427497/498; A3 + B0–B3 18387761–765; A5 18388251; A4 18388318; C0P/C1/C2/C3 18392311 / 18392313 / 18392314 / 18392315 (first launch 18388346–349 cancelled while pending, for the loader speed-up).
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
| `ABL_GPT` (A4) | frozen GenePT v2 text embedding per gene → Y06's adjacency-map MLP (full-vocab table `data/assets/genept_full_occam_rhj_fewshot.pkl`, 17,843/18,216 vocab, 1,652/1,657 train, 621/621 test; builder `scripts/build_genept_full_vocab.py`) | launched, 18388318 |
| `ABL_ESM` (A5) | frozen ESM2 protein embedding per gene → Y06's adjacency-map MLP (esm2 650M, 1280-d, precomputed table from HF genbio-ai/foundation-models-perturbation @ d67ba210; pooling not recorded in the source; coverage 18,149/18,216 vocab, 621/621 test perts) | launched, 18388251 |

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
| `ABL_C0P` | pretrained, frozen (Y06's encoder, non-stream loader) | launched, 18392311 |
| `ABL_C1` | pretrained, fine-tuned, encoder_lr 1e-5 (codebase default) | launched, 18392313 |
| `ABL_C2` | random init, trained from scratch, encoder_lr 1e-3 (= rest of the network) | launched, 18392314 |
| `ABL_C3` | random init, frozen | launched, 18392315 |

Config `configs/rhj_fewshot_encoder.yaml` (= `rhj_fewshot.yaml` + the `encoder_lr` /
`random_encoder` keys; re-sync if the base changes). Walltime 12 h. The non-stream loader was sped up before launch: 531–1,600 → ~31,000 samples/s
(loader alone, CPU; epoch 1 ~3,500 while it fills a 17 GB RAM cache of the 6,983 used genes;
background-thread prefetch, one gather per buffer group, GPU densify). Rows read match the old
loader exactly (max diff 0.0); same 580,212 eligible cells / 608,686 samples per epoch; same
chunk order and proportional interleave. Changed: the control-draw / within-group shuffle random
stream. Expected ~2.5–4 h per 150-epoch arm (estimate; the vcell stage has no mid-run resume).

Non-stream path fixes made before launch (stream_v2 untouched; stream batches do not reach the
edited branch):
1. `delta_cos` / `delta_mag` were silently ZERO on the non-stream path -> group means loaded and
   `group_ids` carried per batch (`dataset.py`, `pipeline.py`). Without this C0P would not be Y06.
2. Checkpoints stored the init-time pretrained encoder snapshot -> a trained or random encoder
   is now saved as it is, and the best-checkpoint swap includes it.
3. `eta_min` 3e-4 is an absolute floor, which annealed a 1e-5 encoder UP to 3e-4 -> encoder floor
   is now `eta_min * encoder_lr / lr`; other groups unchanged.
4. Proportional interleave ported from stream_v2 (was roundrobin only).
Checked on CPU: 580,212 perturbed training cells (same as the Y06 log), zero HepG2 val/test
perturbations eligible for training. Not equalised: control-draw and shuffle random streams
differ between the loaders; controls are encoded per batch (float-tolerance difference).
Caveat for C1/C2: the latent-MSE target (0.0044) moves as the encoder trains.

### Axis D: loss terms REMOVED from the full recipe, on both bases (added 2026-09-24)
Question (user): which of the three secondary terms does the best recipe actually need, and does
the answer depend on the schedule? Each arm drops terms from the FULL recipe of its base; the
reference is that base's own 5 seeds (Y06: 150 ep, eta_min 3e-4; AA03: 180 ep, eta_min 7e-4).
Motivation: axis B shows magnitude + latent MSE trading mae for disc_l1 / OOD pΔ.

| arm | base | removed | kept | job |
|---|---|---|---|---|
| `ABL_Y_DC` | Y06 | magnitude, profile cosine, latent MSE | smooth_l1 + delta-cos | 18435524 |
| `ABL_Y_NOMAG` | Y06 | magnitude | smooth_l1 + profile cos + delta-cos + latent MSE | 18435525 |
| `ABL_Y_NOCOS` | Y06 | profile cosine | smooth_l1 + delta-cos + magnitude + latent MSE | 18435526 |
| `ABL_A_DC` | AA03 | magnitude, profile cosine, latent MSE | smooth_l1 + delta-cos | 18435527 |
| `ABL_A_NOMAG` | AA03 | magnitude | smooth_l1 + profile cos + delta-cos + latent MSE | 18435528 |
| `ABL_A_NOCOS` | AA03 | profile cosine | smooth_l1 + delta-cos + magnitude + latent MSE | 18435529 |

Reading: an arm at or above its base on pΔ / overlap / disc_l1 on the OOD panels, within ~1 sd on
mae, means the removed term is not needed. Same arms on both bases say whether the answer
depends on the schedule. AA03's OOD reference scores were added 2026-09-24 (jobs 18431111–125 +
local scoring on hop01).

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
- A4/A5 use `configs/rhj_fewshot_pertemb.yaml` (= `rhj_fewshot.yaml` + 2 opt-in keys, empty by
  default; `--set` cannot add keys, and adding them to the base config would change Y06's stage
  hash). Module: `FrozenEmbeddingModel` in `src/occam/vcell/adjacency_map.py`.
- Launching goes through `scripts/slurm/rhj_loss_sweep.sbatch`. That script gained two opt-in
  hooks: `COS_W` and `EXTRA_SETS`. Both are absent from every earlier launch.
- The grids are generated from `scripts/grids/rhj_seeds.tsv:S43_Y06`, so they are Y06
  byte-for-byte except for the arm's own change.
- Scoring is automatic: the score ledger daemon picks up the registered units once
  `final_model.pth` exists.
- To add an arm: add one row to `arms.tsv`, then run `grids`, `launch` and `register`.
- The `base` column picks the starting recipe (Y06 or AA03; `BASES` in `ablate.py`).
- `scripts/score_local.py --arm-regex <rx>` scores registered units on the local GPU when the
  queue is full (holds the matching pending cluster tasks, so nothing is scored twice).

### Embedding tables (not in git: 94 MB / 220 MB exceed GitHub's 100 MB limit)

    python scripts/ablation/build_esm2_embeddings.py        # ESM2, --source precomputed, CPU ~1 min
    python scripts/build_genept_full_vocab.py               # GenePT full vocab, CPU ~1 min

Each writes `data/assets/<table>.pkl` plus a `.json` sidecar (committed) with the source,
sha256, retired-name handling and per-split coverage.

## 4. Outputs

- `data/release/ablation/scores_by_seed_split.csv` has one row per (panel, arm, seed, split).
  It contains every cell-eval metric, with columns `axis` and `change`. Y06 is included as `ref`.
- `data/release/ablation/summary.csv` gives, per panel × arm and for the 4 headline metrics:
  - the mean over seeds of the per-seed split-mean;
  - the sd across seeds;
  - n seeds and n rows.

## 5. Results — PARTIAL, read 2026-09-24 ~04:00 (`data/release/ablation/summary.csv`)

Mean ± sd across seeds of the per-seed split-mean; nocal; `mae` is an error (lower is better).
Arms with < 5 seeds are PARTIAL (seeds/rows column). GO-only / STRING-only relaunched 18427497/498
after a loader list-coercion failure; GenePT and C0P–C3 trained, scoring pending (cluster full).

### rhj_fewshot

| arm | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ | seeds/rows |
|---|---|---|---|---|---|
| B0 smooth_l1 only | 0.3586 ± 0.0050 | 0.0206 ± 0.0001 | 0.0762 ± 0.0006 | 0.6696 ± 0.0061 | 5/25 |
| B1 + profile cos | 0.3529 ± 0.0057 | 0.0207 ± 0.0001 | 0.0766 ± 0.0018 | 0.6659 ± 0.0050 | 5/25 |
| B2 + delta-cos | 0.3995 ± 0.0011 | 0.0231 ± 0.0006 | 0.0902 ± 0.0023 | 0.7414 ± 0.0040 | 4/20 |
| B3 + delta-mag | 0.3961 ± 0.0030 | 0.0219 ± 0.0004 | 0.0913 ± 0.0018 | 0.7293 ± 0.0026 | 3/15 |
| **Y06** = B4 + latent MSE = A0 GO+STRING | 0.3957 ± 0.0068 | 0.0220 ± 0.0006 | 0.0885 ± 0.0057 | 0.7233 ± 0.0040 | 5/25 |
| A3 learnable emb. | 0.3809 ± 0.0053 | 0.0230 ± 0.0010 | 0.0847 ± 0.0034 | 0.7192 ± 0.0060 | 5/25 |
| A5 ESM2 | 0.3796 ± 0.0018 | 0.0222 ± 0.0003 | 0.0880 ± 0.0025 | 0.7043 ± 0.0013 | 2/7 |

### h1_seen

| arm | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ | seeds/rows |
|---|---|---|---|---|---|
| B0 smooth_l1 only | 0.0836 ± 0.0010 | 0.1375 ± 0.0010 | 0.3215 ± 0.0002 | 0.5560 ± 0.0014 | 5/25 |
| B1 + profile cos | 0.0828 ± 0.0012 | 0.1372 ± 0.0006 | 0.3227 ± 0.0009 | 0.5530 ± 0.0036 | 5/25 |
| B2 + delta-cos | 0.0972 ± 0.0029 | 0.1407 ± 0.0011 | 0.3236 ± 0.0014 | 0.6139 ± 0.0019 | 4/20 |
| B3 + delta-mag | 0.0957 ± 0.0043 | 0.1400 ± 0.0007 | 0.3235 ± 0.0002 | 0.5979 ± 0.0060 | 3/15 |
| **Y06** = B4 + latent MSE = A0 GO+STRING | 0.0939 ± 0.0032 | 0.1397 ± 0.0015 | 0.3231 ± 0.0014 | 0.5953 ± 0.0088 | 5/25 |
| A3 learnable emb. | 0.0945 ± 0.0049 | 0.1402 ± 0.0009 | 0.3220 ± 0.0009 | 0.5871 ± 0.0044 | 5/25 |
| A5 ESM2 | 0.0965 ± 0.0007 | 0.1390 ± 0.0000 | 0.3236 ± 0.0002 | 0.6038 ± 0.0069 | 2/10 |

### k562_seen

| arm | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ | seeds/rows |
|---|---|---|---|---|---|
| B0 smooth_l1 only | 0.1396 ± 0.0053 | 0.0467 ± 0.0003 | 0.0414 ± 0.0012 | 0.6234 ± 0.0035 | 5/25 |
| B1 + profile cos | 0.1381 ± 0.0033 | 0.0474 ± 0.0005 | 0.0422 ± 0.0008 | 0.6144 ± 0.0191 | 5/25 |
| B2 + delta-cos | 0.1790 ± 0.0048 | 0.0512 ± 0.0005 | 0.0518 ± 0.0003 | 0.7138 ± 0.0062 | 4/20 |
| B3 + delta-mag | 0.1691 ± 0.0058 | 0.0494 ± 0.0005 | 0.0468 ± 0.0011 | 0.7010 ± 0.0063 | 3/15 |
| **Y06** = B4 + latent MSE = A0 GO+STRING | 0.1713 ± 0.0039 | 0.0491 ± 0.0004 | 0.0474 ± 0.0012 | 0.6843 ± 0.0106 | 5/25 |
| A3 learnable emb. | 0.1752 ± 0.0105 | 0.0488 ± 0.0013 | 0.0469 ± 0.0012 | 0.6682 ± 0.0093 | 5/25 |
| A5 ESM2 | 0.1784 ± 0.0011 | 0.0498 ± 0.0007 | 0.0486 ± 0.0002 | 0.6843 ± 0.0185 | 2/7 |

### k562_seen_h1set

| arm | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ | seeds/rows |
|---|---|---|---|---|---|
| B0 smooth_l1 only | 0.1273 ± 0.0080 | 0.0464 ± 0.0004 | 0.0408 ± 0.0016 | 0.5877 ± 0.0059 | 5/25 |
| B1 + profile cos | 0.1258 ± 0.0069 | 0.0469 ± 0.0005 | 0.0421 ± 0.0011 | 0.5773 ± 0.0111 | 5/25 |
| B2 + delta-cos | 0.1750 ± 0.0059 | 0.0507 ± 0.0005 | 0.0569 ± 0.0033 | 0.6718 ± 0.0062 | 4/20 |
| B3 + delta-mag | 0.1604 ± 0.0097 | 0.0491 ± 0.0004 | 0.0482 ± 0.0025 | 0.6611 ± 0.0013 | 3/15 |
| **Y06** = B4 + latent MSE = A0 GO+STRING | 0.1609 ± 0.0034 | 0.0485 ± 0.0004 | 0.0470 ± 0.0025 | 0.6475 ± 0.0079 | 5/25 |
| A3 learnable emb. | 0.1696 ± 0.0131 | 0.0482 ± 0.0016 | 0.0492 ± 0.0020 | 0.6348 ± 0.0073 | 5/25 |
| A5 ESM2 | 0.1722 ± 0.0034 | 0.0491 ± 0.0001 | 0.0505 ± 0.0021 | 0.6559 ± 0.0058 | 2/10 |

**Reading so far.** delta-cosine is the lever on every panel and every metric except mae
(B1 → B2: rhj pΔ +0.047, disc_l1 +0.076; K562 pΔ +0.04–0.05). Profile cosine adds nothing (B0 ≈ B1).
Magnitude (B3) and latent MSE (Y06) buy mae (0.0231 → 0.0219/0.0220 on rhj) and cost disc_l1
(0.7414 → 0.7293 → 0.7233) and OOD pΔ (k562_seen_h1set 0.1750 → 0.1604/0.1609): a trade-off, not
a free gain. Learnable (A3) and ESM2 (A5, 2 seeds) are ~0.015 pΔ below Y06 on rhj but at or
above Y06 on the OOD panels.

**2026-09-24 04:20** — `ABL_L2` seed 43 (task 18387764_1) hit its 8 h limit at epoch 2 on **hop02**,
which ran ~200 s/iteration (normal ~0.07 s). Relaunched with `--exclude hop39,hop02` (job
18439044; `ablate.py launch --exclude` added). hop02 is a slow-node suspect for all training.

**2026-09-24 ~04:50 — PARKED (user): GenePT, ESM2, C0P, C1, C2, C3.** The model is expected to
change before these are interpreted, so they will be retrained. All 30 trainings had finished;
their scoring was cancelled (103 score arrays) and their 120 units removed from
`configs/score_registry.tsv`. Status `parked` in `arms.tsv` (launch/register skip it). ESM2's 75
already-scored splits stay on disk and in the table as PARTIAL (2 seeds on rhj) — not a result.
**2026-09-24 — PARKED (user): GO-only, STRING-only** — scoring cancelled and 40 units removed from
the registry; their trainings (18427497/498) were then CANCELLED too (user: to be redone with the final architecture).

**2026-09-24 ~08:25 — Y06 ablations STOPPED (user: AA03 is the recipe going forward).**
`ABL_Y_DC/NOMAG/NOCOS` parked (all 15 trainings had finished; scoring cancelled, 60 units removed);
`ABL_L2` seed 43 trained (final_model.pth exists) but its scoring was cancelled, so B2 stays at
4 seeds. 48 score arrays cancelled. Only the AA03 loss-removal arms (axis D, base AA03) continue.

## 6. AA03 loss ablation (axis D, base AA03) — COMPLETE, read 2026-09-24 13:05

5 seeds × 5 splits per arm and panel, nocal. Mean ± sd across seeds of the per-seed split-mean;
'vs AA03' = paired by seed (seeds where the arm is higher). mae is an error (lower is better).

### rhj_fewshot
| arm | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ | rows |
|---|---|---|---|---|---|
| **AA03** (full: smooth_l1 + profile cos + delta-cos + mag + latent MSE) | 0.3994 ± 0.0040 | 0.0222 ± 0.0008 | 0.0952 ± 0.0012 | 0.7316 ± 0.0037 | 25 |
| − magnitude | 0.4017 ± 0.0018 | 0.0231 ± 0.0002 | 0.0957 ± 0.0020 | 0.7451 ± 0.0017 | 25 |
| ↳ vs AA03 | +0.0023 (4/5) | +0.0008 (4/5) | +0.0005 (3/5) | +0.0135 (5/5) | |
| − profile cosine | 0.4006 ± 0.0039 | 0.0220 ± 0.0002 | 0.0958 ± 0.0022 | 0.7338 ± 0.0059 | 25 |
| ↳ vs AA03 | +0.0011 (3/5) | -0.0002 (2/5) | +0.0006 (2/5) | +0.0022 (3/5) | |
| − mag − profile cos − latent (smooth_l1 + delta-cos) | 0.4007 ± 0.0008 | 0.0261 ± 0.0007 | 0.0893 ± 0.0035 | 0.7613 ± 0.0030 | 25 |
| ↳ vs AA03 | +0.0013 (3/5) | +0.0038 (5/5) | -0.0059 (0/5) | +0.0298 (5/5) | |

### h1_seen
| arm | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ | rows |
|---|---|---|---|---|---|
| **AA03** (full: smooth_l1 + profile cos + delta-cos + mag + latent MSE) | 0.0948 ± 0.0039 | 0.1398 ± 0.0016 | 0.3232 ± 0.0009 | 0.6056 ± 0.0095 | 25 |
| − magnitude | 0.0913 ± 0.0016 | 0.1391 ± 0.0004 | 0.3235 ± 0.0005 | 0.6050 ± 0.0062 | 25 |
| ↳ vs AA03 | -0.0035 (1/5) | -0.0007 (2/5) | +0.0003 (3/5) | -0.0007 (2/5) | |
| − profile cosine | 0.0984 ± 0.0038 | 0.1407 ± 0.0015 | 0.3231 ± 0.0002 | 0.5985 ± 0.0068 | 25 |
| ↳ vs AA03 | +0.0036 (4/5) | +0.0009 (3/5) | -0.0001 (2/5) | -0.0072 (2/5) | |
| − mag − profile cos − latent (smooth_l1 + delta-cos) | 0.1049 ± 0.0032 | 0.1418 ± 0.0011 | 0.3249 ± 0.0010 | 0.6364 ± 0.0134 | 25 |
| ↳ vs AA03 | +0.0101 (5/5) | +0.0020 (5/5) | +0.0017 (5/5) | +0.0308 (5/5) | |

### k562_seen
| arm | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ | rows |
|---|---|---|---|---|---|
| **AA03** (full: smooth_l1 + profile cos + delta-cos + mag + latent MSE) | 0.1785 ± 0.0123 | 0.0498 ± 0.0004 | 0.0496 ± 0.0011 | 0.6957 ± 0.0108 | 25 |
| − magnitude | 0.1850 ± 0.0037 | 0.0508 ± 0.0003 | 0.0541 ± 0.0012 | 0.7147 ± 0.0048 | 25 |
| ↳ vs AA03 | +0.0064 (4/5) | +0.0010 (5/5) | +0.0046 (5/5) | +0.0190 (5/5) | |
| − profile cosine | 0.1792 ± 0.0075 | 0.0493 ± 0.0004 | 0.0496 ± 0.0011 | 0.6893 ± 0.0121 | 25 |
| ↳ vs AA03 | +0.0007 (3/5) | -0.0006 (1/5) | +0.0001 (3/5) | -0.0064 (1/5) | |
| − mag − profile cos − latent (smooth_l1 + delta-cos) | 0.2055 ± 0.0085 | 0.0560 ± 0.0007 | 0.0634 ± 0.0029 | 0.7504 ± 0.0061 | 25 |
| ↳ vs AA03 | +0.0269 (5/5) | +0.0062 (5/5) | +0.0138 (5/5) | +0.0547 (5/5) | |

### k562_seen_h1set
| arm | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ | rows |
|---|---|---|---|---|---|
| **AA03** (full: smooth_l1 + profile cos + delta-cos + mag + latent MSE) | 0.1715 ± 0.0143 | 0.0493 ± 0.0004 | 0.0519 ± 0.0008 | 0.6576 ± 0.0059 | 25 |
| − magnitude | 0.1758 ± 0.0052 | 0.0502 ± 0.0003 | 0.0589 ± 0.0027 | 0.6665 ± 0.0031 | 25 |
| ↳ vs AA03 | +0.0043 (4/5) | +0.0009 (4/5) | +0.0069 (5/5) | +0.0089 (5/5) | |
| − profile cosine | 0.1740 ± 0.0106 | 0.0487 ± 0.0007 | 0.0528 ± 0.0023 | 0.6611 ± 0.0042 | 25 |
| ↳ vs AA03 | +0.0025 (3/5) | -0.0006 (1/5) | +0.0009 (3/5) | +0.0035 (2/5) | |
| − mag − profile cos − latent (smooth_l1 + delta-cos) | 0.1952 ± 0.0096 | 0.0549 ± 0.0006 | 0.0673 ± 0.0028 | 0.6970 ± 0.0089 | 25 |
| ↳ vs AA03 | +0.0237 (5/5) | +0.0057 (5/5) | +0.0154 (5/5) | +0.0394 (5/5) | |

**Reading.** Removing all three secondary terms (smooth_l1 + delta-cos only) is 5/5 better than
AA03 on pΔ, overlap_at_N and disc_l1 on all three OOD panels (k562_seen pΔ +0.027, disc_l1
+0.055; h1_seen pΔ +0.010) and on rhj disc_l1 (+0.030), at a 5/5 mae cost (+12–17% relative:
rhj 0.0222 → 0.0261, k562_seen 0.0498 → 0.0560) and −0.006 rhj overlap_at_N. Removing magnitude
alone gives a smaller version of the same trade (better K562 overlap/disc, +0.001 mae; H1 pΔ
−0.0035). Removing profile cosine alone is within seed noise everywhere. The terms buy mae; they
cost ranking/discrimination and OOD transfer.

## 7. Ablations on the NEW DEFAULT (AA03 + smooth_l1 + delta-cos), launched 2026-09-24

Base `AA03_DC` in `ablate.py` (grid `scripts/grids/rhj_seeds_aa03_dc.tsv`, COS_W 0.0); reference =
the `ABL_A_DC` 5-seed runs (§6). Each arm = new default with ONE change, seeds 42–46, scored on
rhj + h1_seen + k562_seen + k562_seen_h1set. New arm ids (`ABL_N_*`) so nothing mixes with the
parked Y06-based arms.

| arm | axis | change | job |
|---|---|---|---|
| `ABL_N_GO` | A | graph prior = GO only | 18508696 |
| `ABL_N_STR` | A | graph prior = STRING only | 18508697 |
| `ABL_N_LRN` | A | learnable embedding per gene, no prior | 18508698 |
| `ABL_N_GPT` | A | frozen GenePT (full-vocab table) | 18508699 |
| `ABL_N_ESM` | A | frozen ESM2 | 18508700 |
| `ABL_N_ALL` | A | ALL sources concatenated: GO+STRING adjacency row ‖ GenePT ‖ ESM2 (`ConcatKnowledgeModel`; input 22,568 = 18,216 + 3,072 + 1,280; each table z-scored then scaled so its mean squared row norm = the adjacency's, 46.0; +561k params) | 18509294 |
| `ABL_N_L0` | B | minus delta-cos (smooth_l1 only) | 18508721 |

Reading: an arm below ABL_A_DC by more than ~2 seed-sd on a panel means the default's
representation (or delta-cos, for L0) carries performance there; ALL above the default means the
sources are complementary.

**2026-09-24 13:55 — run check.** 35 trainings running (Norman NF32-DC epoch ~41/150; ABL_N_* epoch
14–17/180, ≈75 s/epoch → ~3.7 h per run), no errors, no NaN, no OOM; ABL_N_ALL (18509294) pending
on QOSGrpGRES. Pre-existing issue found: **wandb has not logged for any orchestrated vcell run**
(`wandb init failed: local variable 'timestamp' referenced before assignment`,
`src/occam/vcell/pipeline.py:4088` — `timestamp` is only defined when no run_dir is passed, and
the orchestrator always passes one). Caught, so training and scoring are unaffected; metrics are
in the logs and eval roots. Not fixed (would start sending runs to wandb) — pending user decision.

### Loss-term balance in the new default (measured 2026-09-24)
Per-term losses are not logged (wandb broken, see above), so the delta-cos term is read as the
total-loss difference between `ABL_A_DC` (smooth_l1 + 0.5·dcos) and `ABL_N_L0` (smooth_l1 only),
same AA03 schedule, seeds 42–46 (assumes smooth_l1 is similar in both runs).

| epoch | smooth_l1 | 0.5 × dcos | dcos share of the loss |
|---|---|---|---|
| 1 | 0.0062 | 0.394 | 98% |
| 10 | 0.0049 | 0.126 | 96% |
| 50 | 0.0048 | 0.059 | 92% |
| 180 | 0.0048 | 0.040–0.045 | 89–90% |

delta-cos dominates (≈90–98%). The dcos weight giving a 50/50 split: **≈0.008–0.01 at init**
(= the M03 base's "dcos at parity" point, 0.01), **0.039 averaged over training**, **≈0.05 at the
end** (raw dcos falls 0.79 → 0.09 while smooth_l1 stays ≈0.005). Norman equivalents scale by
0.32977/0.5 = 0.6595 (`scripts/norman/norman_loss_scale.py` rule): 0.0066 / 0.0255 / 0.033.

**Launched 2026-09-24 (user: both balance points), 5 seeds each:**

| arm | benchmark | delta-cos weight | balance point | job |
|---|---|---|---|---|
| `ABL_N_DCI` | rhj_fewshot (+ H1/K562 scoring) | 0.01 | parity at init | 18511028 |
| `ABL_N_DCA` | rhj_fewshot (+ H1/K562 scoring) | 0.039 | parity averaged over training | 18511029 |
| `NF32DCI` | norman_combo | 0.0066 | parity at init | 18511030 |
| `NF32DCA` | norman_combo | 0.0255 | parity averaged over training | 18511031 |

References: `ABL_A_DC` (rhj, dcos 0.5) and `NF32DC` (Norman, dcos 0.32977), both at ~90–98% dcos.

### PARTIAL results, read 2026-09-24 17:10 — new-default ablations (§7)
Each arm: mean ± sd over its scored (seed, split) rows, and in brackets the mean paired difference vs
the new default (ABL_A_DC) on the SAME pairs. 2–4 seeds per arm; STRING-only and ALL not scored yet.

### rhj_fewshot
| arm (vs new default on the same pairs) | pairs | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ |
|---|---|---|---|---|---|
| **new default** (AA03, smooth_l1 + delta-cos), all 25 | 25 | 0.4007 ± 0.0022 | 0.0261 ± 0.0007 | 0.0893 ± 0.0034 | 0.7613 ± 0.0033 |
| GO only | 10 (2 seeds) | 0.3892 ± 0.0032 (-0.0110) | 0.0275 ± 0.0012 (+0.0019) | 0.0871 ± 0.0017 (-0.0032) | 0.7532 ± 0.0014 (-0.0053) |
| learnable emb. | 15 (3 seeds) | 0.3869 ± 0.0033 (-0.0136) | 0.0287 ± 0.0023 (+0.0024) | 0.0823 ± 0.0055 (-0.0049) | 0.7515 ± 0.0022 (-0.0105) |
| GenePT | 15 (3 seeds) | 0.3948 ± 0.0029 (-0.0062) | 0.0274 ± 0.0007 (+0.0009) | 0.0884 ± 0.0021 (-0.0002) | 0.7627 ± 0.0029 (-0.0005) |
| ESM2 | 14 (3 seeds) | 0.3972 ± 0.0023 (-0.0041) | 0.0278 ± 0.0015 (+0.0017) | 0.0911 ± 0.0041 (+0.0008) | 0.7583 ± 0.0046 (-0.0034) |
| smooth_l1 only (no delta-cos) | 9 (3 seeds) | 0.3572 ± 0.0041 (-0.0433) | 0.0205 ± 0.0000 (-0.0053) | 0.0799 ± 0.0027 (-0.0095) | 0.6705 ± 0.0020 (-0.0894) |

### h1_seen
| arm (vs new default on the same pairs) | pairs | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ |
|---|---|---|---|---|---|
| **new default** (AA03, smooth_l1 + delta-cos), all 25 | 25 | 0.1049 ± 0.0034 | 0.1418 ± 0.0010 | 0.3249 ± 0.0015 | 0.6364 ± 0.0125 |
| GO only | 10 (2 seeds) | 0.1069 ± 0.0040 (+0.0043) | 0.1409 ± 0.0009 (-0.0005) | 0.3236 ± 0.0013 (-0.0008) | 0.6224 ± 0.0043 (-0.0042) |
| learnable emb. | 15 (3 seeds) | 0.1095 ± 0.0083 (+0.0041) | 0.1418 ± 0.0008 (+0.0003) | 0.3250 ± 0.0025 (-0.0003) | 0.6277 ± 0.0141 (-0.0139) |
| GenePT | 15 (3 seeds) | 0.1098 ± 0.0024 (+0.0034) | 0.1418 ± 0.0007 (-0.0002) | 0.3241 ± 0.0012 (-0.0011) | 0.6411 ± 0.0044 (-0.0019) |
| ESM2 | 15 (3 seeds) | 0.1094 ± 0.0036 (+0.0033) | 0.1411 ± 0.0005 (-0.0014) | 0.3247 ± 0.0014 (-0.0002) | 0.6531 ± 0.0097 (+0.0167) |
| smooth_l1 only (no delta-cos) | 17 (4 seeds) | 0.0827 ± 0.0018 (-0.0202) | 0.1379 ± 0.0005 (-0.0035) | 0.3220 ± 0.0011 (-0.0023) | 0.5603 ± 0.0060 (-0.0706) |

### k562_seen
| arm (vs new default on the same pairs) | pairs | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ |
|---|---|---|---|---|---|
| **new default** (AA03, smooth_l1 + delta-cos), all 25 | 25 | 0.2055 ± 0.0078 | 0.0560 ± 0.0006 | 0.0634 ± 0.0028 | 0.7504 ± 0.0058 |
| GO only | 10 (2 seeds) | 0.2232 ± 0.0073 (+0.0227) | 0.0550 ± 0.0006 (-0.0006) | 0.0648 ± 0.0021 (+0.0032) | 0.7370 ± 0.0065 (-0.0096) |
| learnable emb. | 15 (3 seeds) | 0.2101 ± 0.0229 (+0.0009) | 0.0562 ± 0.0008 (+0.0002) | 0.0613 ± 0.0059 (-0.0026) | 0.7463 ± 0.0033 (-0.0062) |
| GenePT | 15 (3 seeds) | 0.2170 ± 0.0071 (+0.0081) | 0.0564 ± 0.0004 (+0.0000) | 0.0662 ± 0.0014 (+0.0016) | 0.7465 ± 0.0035 (-0.0065) |
| ESM2 | 14 (3 seeds) | 0.2210 ± 0.0112 (+0.0164) | 0.0569 ± 0.0008 (+0.0005) | 0.0662 ± 0.0030 (+0.0021) | 0.7471 ± 0.0066 (-0.0024) |
| smooth_l1 only (no delta-cos) | 6 (2 seeds) | 0.1318 ± 0.0049 (-0.0690) | 0.0466 ± 0.0002 (-0.0090) | 0.0411 ± 0.0005 (-0.0205) | 0.6185 ± 0.0049 (-0.1279) |

### k562_seen_h1set
| arm (vs new default on the same pairs) | pairs | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ |
|---|---|---|---|---|---|
| **new default** (AA03, smooth_l1 + delta-cos), all 25 | 25 | 0.1952 ± 0.0097 | 0.0549 ± 0.0006 | 0.0673 ± 0.0034 | 0.6970 ± 0.0085 |
| GO only | 10 (2 seeds) | 0.2200 ± 0.0115 (+0.0321) | 0.0540 ± 0.0001 (-0.0005) | 0.0714 ± 0.0031 (+0.0065) | 0.6888 ± 0.0095 (-0.0005) |
| learnable emb. | 15 (3 seeds) | 0.2031 ± 0.0237 (+0.0042) | 0.0547 ± 0.0005 (-0.0000) | 0.0680 ± 0.0058 (-0.0006) | 0.6926 ± 0.0037 (-0.0081) |
| GenePT | 15 (3 seeds) | 0.2098 ± 0.0094 (+0.0098) | 0.0552 ± 0.0005 (-0.0000) | 0.0729 ± 0.0030 (+0.0039) | 0.6984 ± 0.0038 (-0.0037) |
| ESM2 | 15 (3 seeds) | 0.2151 ± 0.0136 (+0.0190) | 0.0557 ± 0.0008 (+0.0004) | 0.0743 ± 0.0042 (+0.0074) | 0.7004 ± 0.0039 (+0.0063) |
| smooth_l1 only (no delta-cos) | 20 (4 seeds) | 0.1248 ± 0.0045 (-0.0668) | 0.0462 ± 0.0002 (-0.0086) | 0.0415 ± 0.0017 (-0.0253) | 0.5956 ± 0.0068 (-0.1006) |

**`ABL_N_DCO` (launched 2026-09-24, user): new default minus smooth_l1 — loss = delta-cos only.**
smooth_l1 is zeroed through existing knobs: `recon_split=true` with `recon_ctrl_weight = recon_delta_weight = 0`
(recon_loss = 0·L_ctrl + 0·L_delta, exactly 0, no gradient); profile cosine 0, magnitude 0, latent off;
delta-cos weight 0.5 as in the default. Nothing then constrains the predicted effect's SIZE
(delta-cos is scale-invariant), so expect mae to be the metric that moves. `ablate.py run_tag`
now renders the `_rsplit…` suffix the same way the launcher does. Job in `runs/_ledger/ablation_launches.tsv`.
Note found while setting this up: with profile cosine on (Y06/AA03, 0.5) the code blends
`0.5·smooth_l1 + 0.5·profile_cos`, so turning profile cosine off also DOUBLED smooth_l1's weight
in the ABL_A_DC / NOCOS arms.

**ABL_N_ALL training done (2026-09-24 17:15), scoring queued.** Early signal only — best VALIDATION
pΔ from the training logs (HepG2 val, 103 perts; NOT the split-half test score), seeds 42–46:
ALL 0.4392 / 0.4427 / 0.4423 / 0.4448 / 0.4451 (mean 0.4428, sd 0.0024) · new default (ABL_A_DC)
0.4409 / 0.4460 / 0.4475 / 0.4557 / 0.4363 (0.4453, 0.0072) · ESM2 0.4415 · GO 0.4336 · GenePT 0.4315.

### Results read 2026-09-24 ~18:20 — new-default ablations (supersedes the 17:10 partial table)
Mean ± sd across seeds of the per-seed split-mean (partial seeds included); bracket = mean paired
difference vs the new default on the SAME (seed, split) pairs. ESM2, GenePT, learnable, smooth_l1-only
COMPLETE (5×5); GO 4–5 seeds; STRING and ALL partial (ALL rhj 2 seeds scored locally on hop01).

### rhj_fewshot
| arm | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ | seeds × splits |
|---|---|---|---|---|---|
| **new default** (GO+STRING adjacency) | 0.4007 ± 0.0008 | 0.0261 ± 0.0007 | 0.0893 ± 0.0035 | 0.7613 ± 0.0030 | 25 rows (5 full seeds) |
| ALL: adjacency ‖ GenePT ‖ ESM2 | 0.3993 ± 0.0048 (-0.0013) | 0.0279 ± 0.0020 (+0.0024) | 0.0820 ± 0.0055 (-0.0094) | 0.7620 ± 0.0015 (+0.0026) | 14 rows (2 full seeds) |
| ESM2 | 0.3973 ± 0.0015 (-0.0034) | 0.0272 ± 0.0015 (+0.0012) | 0.0926 ± 0.0043 (+0.0034) | 0.7584 ± 0.0036 (-0.0029) | 25 rows (5 full seeds) |
| GenePT | 0.3962 ± 0.0028 (-0.0045) | 0.0269 ± 0.0009 (+0.0008) | 0.0895 ± 0.0021 (+0.0002) | 0.7614 ± 0.0029 (+0.0001) | 25 rows (5 full seeds) |
| GO only | 0.3917 ± 0.0046 (-0.0097) | 0.0267 ± 0.0011 (+0.0006) | 0.0888 ± 0.0025 (-0.0002) | 0.7530 ± 0.0015 (-0.0085) | 23 rows (4 full seeds) |
| STRING only | 0.3947 ± 0.0036 (-0.0063) | 0.0271 ± 0.0016 (+0.0008) | 0.0857 ± 0.0028 (-0.0039) | 0.7518 ± 0.0022 (-0.0104) | 14 rows (2 full seeds) |
| learnable emb. | 0.3879 ± 0.0028 (-0.0128) | 0.0278 ± 0.0023 (+0.0018) | 0.0853 ± 0.0061 (-0.0040) | 0.7516 ± 0.0014 (-0.0097) | 25 rows (5 full seeds) |
| smooth_l1 only (no delta-cos) | 0.3547 ± 0.0049 (-0.0460) | 0.0205 ± 0.0000 (-0.0055) | 0.0783 ± 0.0027 (-0.0110) | 0.6694 ± 0.0022 (-0.0919) | 25 rows (5 full seeds) |

### h1_seen
| arm | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ | seeds × splits |
|---|---|---|---|---|---|
| **new default** (GO+STRING adjacency) | 0.1049 ± 0.0032 | 0.1418 ± 0.0011 | 0.3249 ± 0.0010 | 0.6364 ± 0.0134 | 25 rows (5 full seeds) |
| ALL: adjacency ‖ GenePT ‖ ESM2 | 0.1076 ± 0.0058 (+0.0043) | 0.1424 ± 0.0014 (+0.0007) | 0.3249 ± 0.0012 (+0.0009) | 0.6481 ± 0.0072 (+0.0157) | 15 rows (2 full seeds) |
| ESM2 | 0.1090 ± 0.0028 (+0.0042) | 0.1412 ± 0.0005 (-0.0006) | 0.3250 ± 0.0008 (+0.0002) | 0.6483 ± 0.0106 (+0.0119) | 25 rows (5 full seeds) |
| GenePT | 0.1087 ± 0.0025 (+0.0039) | 0.1413 ± 0.0009 (-0.0005) | 0.3241 ± 0.0008 (-0.0008) | 0.6443 ± 0.0053 (+0.0078) | 25 rows (5 full seeds) |
| GO only | 0.1048 ± 0.0031 (-0.0001) | 0.1409 ± 0.0007 (-0.0009) | 0.3245 ± 0.0011 (-0.0003) | 0.6336 ± 0.0111 (-0.0028) | 25 rows (5 full seeds) |
| STRING only | 0.1074 ± 0.0039 (+0.0013) | 0.1424 ± 0.0012 (-0.0002) | 0.3241 ± 0.0006 (-0.0007) | 0.6309 ± 0.0153 (-0.0055) | 15 rows (3 full seeds) |
| learnable emb. | 0.1074 ± 0.0077 (+0.0026) | 0.1417 ± 0.0011 (-0.0001) | 0.3246 ± 0.0022 (-0.0002) | 0.6261 ± 0.0157 (-0.0103) | 25 rows (5 full seeds) |
| smooth_l1 only (no delta-cos) | 0.0831 ± 0.0003 (-0.0218) | 0.1378 ± 0.0005 (-0.0040) | 0.3223 ± 0.0007 (-0.0026) | 0.5584 ± 0.0060 (-0.0781) | 25 rows (5 full seeds) |

### k562_seen
| arm | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ | seeds × splits |
|---|---|---|---|---|---|
| **new default** (GO+STRING adjacency) | 0.2055 ± 0.0085 | 0.0560 ± 0.0007 | 0.0634 ± 0.0029 | 0.7504 ± 0.0061 | 25 rows (5 full seeds) |
| ALL: adjacency ‖ GenePT ‖ ESM2 | 0.2183 ± 0.0129 (+0.0155) | 0.0565 ± 0.0012 (+0.0003) | 0.0658 ± 0.0032 (+0.0033) | 0.7479 ± 0.0031 (-0.0002) | 9 rows (0 full seeds) |
| ESM2 | 0.2163 ± 0.0122 (+0.0108) | 0.0563 ± 0.0012 (+0.0002) | 0.0655 ± 0.0030 (+0.0021) | 0.7428 ± 0.0105 (-0.0076) | 25 rows (5 full seeds) |
| GenePT | 0.2148 ± 0.0072 (+0.0094) | 0.0560 ± 0.0005 (+0.0000) | 0.0650 ± 0.0020 (+0.0016) | 0.7461 ± 0.0028 (-0.0043) | 25 rows (5 full seeds) |
| GO only | 0.2111 ± 0.0123 (+0.0051) | 0.0558 ± 0.0009 (-0.0003) | 0.0629 ± 0.0025 (-0.0007) | 0.7432 ± 0.0071 (-0.0073) | 22 rows (4 full seeds) |
| STRING only | 0.2087 ± 0.0114 (+0.0003) | 0.0558 ± 0.0008 (-0.0007) | 0.0641 ± 0.0039 (-0.0011) | 0.7436 ± 0.0024 (-0.0060) | 13 rows (2 full seeds) |
| learnable emb. | 0.2017 ± 0.0236 (-0.0038) | 0.0554 ± 0.0014 (-0.0006) | 0.0590 ± 0.0059 (-0.0044) | 0.7426 ± 0.0060 (-0.0078) | 25 rows (5 full seeds) |
| smooth_l1 only (no delta-cos) | 0.1336 ± 0.0041 (-0.0719) | 0.0467 ± 0.0001 (-0.0094) | 0.0409 ± 0.0004 (-0.0225) | 0.6245 ± 0.0114 (-0.1259) | 25 rows (5 full seeds) |

### k562_seen_h1set
| arm | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ | seeds × splits |
|---|---|---|---|---|---|
| **new default** (GO+STRING adjacency) | 0.1952 ± 0.0096 | 0.0549 ± 0.0006 | 0.0673 ± 0.0028 | 0.6970 ± 0.0089 | 25 rows (5 full seeds) |
| ALL: adjacency ‖ GenePT ‖ ESM2 | 0.2031 ± 0.0141 (+0.0104) | 0.0553 ± 0.0008 (+0.0005) | 0.0717 ± 0.0043 (+0.0047) | 0.6979 ± 0.0051 (+0.0018) | 23 rows (4 full seeds) |
| ESM2 | 0.2094 ± 0.0133 (+0.0142) | 0.0552 ± 0.0010 (+0.0003) | 0.0725 ± 0.0038 (+0.0051) | 0.6975 ± 0.0046 (+0.0005) | 25 rows (5 full seeds) |
| GenePT | 0.2053 ± 0.0106 (+0.0101) | 0.0550 ± 0.0005 (+0.0001) | 0.0723 ± 0.0015 (+0.0049) | 0.6939 ± 0.0063 (-0.0031) | 25 rows (5 full seeds) |
| GO only | 0.2052 ± 0.0156 (+0.0100) | 0.0546 ± 0.0006 (-0.0004) | 0.0688 ± 0.0030 (+0.0015) | 0.6967 ± 0.0092 (-0.0003) | 25 rows (5 full seeds) |
| STRING only | 0.2038 ± 0.0099 (+0.0076) | 0.0544 ± 0.0003 (-0.0009) | 0.0730 ± 0.0043 (+0.0061) | 0.6975 ± 0.0021 (+0.0033) | 15 rows (3 full seeds) |
| learnable emb. | 0.1971 ± 0.0225 (+0.0019) | 0.0543 ± 0.0009 (-0.0006) | 0.0674 ± 0.0046 (+0.0000) | 0.6893 ± 0.0050 (-0.0076) | 25 rows (5 full seeds) |
| smooth_l1 only (no delta-cos) | 0.1246 ± 0.0023 (-0.0706) | 0.0462 ± 0.0002 (-0.0088) | 0.0413 ± 0.0008 (-0.0261) | 0.5973 ± 0.0070 (-0.0996) | 25 rows (5 full seeds) |

### Results read 2026-09-24 ~23:05 — new-default ablations COMPLETE (except dcos-only s42), + rescaled dcos, + Norman partial
Mean ± sd across seeds of the per-seed split-mean; bracket = paired diff vs the reference (rhj panels:
ABL_A_DC; Norman: NF32DC) on the same (seed, split) pairs. mae is an error.

### rhj_fewshot
| arm | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ | rows |
|---|---|---|---|---|---|
| **new default** (GO+STRING, dcos 0.5 ≈ 90–98% of loss) | 0.4007 ± 0.0008 | 0.0261 ± 0.0007 | 0.0893 ± 0.0035 | 0.7613 ± 0.0030 | 25 |
| dcos 0.039 (50/50 averaged over training) | 0.3989 ± 0.0014 (-0.0018) | 0.0235 ± 0.0010 (-0.0026) | 0.0934 ± 0.0020 (+0.0041) | 0.7441 ± 0.0049 (-0.0172) | 25 |
| dcos 0.01 (50/50 at init) | 0.3976 ± 0.0019 (-0.0031) | 0.0219 ± 0.0003 (-0.0041) | 0.0949 ± 0.0019 (+0.0057) | 0.7292 ± 0.0030 (-0.0321) | 25 |
| ALL: adjacency ‖ GenePT ‖ ESM2 | 0.3999 ± 0.0035 (-0.0008) | 0.0269 ± 0.0020 (+0.0008) | 0.0860 ± 0.0066 (-0.0033) | 0.7610 ± 0.0018 (-0.0003) | 25 |
| ESM2 | 0.3973 ± 0.0015 (-0.0034) | 0.0272 ± 0.0015 (+0.0012) | 0.0926 ± 0.0043 (+0.0034) | 0.7584 ± 0.0036 (-0.0029) | 25 |
| GenePT | 0.3962 ± 0.0028 (-0.0045) | 0.0269 ± 0.0009 (+0.0008) | 0.0895 ± 0.0021 (+0.0002) | 0.7614 ± 0.0029 (+0.0001) | 25 |
| GO only | 0.3917 ± 0.0044 (-0.0090) | 0.0267 ± 0.0011 (+0.0006) | 0.0888 ± 0.0024 (-0.0005) | 0.7529 ± 0.0013 (-0.0084) | 25 |
| STRING only | 0.3951 ± 0.0029 (-0.0056) | 0.0270 ± 0.0014 (+0.0009) | 0.0869 ± 0.0030 (-0.0024) | 0.7520 ± 0.0016 (-0.0093) | 25 |
| learnable emb. | 0.3879 ± 0.0028 (-0.0128) | 0.0278 ± 0.0023 (+0.0018) | 0.0853 ± 0.0061 (-0.0040) | 0.7516 ± 0.0014 (-0.0097) | 25 |
| smooth_l1 only (no dcos) | 0.3547 ± 0.0049 (-0.0460) | 0.0205 ± 0.0000 (-0.0055) | 0.0783 ± 0.0027 (-0.0110) | 0.6694 ± 0.0022 (-0.0919) | 25 |
| dcos only (no smooth_l1) | 0.3686 ± 0.0052 (-0.0324) | 0.0344 ± 0.0044 (+0.0082) | 0.0653 ± 0.0079 (-0.0238) | 0.7197 ± 0.0083 (-0.0424) | 20 |

### h1_seen
| arm | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ | rows |
|---|---|---|---|---|---|
| **new default** (GO+STRING, dcos 0.5 ≈ 90–98% of loss) | 0.1049 ± 0.0032 | 0.1418 ± 0.0011 | 0.3249 ± 0.0010 | 0.6364 ± 0.0134 | 25 |
| dcos 0.039 (50/50 averaged over training) | 0.0975 ± 0.0020 (-0.0073) | 0.1406 ± 0.0002 (-0.0012) | 0.3242 ± 0.0009 (-0.0007) | 0.6201 ± 0.0077 (-0.0164) | 25 |
| dcos 0.01 (50/50 at init) | 0.0970 ± 0.0018 (-0.0079) | 0.1417 ± 0.0010 (-0.0001) | 0.3238 ± 0.0006 (-0.0011) | 0.5973 ± 0.0075 (-0.0392) | 25 |
| ALL: adjacency ‖ GenePT ‖ ESM2 | 0.1071 ± 0.0057 (+0.0023) | 0.1423 ± 0.0012 (+0.0005) | 0.3250 ± 0.0012 (+0.0001) | 0.6436 ± 0.0132 (+0.0071) | 25 |
| ESM2 | 0.1090 ± 0.0028 (+0.0042) | 0.1412 ± 0.0005 (-0.0006) | 0.3250 ± 0.0008 (+0.0002) | 0.6483 ± 0.0106 (+0.0119) | 25 |
| GenePT | 0.1087 ± 0.0025 (+0.0039) | 0.1413 ± 0.0009 (-0.0005) | 0.3241 ± 0.0008 (-0.0008) | 0.6443 ± 0.0053 (+0.0078) | 25 |
| GO only | 0.1048 ± 0.0031 (-0.0001) | 0.1409 ± 0.0007 (-0.0009) | 0.3245 ± 0.0011 (-0.0003) | 0.6336 ± 0.0111 (-0.0028) | 25 |
| STRING only | 0.1089 ± 0.0045 (+0.0040) | 0.1427 ± 0.0015 (+0.0009) | 0.3240 ± 0.0006 (-0.0009) | 0.6326 ± 0.0112 (-0.0039) | 25 |
| learnable emb. | 0.1074 ± 0.0077 (+0.0026) | 0.1417 ± 0.0011 (-0.0001) | 0.3246 ± 0.0022 (-0.0002) | 0.6261 ± 0.0157 (-0.0103) | 25 |
| smooth_l1 only (no dcos) | 0.0831 ± 0.0003 (-0.0218) | 0.1378 ± 0.0005 (-0.0040) | 0.3223 ± 0.0007 (-0.0026) | 0.5584 ± 0.0060 (-0.0781) | 25 |
| dcos only (no smooth_l1) | 0.1133 ± 0.0030 (+0.0077) | 0.1474 ± 0.0032 (+0.0052) | 0.3275 ± 0.0025 (+0.0027) | 0.6097 ± 0.0098 (-0.0264) | 20 |

### k562_seen
| arm | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ | rows |
|---|---|---|---|---|---|
| **new default** (GO+STRING, dcos 0.5 ≈ 90–98% of loss) | 0.2055 ± 0.0085 | 0.0560 ± 0.0007 | 0.0634 ± 0.0029 | 0.7504 ± 0.0061 | 25 |
| dcos 0.039 (50/50 averaged over training) | 0.1750 ± 0.0086 (-0.0305) | 0.0522 ± 0.0004 (-0.0038) | 0.0524 ± 0.0007 (-0.0110) | 0.7201 ± 0.0068 (-0.0303) | 25 |
| dcos 0.01 (50/50 at init) | 0.1639 ± 0.0052 (-0.0416) | 0.0500 ± 0.0007 (-0.0060) | 0.0484 ± 0.0009 (-0.0150) | 0.6969 ± 0.0126 (-0.0535) | 25 |
| ALL: adjacency ‖ GenePT ‖ ESM2 | 0.2115 ± 0.0125 (+0.0060) | 0.0562 ± 0.0009 (+0.0002) | 0.0639 ± 0.0031 (+0.0005) | 0.7488 ± 0.0029 (-0.0016) | 25 |
| ESM2 | 0.2163 ± 0.0122 (+0.0108) | 0.0563 ± 0.0012 (+0.0002) | 0.0655 ± 0.0030 (+0.0021) | 0.7428 ± 0.0105 (-0.0076) | 25 |
| GenePT | 0.2148 ± 0.0072 (+0.0094) | 0.0560 ± 0.0005 (+0.0000) | 0.0650 ± 0.0020 (+0.0016) | 0.7461 ± 0.0028 (-0.0043) | 25 |
| GO only | 0.2109 ± 0.0124 (+0.0055) | 0.0558 ± 0.0009 (-0.0002) | 0.0630 ± 0.0026 (-0.0004) | 0.7435 ± 0.0074 (-0.0069) | 25 |
| STRING only | 0.2074 ± 0.0100 (+0.0019) | 0.0560 ± 0.0006 (-0.0001) | 0.0630 ± 0.0034 (-0.0003) | 0.7439 ± 0.0032 (-0.0065) | 25 |
| learnable emb. | 0.2017 ± 0.0236 (-0.0038) | 0.0554 ± 0.0014 (-0.0006) | 0.0590 ± 0.0059 (-0.0044) | 0.7426 ± 0.0060 (-0.0078) | 25 |
| smooth_l1 only (no dcos) | 0.1336 ± 0.0041 (-0.0719) | 0.0467 ± 0.0001 (-0.0094) | 0.0409 ± 0.0004 (-0.0225) | 0.6245 ± 0.0114 (-0.1259) | 25 |
| dcos only (no smooth_l1) | 0.2285 ± 0.0204 (+0.0214) | 0.0614 ± 0.0040 (+0.0052) | 0.0618 ± 0.0041 (-0.0024) | 0.6722 ± 0.0171 (-0.0776) | 20 |

### k562_seen_h1set
| arm | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ | rows |
|---|---|---|---|---|---|
| **new default** (GO+STRING, dcos 0.5 ≈ 90–98% of loss) | 0.1952 ± 0.0096 | 0.0549 ± 0.0006 | 0.0673 ± 0.0028 | 0.6970 ± 0.0089 | 25 |
| dcos 0.039 (50/50 averaged over training) | 0.1698 ± 0.0087 (-0.0254) | 0.0517 ± 0.0004 (-0.0032) | 0.0571 ± 0.0011 (-0.0102) | 0.6724 ± 0.0071 (-0.0246) | 25 |
| dcos 0.01 (50/50 at init) | 0.1595 ± 0.0071 (-0.0357) | 0.0497 ± 0.0007 (-0.0052) | 0.0537 ± 0.0033 (-0.0137) | 0.6620 ± 0.0036 (-0.0350) | 25 |
| ALL: adjacency ‖ GenePT ‖ ESM2 | 0.2033 ± 0.0138 (+0.0082) | 0.0553 ± 0.0008 (+0.0004) | 0.0715 ± 0.0045 (+0.0041) | 0.6981 ± 0.0048 (+0.0011) | 25 |
| ESM2 | 0.2094 ± 0.0133 (+0.0142) | 0.0552 ± 0.0010 (+0.0003) | 0.0725 ± 0.0038 (+0.0051) | 0.6975 ± 0.0046 (+0.0005) | 25 |
| GenePT | 0.2053 ± 0.0106 (+0.0101) | 0.0550 ± 0.0005 (+0.0001) | 0.0723 ± 0.0015 (+0.0049) | 0.6939 ± 0.0063 (-0.0031) | 25 |
| GO only | 0.2052 ± 0.0156 (+0.0100) | 0.0546 ± 0.0006 (-0.0004) | 0.0688 ± 0.0030 (+0.0015) | 0.6967 ± 0.0092 (-0.0003) | 25 |
| STRING only | 0.2036 ± 0.0087 (+0.0085) | 0.0546 ± 0.0003 (-0.0004) | 0.0730 ± 0.0032 (+0.0057) | 0.6988 ± 0.0024 (+0.0018) | 25 |
| learnable emb. | 0.1971 ± 0.0225 (+0.0019) | 0.0543 ± 0.0009 (-0.0006) | 0.0674 ± 0.0046 (+0.0000) | 0.6893 ± 0.0050 (-0.0076) | 25 |
| smooth_l1 only (no dcos) | 0.1246 ± 0.0023 (-0.0706) | 0.0462 ± 0.0002 (-0.0088) | 0.0413 ± 0.0008 (-0.0261) | 0.5973 ± 0.0070 (-0.0996) | 25 |
| dcos only (no smooth_l1) | 0.2240 ± 0.0196 (+0.0267) | 0.0589 ± 0.0036 (+0.0037) | 0.0702 ± 0.0053 (+0.0028) | 0.6425 ± 0.0193 (-0.0546) | 20 |

### norman_combo · additive (PARTIAL for DCI/DCA)
| arm | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ | rows |
|---|---|---|---|---|---|
| NF32 full loss | 0.9051 ± 0.0045 (+0.0230) | 0.0114 ± 0.0003 (-0.0003) | 0.3012 ± 0.0108 (-0.0084) | 0.8794 ± 0.0219 (-0.0750) | 25 |
| NF32 smooth_l1+dcos, dcos 0.32977 (reference) | 0.8821 ± 0.0028 | 0.0117 ± 0.0005 | 0.3097 ± 0.0055 | 0.9544 ± 0.0078 | 25 |
| dcos 0.0255 (50/50 avg) | 0.8916 ± 0.0006 (+0.0099) | 0.0109 ± 0.0001 (-0.0006) | 0.3153 ± 0.0070 (+0.0053) | 0.9573 ± 0.0104 (+0.0056) | 20 |
| dcos 0.0066 (50/50 init) | 0.8926 ± 0.0034 (+0.0094) | 0.0110 ± 0.0002 (-0.0006) | 0.3094 ± 0.0162 (-0.0035) | 0.9462 ± 0.0258 (-0.0085) | 10 |

### norman_combo · nonadditive (PARTIAL for DCI/DCA)
| arm | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ | rows |
|---|---|---|---|---|---|
| NF32 full loss | 0.7981 ± 0.0071 (+0.0046) | 0.0133 ± 0.0001 (-0.0008) | 0.2880 ± 0.0054 (-0.0055) | 0.7664 ± 0.0061 (-0.0662) | 25 |
| NF32 smooth_l1+dcos, dcos 0.32977 (reference) | 0.7935 ± 0.0028 | 0.0141 ± 0.0005 | 0.2935 ± 0.0056 | 0.8326 ± 0.0174 | 25 |
| dcos 0.0255 (50/50 avg) | 0.8016 ± 0.0010 (+0.0072) | 0.0134 ± 0.0001 (-0.0006) | 0.3031 ± 0.0064 (+0.0087) | 0.8129 ± 0.0067 (-0.0143) | 20 |
| dcos 0.0066 (50/50 init) | 0.7967 ± 0.0025 (+0.0032) | 0.0135 ± 0.0002 (-0.0006) | 0.2984 ± 0.0138 (+0.0019) | 0.8093 ± 0.0261 (-0.0215) | 10 |

**Slow-node relaunch (23:00).** Four seeds had run ~6 h at a fraction of normal speed and would
have timed out: Norman DCI s42 (hop11, epoch 15/150), Norman DCI s44 (hop05, 5/150), Norman DCA
s42 (hop20, 65/150), rhj dcos-only s42 (hop07, 44/180). Cancelled and relaunched on vco excluding
hop02/05/07/11/20/39: jobs 18512807 (DCI s42,s44), 18512808 (DCA s42), 18512809 (dcos-only s42).

## 8. ALL-sources concatenated = the NEW DEFAULT representation (user, 2026-09-24 23:10)

| arm | benchmark | what | job |
|---|---|---|---|
| `NF32ALL` | norman_combo | NF32 + smooth_l1 + delta-cos 0.32977, ALL sources concatenated | 18512821 |
| `NF32ALLA` | norman_combo | same, delta-cos 0.0255 (50/50 averaged over training) | 18512822 |
| `ABL_N_ALL_T3` | rhj (+ H1/K562) | ALL, GO/STRING edges top_3 per gene | 18512825 |
| `ABL_N_ALL_T5` | rhj (+ H1/K562) | ALL, top_5 | 18512826 |
| `ABL_N_ALL_T10` | rhj (+ H1/K562) | ALL, top_10 | 18512827 |

top_20 is the current setting (= `ABL_N_ALL`). `graph_edge_filter_mode=top_N` keeps each gene's N
strongest edges per KG; the filtered graph is rebuilt from the raw files every run (no cache).
ConcatKnowledgeModel energy-matches each table to the adjacency row's mean squared norm, so a
sparser graph also scales the GenePT/ESM2 blocks down with it. Norman uses the new
`configs/norman_combo_pertemb.yaml` (= norman_combo.yaml + the 3 embedding keys; same master
gene set as rhj, so the rhj tables apply). All 5 seeds each, slow nodes excluded.
