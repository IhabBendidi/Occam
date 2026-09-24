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
