# Final tables — 5 seeds × 5 splits for every model and baseline (target: 2026-09-24/25)

**Goal (user, 2026-09-23).** Within 24–36 h: every model trained on 5 seeds, every baseline scored
over 5 control-draw seeds, every model scored at 5 seeds × 5 splits, and all cell-eval metrics
written per (seed, split) — one row per tuple, labelled — so the final tables can be built.

## The matrix

| benchmark | Occam | competitors (4) | baselines |
|---|---|---|---|
| `rhj_fewshot` | Y06 × seeds 42–46 (s43–46: job 18368487) | TxPert (single-cell ctrl), STATE, CellFlow, PerturbDiff genept × 42–46 | 5 control-draw seeds |
| `norman_combo` | winner of NF31 vs NF40–43 (bs-64 eta_min sweep, job 18368256) × 42–46 — launched when the sweep lands | TxPert, PerturbDiff × 42–46; **CellFlow and STATE × 42–46 RETRAINED** after their fixes (audit defects) | 5 seeds |
| `h1_seen` (80) | Y06 × 42–46; Q02/M06/M03 × 42 | the 4 rhj-trained arms × 42–46 (exports: jobs 18368556–587) | 5 seeds (done) |
| `k562_seen` (528) | same | same (18368588–619) | 5 seeds |
| `k562_seen_h1set` (80) | same | same (18368620–651) | 5 seeds |

Seed = the training seed for models; the half-B control draw for baselines (rng = seed + 1000·split).

## How completeness is enforced

- `configs/score_registry.tsv` lists every score unit (154 units = 770 splits at registration).
- `scripts/score_ledger.py --daemon` (job 18368652, cpu, 4 days) resubmits any split that is
  missing and has no live job, up to 3 attempts; it waits for training (`final_model.pth`) and for
  competitor exports (`pred.npy`). It replaces the per-benchmark auto-eval daemons.
  Status at any time: `python scripts/score_ledger.py` (`#` done, `~` in flight, `.` missing,
  `e` needs export, `t` still training, `X` gave up).
- cell-eval runs through `scripts/cell_eval_shmsafe.py` (the /dev/shm wipe, F-infra1, is fixed).

## The final table

    python scripts/export_split_metrics.py --scope all --out data/release/metrics_all
    python scripts/final_table.py

→ `data/release/final/scores_by_seed_split.csv`: columns `benchmark, panel, model, model_kind,
variant, seed, split, n_perts, eval_root` + all 35 cell-eval metrics; one row per
(benchmark, panel, model, seed, split). `data/release/final/coverage.csv` shows, per model, how
many splits each seed has; the table is complete when every expected model has seeds 42–46 × 5.

## Still to launch (tracked here)

| item | waits on |
|---|---|
| Norman Occam seeds 43–46 | ✅ **NF32** chosen by the user 2026-09-23 (F-sweep7: all bs-64/32 arms within seed noise; NF32 best on mae/overlap/disc_l1); `scripts/grids/norman_seeds_nf32.tsv`, job 18386196 |
| CellFlow norman_combo retrain × 5 seeds | ✅ launched 2026-09-23: chains 18368707–721 (pairs as two component tokens in train/val/selection/export; selection now scores every pass and restores the best; old outputs archived). Check `run_meta.json` selection.n_scored_states = 6 after training |
| STATE norman_combo retrain × 5 seeds | ✅ launched 2026-09-23: chains 18368664–678 (validation now composes pairs like the export; old outputs archived) |
| registry rows | ✅ STATE + CellFlow norman added; Norman Occam seeds added at launch |

Before the CellFlow / STATE norman retrains, their DEFECTIVE outputs (runs/competitors/
{cellflow,state}__norman_combo, runs/norman_combo/compet_{cellflow,state}__norman_combo_s*) move to
`archive/` so the ledger cannot count them as done.

## Updates 2026-09-23 ~15:45
- Norman Occam arm = **NF32** (bs 32) for all 5 seeds; tables use NF32 for norman_combo.
- Relaunched after hangs: S43_Y06 (18385637), NF42 (18385636), RB128 (18368869).
- STATE h1_seen seeds 43/44/46: output library size 8.3x the target's; per the user they are
  scored as is (`OCCAM_WAIVE_LIBSIZE_CHECK=1` waives only the library-size band; recorded in each
  export manifest as `libsize_check_waived`): exports 18386250/53/55 → scores 18386252/54/56.
- Done at 5 seeds × 5 splits: every baseline on every panel; CellFlow, PerturbDiff, TxPert on
  rhj + all OOD panels; STATE on rhj + both K562 panels; CellFlow (fixed), PerturbDiff, TxPert on
  norman. Y06 at seeds 42/44/45/46 everywhere.

## Update 2026-09-24 ~02:50 — Occam headline rows complete
Every row of the matrix is now at 5 seeds × 5 splits: Y06 on rhj + 3 OOD panels, NF32 on
norman_combo, all 4 competitors, all baselines. `data/release/final/scores_by_seed_split.csv`
(6,041 rows). Still running, for the record only: RB064 / RB128 scoring, RB032 training.

## 2026-09-24 08:15 — vco headroom (user)
Keep ≥ 5 vco GPUs free: rebalancer restarted with `VCO_CAP=27` (job 18458833, replaces 18365346);
36 pending vco score jobs moved to hooke-predict; 5 running score tasks requeued onto
hooke-predict (18447661_1/_2, 18447662_0/_1, 18447663_0). No training was touched.
