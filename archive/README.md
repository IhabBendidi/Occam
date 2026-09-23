# archive/ — STALE. Do not use, do not quote, do not build on.

Everything here was moved out of the live tree on 2026-09-23 (user decision) so that no one —
human or agent — mixes a withdrawn number, a superseded benchmark or a pre-recipe checkpoint
into current work. Nothing was deleted: every item was renamed from its live path into the same
relative path under `archive/`, and `MANIFEST.tsv` lists each move (date, old path, new path,
reason). To bring one back: `mv archive/<path> <path>` and add a line to MANIFEST.tsv.

| what | why it is stale |
|---|---|
| `runs/rhj_fewshot_*` (660) and their `runs/rhj_splithalf/*` scores | pre-recipe sweeps (27-arm FiLM grid, 486-arm adjacency-map sweep, D/L/C/E/G/N/P/R/S/T/V/W/X/Z families, other Q/M arms), all before the 2026-09-22 default (Q02 base, 150 ep, eta_min 3e-4) |
| `runs/rhj_fewshot_*`, `runs/rhj_adjmap_s*`, `runs/rhj_loss_*` marker dirs | written by old launchers; they look like runs and are not |
| `runs/norman_combo_{NQ02s,NM03s,NM06s}` + scores | 30-epoch pre-recipe Norman bases |
| `runs/rhj_splithalf/compet_txpert_s42` and `runs/competitors/txpert/{s42,ckpt/occam_s42}` | **WITHDRAWN** TxPert 0.3077 (mean-control training, single-cell inference, missing control token). The live TxPert arm is `txpert_single` (0.3463) |
| `txpert_mean`, `txpert_ep20`, `ckpt_frozen`, `_gate_ctrlcell`, `cellflow_preflight`, `_DEFECTIVE_*`, `CKSWEEP_*` | A/B diagnostics, infrastructure gates, a defective export, checkpoint sweeps of archived arms |
| `data/benchmarks/h1_seen_99`, `runs/h1_seen_99`, `runs/competitors/*__h1_seen_99`, `docs/h1_seen_99_results.md` | the SUPERSEDED 99-perturbation H1 panel; live `h1_seen` has 80 |
| `runs/aborted_3h_limit`, `runs/_hung_NF32_*`, `runs/competitors/txpert/ckpt/_killed_by_pdex_hang_*`, smoke tests | incomplete or broken runs (`runs/occam_small_smoke` stayed live: it is a test fixture) |
| `docs/rhj_fewshot_results.md`, `docs/rhj_adjmap_*.md`, `docs/rhj_all_results.md`, `docs/rhj_scoreboard__snapshot_*` | results tables of the archived arms; live results are in `docs/STATUS.md` |
| `scripts/collect_adjmap_eval.py` | only regenerated the archived adjmap docs |

Large data here (`archive/runs`, `archive/data`) is not in git; the docs, scripts, this README
and MANIFEST.tsv are.
