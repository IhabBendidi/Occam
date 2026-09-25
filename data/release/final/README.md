# final/ — the paper tables' source: one row per (benchmark, panel, model, seed, split)

`scores_by_seed_split.csv`: every cell-eval metric (35 columns) for every model, competitor and
baseline on rhj_fewshot, h1_seen, k562_seen, k562_seen_h1set and norman_combo (panels additive /
nonadditive / ...), nocal. `model_kind` = occam | competitor | baseline; `seed` = training seed
for models, control-draw seed for baselines; `split` = split-half 0–4; `eval_root` = where it came
from. `coverage.csv`: splits per seed for every (benchmark, panel, model).
Regenerate: `python scripts/export_split_metrics.py --scope all --out data/release/metrics_all &&
python scripts/final_table.py`. Mean ± sd tables: `docs/current_tables_2026-09-24.md`.
