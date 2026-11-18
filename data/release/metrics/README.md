# Published perturbation-effect metrics

These tables contain cell-eval metrics for the current all-source Occam models,
competitors, baselines, and experimental reproducibility. RHJ/H1/K562 use ABL_N_ALL;
Norman uses NF32ALLA. Retired model variants are excluded from the published export.

`metrics_long.csv` records one metric per evaluation split. The wide and per-split
files provide alternate views. SEM in a wide file describes evaluation splits within
one training seed; the paper instead reports sample SD across five training-seed means.

Rebuild from the released scores with `python scripts/export_release_metrics.py`.
Use `scripts/export_split_metrics.py --scope published` to collect new local evaluations.
The official comparisons are in `data/release/official_comparisons/`.
