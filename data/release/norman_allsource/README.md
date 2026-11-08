# Norman all-source evaluation supplement

This export contains NF32ALL and NF32ALLA, seeds 42–46, evaluated with
`decode_seq_mean` on splits 0–4 of the frozen `norman_combo` benchmark.
NF32ALLA is the official Norman arm; NF32ALL is a representation/loss ablation.
Scores are uncalibrated (`model_nocal`), using cell-eval 0.6.0.

`scores_by_seed_split.csv` preserves benchmark, panel, model, variant, seed,
split, evaluation root, and all exported metrics. `coverage.csv` checks the
five splits for every seed. `metrics_long.csv` retains scored-arm and calibration
identity. The wide and per-split files are auxiliary exports; their SEM columns
are not the headline uncertainty.

For headline results, average splits within each seed, then report the mean and
sample SD (`ddof=1`) of the five seed means. Only `additive` and `nonadditive`
(36 pairs each) are headline strata. Other panels are retained separately.

The mandatory `ctrlmean` and `ctrlcell` anchors, other baselines, and competitors
remain in `../final/scores_by_seed_split.csv`, with coverage in `../final/coverage.csv`.
`python scripts/official_score_tables.py` combines these frozen sources without
changing either source table or the benchmark definitions, and writes the complete
comparisons to `../official_comparisons/`.

Reproduce this supplement from the existing scored evaluation roots:

```bash
python scripts/export_split_metrics.py --help
python scripts/final_table.py --long data/release/norman_allsource/metrics_long.csv \
    --out-dir data/release/norman_allsource
python scripts/official_score_tables.py
```

The scoring runner is `scripts/run_norman_allsource_local.py`; real scoring requires
the separate cell-eval 0.6.0 environment described in the repository README.
