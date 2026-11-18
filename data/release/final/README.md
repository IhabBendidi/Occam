# Published seed/split scores

`scores_by_seed_split.csv` contains the current RHJ/H1/K562 all-source model (ABL_N_ALL),
competitors, baselines, and experimental reproducibility. Norman NF32ALLA scores are
in `data/release/norman_allsource/` (on Hugging Face: `final/norman_allsource/`).
`coverage.csv` records the five training/control seeds and five evaluation splits.
Retired experiment rows are excluded; retained numerical values are unchanged.

Run `python scripts/official_score_tables.py` to combine these sources into the
complete official comparisons. Run `python scripts/export_release_metrics.py` to
regenerate the long, wide, and per-split views of all cell-eval metrics.
