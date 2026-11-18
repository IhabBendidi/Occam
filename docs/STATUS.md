# Current Occam release

The released perturbation predictors use **GO, STRING, GenePT, and ESM2 together**.
All reported model results use training seeds 42–46 and five evaluation splits per seed.

| Predictor | Recipe | Training configuration | Checkpoint card |
|---|---|---|---|
| RHJ; also evaluated on H1 and K562 without fine-tuning | ABL_N_ALL | [RHJ](../configs/rhj_fewshot_official.yaml) | [RHJ models](../checkpoints/cards/rhj_all_sources_CARD.md) |
| Norman double perturbations | NF32ALLA | [Norman](../configs/norman_combo_official.yaml) | [Norman models](../checkpoints/cards/norman_combo_NF32ALLA_CARD.md) |

[Official comparison tables](../data/release/official_comparisons/official_tables.md)
are the result entry point. The accompanying selected-score CSV contains every metric
for each training seed and evaluation split, including competitors, baselines, and
experimental reproducibility. Older experimental variants are excluded.

- [Download models and data](downloads.md)
- [Train on your own data](training.md)
- [Train a perturbation predictor](virtual-cell-training.md)
- [Reproduce the published training and evaluations](../scripts/repro/README.md)
- [Evaluation commands and aggregation](benchmarking.md)
- [Release publication](hf_release.md)

The benchmark definitions remain in `data/benchmarks/`. Current training recipes and
checkpoint hashes are recorded in `checkpoints/official_manifest.json`.
