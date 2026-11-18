# RHJ few-shot benchmark

The model observes HepG2, Jurkat, and RPE1 during predictor training, but the evaluated
perturbation–HepG2 combinations are held out. Their perturbation identities can be seen
in Jurkat or RPE1. This tests new combinations of a familiar perturbation and cell context.

The current all-source model is ABL_N_ALL. Use the
[official configuration](../configs/rhj_fewshot_official.yaml),
[reproduction commands](../scripts/repro/README.md), and
[split-half protocol](rhj_splithalf.md).
The [official comparison tables](../data/release/official_comparisons/official_tables.md)
report all metrics, seeds, and evaluation settings.
