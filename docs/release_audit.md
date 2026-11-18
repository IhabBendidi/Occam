# Release verification

The current release uses **ABL_N_ALL** for RHJ/H1/K562 and **NF32ALLA** for Norman.
Both combine GO, STRING, GenePT, and ESM2. Reproduction commands use
`configs/rhj_fewshot_official.yaml` and `configs/norman_combo_official.yaml`.

The [official comparison tables](../data/release/official_comparisons/official_tables.md)
cover 11 benchmark/panel combinations and 110 model comparisons, each with five
training seeds and five evaluation splits. Uncertainty is the sample standard
deviation across training-seed means. Cleanup preserves these published tables.

[Checkpoint hashes](../checkpoints/official_manifest.json) identify all ten official
virtual-cell models and their shared encoder–decoder initialization. The RHJ recipe
was checked against the resolved configurations of all five released checkpoints.

Superseded Occam models, ablations, experiment plans, and intermediate result exports
are excluded from the release. Benchmark data, current model seeds, competitors,
baselines, experimental reproducibility, and all Norman evaluation subsets are retained.
The intermediate-response Norman subset is a biological evaluation subset, not an
intermediate training experiment.

See [publication instructions](hf_release.md), [reproduction commands](../scripts/repro/README.md),
and the [evaluation protocol](rhj_splithalf.md). Metadata and hash verification do not
constitute a new training run or a new benchmark evaluation.
