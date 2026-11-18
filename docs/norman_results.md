# Norman double-perturbation evaluation

The Norman CRISPR-activation benchmark trains on single-gene perturbations in K562
and evaluates held-out double perturbations, each targeting two genes together.
Validation uses 20 double-perturbation labels and testing uses 111 distinct labels.
Validation checkpoint selection and testing both use `decode_seq_mean` composition.

The released predictor is **NF32ALLA**, using GO, STRING, GenePT, and ESM2 together.
[The official comparison tables](../data/release/official_comparisons/official_tables.md)
include the full test set and the strongly additive and weakly additive subsets.
The on-disk subset keys remain `additive` and `nonadditive` for compatibility.
Each subset contains 36 double perturbations; the full set also includes the
39 intermediate responses. These are evaluation subsets, not intermediate training runs.

The additive baseline sums measured single-perturbation effects. Experimental
reproducibility and every predictor use the matched split-half evaluation protocol.
Average five evaluation splits within each training seed, then report the mean and
sample SD across the five seed means.

- [Frozen benchmark and subset construction](norman_benchmark.md)
- [Official training configuration](../configs/norman_combo_official.yaml)
- [Reproduction commands](../scripts/repro/README.md)
- [Complete NF32ALLA seed/split scores](../data/release/norman_allsource/)
