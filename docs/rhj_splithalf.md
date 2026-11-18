# RHJ split-half evaluation

RHJ combines the HepG2, Jurkat, and RPE1 CRISPR-knockdown datasets. Evaluation uses
611 held-out HepG2 gene perturbations. Their perturbation identities can be observed
in the training cell lines, while the corresponding HepG2 responses are held out.
The frozen validation and test partitions are in `data/benchmarks/rhj_fewshot/`.

## Scoring protocol

For each perturbation, measured perturbed cells are divided into reference half A and
comparison half B. Controls are split separately. The model receives controls from B;
its predictions and the measured perturbed cells in B are both evaluated against A.
The measured B-versus-A comparison is **experimental reproducibility**.

Pearson delta compares effects relative to controls: the reference uses A controls,
while predictions and the measured B comparison use B controls. This avoids sharing
the same control estimate between the reference and comparison deltas. Cell counts
and scoring genes are matched so DEG tests use comparable sample sizes.

Five evaluation splits are scored for each of five training seeds. Average the splits
within each training seed, then report the mean and sample SD across those five means.

## Reproduce and inspect

The released predictor is **ABL_N_ALL**, combining GO, STRING, GenePT, and ESM2.
See [evaluation commands](benchmarking.md), [training recipes](default_recipe.md),
and [official results](../data/release/official_comparisons/official_tables.md).
The protocol and baseline implementations are in `src/occam/vcell/panel/`.
