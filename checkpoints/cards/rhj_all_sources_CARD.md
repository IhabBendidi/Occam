# RHJ all-source virtual-cell models (ABL_N_ALL)

These are the five published Occam perturbation predictors, with GO, STRING, GenePT,
and ESM2 combined. Training uses HepG2, Jurkat, and RPE1 with the frozen RHJ few-shot
partition. The same checkpoints are evaluated on H1 and K562 without fine-tuning.

Each checkpoint contains the encoder, decoder, perturbation module, FiLM module,
gene vocabulary, and preprocessing metadata. No separate knowledge download is needed
for inference. Use `configs/rhj_fewshot_official.yaml` to reproduce training.

| Training seed | SHA-256 (sha256) |
|---|---|
| 42 | `11d5ba5515e79710b3a952532c60aa29515046117292528c59d452a0937685ff` |
| 43 | `a16e94a7a4a2f17bac8b57952997119326e611ec48dda1f15b870e6d25489246` |
| 44 | `5808cc5c721222d75b4f213934a7fb044c61a3af32b2201fb57a1de9194da0c5` |
| 45 | `58464abc54e975ac367b272351a61bdfcb63fb45737f7a60b982a4e8a782c997` |
| 46 | `06dbbfcd709d8a1bdc0190b99bd33f3117a2f9a1b5aa9349d83c800573612d81` |

The [official tables](../../data/release/official_comparisons/official_tables.md)
report all metrics against competitors, ctrlmean, ctrlcell, meanpert, transfer,
and experimental reproducibility. They average five evaluation splits per training
seed and report sample SD across the five training-seed means.

## Limitations

These are genetic-perturbation predictors trained on the stated cell lines and
assays. Performance in new biological contexts must be evaluated; the released
H1 and K562 comparisons do not establish generalization to arbitrary cell types.
