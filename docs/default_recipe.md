# Published all-source training recipes

Both perturbation predictors combine GO and STRING adjacency information with GenePT
and ESM2 embeddings. The encoder is frozen; the perturbation module, FiLM module, and
expression decoder are trained. Single control cells are inputs and single perturbed
cells are training targets.

| Setting | RHJ (ABL_N_ALL) | Norman (NF32ALLA) |
|---|---|---|
| Configuration | [RHJ](../configs/rhj_fewshot_official.yaml) | [Norman](../configs/norman_combo_official.yaml) |
| Epochs | 180 | 150 |
| Batch size | 2,048 | 32 |
| Learning rate, including decoder | 0.001 | 0.001 |
| Minimum learning rate | 0.0007 | 0.0003 |
| Smooth-L1 beta | 2 | 2 |
| Delta cosine weight | 0.5 | 0.0255 |
| Training seeds | 42–46 | 42–46 |

Both use expression-space smooth-L1 reconstruction plus the delta cosine loss.
Profile cosine, delta magnitude, latent losses, and VICReg terms are disabled.
Norman validation and test prediction use `decode_seq_mean` for double perturbations.

RHJ settings were checked against the resolved configurations attached to all five
released checkpoints. The training grids are [RHJ](../scripts/grids/rhj_all_sources.tsv)
and [Norman](../scripts/grids/norman_seeds_nf32_alla.tsv).

Use the [reproduction commands](../scripts/repro/README.md) for the published benchmarks.
For a new dataset, start with the [custom-data guide](virtual-cell-training.md).
