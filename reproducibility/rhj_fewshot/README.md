# RHJ reproduction records

The current five-seed comparison is in the
[official results](../../data/release/official_comparisons/rhj_fewshot__all.md).
Occam uses the all-source ABL_N_ALL model. The folders here retain competitor
configuration and provenance records; use the official results for aggregate scores.

Each competitor was retrained on the same benchmark. Its manifest records checkpoint
hashes, training settings, evaluation settings, and deliberate deviations from the
upstream recipe. Read those deviations before interpreting the comparison as a
reproduction of a competitor's original published result.

See [current reproduction commands](../../scripts/repro/README.md),
[competitor training](../../docs/rhj_competitors.md), and the
[split-half protocol](../../docs/rhj_splithalf.md). Complete competitor checkpoints are
available from [Hugging Face](https://huggingface.co/bendidiihab/Occam-competitor-checkpoints).
