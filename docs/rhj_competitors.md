# Competitor reproduction

TxPert, STATE, CellFlow, and PerturbDiff are retrained on the same frozen RHJ and
Norman benchmark definitions used for Occam. Their adapters and training configurations
are in `competitors/`; [the manifest](../competitors/MANIFEST.json) records their sources.

Use [the reproduction guide](../scripts/repro/README.md) to train, export predictions,
and score them. All comparisons use the shared genes, evaluation splits, control draws,
and cell-eval scorer. The shared data adapter is `competitors/shared/occam_data.py`.

On H1 and K562, use the RHJ-trained checkpoints without fine-tuning. For Norman,
validation and prediction must handle both target genes of a double perturbation;
see [double-perturbation adapters](competitors_on_doubles.md).

The [official tables](../data/release/official_comparisons/official_tables.md) contain
five training seeds and five evaluation splits per seed for every competitor.
[Competitor checkpoints](https://huggingface.co/bendidiihab/Occam-competitor-checkpoints)
are hosted separately. Their original licenses remain with the implementations and
checkpoint files.
