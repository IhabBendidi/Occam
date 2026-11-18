# Occam: models, data, and results

[Paper](https://arxiv.org/abs/2611.21959) · [Code and instructions](https://github.com/IhabBendidi/Occam)

Occam learns cell representations and predicts perturbation effects using gene expression,
GO, STRING, GenePT, and ESM2 knowledge sources.

- **Starting encoder–decoder:** `checkpoints/occam_small_encoder_decoder.pth`.
- **Trained RHJ virtual-cell models:** `checkpoints/occam_seeds/rhj_fewshot/all_sources/seed_42/`
  through `seed_46/`. These are the ABL_N_ALL models used on RHJ, H1, and K562.
- **Trained Norman virtual-cell models:** `checkpoints/vcell/norman_combo_NF32ALLA/seed_42/`
  through `seed_46/`, for double-perturbation prediction.
- **Benchmark data:** `eval/` and `benchmarks/`. **Knowledge sources:** `graphs/` and `assets/`.
- **Published results:** `official_comparisons/`, `final/`, `metrics/`, and `eval_outputs/`.

Complete virtual-cell checkpoints already contain their encoder and decoder.
Use the [selective downloader](https://github.com/IhabBendidi/Occam/blob/main/docs/downloads.md)
to get only the files you need. Checkpoint hashes are in `checkpoints/official_manifest.json`.
The full stage-two MAE checkpoint is retained under `checkpoints/stage2/` for fine-tuning.

[Competitor checkpoints](https://huggingface.co/bendidiihab/Occam-competitor-checkpoints)
are hosted separately. The DiverseRNA pretraining corpus is not hosted here.
