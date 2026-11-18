# Publishing the current release

The [Occam Hub repository](https://huggingface.co/bendidiihab/Occam) contains the
encoder–decoder, five RHJ all-source models, five Norman all-source models, benchmark
inputs, knowledge sources, and published evaluation results. Competitor checkpoints
remain in [their separate repository](https://huggingface.co/bendidiihab/Occam-competitor-checkpoints).

[Download instructions](downloads.md) distinguish the encoder–decoder used for
initialization from complete perturbation predictors. The current recipes are
ABL_N_ALL for RHJ/H1/K562 and NF32ALLA for Norman.

## Publication inputs

`checkpoints/official_manifest.json` records the remote paths, local paths, and
SHA-256 hashes of the ten official virtual-cell checkpoints and the encoder–decoder.
The publisher verifies locally available checkpoint bytes against those hashes.

| GitHub source | Hub destination |
|---|---|
| `data/release/official_comparisons/` | `official_comparisons/` |
| `data/release/final/` | `final/` |
| `data/release/norman_allsource/` | `final/norman_allsource/` |
| `data/release/metrics/` | `metrics/` |
| `data/release/eval_outputs/` | `eval_outputs/` |
| `scripts/repro/` | `repro/` |

The metric files retain current Occam models, competitors, baselines, and experimental
reproducibility. All Norman evaluation subsets are retained, including the intermediate
response subset. Ablation models and intermediate training experiments are excluded.
The stage-two MAE checkpoint remains available for encoder fine-tuning.

## Publish

```bash
python scripts/official_score_tables.py
python scripts/export_release_metrics.py
python scripts/publish_hf.py --repo bendidiihab/Occam --dry-run
python scripts/publish_hf.py --repo bendidiihab/Occam
```

The dry run performs no network access. Uploading requires a write token and retains
the repository's existing visibility. `--only REMOTE_PATH` selects an artifact from
the publication list. Missing local files are reported; existing Hub files are not
removed by an upload. Publication excludes retired experiment folders even if local
copies remain in an older staging directory.

The DiverseRNA pretraining corpus is not included in the Hub release.
