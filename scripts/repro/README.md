# Reproduce the published all-source models

Use **ABL_N_ALL** for RHJ, H1, and K562, and **NF32ALLA** for Norman double perturbations.
Both combine GO, STRING, GenePT, and ESM2. The official configurations are
[RHJ](../../configs/rhj_fewshot_official.yaml) and [Norman](../../configs/norman_combo_official.yaml).

## Prepare

Follow the [installation and downloads guide](../../docs/downloads.md). Training needs
an encoder–decoder, benchmark data, and knowledge sources. Evaluation of a released
virtual-cell checkpoint needs only that checkpoint and its benchmark data.

```bash
python scripts/download_release.py encoder-decoder
python scripts/download_release.py knowledge
python scripts/download_release.py data --benchmark rhj
python scripts/download_release.py data --benchmark norman
```

For Norman, copy `configs/paths.example.yaml` to `configs/paths.yaml` if it does not
already exist. The defaults point to the download locations. Perturbation-effect
scoring requires the [separate cell-eval environment](../../docs/benchmarking.md).

## Train

On a Slurm GPU cluster:

```bash
for seed in 42 43 44 45 46; do
    SEED=$seed sbatch scripts/repro/rhj_train.sh
    SEED=$seed sbatch scripts/repro/norman_train.sh
done
```

For a single GPU without Slurm, run the same configurations directly:

```bash
python -m occam.train --config configs/rhj_fewshot_official.yaml --set seed=42 --run-suffix repro_s42
python -m occam.train --config configs/norman_combo_official.yaml --set seed=42 --run-suffix repro_s42
```

Use a distinct run suffix for each training seed. The Slurm wrappers set it automatically.
Checkpoints are selected on validation Pearson delta. The test sets are not used for
checkpoint selection. Full schedules and losses are in [the recipe guide](../../docs/default_recipe.md).

## Evaluate

Download each model with `scripts/download_release.py virtual-cell --benchmark rhj --seed 42`
(or `--benchmark norman`). Repeat for seeds 43–46 to reproduce the reported uncertainty.

```bash
for seed in 42 43 44 45 46; do
    SEED=$seed sbatch scripts/repro/rhj_eval.sh
    SEED=$seed sbatch scripts/repro/norman_eval.sh
done
```

Each command submits five evaluation splits. Pass your own checkpoint as the first
argument. `MODE=reference` scores baselines and experimental reproducibility;
`ARMS=ctrlmean,ctrlcell` optionally selects reference methods. Keep `--calibrate none`.
Norman uses `decode_seq_mean` for double perturbations. Keep the `%6` evaluation throttle.
The [evaluation guide](../../docs/benchmarking.md) gives direct commands without Slurm.

For H1 or K562, download the corresponding dataset and use the RHJ-trained checkpoints
without fine-tuning. `BENCH=h1_seen bash scripts/repro/ood_seen.sh` submits all five RHJ
training seeds, references, and competitors; use `k562_seen` or `k562_seen_h1set` for K562.

## Competitors and benchmark reconstruction

Use `competitors_rhj.sh` or `competitors_norman.sh` with `ARM=txpert`, `state`, `cellflow`,
or `perturbdiff`. The implementations, adapters, and original licenses are under
`competitors/`. Checkpoint downloads are hosted in the separate competitor Hub repository.

The downloaded benchmark definitions are ready to use. To verify reconstruction,
use `build_benchmark_rhj.sh`, `build_benchmark_norman.sh`, `build_benchmark_h1.sh`, or
`build_benchmark_k562.sh` with a separate output directory. The scripts compare the
rebuilt scientific content and refuse to overwrite an existing benchmark by default.

## Read results

```bash
python scripts/official_score_tables.py
python scripts/export_release_metrics.py
```

These commands use the released score records and require no model or dataset download.
[Official tables and selected scores](../../data/release/official_comparisons/)
contain all published metrics. Average the five splits within each training seed,
then report the mean and sample SD across five seed means. To collect fresh local
scorer outputs, use `scripts/export_split_metrics.py --scope published`.
