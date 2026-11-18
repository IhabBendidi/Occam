# Evaluate the released models

[Back to README](../README.md) ·
[Official comparison tables](../data/release/official_comparisons/official_tables.md)

## Install the scorer separately

Occam training uses Python 3.8. Perturbation-effect scoring uses `cell-eval==0.6.0` in a
separate Python 3.10 environment because its dependencies differ:

```bash
python3.10 -m venv .venv-celleval
.venv-celleval/bin/pip install 'cell-eval==0.6.0'
export OCCAM_CELLEVAL_PYTHON="$PWD/.venv-celleval/bin/python"
```

Run the commands below with your Occam training environment active. The evaluation script
calls the separate scorer automatically.

## Score one checkpoint

```bash
python scripts/download_release.py virtual-cell --benchmark rhj --seed 42
python scripts/download_release.py data --benchmark rhj
python scripts/eval_splithalf.py \
    --checkpoint checkpoints/vcell/rhj/seed_42/best_model.pth \
    --panel-dir data/benchmarks/rhj_fewshot/splithalf \
    --out-root runs/evaluation/rhj_seed42 \
    --split 0 --calibrate none --mode model
```

For Norman, download `--benchmark norman`, select
`checkpoints/vcell/norman/seed_42/best_model.pth`, and use
`data/benchmarks/norman_combo/splithalf`. Keep `--compose decode_seq_mean`, which is the
script's default for double perturbations.

For H1 or K562, download the corresponding [dataset](downloads.md#data), use the **RHJ**
checkpoint, and pass `--ood-context` with its panel directory. No fine-tuning is performed.

| Setting | Panel directory |
|---|---|
| HepG2 few-shot | `data/benchmarks/rhj_fewshot/splithalf` |
| Unseen K562, seen assay | `data/benchmarks/k562_seen/splithalf` |
| Unseen H1, unseen assay | `data/benchmarks/h1_seen/splithalf` |
| K562 with H1's target genes | `data/benchmarks/k562_seen_h1set/splithalf` |
| Norman double perturbations | `data/benchmarks/norman_combo/splithalf` |

## Reproduce means and standard deviations

Run each training checkpoint (seeds 42–46) on each evaluation split (`--split 0` through
`--split 4`), with a separate output directory per checkpoint. Average the five splits
within each training seed, then report the mean and sample SD across the five seed means.
Do not treat the 25 evaluations as 25 independent training runs.

The split-half protocol compares both predictions and experimental reproducibility with
the same reference half, using matched cell counts. `--mode reference` scores the baselines
and experimental reproducibility without a model checkpoint. Full protocol details are in
the [paper](https://arxiv.org/abs/2611.21959).

The [official tables](../data/release/official_comparisons/official_tables.md) and their
[CSV source](../data/release/official_comparisons/) are already in Git; reading results
requires no model or dataset downloads.

## Representation evaluation

```bash
pip install -e ".[eval]" scib
python scripts/download_release.py encoder-decoder
python scripts/download_release.py data --benchmark representation
test -f configs/paths.yaml || cp configs/paths.example.yaml configs/paths.yaml
occam-benchmark --checkpoint checkpoints/release/occam_small_encoder_decoder.pth
occam-scib --checkpoint checkpoints/release/occam_small_encoder_decoder.pth --datasets pancreas_scib,pbmc_68k
```

If you already have `configs/paths.yaml`, keep your existing file and check its `eval_dir`
and `scib_dir`. The paper reports HepG2, Jurkat, and RPE1 perturbation benchmarks plus
Pancreas and PBMC68k cell-integration benchmarks. The encoder stays frozen during evaluation.
