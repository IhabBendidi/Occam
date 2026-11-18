# Occam Norman virtual cell — official NF32ALLA checkpoint

NF32ALLA predicts unseen gene pairs in the frozen Norman 2019 K562 benchmark after training on single-gene perturbations. The official release contains all five independently trained checkpoints, seeds 42–46. The older NQ02 checkpoint and other Norman Occam variants are ablations, not the current Norman release.

## Identity and recipe

| Field | Value |
|---|---|
| Files | `seed_<42..46>/best_model.pth`; root `best_model.pth` aliases seed 42 |
| Training arms | `S42_NF32ALLA` through `S46_NF32ALLA` |
| Benchmark | `norman_combo`, frozen benchmark sha256 `8ac7da0d5adb3ae49e3a3dfe084494f4dce051286d219e346b4ea11c467cb3cf` |
| Perturbation input | GO+STRING adjacency rows concatenated with GenePT and ESM2 embeddings |
| Loss | `smooth_l1 + 0.0255 * delta_cos`; no profile cosine, magnitude, or latent MSE term |
| Schedule | NF32: batch 32, 150 epochs, 5 warmup epochs, learning rate 1e-3, eta_min 3e-4 |
| Pair composition | `decode_seq_mean` |
| Reproduction config | `configs/norman_combo_official.yaml` |

The 0.0255 delta-cos weight approximates equal smooth-L1 and delta-cos contributions averaged over training. It corresponds to the 0.039 scale in the RHJ loss-balance experiment; the weights differ because loss magnitudes differ by corpus.

| Seed | Checkpoint SHA-256 |
|---|---|
| 42 | `2c73b717bb5391cd4d1dec4f76fc22fbd680a147a124442db354ff8c87597af8` |
| 43 | `e18fff6e6c676f9daad9b7ba0828e747b39e929b77e791b0ce2e5429bb25022d` |
| 44 | `c181809206fa1e1b06c9f5837696e6a11363e71fbf726e3da6c85953c353660a` |
| 45 | `f54ab574a80f725171fe6935ae5a889da27ed2a845327fa30551f8f8a268c0fd` |
| 46 | `b37bb66244378ea32709e3f7424c05af1b397e804762ca0e81483a42fcba0720` |

## Evaluation

Uncalibrated (`model_nocal`) cell-eval 0.6.0 scores on five seeded split-halves for each of five training seeds. Each value is the mean and sample SD of the five seed means, with five splits averaged within each seed. Each panel contains 36 held-out pairs. Higher is better except MAE.

| Panel and arm | Pearson delta ↑ | MAE ↓ | Overlap@N ↑ | Discrimination L1 ↑ |
|---|---:|---:|---:|---:|
| Additive · NF32ALLA | 0.8955 ± 0.0032 | 0.0108 ± 0.0002 | 0.2843 ± 0.0065 | 0.9662 ± 0.0046 |
| Additive · sum_of_singles | 0.8822 ± 0.0003 | 0.0126 ± 0.0000 | 0.2864 ± 0.0012 | 0.9890 ± 0.0003 |
| Additive · meanpert | 0.4631 ± 0.0032 | 0.0171 ± 0.0001 | 0.1344 ± 0.0009 | 0.5438 ± 0.0044 |
| Additive · experimental | 0.9165 ± 0.0000 | 0.0110 ± 0.0000 | 0.5247 ± 0.0000 | 0.9978 ± 0.0000 |
| Nonadditive · NF32ALLA | 0.8013 ± 0.0044 | 0.0134 ± 0.0001 | 0.2720 ± 0.0048 | 0.8249 ± 0.0123 |
| Nonadditive · sum_of_singles | 0.7749 ± 0.0012 | 0.0168 ± 0.0000 | 0.2751 ± 0.0011 | 0.8567 ± 0.0029 |
| Nonadditive · meanpert | 0.4023 ± 0.0028 | 0.0188 ± 0.0000 | 0.1225 ± 0.0009 | 0.4980 ± 0.0021 |
| Nonadditive · experimental | 0.8377 ± 0.0000 | 0.0129 ± 0.0000 | 0.4718 ± 0.0000 | 0.9644 ± 0.0000 |

Mandatory `ctrlmean` and `ctrlcell` anchors and every per-seed, per-split metric remain in `data/release/final/scores_by_seed_split.csv` and `data/release/norman_allsource/scores_by_seed_split.csv`. `data/release/norman_allsource/coverage.csv` verifies full coverage for NF32ALLA.

## Limitations

- Evaluation covers Norman K562 pairs whose individual genes were seen during training. It does not establish transfer to unseen genes or other cell types.
- The root `best_model.pth` is a convenience alias for seed 42. Reproducing the reported mean and SD requires evaluating all five `seed_<seed>/best_model.pth` files across the five splits.
- NF32ALLA improves Pearson delta and MAE over the high-weight all-source arm on both panels, but its overlap@N is lower. Its selection as the official Norman setup is a user decision across the reported metrics.
