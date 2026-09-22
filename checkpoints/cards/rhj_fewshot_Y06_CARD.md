# Occam vcell — `rhj_fewshot` main checkpoint (arm **Y06**)

## What this model does

Predicts a perturbation's transcriptional effect in a cell line where **that perturbation was
never measured**, from a **single control cell** in the target cell line. Trained on HepG2 +
Jurkat + RPE1 with a real perturbation holdout; scored on **611 held-out HepG2 perturbations**.

This is the **main checkpoint for the `rhj_fewshot` benchmark** and the configuration that
became Occam's default recipe on 2026-09-22 (`docs/default_recipe.md`).

## Identity

| | |
|---|---|
| file | `best_model.pth` (2,230,677,544 bytes) |
| sha256 | `fe93eeac08d345f3ec53a411879d6331985b4ef9df2b5c856cfc8f2b50fce219` |
| run | `rhj_fewshot_Y06__smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_e150_em3e-4_s42` |
| selected epoch | **142** of 150 (best validation `pearson_delta`) |
| benchmark | `rhj_fewshot`, sha256 `d934211e5eb1e37e28c893fb8c5ff4211d607a1d77b0103d5c2b79221d21677f` |
| scoring axis | 6,983 genes, sha256 `ef64f2ba7ff7816ed947796066cd8a68014884c6176619d750ef79462c78d4c0`, `axis_scope: input_and_loss` |

## Score — 611 held-out HepG2 perturbations

Protocol: 5 paired split-halves (half-A is the reference, half-B perturbed cells are the
target, the model is fed half-B **controls**), cell-eval 0.6.0, **`nocal`** (no calibration).
Mean over the 5 splits. **`mae` is an ERROR — lower is better.** `discrimination_score_l1`
chance is 0.5.

| arm | pearson_delta ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ |
|---|---|---|---|---|
| `ctrlcell` (zero-signal floor) | −0.0018 ± 0.0031 | 0.0296 | 0.0000 | 0.5038 |
| `ctrlmean` (zero-signal floor) | 0.0016 ± 0.0028 | 0.0231 | 0.0081 | 0.5014 |
| `meanpert` (predict a constant) | 0.1324 ± 0.0030 | 0.0290 | 0.0108 | 0.5037 |
| `transfer` (same pert, other cell line) | 0.2323 ± 0.0021 | 0.0299 | 0.0351 | 0.6404 |
| **Y06 — this checkpoint** | **0.4043** ± 0.0019 | **0.0215** | **0.0946** | **0.7288** |
| `experimental` (replicate agreement) | 0.3722 ± 0.0010 | 0.0257 | 0.1942 | 0.7821 |

`overlap_at_N` 0.0946 and `disc_l1` 0.7288 are campaign maxima; `mae` 0.0215 ties the
Q02-family best. Best competitor on the same panel and protocol: **TxPert 0.3463** ± 0.0035.

**Y06 scores ABOVE `experimental`** (0.4043 vs 0.3722). `experimental` is replicate agreement
at the panel's own cell counts (median half 22 cells), **not an upper bound** — a model that
predicts the group mean can beat a noisy empirical half. Read it as a reference, not a ceiling.

⚠️ `best_metric` stored in the checkpoint is **0.4409**, which is the **validation** score on
the 103 HepG2 val perturbations, a different quantity from the 0.4043 test number above. Do not
quote it as the result.

⚠️ Do **not** rank on `overlap_at_N` for this benchmark: a zero-signal predictor beats a real
model on it (the emission-floor artifact).

## Configuration

| | |
|---|---|
| encoder | Occam small, 512-dim latent, 44,728 genes, **FROZEN** |
| perturbation model | `adjacency_map` (MLP over each gene's adjacency row) over 18,216 nodes, graphs `go` + `string`, `top_20` edge filter, `pert_latent_dim` 128 |
| interaction | FiLM, `multiplicative_only`, `film_mean_free: true`, hidden 256 |
| loss | Q02 = `smooth_l1`(β 2.0) + `cosine` 0.5 + `delta_cos` **0.5** + `delta_mag` **0.01363** + latent MSE on `delta_z` **0.0044** |
| schedule | 150 epochs, 5 warmup, `lr` 1e-3, `decoder_lr` 1e-3, `eta_min` 3e-4, batch 2048, grad clip 1.0 |
| input / output | **one** control cell in, **one** cell out (`n_controls_per_sample: 1`, `target_pb_k: 1`) |
| `target_sum` | 2000.0 — must match at inference |

Parameters: encoder 94,235,136 (frozen) · decoder 128,456,385 · perturbation model
334,170,944 · FiLM 329,216 · **total 557,191,681**.

Reproduce with `configs/rhj_fewshot.yaml` as shipped — it resolves to exactly this arm,
with no overrides. Scripts: `scripts/repro/rhj_train.sh`, `scripts/repro/rhj_eval.sh`,
`scripts/repro/occam_predict.py`; see `scripts/repro/README.md`.

## How to load

```python
import torch
ck = torch.load("best_model.pth", map_location="cpu", weights_only=False)
# self-contained: encoder_state_dict, decoder_state_dict, gnn_state_dict,
# interaction_state_dict, gene_names (44,728, ordered), gene_to_index, pert2id (18,216),
# target_sum, training_config, benchmark, split_info
```

Scoring entry point: `python scripts/eval_splithalf.py --mode model --checkpoint <this file>
--panel-dir data/benchmarks/rhj_fewshot/splithalf --split 0..4 --calibrate none`.

⚠️ `training_config["encoder_checkpoint"]` holds the **absolute local path** of the pretrained
encoder on the machine that trained this run. It is informational only — the encoder weights
are inside this file as `encoder_state_dict`, so nothing needs to resolve that path.

## Limitations

- **n=1, no seed replicates.** Every margin quoted here is a single seed (42).
- Validation **selects the epoch well but misranks arms** (finding F-ckpt4, held on 5
  consecutive grids). The 0.4409 val / 0.4043 test gap is normal, not a defect.
- Single cell line family: HepG2 / Jurkat / RPE1. Nothing here establishes transfer to a
  cell type outside that set.
- The benchmark data is third-party and its licensing is unresolved; see `docs/hf_release.md`.
