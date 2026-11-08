# Occam vcell — historical `norman_combo` ablation checkpoint (arm **NQ02**)

**Historical record.** NF32ALLA is the current official Norman checkpoint. NQ02 is retained
under ablations for reproducibility; the scores below describe its original seed-42 run.

## What this model does

Trained on **single-gene** CRISPRa perturbations in K562 (Norman et al. 2019, GSE133344);
predicts the effect of a **PAIR** of genes where both were seen individually and the pair never
was. Input is a **single control cell**. This is the composition task — the direct test of
Occam's own operator.

This was the original main checkpoint for the `norman_combo` benchmark. Its configuration was
the earlier Q02 recipe, scale-matched to this corpus.

## Identity

| | |
|---|---|
| file | `best_model.pth` (2,230,625,192 bytes) |
| sha256 | `91aed55e3db04afb49e225e6d49193c14f5bdf72b74b3815111c7da8cff26d4a` |
| run | `norman_combo_NQ02__smooth_l1_nbw0_nbs1.0_dcos0.32977_mag0.0400866_lmse0.00198826_e150_em3e-4_s42` |
| selected epoch | **44** of 150 (best validation `pearson_delta` on the 20 held-out val pairs) |
| benchmark | `norman_combo`, sha256 `8ac7da0d5adb3ae49e3a3dfe084494f4dce051286d219e346b4ea11c467cb3cf` |
| scoring axis | 8,000 genes, sha256 `6efa91af1af5c8b88fa5b024c683a53962ab9b7aa0c02b82173babb45966c77d`, `axis_scope: input_and_loss` |
| composition mode | **`decode_seq_mean`** (the default; see below) |

## Score — two effect-size-controlled test panels, 36 pairs each

Protocol: 5 paired split-halves, cell-eval 0.6.0, **`nocal`**, composition
**`decode_seq_mean`**. Mean ± SEM over the 5 splits. **`mae` is an ERROR — lower is better.**
`discrimination_score_l1` chance is 0.5. The panels split the 111 test pairs by additivity
residual `‖Δ_A + Δ_B − Δ_real‖ / ‖Δ_real‖`, terciled **within** effect-size terciles, so the
two panels are matched on effect size (`splithalf/CARD.md`).

### `additive` panel

| arm | pearson_delta ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ |
|---|---|---|---|---|
| `meanpert` (predict a constant) | 0.4634 | — | — | — |
| **NQ02 × `decode_seq_mean`** | **0.8169** ± 0.0009 | **0.0123** ± 0.0000 | **0.3059** ± 0.0006 | 0.9091 ± 0.0009 |
| **`sum_of_singles`** ⭐ the bar | **0.8824** | 0.0126 | 0.2865 | 0.9891 |
| `experimental` (replicate agreement) | 0.9165 | — | — | — |

### `nonadditive` panel

| arm | pearson_delta ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ |
|---|---|---|---|---|
| `meanpert` | 0.4016 | — | — | — |
| **NQ02 × `decode_seq_mean`** | **0.7380** ± 0.0018 | **0.0146** ± 0.0000 | **0.2843** ± 0.0008 | 0.7708 ± 0.0011 |
| **`sum_of_singles`** ⭐ the bar | **0.7757** | 0.0168 | 0.2750 | 0.8578 |
| `experimental` | 0.8377 | — | — | — |

**Read this honestly: the model does NOT beat additivity on `pearson_delta`** — it is 0.0655
below `sum_of_singles` on the additive panel and 0.0377 below on the non-additive one.
`sum_of_singles` (`Δ_A + Δ_B`, measured from the singles) IS this architecture's hypothesis in
closed form, so that gap means the model has not learned genuine interaction.

**What it does beat the bar on:** `mae` on both panels (0.0123 vs 0.0126; 0.0146 vs 0.0168) and
`overlap_at_N` on both (0.3059 vs 0.2865; 0.2843 vs 0.2750). So it ranks and reconstructs
better than additivity while predicting the pseudobulk delta profile slightly worse.
`disc_l1` stays below the bar on both panels.

⚠️ `best_metric` in the checkpoint is **0.7800** — the **validation** score on the 20 val
pairs under `latent_seq_mean`, a different quantity and a different mode from every number
above.

## Why `decode_seq_mean`

Chosen from the full 72-cell matrix (6 arms × 12 composition modes × 2 panels, 360 eval tasks,
0 failures; `docs/norman_results.md`). Over the 8 distinct modes it has the best mean rank
(2.08 of 8), is top-two on 5 of 6 arms — no other mode is — and gives the best single cell on
`pearson_delta` and `mae`. It chains the perturbations through the full pipeline
(`control → encode → FiLM(A) → decode → re-encode → FiLM(B) → decode`) and averages the two
orders, since order matters here (max |ab − ba| = 8.96e-02).

⚠️ `sum` is the mode to beat, not a discarded option: higher additive `pearson_delta` marginal
(0.7402 vs 0.7351) and one decoder pass instead of four. It degrades faster into
non-additivity (ratio 0.854 vs the bar's 0.879).

⚠️ The three `latent_seq_*` modes agree to **1.07e-08** — mean-free multiplicative FiLM makes
each step a per-dimension multiply, which commutes, so latent chaining cannot express order or
interaction at all. That is a property of the architecture, not a bug.

## Configuration

Identical to the `rhj_fewshot` main checkpoint except the three loss weights, which are
**scale-matched to this corpus** (`scripts/norman/norman_loss_scale.py`):

| | |
|---|---|
| encoder | Occam small, 512-dim latent, 44,728 genes, **FROZEN** |
| perturbation model | `adjacency_map` over 18,216 nodes, `go` + `string`, `top_20`, `pert_latent_dim` 128 |
| interaction | FiLM, `multiplicative_only`, `film_mean_free: true`, hidden 256 |
| loss | Q02 = `smooth_l1`(β 2.0) + `cosine` 0.5 + `delta_cos` **0.32977** + `delta_mag` **0.0400866** + latent MSE on `delta_z` **0.00198826** |
| schedule | 150 epochs, 5 warmup, `lr` 1e-3, `decoder_lr` 1e-3, `eta_min` 3e-4, batch 2048 |
| `combo_decomposition` | **true** — required: `"A+B"` is not a graph node, and without it the val split resolves EMPTY |
| `target_sum` | 2000.0 — must match at inference |

Parameters: encoder 94,235,136 (frozen) · decoder 128,456,385 · perturbation model
334,170,944 · FiLM 329,216 · **total 557,191,681**.

⚠️ **Do not copy the loss weights to another dataset.** The same base on `rhj_fewshot` uses
0.5 / 0.01363 / 0.0044 — 1.5× to 2.2× different. `tests/test_default_recipe.py` fails if the
two configs ever carry identical weights.

## Reproduce

For the historical NQ02 training schedule, use `configs/norman_combo.yaml` with
`stages.vcell.optim.batch_size=2048` and `stages.vcell.optim.eta_min=3e-4` overrides.
`scripts/repro/norman_eval.sh` accepts this checkpoint as an explicit argument; its default
now evaluates NF32ALLA.

## Limitations

- **n=1, no seed replicates.** The margin over `NM03__delta_sum` is 0.0178.
- The cell line is **K562 only**; nothing here establishes transfer to another context.
- Only **pairs whose two genes were both seen as singles**. Composing an unseen gene is
  untested.
- The upstream file `norman_2016_a549_preprocessed.h5ad` is mislabelled — it is 2019 and K562.
  This checkpoint was trained on the GEO rebuild (`docs/norman_benchmark.md`).
- `training_config["encoder_checkpoint"]` holds an absolute local path; informational only,
  the encoder weights are inside this file.
