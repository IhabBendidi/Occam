# `src/models/diffusion/` — the diffusion process

Companion to [../../../ZARR_BENCHMARK.md](../../../ZARR_BENCHMARK.md).

| File | Role |
|---|---|
| `diffusion_core.py` | `GaussianDiffusion` (assembled from the two mixins) + `ModelMeanType` / `ModelVarType` / `LossType` enums |
| `diffusion_schedules.py` | beta schedules (`linear`, …) |
| `diffusion_training.py` | training-loss mixin (MSE + geomloss energy MMD) |
| `diffusion_sampling.py` | `p_sample_loop` (DDPM) and `ddim_sample_loop` (DDIM) |

**Zarr impact: none.** Documented for completeness and because the *sampling* entrypoint's
behaviour is configured from here.

---

## Construction

`create_diffusion(model_cfg)` (`../lightning/lightning_factories.py:52`) builds the object from
`configs/model/base.yaml`:

```
steps: 1000                 predict_xstart: True        learn_sigma: False
noise_schedule: "linear"    noise_schedule_gamma: 0.3   sigma_small: False
use_kl: False               rescale_timesteps: False    rescale_learned_sigmas: False
```

`predict_xstart: True` means the network predicts x₀ directly, which is why `Cross_DiT`'s output
head is a ReLU over gene space rather than a noise prediction.

## Training losses

`diffusion_training.training_losses(model, x_start, t, self_condition, …, p_drop_cond, MMD_loss_fn)`
is called from `PlModel._compute_loss` (`../lightning/lightning_module.py:235-244`). It returns
`loss1` (per-sample MSE-style term) and `mmd1` / `mmd1_list` (geomloss `SamplesLoss(loss="energy",
blur=0.05)`), combined in the Lightning module as

```python
losses["loss"] = losses["loss1"] + losses["mmd1"] * optimizer_cfg.MMD_loss_factor
```

with `use_mse_loss=false` zeroing the first term (`lightning_module.py:258-263`).

The energy-MMD term is the distribution-level objective that motivates the cell-set (`[B, S, G]`)
batching — it is computed **across the S axis**, so `data.use_cell_set` is a genuine modelling
hyperparameter, not just a batching detail. Set it to **64** to match STATE's
`model.kwargs.cell_set_len=64`.

`p_drop_cond` / `p_drop_control` (`configs/model/base.yaml`) implement classifier-free-guidance
style conditioning dropout; the README's finetune and sampling blocks both set
`model.p_drop_control=0`.

## Sampling

`resolve_sampling_runner` (`../../apps/sampling/sampling_generation_helpers.py:14-45`) picks:

* **DDIM** — `ddim_sample_loop(start_time, eta, guidance_strength)`
* **DDPM** — `p_sample_loop(start_time, nw, start_guide_steps, guidance_strength)`

Defaults in `configs/rawdata_diffusion_sampling.yaml`: `use_ddim: true`, `start_time: 100`,
`eta: 0.0`, `guidance_strength: 1.0`, `clip_denoised: true`.

`start_time` is clipped to `model.steps` (helpers, lines 23-24). Note `start_time: 100` with
`steps: 1000` means sampling starts from t=100, **not** from pure noise — a deliberate
partial-noising choice. Record whatever you use, since it materially affects the benchmark numbers
and has no STATE analogue (STATE is a single deterministic forward pass).

## Benchmarking note

Because generation is stochastic, a fair comparison against STATE should either fix the seed
(`cfg.optimization.seed`, already passed to `pl.seed_everything` in the sampling entrypoint) or
average metrics over several sampling runs. STATE's `tx predict` is deterministic given a
checkpoint, so a single PerturbDiff draw is the noisier estimator of the two.
