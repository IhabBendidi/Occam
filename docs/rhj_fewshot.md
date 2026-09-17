# `rhj_fewshot` — the 27-arm grid and the non-learned baselines

**This file owns every number the TRAINING side of this benchmark produces.** The benchmark
result is the split-half protocol (`docs/rhj_splithalf.md`); everything here is on the
**validation** path and its group-mean metric, which exists for checkpoint selection.

## What this is for

**Overall objective.** Predict a perturbation's effect in a context where that perturbation
was never measured, from a single control cell.

**The grid's question.** Which parts of the FiLM conditioning actually matter on this
benchmark, and does the negative-binomial loss beat smooth-L1 here?

**Varied:** `film_mode` × `film_hidden_dim` × `film_mean_free` × `loss_type` = 24 arms, plus
2 no-FiLM anchors and 1 baseline arm. **Held fixed:** the frozen benchmark artifact,
`film_scale` 1.0, 30 epochs, seed 42, frozen encoder, `target_pb_k` 1, `num_workers` 8.

**The metric everywhere below** is `pearson_delta`, group-mean, on the 103 held-out HepG2
**validation** perturbations, 6,983 genes. It is the checkpoint-selection metric. It is
**not** the benchmark result and is not comparable to any published number.

---

## ⛔ F5 — a non-learned baseline beats every one of the 24 arms, by 0.20

**Status: MEASURED, n=1 per arm, no seed replicates.** Grid jobs `18021944` / `18021945` /
`18022391` / `18022404` / `18022405`, baseline arm `18022439`. 27/27 completed, 0 failed.

| | val `pearson_delta` (group-mean, 103 HepG2 perts) |
|---|---|
| `ControlBaseline` — the HepG2 control mean | 0.0094 |
| `SimpleMeanBaseline` — global mean of training perturbation cells | 0.0113 |
| `ExperimentalAccuracy` — cells split in half, paired 1:1 | 0.3445 |
| **no-FiLM anchor** (`a24`) | 0.3222 |
| **best model arm** (`a16`, additive_only, fh0, fmf0, smooth_l1) | **0.3543** |
| **`MeanBaseline` — cross-cell-line transfer** | **0.5567** |

**`MeanBaseline` beats the best of 24 FiLM arms by +0.2024** and beats the no-FiLM anchor by
+0.2345. It is not learned and it costs nothing.

### What `MeanBaseline` actually is, and why that is the point

Confirmed by its own log line: **`MeanBaseline: 6731 specific + 0 global`** — exactly the F4
prediction. Because `baselines.py:120` tests `has_specific` across **every training cell
type at once**, and all 103 held-out HepG2 perturbations *are* trained in Jurkat and RPE1,
its prediction is:

> HepG2 control mean **+** the count-weighted mean of that perturbation's Jurkat and RPE1
> deltas.

That is the **non-learned version of exactly what the model is asked to do**. So F5 says:
*a plain average of the same perturbation's effect in two other cell lines carries more of
the HepG2 effect than anything the FiLM models learned.* The models are not failing to find
a hard signal — they are failing to exploit a signal that is trivially available in their
own training data.

The same ordering holds on the 621 test perturbations, in all three contexts:

| context | `ControlBaseline` | `SimpleMean` | `ExperimentalAccuracy` | **`MeanBaseline`** |
|---|---|---|---|---|
| HepG2 (621 held-out) | 0.0027 | 0.0234 | 0.3671 | **0.5690** |
| RPE1 (621, seen) | 0.0226 | 0.0520 | 0.5813 | **0.7716** |
| Jurkat (621, seen) | −0.0005 | 0.1017 | 0.3988 | **0.7202** |

HepG2 is the hardest of the three for `MeanBaseline` (0.5690 vs 0.7716 / 0.7202), which is
the expected direction: for HepG2 the delta is transferred *in*, for the other two it is
partly the context's own.

### ⚠️ `ExperimentalAccuracy` here is NOT a ceiling — it sits BELOW `MeanBaseline`

0.3445 against `MeanBaseline`'s 0.5567. That is not a paradox; it is the attenuation
asymmetry, in its strongest form. `ExperimentalAccuracy` pairs **single cells 1:1 with no
averaging**, so both sides of its correlation are one-cell-noisy, while `MeanBaseline`
predicts a smooth mean profile against a group mean and is attenuated on one side only. Call
it *"1-cell-to-1-cell replicate agreement"*, never *"the ceiling"*.

This is the same effect the split-half protocol is designed around, and it is why that
protocol reports a count-stratified gap rather than a single ceiling number.

---

## F6 — the ablation's own ordering, and the falsifier firing

**Every pre-registered prediction about the axes held. The one that matters went against the
hypothesis the grid was built to test.**

### smooth_l1 beats nb, 12/12

| loss | n | best | worst | mean |
|---|---|---|---|---|
| `smooth_l1` | 12 | 0.3543 | 0.3276 | **0.3400** |
| `nb` | 12 | 0.3019 | 0.2765 | **0.2919** |

Pre-registered: *"NB completes, and scores BELOW smooth_l1."* **Held**, with no overlap
between the two ranges at all. ⚠️ The confound was pre-registered too and still applies: at
`cosine_loss_weight` 0.5 the smooth_l1 arms are genuinely half-cosine while the NB arms are
effectively pure-NB, because the two recon terms differ by an order of magnitude. This is a
loss-*configuration* comparison, not a clean likelihood comparison.

### `additive_only` wins — and it is the arm that CANNOT be context-dependent

| `film_mode` (smooth_l1 only) | n | best | mean |
|---|---|---|---|
| `full` | 4 | 0.3454 | 0.3400 |
| `multiplicative_only` | 4 | 0.3435 | 0.3388 |
| **`additive_only`** | 4 | **0.3543** | **0.3413** |

`additive_only` was included as **the falsifier**: it has no `(γ−1)⊙z` term, so it cannot be
context-dependent, and its score was supposed to be the floor that makes
"multiplicative_only is the useful part" mean something. It is instead the **best** mode, on
both best-arm and mean.

**Reading: the multiplicative interaction buys nothing on this benchmark.** The research
tree's `multiplicative_only` finding does not reproduce here. The spread across modes is
0.0025 in the means, which is smaller than the ±0.007 BatchNorm noise floor — so the honest
statement is *the grid does not resolve `film_mode` at all*, and the imported
`multiplicative_only` recommendation has no support here.

### FiLM buys +0.032 over no FiLM

Best FiLM arm 0.3543 against the no-FiLM additive anchor 0.3222. Real but small, and
dwarfed by the 0.2024 the model is behind `MeanBaseline`.

### `film_mean_free` is a wash, and its degenerate case behaved

| | n | best | mean |
|---|---|---|---|
| `fmf0` | 6 | 0.3543 | 0.3395 |
| `fmf1` | 6 | 0.3479 | 0.3406 |

Pre-registered: *"`additive_only` × `mean_free` × `hidden 0` is analytically near-degenerate
(`Wβ(e_p) − Wβ(ē) = W(e_p − ē)` is embedding-centring, which the bias can absorb). If that
arm moves a lot, the `film_mean_free` port is wrong."* `a16` (fmf0) 0.3543 → `a18` (fmf1)
0.3479, a move of **−0.0064**. It did not move. **The port is not obviously wrong**, and the
mean-free term is doing approximately nothing in that configuration, as predicted.

---

## Full table

All 27 arms: [`docs/rhj_fewshot_results.md`](rhj_fewshot_results.md), regenerated with
`python scripts/collect_rhj_results.py --out docs/rhj_fewshot_results.md`.

Top 8, all smooth_l1:

| arm | val pΔ | film_mode | fh | fmf |
|---|---|---|---|---|
| `a16` | **0.3543** | additive_only | 0 | 0 |
| `a18` | 0.3479 | additive_only | 0 | 1 |
| `a06` | 0.3454 | full | 256 | 1 |
| `a10` | 0.3435 | multiplicative_only | 0 | 1 |
| `a08` | 0.3427 | multiplicative_only | 0 | 0 |
| `a02` | 0.3397 | full | 0 | 1 |
| `a04` | 0.3379 | full | 256 | 0 |
| `a12` | 0.3378 | multiplicative_only | 256 | 0 |

---

## What this does NOT establish

Stated because someone could act wrongly on F5 and F6 specifically.

1. **n = 1 per arm.** The `a16`-over-`a18` margin is 0.0064 and the mode means span 0.0025,
   both **below the ±0.007 noise floor**. No arm ordering here is resolved except
   `smooth_l1` > `nb`, which is a 0.048 gap with no range overlap. Seed replication
   (`scripts/build_seed_grid.py`) exists for exactly this.
2. **This is the selection metric, not the benchmark.** Group-mean `pearson_delta` on 103
   validation perturbations. This project has a documented case of selection and benchmark
   metrics ranking models *oppositely*. `MeanBaseline` beating the models on val does not
   by itself predict the split-half result — but it is a strong enough signal that the
   split-half run should include a transfer arm.
3. **`MeanBaseline` is not available at deployment for a genuinely novel perturbation.** It
   needs that perturbation measured in at least one other cell line. It is the right
   comparator for *this* benchmark, whose design guarantees exactly that, and the wrong one
   for a zero-shot claim.
