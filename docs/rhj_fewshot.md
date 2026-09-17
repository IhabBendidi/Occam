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

## ⭐ F10 — the F7 oscillation is 5.9× smaller in the new setting, cause not yet isolated

**Status: MEASURED on one configuration; the isolating experiment is running.**

Last 10 epochs of the validation `pearson_delta`, same metric, same benchmark:

| setting | curve | span | mean \|Δ epoch-to-epoch\| |
|---|---|---|---|
| **old**: `gnn`, batch 256, lr 1e-3 | 0.324 0.328 0.226 0.193 0.223 0.139 **0.341 0.174 0.354 0.114** | **0.2400** | **0.1158** |
| **new**: `adjacency_map`, batch 2048, lr 3e-4 | 0.297 0.288 0.276 0.300 0.267 0.299 0.310 0.291 0.303 0.279 | **0.0433** | **0.0196** |

The phase-locked ±0.20 alternation is **gone**. Epoch-to-epoch movement is down **5.9×** and
the curve now looks like a converged model with ordinary noise. Four arms checked, all the
same shape.

### ⚠️ Three things changed at once, so the cause is still open

`gnn → adjacency_map`, `batch 256 → 2048`, and `lr 1e-3 → 3e-4` all moved together. Each is
a plausible cause on its own:

* **lr** — the original F7(a) hypothesis. 3e-4 is 3.3× lower.
* **batch size** — 8× larger means 8× fewer optimiser steps per epoch and a much lower
  gradient variance per step, which would damp exactly this.
* **the perturbation model** — the adjacency map has no message passing, so it cannot
  amplify a graph-wide perturbation of the embedding table between epochs.

⚠️ **Note what this does NOT support.** The F7(b) hypothesis was eval-time BatchNorm
statistics from an unbalanced epoch tail. The decoder still has its 11 `BatchNorm1d` layers
and `proportional` interleave is unchanged, so if BN were the whole story the oscillation
should have survived. That it did not makes (b) less likely — but batch 2048 also changes
how many batches contribute to the running statistics, so (b) is not excluded either.

### ⭐ RESOLVED: it is NOT the learning rate

Sweep indices 0 / 162 / 324 differ **only** in `lr`. All three completed (job `18034957`,
~29 min each). Last 9–10 epochs of the validation `pearson_delta`:

| `lr` | curve | span |
|---|---|---|
| 3e-4 | 0.297 0.288 0.276 0.300 0.267 0.299 0.310 0.291 0.303 0.279 | **0.043** |
| **1e-3** *(the old grid's value)* | 0.283 0.273 0.294 0.239 0.269 0.299 0.298 0.294 0.265 | **0.060** |
| 3e-3 | 0.239 0.201 0.264 0.235 0.242 0.265 0.230 0.248 0.195 | **0.070** |

**At `lr` 1e-3 — exactly the old grid's learning rate — the span is 0.060 against the old
grid's 0.240.** Four times smaller, with the learning rate held identical. Even at 3e-3,
three times higher than the old grid, the span is 0.070.

**So F7(a) is falsified: the ±0.20 oscillation was not a learning-rate artefact.** The
credit belongs to the other two changes — batch 256 → 2048, or `gnn` → `adjacency_map`, or
both. Given F8 showed the GNN was a fixed per-step cost recomputed every batch while being
trained, and the adjacency map has no message passing to amplify a graph-wide embedding
perturbation between epochs, either remains plausible; separating them would need a
`gnn` × batch-2048 arm, which is not currently planned.

⚠️ **What this means for the old 27-arm grid: its rankings stay dead.** The oscillation was
real and F6 is still uninterpretable. But the cause does not apply to any run in the new
setting, so the adjacency-map sweep's rankings are trustworthy at the ~0.02 level rather
than the ~0.20 level.

**Side finding, and it is a real one:** lower `lr` is better here, monotonically — best
observed value 0.310 at 3e-4, 0.299 at 1e-3, 0.265 at 3e-3. The sweep will confirm across
the other axes.

### The isolating experiment (as originally planned)

The sweep's TSV puts `lr` as its outermost axis, so all 162 `lr=3e-4` arms run before the
first `lr=1e-3` — which means the sweep alone will not answer this for hours. Two dedicated
arms, identical to sweep index 0 except for `lr` (1e-3 and 3e-3), are running under the
`f7probe_` prefix so they cannot collide with the sweep's run dirs.

Reading, pre-registered:
* **oscillation returns at lr 1e-3** → F7 is a learning-rate artefact. The old grid's
  rankings are noise, and the fix is simply a lower lr.
* **curve stays smooth at lr 1e-3 and 3e-3** → lr is *not* the cause, and the credit goes to
  batch size or the adjacency map. Then the old grid's F6 ranking is still dead, but for a
  reason that no longer applies to any new run.

---

## F8 — the 8.9× slowdown is batch size and the GNN, not I/O and not the estimand

**Status: MEASURED.** Job `18033959`, 6 arms × 200 real training batches on one H100.
Steady-state throughput (second half of each run, after warmup):

| arm | batch | workers | pert model | samples/s | vs baseline |
|---|---|---|---|---|---|
| 0 | 256 | 8 | `gnn` | **1,615** | 1.00× (the 27-arm grid) |
| 1 | 1024 | 8 | `gnn` | 5,818 | **3.60×** |
| 2 | 2048 | 8 | `gnn` | 10,435 | **6.46×** |
| 3 | 256 | **16** | `gnn` | 1,626 | 1.01× |
| 4 | 2048 | 16 | `gnn` | 10,732 | 6.64× |
| 5 | 256 | 8 | **`adjacency_map`** | **9,517** | **5.89×** |

### Three conclusions, two of which overturn what I previously wrote

1. **Batch size is the lever.** 256 → 1024 → 2048 gives 3.60× → 6.46×, close to linear.
   That is the signature of a **fixed per-step cost**, and it is the single biggest
   difference from the tuned research job, which used `--batch_size 2048`.
2. **Workers do nothing.** 8 → 16 is +0.7% at batch 256 and +2.8% at batch 2048. The run is
   **not I/O bound**, so the "per-job I/O bound" explanation I gave earlier is **wrong**.
   The research job's 14 workers were not the difference either.
3. **The adjacency map alone buys 5.89× at batch 256** — more than moving to batch 1024
   with the GNN. So the GNN is the fixed per-step cost, and its share is far larger than
   the 30% I measured for it in isolation: that isolated timing used a synthetic random
   graph, whereas the real path merges 515,063 edges across two KGs, carries edge weights,
   and reduces to perturbation nodes.

### `target_pb_k` is NOT the difference, and a plan-era claim was wrong

`scripts/neurips_fast.sh` has `PBK=${PBK:-1}` and this benchmark uses `target_pb_k: 1`.
**Identical.** The plan asserted that the best-known recipe used `target_pb_k=0` (exact
group-mean targets, ~√59× less label noise), and F7 leaned on that to argue `lr` 1e-3 was
mistuned for our estimand. **That argument is withdrawn**: the research default is 1, so
the research job ran the same estimand at the same learning rate. The LR hypothesis in F7
survives only on its own merits, not on an estimand mismatch.

### What this changes

At batch 2048 with the adjacency map, 30 epochs should take **well under half an hour**
rather than 3.2 h — the two effects are largely independent, though the combination was not
measured (the probe has no `adjacency_map` × 2048 cell). Sweeps that were unaffordable at
1,615 samples/s are now routine.

⚠️ **Batch size is not a free knob.** It changes the effective learning rate, so any batch
size adopted here needs its own `lr`, and a batch-size change inside an ablation confounds
the ablation. That is why the LR sweep has to move with it rather than after it.

---

## ⛔⛔ F7 — the validation metric oscillates by ±0.20 between consecutive epochs

**This invalidates the arm ranking in F6, and it is the most important finding of the run.**

The val `pearson_delta` does not converge. It alternates, violently, with epoch parity.
`a16`, every epoch of 30:

```
 1:0.0575   2:0.0799   3:0.1605   4:0.1331   5:0.1120   6:0.0583
 7:0.2523   8:0.1264   9:0.0739  10:0.0591  11:0.2165  12:0.1420
13:0.0140  14:0.1261  15:0.1277  16:0.3275  17:0.3141  18:0.1578
19:0.3164  20:0.0858  21:0.3241  22:0.3280  23:0.2256  24:0.1926
25:0.2226  26:0.1387  27:0.3407  28:0.1740  29:0.3543  30:0.1143
```

The last four epochs are **0.3407 → 0.1740 → 0.3543 → 0.1143**. The same phase-locked
alternation appears in every arm checked:

| arm | ep 25 | 26 | 27 | 28 | 29 | 30 |
|---|---|---|---|---|---|---|
| `a16` | 0.2226 | 0.1387 | **0.3407** | 0.1740 | **0.3543** | 0.1143 |
| `a06` | 0.2172 | 0.1628 | **0.3359** | 0.1332 | **0.3454** | 0.0899 |
| `a17` | 0.2349 | 0.1850 | **0.3019** | 0.1401 | **0.2889** | 0.0715 |

### Two candidate mechanisms, neither yet tested

**Both are live. The diagnosis below is a hypothesis, not a finding.**

**(a) Learning rate.** `lr` is 1e-3 with a cosine schedule. ⚠️ **The estimand-mismatch
argument I first gave for this is withdrawn — see F8**: the research job's default is
`target_pb_k=1`, the same as ours, so 1e-3 was not tuned on a quieter target. What remains
is that 1e-3 at **batch 256** is a different effective learning rate from 1e-3 at the
research job's **batch 2048**, which is a real and untested difference. **A learning-rate
sweep, moving with batch size, is the direct test and is planned.** The one thing that does
not obviously fit: a cosine schedule decays smoothly, so it does not by itself explain
strict epoch-*parity* alternation.

**(b) Eval-time BatchNorm statistics.**
`src/occam/decoder/external.py` — the decoder carries **11 `BatchNorm1d` layers**.
`pipeline.py:2622` calls `model.eval()` before validating, so those layers score the val
set with their **running** mean/var, accumulated over the training epoch just finished. With
`proportional` interleave over three files of different sizes, the tail of an epoch is not
balanced across cell lines, so the running statistics carry whichever context the epoch
ended on — and the val set is HepG2 only.

This is precisely the failure the plan anticipated and chose `proportional` interleave to
avoid: *"round-robin leaves a single-source epoch tail that poisons eval-time BatchNorm
stats (±0.007, larger than the effect being chased)."* **The anticipated size was ±0.007.
The measured size is ±0.20 — about 30× larger**, and `proportional` did not prevent it.
This mechanism does explain the parity alternation, which is why it is the first suspect —
but it is a suspect, and (a) has not been ruled out.

### What it invalidates

1. **`best_metric` selects the peak of an oscillation, not a converged model.** Every arm's
   `best_model.pth` is whichever high-parity epoch happened to land highest. The
   between-arm differences in F6 (0.3543 vs 0.3454 vs 0.3427, spread 0.012) are an order of
   magnitude **smaller** than the within-arm epoch-to-epoch swing (0.20).
2. **No arm ordering in F6 survives except `smooth_l1` > `nb`**, whose 0.048 gap with no
   range overlap is still larger than... no: it is *smaller* than 0.20, so strictly even
   that is not safe on this metric alone. It is supported instead by being 12/12
   consistent — every one of the twelve pairs agrees — which a symmetric ±0.20 noise process
   would produce with probability 2⁻¹² ≈ 0.0002.
3. **The ±0.007 noise floor quoted in F6 is wrong by 30×** and any caveat framed around it
   understates the problem.

### What it does NOT invalidate

* **F5 stands.** `MeanBaseline` (0.5567) is computed **before training starts**, from data
  only — no BatchNorm, no checkpoint, no epoch. Its 0.2024 margin over the best arm is
  larger than the full oscillation amplitude, and the best arm's number is the *peak* of
  that oscillation, so the true gap is if anything wider.
* The split-half benchmark is **not** affected the same way: it scores a fixed checkpoint
  once, in `eval()` mode, with no further training, so there is no epoch-tail to vary. What
  it inherits is a **checkpoint chosen on a noisy peak**. `final_model.pth` is not a remedy
  here and is not being scored: at epoch 30 every arm sits in the LOW phase of the
  oscillation (`a16` 0.1143, `a06` 0.0899, `a17` 0.0715), so the final weights are
  systematically the unlucky ones rather than an unbiased alternative.

### What would fix it, for whoever runs this next

Two independent tests, and they separate the two hypotheses:

1. **A learning-rate sweep** (planned). If lower `lr` removes the oscillation, (a) is the
   cause and the recipe simply does not transfer to this estimand.
2. **Recompute the BatchNorm running statistics on a fixed, cell-line-balanced batch stream
   immediately before each eval** — or switch the 11 `BatchNorm1d` layers to `LayerNorm`,
   which has no running state, or freeze the statistics after a warmup. If the oscillation
   disappears with the *same* `lr`, (b) is the cause.

Neither is run here. Run (1) first since it is already planned; if the oscillation survives
a 10× lower `lr`, (b) is implicated and the BN fix is the next move.

---

## F6 — the ablation's own ordering, and the falsifier firing

⚠️ **Read F7 first.** The within-arm epoch-to-epoch swing is ±0.20; the between-arm
differences below span 0.012. Only the `smooth_l1` > `nb` result survives, and it survives
on its 12/12 consistency rather than on its margin.

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

1. **n = 1 per arm, and the metric itself swings ±0.20 (F7).** The `a16`-over-`a18` margin
   is 0.0064 and the mode means span 0.0025, against a within-arm epoch-to-epoch swing of
   0.20. **No arm ordering here is resolved by margin.** `smooth_l1` > `nb` survives only
   because it is 12/12 consistent across pairs, which symmetric noise would produce with
   probability ~0.0002. Seed replication (`scripts/build_seed_grid.py`) will not fix F7 —
   the BatchNorm statistics need fixing first, or the replicates measure the same
   oscillation three times.
2. **This is the selection metric, not the benchmark.** Group-mean `pearson_delta` on 103
   validation perturbations. This project has a documented case of selection and benchmark
   metrics ranking models *oppositely*. `MeanBaseline` beating the models on val does not
   by itself predict the split-half result — but it is a strong enough signal that the
   split-half run should include a transfer arm.
3. **`MeanBaseline` is not available at deployment for a genuinely novel perturbation.** It
   needs that perturbation measured in at least one other cell line. It is the right
   comparator for *this* benchmark, whose design guarantees exactly that, and the wrong one
   for a zero-shot claim.
