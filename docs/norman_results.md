# `norman_combo` — results: 6 arms × 12 composition modes, 5 splits, 4 metrics

**Objective.** Predict a perturbation's effect in a context where that exact perturbation was
never measured. Here the form is **COMPOSITION**: train on single-gene perturbations, predict
PAIRS, where both genes were seen individually and the pair never was.

**Varied.** The loss base (M03 / M06 / Q02) × the schedule (150 ep @ `eta_min` 3e-4, the rhj
best; and 30 ep @ 1e-7, the bases' original setup) × the composition mode (12).
**Held fixed.** The 111-pair cohort, the 8,000-gene axis, the 5 seeded split-halves, the
singles-only corpus, and the space — library-norm 2,000 → log1p over the full 33,694-gene row
sum — so `mae` is comparable across every arm, baseline and (future) competitor.

**Directions.** `pearson_delta` ↑ · `mae` **↓ (ERROR)** · `overlap_at_N` ↑ · `disc_l1` ↑.

**The bar.** `sum_of_singles` = `ctrl + Δ_A + Δ_B` from the measured single arms. It is this
architecture's own operator in closed form (`model.forward` sums the perturbations' graph
embeddings), so **an arm at or below it has learned nothing about interaction.**

| reference (5/5) | `additive` pd | `nonadditive` pd | add `mae` | non `mae` | add `ovl@N` | non `ovl@N` |
|---|---|---|---|---|---|---|
| `experimental` | 0.9165 | 0.8377 | 0.0110 | 0.0129 | 0.5247 | 0.4718 |
| **`sum_of_singles`** ⭐ | **0.8824** | **0.7757** | 0.0126 | 0.0168 | 0.2865 | 0.2750 |
| `meanpert` | 0.4634 | 0.4016 | 0.0171 | 0.0188 | 0.1349 | 0.1225 |
| `ctrlmean` | 0.0033 | 0.0055 | 0.0179 | 0.0189 | 0.1171 | 0.1065 |

Jobs: `18319064` + `18319081` (train, 6/6) → `18319370` (eval, **360/360 COMPLETED**, 0 failures).

## The headline: no arm×mode beats the additive baseline, and the best gets within 0.038

| rank | arm × mode | `additive` pd | `nonadditive` pd | add `mae` ↓ | non `mae` ↓ | add `ovl@N` | non `ovl@N` |
|---|---|---|---|---|---|---|---|
| 1 | **`NQ02` × `decode_seq_mean`** | **0.8169** | **0.7380** | **0.0123** | **0.0146** | **0.3059** | **0.2843** |
| 2 | `NM03s` × `resid_seq_mean` | 0.8148 | 0.7171 | 0.0153 | 0.0190 | 0.2815 | 0.2477 |
| 3 | `NM03s` × `resid_seq_ab` | 0.8127 | 0.7165 | 0.0154 | 0.0190 | 0.2781 | 0.2465 |
| 4 | `NQ02` × `delta_sum` | 0.7989 | 0.7277 | 0.0189 | 0.0230 | **0.3180** | **0.2959** |
| 5 | `NM03` × `delta_sum` | 0.7905 | 0.7287 | 0.0158 | 0.0179 | 0.2829 | 0.2569 |
| — | `NM03` × `sum` (the naive default) | 0.7639 | 0.6190 | 0.0153 | 0.0177 | 0.2145 | 0.1990 |

**Gap to `sum_of_singles`, best cell:** −0.066 on `additive`, **−0.038** on `nonadditive`.
Against the naive `sum` on the weakest base it was −0.119 / −0.157, so **the composition mode
is worth more than the loss base**: choosing the right mode closes 44% of the additive gap and
76% of the non-additive one.

✅ **Three cells BEAT the additive baseline on `ovl@N`** — `NQ02 × delta_sum` 0.3180 / 0.2959,
`NM03s × delta_sum_realctrl` 0.3200 / 0.2767, `NQ02 × decode_seq_mean` 0.3059 / 0.2843 against
the bar's 0.2865 / 0.2750. So on the DE-overlap metric the model already composes better than
adding the measured deltas; it is `pearson_delta` where it does not.

⚠️ **The non-additive gap is now SMALLER than the additive one** (−0.038 vs −0.066) for the best
cell, reversing the naive mode's ordering (−0.157 vs −0.119). That is the signature a model with
genuine interaction knowledge should show. ⚠️ But the `nonadditive` panel is drawn from
`sum_of_singles`' own residual, so part of that reversal is selection — the claim to make is the
`additive` margin, where the baseline is near its best.

## ⭐ `latent_seq` is PROVABLY order-invariant here, so modes 4–6 are one mode

Measured over 6 arms × 2 panels × 5 splits, `max |latent_seq_ab − latent_seq_ba|` =
**1.07e-08** — floating-point noise. `decode_seq`'s is **8.96e-02**.

That is not a bug, it is the interaction's algebra. Arm s313 uses FiLM
`film_mode: multiplicative_only` with `film_mean_free: true`, so a step is
`z ← z + z·(m(e) − m(ē)) = z·(1 + m(e) − m(ē))`, and chaining two steps multiplies two scalars
per dimension. **Multiplication commutes, so the latent chain cannot depend on order.**
Consequence: `latent_seq_ab`, `_ba` and `_mean` are one mode, and **order effects cannot be
studied in latent space with a multiplicative-only interaction** — only in gene space, where
the encode/decode round trip breaks the commutation.

`decode_seq_ba` > `decode_seq_ab` for **all 6 arms on both panels** (e.g. `NQ02` additive
0.7422 vs 0.6668), and `_mean` beats both. Since the two orders are the same biology, that
spread is round-trip artifact, and averaging removes it — which is why `decode_seq_mean` wins.

## The full matrix — `pearson_delta`, mean over 5 splits

### `additive` (36 pairs)

| mode | NQ02 | NM03 | NM06 | NQ02s | NM03s | NM06s |
|---|---|---|---|---|---|---|
| `sum` | 0.8030 | 0.7639 | 0.7253 | 0.7640 | 0.6921 | 0.6930 |
| `delta_sum` | 0.7989 | 0.7905 | 0.6645 | 0.5240 | 0.7915 | 0.5645 |
| `delta_sum_realctrl` | 0.6422 | 0.5024 | 0.2980 | 0.1383 | **0.8099** | 0.1502 |
| `latent_seq_ab` | 0.6170 | 0.7214 | 0.5775 | 0.4697 | 0.6825 | 0.4361 |
| `latent_seq_ba` | 0.6170 | 0.7214 | 0.5775 | 0.4697 | 0.6825 | 0.4361 |
| `latent_seq_mean` | 0.6170 | 0.7214 | 0.5775 | 0.4697 | 0.6825 | 0.4361 |
| `decode_seq_ab` | 0.6668 | 0.6954 | 0.6103 | 0.6087 | 0.6603 | 0.5259 |
| `decode_seq_ba` | 0.7422 | 0.7500 | 0.6522 | 0.6905 | 0.6720 | 0.6109 |
| **`decode_seq_mean`** | **0.8169** | **0.8039** | 0.6964 | 0.7277 | 0.7406 | 0.6250 |
| `resid_seq_ab` | 0.5914 | 0.4651 | 0.2843 | 0.1218 | 0.8127 | 0.1338 |
| `resid_seq_ba` | 0.5799 | 0.5064 | 0.2761 | 0.1200 | 0.8050 | 0.1234 |
| `resid_seq_mean` | 0.5884 | 0.4910 | 0.2819 | 0.1211 | **0.8148** | 0.1289 |

### `nonadditive` (36 pairs)

| mode | NQ02 | NM03 | NM06 | NQ02s | NM03s | NM06s |
|---|---|---|---|---|---|---|
| `sum` | 0.6804 | 0.6190 | 0.6324 | 0.6376 | 0.6019 | 0.6210 |
| `delta_sum` | 0.7277 | **0.7287** | 0.6159 | 0.4826 | 0.6918 | 0.5432 |
| `delta_sum_realctrl` | 0.6131 | 0.4878 | 0.2760 | 0.1308 | 0.7037 | 0.1621 |
| `latent_seq_*` (all three) | 0.5403 | 0.5879 | 0.5027 | 0.4266 | 0.5500 | 0.3836 |
| `decode_seq_ab` | 0.5894 | 0.5961 | 0.5382 | 0.5280 | 0.5445 | 0.4489 |
| `decode_seq_ba` | 0.6228 | 0.6142 | 0.5563 | 0.5606 | 0.5766 | 0.5170 |
| **`decode_seq_mean`** | **0.7380** | 0.7004 | 0.6291 | 0.6336 | 0.6448 | 0.5647 |
| `resid_seq_ab` | 0.5821 | 0.4571 | 0.2563 | 0.1116 | 0.7165 | 0.1510 |
| `resid_seq_ba` | 0.5739 | 0.4538 | 0.2540 | 0.1098 | 0.7004 | 0.1413 |
| `resid_seq_mean` | 0.5852 | 0.4629 | 0.2574 | 0.1111 | **0.7171** | 0.1468 |

## What the matrix says

1. **The mode matters more than the base.** Within `NQ02` the modes span 0.589–0.817 on
   `additive` (0.228); across bases at fixed `decode_seq_mean` they span 0.625–0.817. Picking
   the composition is the bigger lever, and nothing about it requires retraining.
2. **`decode_seq_mean` is the most robust mode** — best or second for 5 of 6 arms — and
   `delta_sum` is the best on `nonadditive` for the two M03/Q02 150-epoch arms.
3. **Q02 > M03 > M06, as on rhj**, at fixed mode (`decode_seq_mean`: 0.8169 / 0.8039 / 0.6964).
   The base ordering transfers across dataset AND task.
4. **150 epochs beats 30 at fixed mode** (`NQ02` 0.8169 vs `NQ02s` 0.7277 on `decode_seq_mean`)
   — the rhj schedule finding (+0.0260 there) transfers, and is worth ~0.09 here.
   ⚠️ EXCEPT for the real-ctrl family, where `NM03s` (30 ep) is the best arm in the whole
   matrix and its 150-epoch twin collapses to 0.49. See below.
5. ⚠️ **The real-ctrl family is bimodal and arm-dependent**: 0.12–0.64 for five arms, but
   0.80–0.81 for `NM03s`. Adding deltas to the real control amplifies whatever scale error the
   arm has, and only `NM03s` has a small enough one. **Do not read `resid_seq_*` as a mode
   ranking** — it is a scale-calibration test that one arm passes.

## ⛔ Not established

* No arm×mode beats `sum_of_singles` on `pearson_delta` or `mae` — only on `ovl@N`.
* Every number is one seed. The `NM03s` real-ctrl result (0.81, against 0.49 for its
  150-epoch twin) is the single most surprising cell and rests on one run.
* The `nonadditive` panel is selected by the baseline's own residual (`splithalf/CARD.md`),
  so margins there are not independent of the baseline.

---

## Cross-cutting analysis of the 72-cell matrix (written 2026-09-22)

**Objective.** Norman double perturbations exist to ask whether a model that was trained on
SINGLE perturbations can compose them. The whole point of the two panels is that
`sum_of_singles` already answers "just add the two deltas"; anything we do has to be read
against it.

**Question this analysis answers.** Across the 6 trained arms × 12 composition modes × 2 panels,
(a) which LOSS BASE family generalises best, (b) which COMPOSITION MODE is best and on which
panel, and (c) does the mode ranking depend on the base.

**Varied:** loss base (Q02/M03/M06), schedule (150 ep eta_min 3e-4 vs 30 ep eta_min 1e-7),
composition mode, panel. **Fixed:** frozen 8,000-gene axis, benchmark sha256
`8ac7da0d5adb3ae4…`, split-half protocol, cell-eval 0.6.0, `nocal` only.

**Anchors (Norman, additive / nonadditive):** `meanpert` 0.4634 / 0.4016 · **`sum_of_singles`
0.8824 / 0.7757** · `experimental` (replicate ceiling) 0.9165 / 0.8377. `mae` bar 0.0126 /
0.0168 (ERROR, lower better); `overlap_at_N` bar 0.2865 / 0.2750; `discrimination_score_l1`
bar 0.9891 / 0.8578 (chance 0.5).

All aggregations below use the **8 distinct modes** (`latent_seq_{ab,ba}` are numerically
identical to `latent_seq_mean` to 1.07e-08, so counting them thrice would triple-weight a single
quantity).

### A1 — Base marginal, mean over the 8 distinct modes, `pearson_delta`

| arm | mean add | median add | mean non | median non | its best mode | best-mode add |
|---|---|---|---|---|---|---|
| **NM03s** (30 ep) | **0.7330** | 0.7163 | 0.6288 | 0.6234 | `resid_seq_mean` | 0.8148 |
| **NQ02** (150 ep) | 0.7094 | 0.7045 | **0.6371** | 0.6179 | `decode_seq_mean` | **0.8169** |
| NM03 (150 ep) | 0.6898 | 0.7357 | 0.5996 | 0.6051 | `delta_sum` | 0.7905 |
| NM06 (150 ep) | 0.5633 | 0.6313 | 0.5010 | 0.5472 | `sum` | 0.7253 |
| NQ02s (30 ep) | 0.5055 | 0.5663 | 0.4389 | 0.5053 | `sum` | 0.7640 |
| NM06s (30 ep) | 0.4668 | 0.5452 | 0.4234 | 0.4829 | `sum` | 0.6930 |

Restricting to the **6 recon-emission modes** (dropping the two real-ctrl modes that collapse on
4 of 6 arms) removes NM03s's advantage and makes the ordering monotone in the schedule:

| arm | mean add | mean non | mean both |
|---|---|---|---|
| NM03 | 0.7542 | 0.6410 | **0.6976** |
| NQ02 | 0.7408 | 0.6498 | **0.6953** |
| NM03s | 0.7065 | 0.6016 | 0.6540 |
| NM06 | 0.6544 | 0.5791 | 0.6167 |
| NQ02s | 0.6308 | 0.5448 | 0.5878 |
| NM06s | 0.5759 | 0.5130 | 0.5445 |

### A2 — Loss-base family and schedule marginals (recon modes × both panels)

| base family | mean `pearson_delta` | | schedule | mean `pearson_delta` |
|---|---|---|---|---|
| **M03** (dcos 0.01, balanced) | **0.6758** | | **150 ep / eta_min 3e-4** | **0.6699** |
| Q02 (M06 + latent MSE) | 0.6415 | | 30 ep / eta_min 1e-7 | 0.5954 |
| M06 (dcos 0.5, direction-dominated) | 0.5806 | | | |

**F-nor5. M06 is the worst base on Norman, by 0.095 `pearson_delta`.** The
direction-dominated loss (dcos 0.5 + delta_mag) that is competitive on RHJ single
perturbations is the weakest of the three when the task is composition. Adding the latent-MSE
term back (Q02 = M06 + latent MSE 0.0044) recovers +0.061 of that; the balanced M03 recovers all
of it. Reading: composition needs a faithful magnitude/latent geometry more than it needs a
sharpened direction.

**F-nor6. The long schedule transfers to Norman: +0.0840 `pearson_delta` mean, positive in
43/48 paired cells** (same base, same mode, 150 ep/3e-4 minus 30 ep/1e-7). The 5 negative cells
are all `NM03` vs `NM03s` on the two real-ctrl modes plus `delta_sum` (−0.3076 … −0.0009), i.e.
the one arm where the 30-epoch checkpoint happens to have a usable real-ctrl branch. On the 6
recon modes the long schedule wins **35/36**.

### A3 — Mode marginal, mean/median/mean-rank over the 6 arms, `pearson_delta`

| mode | mean add | med add | mean non | med non | mean rank (1–8) | times best |
|---|---|---|---|---|---|---|
| **`decode_seq_mean`** | 0.7351 | 0.7341 | **0.6518** | **0.6392** | **2.08** | 3 |
| **`sum`** | **0.7402** | **0.7446** | 0.6320 | 0.6267 | 2.25 | **6** |
| `delta_sum` | 0.6890 | 0.7275 | 0.6316 | 0.6539 | 3.08 | 1 |
| `decode_seq_ba` | 0.6863 | 0.6813 | 0.5746 | 0.5686 | 4.17 | 0 |
| `decode_seq_ab` | 0.6279 | 0.6353 | 0.5408 | 0.5413 | 5.50 | 0 |
| `delta_sum_realctrl` | 0.4235 | 0.4002 | 0.3956 | 0.3819 | 5.92 | 0 |
| `latent_seq_mean` | 0.5840 | 0.5972 | 0.4985 | 0.5215 | 6.25 | 0 |
| `resid_seq_mean` | 0.4043 | 0.3864 | 0.3801 | 0.3602 | 6.75 | 2 |

Mean and mean-rank disagree on the top slot, so both are reported: `sum` has the higher additive
mean and the most per-arm wins, `decode_seq_mean` the better mean rank and the better
non-additive mean. They are the only two modes never worse than rank 4 on any arm.

### A4 — Mode marginal on all four metrics (mean over the 6 arms)

| mode | pΔ add | pΔ non | mae add ↓ | mae non ↓ | ovl add | ovl non | dl1 add | dl1 non |
|---|---|---|---|---|---|---|---|---|
| `sum` | **0.7402** | 0.6320 | 0.0150 | 0.0181 | 0.2551 | 0.2400 | 0.9129 | 0.7433 |
| `delta_sum` | 0.6890 | 0.6316 | 0.0188 | 0.0212 | 0.2536 | 0.2393 | **0.9196** | **0.8013** |
| `latent_seq_mean` | 0.5840 | 0.4985 | 0.0178 | 0.0199 | 0.1848 | 0.1803 | 0.8042 | 0.6778 |
| `decode_seq_ab` | 0.6279 | 0.5408 | 0.0160 | 0.0185 | 0.2275 | 0.2049 | 0.8208 | 0.6632 |
| `decode_seq_ba` | 0.6863 | 0.5746 | 0.0155 | 0.0187 | **0.2595** | 0.2373 | 0.8759 | 0.6650 |
| **`decode_seq_mean`** | 0.7351 | **0.6518** | **0.0145** | **0.0169** | 0.2499 | 0.2342 | 0.8683 | 0.7066 |
| `delta_sum_realctrl` | 0.4235 | 0.3956 | 0.0288 | 0.0293 | 0.2107 | 0.1998 | 0.7948 | 0.7481 |
| `resid_seq_mean` | 0.4043 | 0.3801 | 0.0278 | 0.0282 | 0.1984 | 0.1907 | 0.7908 | 0.7397 |
| **BAR `sum_of_singles`** | 0.8824 | 0.7757 | 0.0126 | 0.0168 | 0.2865 | 0.2750 | 0.9891 | 0.8578 |

**F-nor7. No composition mode beats `sum_of_singles` on `pearson_delta` — the best single cell
is 0.0655 (additive) and 0.0377 (non-additive) below it — but three metrics ARE beaten.**
Best cells against the bar:

| metric | panel | best cell | value | bar | gap |
|---|---|---|---|---|---|
| `pearson_delta` | additive | `NQ02__decode_seq_mean` | 0.8169 | 0.8824 | **−0.0655** |
| `pearson_delta` | nonadditive | `NQ02__decode_seq_mean` | 0.7380 | 0.7757 | **−0.0377** |
| `mae` ↓ | additive | `NQ02__decode_seq_mean` | 0.0123 | 0.0126 | +0.0003 |
| `mae` ↓ | nonadditive | `NQ02__decode_seq_mean` | 0.0146 | 0.0168 | **+0.0022** |
| `overlap_at_N` | additive | `NM03s__delta_sum_realctrl` | 0.3200 | 0.2865 | **+0.0335** |
| `overlap_at_N` | nonadditive | `NQ02__delta_sum` | 0.2959 | 0.2750 | **+0.0209** |
| `discrimination_score_l1` | additive | `NQ02__delta_sum_realctrl` | 0.9826 | 0.9891 | −0.0065 |
| `discrimination_score_l1` | nonadditive | `NM03__delta_sum` | 0.8735 | 0.8578 | **+0.0157** |

`NQ02__decode_seq_mean` is simultaneously the best cell on `pearson_delta` (both panels) and on
`mae` (both panels), and it beats the additive bar on `mae`. ⚠️ `pearson_delta` and `mae` read
the same pseudobulk profile, so those are not four independent wins; `overlap_at_N` and
`discrimination_score_l1` are.

### A5 — Robustness to non-additivity: `pearson_delta` drop, additive → non-additive

| mode | drop | ratio non/add |
|---|---|---|
| `resid_seq_mean` | −0.0243 | **0.9400** |
| `delta_sum_realctrl` | −0.0279 | 0.9341 |
| `delta_sum` | −0.0573 | 0.9168 |
| **`experimental` (ceiling)** | −0.0788 | 0.9140 |
| `decode_seq_mean` | −0.0833 | 0.8867 |
| `decode_seq_ab` | −0.0871 | 0.8613 |
| `latent_seq_mean` | −0.0855 | 0.8536 |
| **`sum_of_singles` (bar)** | −0.1067 | 0.8791 |
| `sum` | −0.1082 | 0.8538 |
| `decode_seq_ba` | −0.1118 | 0.8372 |

**F-nor8. The two real-control modes are the only ones that degrade LESS than the replicate
ceiling does on the non-additive panel** (ratio 0.940 / 0.934 vs `experimental` 0.914) — they
are just doing it from a much lower absolute level (0.40 vs 0.75). Of the modes that are
actually competitive, **`delta_sum` is the most non-additivity-robust** (ratio 0.9168, better
than `experimental`), and it is also the best mode on `discrimination_score_l1` on the
non-additive panel (0.8013, the only mode above the 0.8578 bar in a single cell). `sum` and
`decode_seq_ba` degrade fastest — `sum` loses more than the additive bar does.

### A6 — Does the mode ranking depend on the base? Spearman ρ of the 8 mode scores, arm vs arm

| pair | ρ | pair | ρ |
|---|---|---|---|
| NQ02 vs NM06 | +0.952 | NQ02 vs NM03s | **−0.214** |
| NM06 vs NM06s | +0.976 | NM03 vs NM03s | **−0.214** |
| NQ02s vs NM06s | +0.976 | NM06 vs NM03s | **−0.333** |
| NQ02 vs NM03 | +0.905 | NQ02s vs NM03s | **−0.548** |
| NM03 vs NM06 | +0.905 | NM03s vs NM06s | **−0.429** |

Mean ρ = +0.489, but the structure is binary: **the 5 arms that are not NM03s agree almost
perfectly (ρ +0.76 … +0.98); NM03s anti-correlates with every one of them (ρ −0.21 … −0.55).**
Cause: for 5 of 6 arms the real-ctrl family (`delta_sum_realctrl`, `resid_seq_*`) collapses to
0.12–0.40, while for NM03s it is the BEST family (0.7568 / 0.7659 mean-of-panels). So the mode
ranking is stable except for the recon/real-ctrl family switch, which is arm-specific.

**F-nor9. Per-base spread across modes is the single largest source of variance in the grid.**
Sums of squares over the 6 × 8 × 2 design, grand mean 0.5747, SS(total) 3.022:
**SS(mode) 1.185 > SS(base) 0.880 ≫ SS(panel) 0.129.** Mode choice moves the score more than
base choice does, and both move it far more than which panel is scored. Per-arm spread
(best mode − worst mode, mean of panels):

| arm | best | | 2nd | | worst | | spread |
|---|---|---|---|---|---|---|---|
| NM03s | `resid_seq_mean` | 0.7659 | `delta_sum_realctrl` | 0.7568 | `decode_seq_ab` | 0.6024 | **0.1635** |
| NQ02 | `decode_seq_mean` | **0.7774** | `delta_sum` | 0.7633 | `latent_seq_mean` | 0.5786 | 0.1988 |
| NM03 | `delta_sum` | 0.7596 | `decode_seq_mean` | 0.7521 | `resid_seq_mean` | 0.4770 | 0.2826 |
| NM06 | `sum` | 0.6789 | `decode_seq_mean` | 0.6627 | `resid_seq_mean` | 0.2696 | 0.4092 |
| NM06s | `sum` | 0.6570 | `decode_seq_mean` | 0.5948 | `resid_seq_mean` | 0.1378 | 0.5192 |
| NQ02s | `sum` | 0.7008 | `decode_seq_mean` | 0.6806 | `resid_seq_mean` | 0.1161 | **0.5847** |

`decode_seq_mean` is in the top two for 5 of 6 arms — no other mode is.

### What each outcome meant, decided in advance

- A base family winning on the mode-marginal AND on its best cell ⇒ adopt it for Norman. **M03
  wins the marginal (0.6758), Q02-150ep wins the best cell (0.7774).** Split verdict: M03 is the
  safer base, Q02 the higher ceiling; M06 is ruled out on both.
- A composition mode winning on mean AND rank ⇒ make it the default. **`decode_seq_mean` and
  `sum` split this**, so the default is not yet forced; `decode_seq_mean` is the choice if one
  number must be picked, on mean rank (2.08) and top-two presence (5/6 arms).
- Beating `sum_of_singles` on `pearson_delta` would have meant the model learns genuine
  non-additive composition. **It does not (−0.038 at best).** It beats the bar on
  `overlap_at_N` (+0.021 … +0.034) and on non-additive `discrimination_score_l1` (+0.016), so it
  ranks and discriminates perturbations better than additivity does while predicting the
  pseudobulk profile slightly worse.

---

## DECISION 2026-09-22 — `decode_seq_mean` and Q02/150 ep/eta_min 3e-4 are now the DEFAULTS

User decision, taken on the analysis above. Recorded in `docs/default_recipe.md`, applied in
`configs/rhj_fewshot.yaml`, `configs/norman_combo.yaml`,
`VirtualCellModel.forward_composed`, `scripts/eval_splithalf.py --compose`, and pinned by
`tests/test_default_recipe.py` (5 tests).

⚠️ Two things this decision does NOT assert, so nobody reads more into it later:
1. **Q02 does not win Norman's base marginal — M03 does** (0.6758 vs 0.6415 pooled over
   schedules; 0.6976 vs 0.6953 on the 150-epoch recon-mode marginal, i.e. tied). Q02 is the
   default because it wins `rhj_fewshot` outright (0.4043) and wins Norman's best single cell
   (`NQ02__decode_seq_mean`, 0.7774 mean-of-panels vs `NM03__delta_sum` 0.7596). M03 remains
   the arm to check first if the default disappoints on a third dataset.
2. **`decode_seq_mean` changes nothing on `rhj_fewshot`** — every recon-emission mode is
   bit-identical on a single perturbation. The mode default only binds on combinatorial
   panels.

Both top-cell margins are **n=1, no seed replicates**: Q02-over-M03 is 0.0178 and
`decode_seq_mean`-over-`sum` is 0.0141 (mean of panels).

---

## COMPETITOR RESULTS on `norman_combo` (2026-09-22, seed 42, 5 splits, `nocal`)

**The question.** Train on single-gene perturbations; predict a PAIR where both genes were
seen individually and the pair never was. Does any published method beat `sum_of_singles`
(`Δ_A + Δ_B` in closed form), which IS this whole architecture family's hypothesis?

**Each arm ran its own composition mechanism** (`docs/competitors_on_doubles.md`): TxPert's
native additive sum, CellFlow's native set aggregation at `max_combination_length: 2`,
State's `decode_seq_mean` (our extension of its own consecutive forward passes). All on the
SAME frozen artifact Occam is scored on — axis `6efa91af1af5c8b8`, cohort
`dc1dd7806ba60d34`, 8,000 genes, the same 5 seeded split-halves.

### `additive` panel (36 pairs)

| arm | pearson_delta ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ |
|---|---|---|---|---|
| `meanpert` | 0.4634 | — | — | — |
| **State** (decode_seq_mean) | 0.6947 ± 0.0034 | 0.0169 | 0.2673 | 0.8831 |
| **CellFlow** (set aggregation) | 0.7603 ± 0.0003 | 0.0140 | 0.2633 | 0.9288 |
| **Occam** NQ02 × decode_seq_mean | 0.8169 ± 0.0009 | **0.0123** | 0.3059 | 0.9091 |
| **TxPert** (additive sum) | **0.8580** ± 0.0012 | 0.0135 | **0.3247** | **0.9776** |
| **`sum_of_singles`** ⭐ the bar | **0.8824** | 0.0126 | 0.2865 | 0.9891 |
| `experimental` | 0.9165 | — | — | — |

### `nonadditive` panel (36 pairs)

| arm | pearson_delta ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ |
|---|---|---|---|---|
| `meanpert` | 0.4016 | — | — | — |
| **State** | 0.5905 ± 0.0023 | 0.0198 | 0.2367 | 0.7321 |
| **CellFlow** | 0.6921 ± 0.0027 | 0.0170 | 0.2668 | **0.7760** |
| **TxPert** | 0.7139 ± 0.0027 | 0.0191 | **0.3041** | 0.7245 |
| **Occam** NQ02 × decode_seq_mean | **0.7380** ± 0.0018 | **0.0146** | 0.2843 | 0.7708 |
| **`sum_of_singles`** ⭐ the bar | **0.7757** | 0.0168 | 0.2750 | 0.8578 |
| `experimental` | 0.8377 | — | — | — |

## F-comp2. The two panels RANK THE METHODS DIFFERENTLY, and that is the finding

| | additive | nonadditive |
|---|---|---|
| 1st | **TxPert 0.8580** | **Occam 0.7380** |
| 2nd | Occam 0.8169 | TxPert 0.7139 |
| 3rd | CellFlow 0.7603 | CellFlow 0.6921 |
| 4th | State 0.6947 | State 0.5905 |

**TxPert is the best method where additivity already works; Occam is the best method where
it does not.** TxPert leads by +0.0411 on `additive` and trails by −0.0241 on
`nonadditive` — a swing of 0.065 between the two panels, against split SEMs of 0.001–0.003.

That is exactly the axis the effect-size-controlled strata were built to separate, and it
is consistent with each method's own mechanism: **TxPert composes a pair by SUMMING the two
perturbation embeddings**, which is the additive hypothesis expressed in latent space, so
it should — and does — excel precisely where the additive baseline excels. Occam's
`decode_seq_mean` chains the two perturbations through the full pipeline, re-encoding in
between, which is the only mechanism in this panel that is not additive by construction.

⚠️ Do not over-read it. Both margins are **n=1 seed**, and `decode_seq_mean` was selected
as Occam's default ON this benchmark (`docs/default_recipe.md`) while TxPert's additive sum
is its published mechanism — so Occam has had a mode-selection pass here that TxPert has
not. A fair reading is "these two are close and trade places by stratum", not "Occam wins
non-additivity".

## F-comp3. NOTHING beats `sum_of_singles` on `pearson_delta`. Four methods, two panels, no exceptions.

| arm | gap on additive | gap on nonadditive |
|---|---|---|
| TxPert | **−0.0244** | −0.0618 |
| Occam | −0.0655 | **−0.0377** |
| CellFlow | −0.1221 | −0.0836 |
| State | −0.1877 | −0.1852 |

The closest any method gets is TxPert, 0.0244 short on the panel where additivity is
easiest. **On the non-additive panel — the one that exists to reward interaction — the best
method is still 0.0377 short of simply adding the two single effects.**

⚠️ `sum_of_singles` is measured from the SINGLES on the same cells, so it is not a model
and cannot be deployed on an unseen pair without those singles. That is what makes it a
bar rather than a competitor. But every method here is given exactly the same singles in
training, so "worse than adding them" is a real indictment, not an artifact.

⚠️ The stratification caveat cuts the other way on `nonadditive`: the strata are defined by
`sum_of_singles`'s OWN residual, so the bar is low there BY CONSTRUCTION
(`splithalf/CARD.md`). Beating it on `nonadditive` is therefore *easier* than the number
suggests — and nothing does.

## Where the methods DO beat the bar

| metric | panel | best | value | bar | gap |
|---|---|---|---|---|---|
| `overlap_at_N` | additive | **TxPert** | **0.3247** | 0.2865 | **+0.0382** |
| `overlap_at_N` | nonadditive | **TxPert** | **0.3041** | 0.2750 | **+0.0291** |
| `mae` ↓ | additive | **Occam** | **0.0123** | 0.0126 | +0.0003 |
| `mae` ↓ | nonadditive | **Occam** | **0.0146** | 0.0168 | **+0.0022** |
| `disc_l1` | either | TxPert 0.9776 / CellFlow 0.7760 | | 0.9891 / 0.8578 | below on both |

**TxPert ranks DE genes better than additivity does on both panels** (+0.038 / +0.029),
and by a clear margin over Occam (+0.019 / +0.020). Occam is the only arm that beats the
bar on `mae`, on both panels. So the honest summary is that the methods add real value on
ranking and on absolute error while losing on the pseudobulk delta profile.

## Caveats that belong next to these numbers

- **n=1 seed (42), 5 split-halves.** SEMs on `pearson_delta` are 0.0003–0.0034, so the
  split-to-split noise is small, but the seed-to-seed noise is unmeasured.
- **Every competitor carries deliberate deviations**, recorded per arm in
  `docs/competitors_on_doubles.md` and `docs/rhj_competitors_results.md`. None of these
  numbers is a reproduction of a published figure, and none should be quoted as one.
- **State's composition rule is ours**, not its published protocol — it has no genetic
  combinatorial experiment. Read its number as "State's architecture under our sequential
  composition".
- **PerturbDiff is missing** from this table: its training was still running. It has no
  published combinatorial protocol at all, so when it lands its rule will be ours too.
- `pearson_delta` is a rank WITHIN its panel. These numbers are not comparable to
  `rhj_fewshot`'s.
