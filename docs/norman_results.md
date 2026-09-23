# `norman_combo` — results: 6 arms × 12 composition modes, 5 splits, 4 metrics

> **Chronological findings log.** Current numbers: `docs/STATUS.md`. Arms, checkpoints and
> scores that this log mentions but that are not in STATUS were moved to `archive/` on
> 2026-09-23 (`archive/MANIFEST.tsv` gives each one's new path); treat them as history.

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

---

## FULL per-panel tables — every arm, all four metrics, ± SEM over 5 splits

Sorted by `pearson_delta`. `nocal`, seed 42. **`mae` is an ERROR: lower is better.**
`disc_l1` chance is 0.5. The zero-signal floors (`ctrlcell`, `ctrlmean`) are included so
every column has a floor as well as a ceiling — `overlap_at_N` in particular has a
*non-zero* floor (`ctrlmean` 0.1171 / 0.1065), which is why an arm near 0.13 is at chance
on it rather than doing well.

### `additive` panel (36 pairs)

| arm | pearson_delta ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ |
|---|---|---|---|---|
| `experimental` (replicate ceiling) | 0.9165 ± 0.0006 | 0.0110 ± 0.0000 | 0.5247 ± 0.0019 | 0.9978 ± 0.0002 |
| **`sum_of_singles`** ⭐ the bar | **0.8824** ± 0.0006 | 0.0126 ± 0.0000 | 0.2865 ± 0.0017 | **0.9891** ± 0.0013 |
| **TxPert** (native additive sum) | **0.8580** ± 0.0012 | 0.0135 ± 0.0000 | **0.3247** ± 0.0009 | 0.9776 ± 0.0014 |
| **Occam** NQ02 × decode_seq_mean | 0.8169 ± 0.0009 | **0.0123** ± 0.0000 | 0.3059 ± 0.0006 | 0.9091 ± 0.0009 |
| **CellFlow** (native set aggregation) | 0.7603 ± 0.0003 | 0.0140 ± 0.0000 | 0.2633 ± 0.0008 | 0.9288 ± 0.0024 |
| **State** (decode_seq_mean, ours) | 0.6947 ± 0.0034 | 0.0169 ± 0.0001 | 0.2673 ± 0.0018 | 0.8831 ± 0.0023 |
| `meanpert` (one constant delta) | 0.4634 ± 0.0021 | 0.0171 ± 0.0000 | 0.1349 ± 0.0005 | 0.5436 ± 0.0026 |
| `ctrlmean` (zero-signal floor) | 0.0033 ± 0.0086 | 0.0179 ± 0.0000 | 0.1171 ± 0.0007 | 0.5432 ± 0.0012 |
| `ctrlcell` (zero-signal floor) | 0.0004 ± 0.0057 | 0.0195 ± 0.0000 | 0.0000 ± 0.0000 | 0.5426 ± 0.0016 |

### `nonadditive` panel (36 pairs)

| arm | pearson_delta ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ |
|---|---|---|---|---|
| `experimental` (replicate ceiling) | 0.8377 ± 0.0006 | 0.0129 ± 0.0000 | 0.4718 ± 0.0031 | 0.9644 ± 0.0018 |
| **`sum_of_singles`** ⭐ the bar | **0.7757** ± 0.0021 | 0.0168 ± 0.0001 | 0.2750 ± 0.0011 | **0.8578** ± 0.0033 |
| **Occam** NQ02 × decode_seq_mean | **0.7380** ± 0.0018 | **0.0146** ± 0.0000 | 0.2843 ± 0.0008 | 0.7708 ± 0.0011 |
| **TxPert** | 0.7139 ± 0.0027 | 0.0191 ± 0.0001 | **0.3041** ± 0.0010 | 0.7245 ± 0.0022 |
| **CellFlow** | 0.6921 ± 0.0027 | 0.0170 ± 0.0001 | 0.2668 ± 0.0012 | **0.7760** ± 0.0044 |
| **State** | 0.5905 ± 0.0023 | 0.0198 ± 0.0001 | 0.2367 ± 0.0006 | 0.7321 ± 0.0025 |
| `meanpert` | 0.4016 ± 0.0020 | 0.0188 ± 0.0000 | 0.1225 ± 0.0006 | 0.4988 ± 0.0034 |
| `ctrlmean` | 0.0055 ± 0.0035 | 0.0189 ± 0.0000 | 0.1065 ± 0.0006 | 0.5135 ± 0.0017 |
| `ctrlcell` | −0.0045 ± 0.0056 | 0.0209 ± 0.0001 | 0.0000 ± 0.0000 | 0.5173 ± 0.0026 |

### Two things these full tables show that the summary did not

**F-comp4. `mae` splits the field differently from every other metric.** On `additive`,
Occam's 0.0123 is the ONLY value below the bar's 0.0126 — and TxPert, which wins
`pearson_delta` by +0.0411, is WORSE on `mae` (0.0135). On `nonadditive` the same reversal
is larger: Occam 0.0146 against TxPert 0.0191, a 31% higher error for the arm that is only
0.024 behind on `pearson_delta`. `pearson_delta` is scale-free and `mae` is not, so an arm
can get the SHAPE of the delta right while mis-scaling it. TxPert does exactly that.

**F-comp5. `disc_l1` is the one metric where no method is close to the bar**, on either
panel: best additive 0.9776 vs 0.9891, best non-additive 0.7760 vs 0.8578 (a 0.082 gap).
And `experimental` sits at 0.9978 / 0.9644, so the headroom is real rather than a ceiling
effect. Discriminating WHICH pair produced a profile is the hardest of the four tasks here,
and adding the two single deltas still does it better than any trained model.

⚠️ `overlap_at_N` has a **non-zero floor** — `ctrlmean` scores 0.1171 / 0.1065 by
predicting nothing at all. Read 0.30 against that floor and the 0.52 / 0.47 ceiling, not
against 0.

---

## PER-SPLIT values — every arm, every metric, all 5 split-halves

The 5 splits are the SAME seeded halves for every arm (`blake2b(SPLIT_SALT|seed|split|group)`),
so a per-split column is directly comparable down the table. `range` = max − min across the
five, which is the honest read on whether a margin between two arms survives split noise.


### `pearson_delta` ↑ · `additive` panel

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | range |
|---|---|---|---|---|---|---|
| `experimental` | 0.9154 | 0.9173 | 0.9161 | 0.9154 | 0.9182 | 0.0028 |
| **`sum_of_singles`** ⭐ | 0.8821 | 0.8820 | 0.8847 | 0.8814 | 0.8816 | 0.0032 |
| **TxPert** | 0.8556 | 0.8562 | 0.8582 | 0.8624 | 0.8575 | 0.0067 |
| **Occam** NQ02 dsm | 0.8166 | 0.8151 | 0.8168 | 0.8202 | 0.8155 | 0.0051 |
| **CellFlow** | 0.7611 | 0.7607 | 0.7601 | 0.7601 | 0.7596 | 0.0015 |
| **State** | 0.6868 | 0.6957 | 0.6988 | 0.7046 | 0.6877 | 0.0179 |
| `meanpert` | 0.4685 | 0.4570 | 0.4607 | 0.4641 | 0.4665 | 0.0115 |

### `pearson_delta` ↑ · `nonadditive` panel

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | range |
|---|---|---|---|---|---|---|
| `experimental` | 0.8376 | 0.8364 | 0.8372 | 0.8400 | 0.8375 | 0.0036 |
| **`sum_of_singles`** ⭐ | 0.7823 | 0.7735 | 0.7710 | 0.7729 | 0.7787 | 0.0113 |
| **TxPert** | 0.7243 | 0.7083 | 0.7137 | 0.7114 | 0.7119 | 0.0160 |
| **Occam** NQ02 dsm | 0.7453 | 0.7363 | 0.7352 | 0.7362 | 0.7371 | 0.0101 |
| **CellFlow** | 0.6898 | 0.6855 | 0.6890 | 0.6959 | 0.7004 | 0.0149 |
| **State** | 0.5861 | 0.5979 | 0.5924 | 0.5852 | 0.5910 | 0.0127 |
| `meanpert` | 0.3991 | 0.4059 | 0.3972 | 0.4068 | 0.3988 | 0.0096 |

### `mae` ↓ (ERROR) · `additive` panel

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | range |
|---|---|---|---|---|---|---|
| `experimental` | 0.0111 | 0.0110 | 0.0111 | 0.0111 | 0.0110 | 0.0002 |
| **`sum_of_singles`** ⭐ | 0.0127 | 0.0126 | 0.0125 | 0.0126 | 0.0127 | 0.0002 |
| **TxPert** | 0.0135 | 0.0135 | 0.0135 | 0.0134 | 0.0135 | 0.0001 |
| **Occam** NQ02 dsm | 0.0123 | 0.0124 | 0.0122 | 0.0123 | 0.0123 | 0.0001 |
| **CellFlow** | 0.0140 | 0.0140 | 0.0139 | 0.0140 | 0.0140 | 0.0001 |
| **State** | 0.0170 | 0.0168 | 0.0168 | 0.0169 | 0.0171 | 0.0003 |
| `meanpert` | 0.0171 | 0.0171 | 0.0171 | 0.0172 | 0.0171 | 0.0001 |

### `mae` ↓ (ERROR) · `nonadditive` panel

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | range |
|---|---|---|---|---|---|---|
| `experimental` | 0.0129 | 0.0129 | 0.0130 | 0.0127 | 0.0128 | 0.0003 |
| **`sum_of_singles`** ⭐ | 0.0167 | 0.0168 | 0.0170 | 0.0169 | 0.0165 | 0.0005 |
| **TxPert** | 0.0190 | 0.0192 | 0.0191 | 0.0191 | 0.0189 | 0.0003 |
| **Occam** NQ02 dsm | 0.0145 | 0.0147 | 0.0147 | 0.0147 | 0.0145 | 0.0002 |
| **CellFlow** | 0.0170 | 0.0172 | 0.0170 | 0.0171 | 0.0166 | 0.0006 |
| **State** | 0.0199 | 0.0198 | 0.0196 | 0.0198 | 0.0197 | 0.0003 |
| `meanpert` | 0.0189 | 0.0188 | 0.0188 | 0.0188 | 0.0189 | 0.0001 |

### `overlap_at_N` ↑ · `additive` panel

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | range |
|---|---|---|---|---|---|---|
| `experimental` | 0.5243 | 0.5215 | 0.5199 | 0.5270 | 0.5307 | 0.0108 |
| **`sum_of_singles`** ⭐ | 0.2886 | 0.2858 | 0.2918 | 0.2818 | 0.2844 | 0.0100 |
| **TxPert** | 0.3253 | 0.3238 | 0.3217 | 0.3274 | 0.3254 | 0.0057 |
| **Occam** NQ02 dsm | 0.3058 | 0.3068 | 0.3038 | 0.3073 | 0.3060 | 0.0035 |
| **CellFlow** | 0.2620 | 0.2645 | 0.2617 | 0.2658 | 0.2627 | 0.0041 |
| **State** | 0.2657 | 0.2692 | 0.2727 | 0.2669 | 0.2619 | 0.0108 |
| `meanpert` | 0.1348 | 0.1344 | 0.1369 | 0.1346 | 0.1339 | 0.0030 |

### `overlap_at_N` ↑ · `nonadditive` panel

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | range |
|---|---|---|---|---|---|---|
| `experimental` | 0.4723 | 0.4712 | 0.4610 | 0.4797 | 0.4749 | 0.0188 |
| **`sum_of_singles`** ⭐ | 0.2750 | 0.2782 | 0.2739 | 0.2715 | 0.2762 | 0.0066 |
| **TxPert** | 0.3058 | 0.3031 | 0.3005 | 0.3048 | 0.3061 | 0.0056 |
| **Occam** NQ02 dsm | 0.2857 | 0.2841 | 0.2816 | 0.2835 | 0.2864 | 0.0047 |
| **CellFlow** | 0.2630 | 0.2672 | 0.2655 | 0.2687 | 0.2697 | 0.0067 |
| **State** | 0.2358 | 0.2370 | 0.2350 | 0.2369 | 0.2386 | 0.0036 |
| `meanpert` | 0.1217 | 0.1227 | 0.1226 | 0.1209 | 0.1246 | 0.0037 |

### `discrimination_score_l1` ↑ · `additive` panel

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | range |
|---|---|---|---|---|---|---|
| `experimental` | 0.9972 | 0.9980 | 0.9980 | 0.9975 | 0.9982 | 0.0010 |
| **`sum_of_singles`** ⭐ | 0.9865 | 0.9925 | 0.9865 | 0.9920 | 0.9880 | 0.0060 |
| **TxPert** | 0.9755 | 0.9802 | 0.9730 | 0.9792 | 0.9800 | 0.0073 |
| **Occam** NQ02 dsm | 0.9092 | 0.9127 | 0.9084 | 0.9077 | 0.9077 | 0.0050 |
| **CellFlow** | 0.9229 | 0.9337 | 0.9229 | 0.9322 | 0.9324 | 0.0108 |
| **State** | 0.8846 | 0.8891 | 0.8751 | 0.8849 | 0.8816 | 0.0140 |
| `meanpert` | 0.5443 | 0.5503 | 0.5358 | 0.5403 | 0.5475 | 0.0145 |

### `discrimination_score_l1` ↑ · `nonadditive` panel

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | range |
|---|---|---|---|---|---|---|
| `experimental` | 0.9615 | 0.9620 | 0.9617 | 0.9705 | 0.9665 | 0.0090 |
| **`sum_of_singles`** ⭐ | 0.8491 | 0.8509 | 0.8609 | 0.8614 | 0.8666 | 0.0175 |
| **TxPert** | 0.7270 | 0.7195 | 0.7235 | 0.7210 | 0.7315 | 0.0120 |
| **Occam** NQ02 dsm | 0.7728 | 0.7725 | 0.7690 | 0.7675 | 0.7720 | 0.0053 |
| **CellFlow** | 0.7650 | 0.7668 | 0.7773 | 0.7833 | 0.7875 | 0.0225 |
| **State** | 0.7227 | 0.7322 | 0.7367 | 0.7365 | 0.7325 | 0.0140 |
| `meanpert` | 0.4920 | 0.4960 | 0.5108 | 0.4940 | 0.5015 | 0.0188 |

### F-comp6. The TxPert/Occam stratum reversal is 5/5 PAIRED, not split noise

The splits are the same seeded halves for every arm, so the contrast can be taken
split-by-split rather than mean-vs-mean. Occam minus TxPert, `pearson_delta`:

| panel | s0 | s1 | s2 | s3 | s4 | verdict |
|---|---|---|---|---|---|---|
| `additive` | −0.0390 | −0.0411 | −0.0414 | −0.0422 | −0.0420 | **TxPert wins 5/5** |
| `nonadditive` | +0.0209 | +0.0280 | +0.0215 | +0.0248 | +0.0252 | **Occam wins 5/5** |

Every split agrees, in both directions, and the smallest margin (0.0209) is **4x** the
largest per-split range of either arm on that panel. The reversal is therefore a property
of the strata, not of the sampling — which is what the effect-size-controlled cut was built
to make visible.

Occam over CellFlow is 5/5 on both panels too (+0.054…+0.060 additive, +0.037…+0.056
non-additive), so the middle of the field is cleanly ordered; only the top two swap.

⚠️ Still n=1 SEED. Paired-over-splits removes split noise, not seed noise, and
`decode_seq_mean` was selected on this benchmark while TxPert's additive sum is its
published mechanism.

---

## NORMAN SWEEP RESULTS (2026-09-23, overnight) — 27 of 30 arms scored

**The question.** The shipped default (NQ02) was tuned on `rhj_fewshot` and moved over
wholesale. Does a different schedule, batch size, loss scale or lr do better on Norman?
`nocal`, seed 42, 5 splits, `decode_seq_mean` at eval. Sorted by `nonadditive`
`pearson_delta`. **`mae` is an ERROR, lower better.** Bar: `sum_of_singles` 0.8824 / 0.7757.

| arm | config | pd add | pd non | mae add ↓ | mae non ↓ | ovl add | ovl non | dl1 add | dl1 non |
|---|---|---|---|---|---|---|---|---|---|
| **NE25** | **bs 128**, 150 ep, eta 3e-4 | **0.9003** | **0.7940** | **0.0114** | **0.0134** | 0.3080 | 0.2942 | 0.8717 | 0.7489 |
| **NB10** | **bs 256**, eta 1e-3 | **0.8910** | **0.7763** | **0.0112** | 0.0137 | **0.3207** | **0.3037** | 0.8804 | 0.7564 |
| **NB12** | **bs 512**, eta 1e-3 | **0.8877** | 0.7700 | 0.0116 | 0.0139 | 0.3043 | 0.2953 | 0.8579 | 0.7548 |
| NE22 | bs 2048, eta **3e-3** (> lr!) | 0.8595 | 0.7686 | 0.0124 | 0.0141 | 0.2654 | 0.2630 | 0.8013 | 0.7365 |
| NB09 | bs 256, eta 3e-4 | 0.8785 | 0.7660 | 0.0114 | 0.0140 | 0.3092 | 0.2881 | 0.8928 | 0.7425 |
| NE29 | lr **3e-3** (5/5 splits; was 0.8335/0.7384 on 3/5) | 0.8326 | 0.7401 | 0.0129 | 0.0146 | 0.2392 | 0.2274 | 0.7775 | 0.7325 |
| *NQ02* | *PUBLISHED* bs 2048, 150 ep | *0.8169* | *0.7380* | *0.0123* | *0.0146* | *0.3059* | *0.2843* | *0.9091* | *0.7708* |
| NB11 | bs 512, eta 3e-4 | 0.8336 | 0.7304 | 0.0120 | 0.0145 | 0.2877 | 0.2804 | 0.8890 | 0.7441 |
| NC15 | bs 4096, 600 ep | 0.7841 | 0.7289 | 0.0135 | 0.0149 | 0.2411 | 0.2322 | 0.7992 | 0.7055 |
| NE24 | 240 ep | 0.8183 | 0.7222 | 0.0125 | 0.0147 | 0.2813 | 0.2682 | 0.8417 | 0.7493 |
| NE27 | loss scale **4×** | 0.7893 | 0.7044 | 0.0128 | 0.0150 | 0.2664 | 0.2555 | 0.8463 | 0.7202 |
| ND19 | loss scale 2× | 0.7410 | 0.6964 | 0.0134 | 0.0150 | 0.2562 | 0.2554 | 0.8410 | 0.7278 |
| NA06 | 180 ep, eta 1e-4 | 0.7687 | 0.6850 | 0.0133 | 0.0153 | 0.2927 | 0.2642 | 0.8619 | 0.7328 |
| NE20 | eta **1e-5** | 0.7789 | 0.6838 | 0.0143 | 0.0169 | 0.2129 | 0.2027 | 0.8967 | 0.7273 |
| NE21 | eta **1e-6** | 0.7819 | 0.6829 | 0.0127 | 0.0154 | 0.3115 | 0.2813 | 0.9016 | 0.7385 |
| NE23 | **60 ep** | 0.7397 | 0.6688 | 0.0138 | 0.0161 | 0.2655 | 0.2551 | 0.8684 | 0.7332 |
| NC14 | bs 4096, 300 ep | 0.7639 | 0.6675 | 0.0136 | 0.0158 | 0.2482 | 0.2417 | 0.8471 | 0.7023 |
| NC13 | bs 1024, 300 ep | 0.7734 | 0.6645 | 0.0132 | 0.0159 | 0.2127 | 0.2090 | 0.8515 | 0.7200 |
| NA07 | 180 ep, eta 3e-4 | 0.7461 | 0.6625 | 0.0140 | 0.0161 | 0.2820 | 0.2590 | 0.8643 | 0.7235 |
| NA01 | 120 ep, eta 3e-4 | 0.7451 | 0.6611 | 0.0134 | 0.0154 | 0.2890 | 0.2718 | 0.8622 | 0.7361 |
| NA02 · NA05 · NA08 | 120/150/180 ep, eta 1e-3 | 0.7409 | 0.6611 | 0.0141 | 0.0162 | 0.2324 | 0.2222 | 0.8387 | 0.7123 |
| NA03 | 150 ep, eta 1e-4 | 0.7705 | 0.6601 | 0.0142 | 0.0171 | 0.2312 | 0.2016 | 0.9112 | 0.7033 |
| NE26 | loss scale 0.25× | 0.7233 | 0.6480 | 0.0152 | 0.0170 | 0.2426 | 0.2287 | 0.8656 | 0.7245 |
| NA00 | 120 ep, eta 1e-4 | 0.7037 | 0.6265 | 0.0166 | 0.0187 | 0.1653 | 0.1604 | 0.8726 | 0.7246 |
| ND18 | loss scale 0.5× | 0.7554 | 0.6135 | 0.0137 | 0.0180 | 0.2529 | 0.2212 | 0.8761 | 0.7138 |
| NE28 | lr **3e-4** | 0.6581 | 0.5783 | 0.0162 | 0.0186 | 0.2207 | 0.2268 | 0.8583 | 0.6616 |

Pending: **NA04** (the selection-metric control, relaunched), **NC16 / NC17** (bs 4096 /
8192 at 1200 ep; hit the 2 h walltime at epoch ~685, relaunched at 6 h). NE29 splits 0 and
2 are re-running after the pdex shared-memory race.

### F-sweep1. SMALLER BATCH WINS, and bs 128 is the first model to BEAT THE BAR on `pearson_delta`

**NE25 (bs 128) scores 0.9003 / 0.7940 against `sum_of_singles` 0.8824 / 0.7757 — above
the additive bar on BOTH panels.** No arm and no competitor had done that before. It is
also best on `mae` on both panels (0.0114 / 0.0134, bar 0.0126 / 0.0168).

The ordering is monotone in batch size on `pearson_delta` at eta 3e-4 / 150 ep:
bs 128 **0.9003** > bs 256 0.8785 > bs 512 0.8336 > bs 2048 0.8169 (additive). That is the
step-count hypothesis the grid was built to test: at bs 2048 Norman gets **27 optimiser
steps per epoch** (4,050 total), against rhj_fewshot's 298 (44,700). bs 128 gives 66,750.
The small-batch winners also selected LATE epochs (NB10 108, NB12 142, NE25 119 of 150)
where the bs-2048 arms peaked at 14–57 — consistent with "still improving when the
schedule ends".

⚠️ **n=1 seed, and the margin over the bar on `nonadditive` is +0.018.** Before calling this
"beats additivity", it needs seed replicates: State's seed-to-seed SD on this panel is
0.067 and CellFlow's 0.029 (below). A 0.018 margin is inside either.

⚠️ **The large-batch arms did NOT recover by training longer.** bs 4096 at 600 ep (7,800
steps) is 0.7289 non-additive, below bs 512 at 150 ep (16,650 steps, 0.7304) — so step
count is not the whole story; smaller batches win at matched steps too.

### F-sweep2. The loss-weight SCALE is already at its optimum

| scale | 0.25× | 0.5× | **1×** | 2× | 4× |
|---|---|---|---|---|---|
| pd non-additive | 0.6480 | 0.6135 | **0.7380** | 0.6964 | 0.7044 |

The scale-matched point (1×, `norman_loss_scale.py`) beats every multiplier in both
directions. That axis is closed.

### F-sweep3. Two arms are the SAME measurement, and one schedule is inverted

**NA02, NA05 and NA08 are identical to 4 decimals on all 8 numbers.** They are 120 / 150 /
180 epochs at `eta_min` 1e-3 — and `lr` is also 1e-3, so the cosine runs from 1e-3 to 1e-3:
a **constant** learning rate. Their first 120 epochs are therefore the same trajectory, all
three selected epoch 51, and they are one measurement, not three.

**NE22 has `eta_min` 3e-3 > `lr` 1e-3**, so its cosine *rises* over training. It still ranks
4th (0.7686 non-additive), which fits F-sweep1: a higher late learning rate behaves like
more effective optimisation on a step-starved corpus.

⚠️ **`lr` 3e-4 is the worst arm in the sweep** (0.5783). Lower lr hurts here, the opposite
of what a smaller batch would want by the linear-scaling rule — which says the win from
smaller batches is not a learning-rate artifact.

### ⚠️ CORRECTION: all 30 arms were selected on `sum`, not `decode_seq_mean`

`panel_compose` was first wired into the **panel** evaluation path (`predict_panel`), which
is an end-of-training measurement and **selects nothing**. The metric that chooses
`best_model.pth` is `val_pearson_delta`, computed in `evaluate()`, which still called
`forward()` — the `sum` mode. So every arm above ran with `panel_compose: decode_seq_mean`
in its resolved config and was **still selected on `sum`**: the confirmation line never
printed once. Found by grepping the logs for it, not from the numbers.

**Consequence: the 27 arms above are all `sum`-selected, exactly like the published NQ02,
so they are directly comparable to it and to each other.** Only the relaunched NA04 uses
the new selection (`[val] selection metric composes pairs with: decode_seq_mean`,
confirmed in its log), so NA04-vs-NQ02 is now a clean test of the selection metric alone.

---

## COMPETITOR SEED REPLICATES on `norman_combo` — `pearson_delta` per seed

5 splits per seed, averaged within seed (the seed is the unit of replication —
`scripts/seed_split_test.py`).

| arm | panel | s42 | s43 | s44 | s45 | s46 | **mean** | **seed SD** |
|---|---|---|---|---|---|---|---|---|
| TxPert | additive | 0.8580 | 0.8603 | 0.8407 | 0.8502 | *running* | **0.8523** | **0.0089** |
| TxPert | nonadditive | 0.7139 | 0.7138 | 0.7057 | 0.6982 | *running* | **0.7079** | **0.0075** |
| CellFlow | additive | 0.7603 | 0.7661 | 0.7608 | 0.7538 | 0.6941 | **0.7470** | **0.0299** |
| CellFlow | nonadditive | 0.6921 | 0.6871 | 0.6834 | 0.6807 | 0.6229 | **0.6733** | **0.0285** |
| State | additive | 0.6947 | 0.7131 | 0.5227 | 0.4986 | 0.6141 | **0.6086** | **0.0973** |
| State | nonadditive | 0.5905 | 0.6459 | 0.4964 | 0.4835 | 0.5531 | **0.5539** | **0.0672** |

### F-seed1. Seed variance differs 10x between methods, and seed 42 flattered State

- **TxPert is stable** — seed SD 0.009 / 0.008. Its seed-42 number was representative
  (0.8580 vs a 4-seed mean of 0.8523).
- **CellFlow is stable except one seed** — s46 is 0.07 below the other four on both panels.
- **State is unstable** — seed SD **0.097** additive. s44 and s45 score ~0.50, against
  0.69 at s42. **Its seed-42 number overstated it by 0.086** (0.6947 vs a mean of 0.6086).

⚠️ This is exactly why every single-seed number in this document needs its replicate
before a margin is quoted. Occam's own seed variance is still UNMEASURED; the 4 replicate
seeds are written (`scripts/grids/norman_seeds.tsv`) and not launched.

---

## INFRASTRUCTURE, overnight

**Two jobs hung for 7+ h each**, silent at the same line: TxPert s46 and PerturbDiff s42,
both inside pdex's parallel pool (`Processing batches`) during in-training validation, each
holding an H100 idle. Found by reading log mtimes, killed, fixed, relaunched:
- `CELLEVAL_NUM_THREADS=1` in `_competitor_common.sh` makes in-training validation run pdex
  **serially** — no pool, nothing to deadlock. Validation is small (20 pairs), so it costs
  seconds.
- `competitor_watchdog` kills a trainer whose `.out` and `.err` have both been silent for
  45 min, so a future hang frees its GPU instead of holding it. Verified on a synthetic
  stall (killed) and a healthy process (spared) before it was relied on.

**Three sweep arms hit the 2 h walltime** — `rhj_loss_sweep.sbatch` carries
`--time=02:00:00`, sized for rhj's 30-epoch runs. My error, not the arms'. NA04 reached
epoch 113 in 2 h where NA03 and NA05 at the same config finished in ~30 min — a slow node.

**PerturbDiff's control-count fix held**: its first validation published a score
(`pearson_delta` 0.514) at 18:17 before the hang.

---

## FULL COMPARISON on `norman_combo` — Occam arms, competitors, baselines

**Competitors are MEANS OVER SEEDS (TxPert n=4, CellFlow n=5, State n=5), each seed the
mean of its 5 splits. Occam arms and baselines are ONE SEED (42), mean of 5 splits** —
so the ± on a competitor row is seed SD and the Occam rows carry no seed error yet.
`nocal`, `decode_seq_mean` for every Occam row. **`mae` is an ERROR, lower better.**

### `additive` panel (36 pairs)

| arm | seeds | pearson_delta ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ |
|---|---|---|---|---|---|
| `experimental` (replicate ceiling) | 1 | 0.9165 | 0.0110 | 0.5247 | 0.9978 |
| **Occam NE25** (bs 128) | 1 | 0.9003 | 0.0114 | 0.3080 | 0.8717 |
| Occam NB10 (bs 256, eta 1e-3) | 1 | 0.8910 | 0.0112 | 0.3207 | 0.8804 |
| Occam NB12 (bs 512, eta 1e-3) | 1 | 0.8877 | 0.0116 | 0.3043 | 0.8579 |
| **`sum_of_singles`** ⭐ the bar | 1 | 0.8824 | 0.0126 | 0.2865 | 0.9891 |
| **TxPert** | 4 | 0.8523 ± 0.0089 | 0.0136 ± 0.0001 | 0.3238 ± 0.0024 | 0.9743 ± 0.0035 |
| Occam NQ02 (published, bs 2048) | 1 | 0.8169 | 0.0123 | 0.3059 | 0.9091 |
| **CellFlow** | 5 | 0.7470 ± 0.0299 | 0.0142 ± 0.0004 | 0.2473 ± 0.0215 | 0.9164 ± 0.0238 |
| **State** | 5 | 0.6086 ± 0.0973 | 0.0174 ± 0.0020 | 0.2927 ± 0.0321 | 0.8762 ± 0.0465 |
| `meanpert` | 1 | 0.4634 | 0.0171 | 0.1349 | 0.5436 |

### `nonadditive` panel (36 pairs)

| arm | seeds | pearson_delta ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ |
|---|---|---|---|---|---|
| `experimental` (replicate ceiling) | 1 | 0.8377 | 0.0129 | 0.4718 | 0.9644 |
| **Occam NE25** (bs 128) | 1 | 0.7940 | 0.0134 | 0.2942 | 0.7489 |
| Occam NB10 (bs 256, eta 1e-3) | 1 | 0.7763 | 0.0137 | 0.3037 | 0.7564 |
| **`sum_of_singles`** ⭐ the bar | 1 | 0.7757 | 0.0168 | 0.2750 | 0.8578 |
| Occam NB12 (bs 512, eta 1e-3) | 1 | 0.7700 | 0.0139 | 0.2953 | 0.7548 |
| Occam NQ02 (published, bs 2048) | 1 | 0.7380 | 0.0146 | 0.2843 | 0.7708 |
| **TxPert** | 4 | 0.7079 ± 0.0075 | 0.0191 ± 0.0001 | 0.3008 ± 0.0022 | 0.7165 ± 0.0093 |
| **CellFlow** | 5 | 0.6733 ± 0.0285 | 0.0170 ± 0.0001 | 0.2453 ± 0.0203 | 0.7663 ± 0.0119 |
| **State** | 5 | 0.5539 ± 0.0672 | 0.0202 ± 0.0021 | 0.2680 ± 0.0322 | 0.7037 ± 0.0403 |
| `meanpert` | 1 | 0.4016 | 0.0188 | 0.1225 | 0.4988 |

### F-sweep4. Small batch buys accuracy and PAYS in discrimination

| arm | pd non-add ↑ | mae non-add ↓ | disc_l1 additive ↑ | disc_l1 non-add ↑ |
|---|---|---|---|---|
| Occam NE25 (bs 128) | **0.7940** | **0.0134** | 0.8717 | 0.7489 |
| Occam NQ02 (bs 2048) | 0.7380 | 0.0146 | **0.9091** | **0.7708** |
| TxPert (4 seeds) | 0.7079 | 0.0191 | **0.9743** | 0.7165 |
| `sum_of_singles` (bar) | 0.7757 | 0.0168 | **0.9891** | **0.8578** |

Going from bs 2048 to bs 128 gains **+0.056** `pearson_delta` and cuts `mae` 8%, but LOSES
**0.037** `disc_l1` on the additive panel and 0.022 on the non-additive one. So NE25 beats
the bar on the two metrics that read the pseudobulk profile and sits **0.117 below it** on
the one that asks *which pair produced this profile*. "Beats additivity" is therefore true
for `pearson_delta` and `mae` and false for `disc_l1` — say which.

⚠️ This matters for WHICH arm gets seed replicates: NE25 is the `pearson_delta` winner, not
an across-the-board winner. `overlap_at_N` still goes to TxPert (0.3238 vs 0.3080).

Paper-grade per-seed × per-split × all-27-metric tables for the competitors:
**`docs/norman_competitors_seeds.md`** (regenerated by
`scripts/norman/write_competitor_seed_report.py`); raw values with their source
`results.csv` paths in `data/release/metrics/norman_competitors_by_seed_split.csv`.


## ⏳ PerturbDiff on Norman — seed replicates LAUNCHED 2026-09-23

s42 is still training (18364988, ~0.40 s/step, export 18364989 → score 18364990 chained).
Seeds 43–46 launched in parallel via `SEED=4x ARMS=perturbdiff bash scripts/repro/competitors_norman.sh`:
train jobs 18365070 / 18365073 / 18365076 / 18365079, each with its export + score chained
(`afterok`). Same benchmark, 8,000-gene axis, `decode_seq_mean` composition at export.
When all five land, regenerate `docs/norman_competitors_seeds.md` with
`scripts/norman/write_competitor_seed_report.py`.


## DECISION 2026-09-23 — NB10 (bs 256, eta_min 1e-3) is the Norman DEFAULT

User decision after the F-sweep1 table. `configs/norman_combo.yaml` now has batch_size 256,
eta_min 1e-3 (= lr, no decay). NB10: pΔ 0.8910 / 0.7763, mae 0.0112 / 0.0137,
ovl 0.3207 / 0.3037, dl1 0.8804 / 0.7564 (add / non). Published HF checkpoint stays NQ02;
repro script pins its settings. Tests: `tests/test_default_recipe.py`. Details and the
eta_min coverage at bs 256 (only 3e-4 and 1e-3 tried) in `docs/default_recipe.md`.

## ⏳ Sweep 4 — NB10 with a smaller eta_min (LAUNCHED 2026-09-23, job 18365088)

**Question.** At bs 256 only eta_min 3e-4 (NB09 0.8785 / 0.7660) and 1e-3 (NB10 0.8910 / 0.7763,
no decay) were tried. Does a lower floor help at small batch? Grid `scripts/grids/norman_sweep4.tsv`.

| arm | eta_min | role |
|---|---|---|
| NF33 | 1e-3 | CONTROL: NB10 re-run under decode_seq_mean selection |
| NF34 | 1e-4 | |
| NF35 | 1e-5 | |
| NF36 | 1e-6 | |
| NF37 | 7e-4 | added 2026-09-23, job 18365094 |
| NF38 | 5e-4 | added 2026-09-23, job 18365094 |
| NF39 | 2e-4 | added 2026-09-23, job 18365094 (3e-4 = NB09, old selection) |

Fixed: Q02 Norman weights, 150 ep, bs 256, lr 1e-3, seed 42. Compare NF34–36 to **NF33**, not
NB10 (NB10 selected on the old `sum` val metric). Auto-eval daemon 18364999 covers `NF[0-9]+`.

### What the bs-2048 sweep already says about eta_min and epochs (answer, 2026-09-23)

At bs 2048 **no epoch or eta_min change beat NQ02** (150 ep, eta 3e-4, pd 0.8169 / 0.7380) on
nonadditive pd, except NE22 (eta 3e-3 > lr, pd 0.8595 / 0.7686, but dl1 0.8013 / 0.7365 vs
0.9091 / 0.7708). Epochs 60 / 120 / 180 / 240 → non pd 0.6688 / 0.6611 / 0.6625 / 0.7222.
eta 1e-6 / 1e-5 / 1e-4 / 1e-3 → 0.6829 / 0.6838 / 0.6601 / 0.6611. Batch size is the only
lever that won. ⚠️ 150→180 ep at the same floor moved non pd by −0.076 (NQ02 vs NA07), n=1:
that is larger than most between-arm gaps, so the NA04 retrain control is what sizes this noise.


### Update 2026-09-23 — NE29 complete (5/5 splits)
NE29 (lr 3e-3, bs 2048, 150 ep, eta 3e-4) after the split 0/2 re-eval (job 18364991):
pΔ 0.8326 / 0.7401, mae 0.0129 / 0.0146 ↓, ovl 0.2392 / 0.2274, dl1 0.7775 / 0.7325 (add / non).
The 3/5 reading (0.8335 / 0.7384) moved by < 0.002; its place in the sweep table is unchanged.

### F-sweep5 (2026-09-23). Validation composed with `decode_seq_mean` picks the SAME checkpoint; bs 64 is the new best

**Question.** NQ02 and NE25 selected their checkpoint on a `sum`-composed validation metric
while being reported on `decode_seq_mean`. Does aligning the selection metric pick a better
epoch? And does the batch-size trend continue below 128?

Seed 42, 5 splits, nocal, `decode_seq_mean`. `mae` ↓. Bar: `sum_of_singles` pΔ 0.8824 / 0.7757.

| arm | what | best epoch | pΔ add / non | mae add / non ↓ | ovl add / non | dl1 add / non |
|---|---|---|---|---|---|---|
| NQ02 | bs 2048, `sum` selection (published) | — | 0.8169 / 0.7380 | 0.0123 / 0.0146 | 0.3059 / 0.2843 | 0.9091 / 0.7708 |
| NA04 | = NQ02, `decode_seq_mean` selection | 44 | 0.8169 / 0.7380 | 0.0123 / 0.0146 | 0.3059 / 0.2843 | 0.9091 / 0.7708 |
| NE25 | bs 128, `sum` selection | 119 | 0.9003 / 0.7940 | 0.0114 / 0.0134 | 0.3080 / 0.2942 | 0.8717 / 0.7489 |
| NF30 | = NE25, `decode_seq_mean` selection | 119 | 0.9003 / 0.7940 | 0.0114 / 0.0134 | 0.3080 / 0.2942 | 0.8717 / 0.7489 |
| NB10 | bs 256, eta 1e-3 (current default) | 108 | 0.8910 / 0.7763 | 0.0112 / 0.0137 | 0.3207 / 0.3037 | 0.8804 / 0.7564 |
| **NF31** | **bs 64**, eta 3e-4, `decode_seq_mean` selection | 144 | **0.9096 / 0.7982** | **0.0112 / 0.0133** | 0.3153 / 0.3000 | 0.8878 / 0.7644 |

**Reading.** (1) The selection change is a no-op on both checked arms: NA04 and NF30 reproduce
NQ02 and NE25 to 4 decimals on all 8 numbers, with the same selected epoch — training is
deterministic and both metrics peak at the same epoch. The NQ02 / NE25 numbers stand.
(2) bs 64 (NF31) beats bs 128 on all 8 numbers and clears the additivity bar on both panels
(+0.027 / +0.023 pΔ). It selected epoch 144 of 150 — still improving at the end of the schedule.
n = 1 seed. NF32 (bs 32) still training.

⚠️ **NF32 (bs 32) HUNG and was restarted** (2026-09-23): job 18364996_2 went silent at epoch 92
step 684 at 04:06 (data-loader workers idle, no output for 1.5 h). Cancelled; run dir moved to
`runs/_hung_NF32_2026-09-23__*`; relaunched from scratch on vco as job 18365474 (no resume support).


## ⏳ Sweep 5 — NF31 (bs 64) with a LOWER eta_min (launched 2026-09-23, job 18365491)

NF31 (bs 64, eta 3e-4) is the best arm (0.9096 / 0.7982) and eta_min was never varied at bs 64.
Grid `scripts/grids/norman_sweep5.tsv`, rows identical to NF31 except eta_min, decode_seq_mean
selection. Read against NF31 and the bs-256 ladder (sweep 4) BEFORE seed replicates of bs 64
(user, 2026-09-23: no bs-64 seeds yet).

| arm | eta_min |
|---|---|
| NF31 | 3e-4 (reference, done) |
| NF40 | 2e-4 |
| NF41 | 1e-4 |
| NF42 | 1e-5 |
| NF43 | 1e-6 |

Baselines at control-draw seeds 43–46 launched the same day: jobs 18365487–90
(`runs/norman_combo/reference_s4N`).

### F-sweep6 (2026-09-23). bs 256 eta_min ladder (sweep 4) — no floor beats constant LR by more than noise; NC16/NC17 long-large-batch arms lose

Seed 42, 5 splits, nocal, `decode_seq_mean`, selection on `decode_seq_mean`. `mae` ↓. Bar:
`sum_of_singles` pΔ 0.8824 / 0.7757. All arms bs 256, lr 1e-3, 150 ep unless stated.

| arm | eta_min | pΔ add / non | mae add / non ↓ | ovl add / non | dl1 add / non |
|---|---|---|---|---|---|
| **NF33** | 1e-3 (= NB10 re-run, new selection) | **0.8957 / 0.7862** | 0.0115 / 0.0136 | 0.3196 / 0.3004 | 0.8588 / 0.7551 |
| NF37 | 7e-4 | 0.8915 / 0.7788 | 0.0115 / 0.0135 | 0.3048 / 0.2946 | 0.8686 / 0.7498 |
| NF38 | 5e-4 | 0.8914 / 0.7803 | 0.0115 / 0.0136 | 0.3098 / 0.2965 | 0.8559 / 0.7318 |
| NB09 | 3e-4 (old selection) | 0.8785 / 0.7660 | 0.0114 / 0.0140 | 0.3092 / 0.2881 | 0.8928 / 0.7425 |
| NF39 | 2e-4 | 0.8615 / 0.7455 | 0.0118 / 0.0140 | 0.2835 / 0.2779 | 0.8688 / 0.7391 |
| NF34 | 1e-4 | 0.8778 / 0.7647 | 0.0116 / 0.0139 | 0.3065 / 0.2919 | 0.8659 / 0.7342 |
| NF35 | 1e-5 | 0.8821 / 0.7750 | 0.0113 / 0.0140 | 0.3108 / 0.2951 | **0.9073 / 0.7778** |
| NF36 | 1e-6 | 0.8920 / 0.7773 | 0.0116 / 0.0136 | 0.3093 / 0.2942 | 0.8580 / 0.7431 |
| *NB10* | *1e-3 (default, old selection)* | *0.8910 / 0.7763* | *0.0112 / 0.0137* | *0.3207 / 0.3037* | *0.8804 / 0.7564* |
| NC16 | bs 4096, 1200 ep | 0.8066 / 0.7043 | 0.0129 / 0.0146 | 0.2450 / 0.2549 | 0.7606 / 0.6930 |
| NC17 | bs 8192, 1200 ep | 0.7595 / 0.6584 | 0.0139 / 0.0161 | 0.2270 / 0.2072 | 0.8208 / 0.6859 |
| ref: NF31 | bs 64, eta 3e-4 | 0.9096 / 0.7982 | 0.0112 / 0.0133 | 0.3153 / 0.3000 | 0.8878 / 0.7644 |

**Reading.** (1) At bs 256 the pΔ ladder is non-monotone: 1e-3 (NF33) is best on pΔ, 2e-4 worst
(0.8615 / 0.7455), and 1e-6 recovers to 0.8920 / 0.7773 — a 0.035 swing between neighbours on
n = 1 seed, so the ordering inside 0.86–0.90 is not resolved. (2) NF35 (1e-5) has the best
`disc_l1` of any Norman arm (0.9073 / 0.7778, above NQ02's 0.9091 / 0.7708 on nonadditive) at
a pΔ 0.014 below NF33. (3) NF33 vs NB10 (same config, selection changed): +0.0047 / +0.0099 pΔ,
−0.022 dl1 add — the new selection metric picked a different epoch here, unlike NA04/NF30.
(4) Very long, very-large-batch training (NC16/NC17) does not recover what small batches give:
0.8066 / 0.7043 at bs 4096 × 1200 ep. (5) bs 64 (NF31) is still best on pΔ and mae.

⚠️ NF32 (bs 32) relaunch and NF40–NF43 (bs 64 eta ladder) FAILED at start (4 s) on
`ModuleNotFoundError: yaml` — submitted without the Occam python env (my error). Relaunched
2026-09-23: NF32 job 18368255, NF40–43 job 18368256.

### Competitor chain failures and relaunches (2026-09-23)

| chain | failure | fix | relaunch |
|---|---|---|---|
| RHJ TxPert s44 | pdex `mp.Pool(1)` fork deadlock in in-training validation (started before the in-process fix) | in-process pool (already in code) | 18368276 → 77 → 78 |
| Norman PerturbDiff s42 export | checkpoint lookup matched 3 seed-42 markers (norman + 2 rhj encodings) and refused | resolver now filters by benchmark tag (`__norman_combo`) | export 18368285 → score 18368286 |
| Norman PerturbDiff s44, s45 | DataLoader `could not unlink the shared memory file /torch_*`, then hang; watchdog kill after ~2.5 h | `TORCH_SHARING_STRATEGY=file_system` (infra only) | s44 18368279 → 80 → 81 · s45 18368282 → 83 → 84 |

Partial checkpoints of the killed chains were moved to `archive/` (MANIFEST.tsv). Norman
PerturbDiff s43 / s46 were already training before the shm fix and are left running.
