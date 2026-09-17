# `rhj_fewshot` adjacency-map sweep — validation vs cell-eval

**The question:** does the validation metric used for checkpoint selection predict the benchmark result? One GNN arm showed val 0.3368 against cell-eval 0.0020 (`docs/rhj_splithalf.md` F11), so they are not interchangeable.

Comparison metrics (decision 2026-09-17): `pearson_delta`, `mae` ↓, `overlap_at_N`, `discrimination_score_l1` *(chance 0.5)*. Both calibration settings are shown because they win different metrics of the four (F12).

`val` is group-mean `pearson_delta` on the 103 HepG2 **validation** perturbations — the checkpoint-selection metric, read at the epoch that scored best. The cell-eval columns are the 611 held-out **test** perturbations under the split-half protocol.

94 of 314 completed arms scored so far.

## Baselines, all 5 splits

| arm | `pearson_delta` | `mae` ↓ | `overlap_at_N` | `disc_l1` |
|---|---|---|---|---|
| `ctrlcell` | -0.0018 | 0.0296 | 0.0000 | 0.5038 |
| `ctrlmean` | 0.0016 | 0.0231 | 0.0081 | 0.5014 |
| `meanpert` | 0.1324 | 0.0290 | 0.0108 | 0.5037 |
| `transfer` | 0.2323 | 0.0299 | 0.0351 | 0.6404 |
| `experimental` | 0.3722 | 0.0257 | 0.1942 | 0.7821 |

## ⭐ F-adj2 — CONFIRMED: the validation ARGMAX is the cell-eval argmax. But the top-10 RANK is not.

**Decision-relevant finding, measured 2026-09-17 on 95 uncalibrated arms (up from 64).**

The correlation holds, and the thing the selection policy actually needs holds strongly:

| | value |
|---|---|
| Spearman(val, cell-eval `pearson_delta`), full scored set | **+0.941** (n=95) |
| best VAL arm within the scored set | `s129`, val 0.3676 |
| best CELL-EVAL arm within the scored set | `s129`, 0.3348 |
| **same arm** | **YES** |

⚠️ **BUT THE CORRELATION COLLAPSES INSIDE THE TOP OF THE RANGE.** This is the restricted-range
effect and it is the caveat that bounds what the policy can be used for:

| subset | val range | rho(`pearson_delta`) |
|---|---|---|
| all 95 scored | 0.2682 – 0.3676 | **+0.941** |
| top 47 by val | 0.3068 – 0.3676 | +0.805 |
| top 30 by val | 0.3140 – 0.3676 | +0.700 |
| top 20 by val | 0.3159 – 0.3676 | +0.729 |
| **top 10 by val** | 0.3214 – 0.3676 | **+0.491** |

**Reading: validation reliably separates a good arm from a mediocre one, and reliably picks the
single best arm, but it CANNOT rank ten near-tied arms.** Those are different claims and only
the first two support an argmax-only policy.

⚠️ **VALIDATION PREDICTS `pearson_delta` AND ESSENTIALLY NOTHING ELSE.** Over the same 95 arms:

| cell-eval metric | rho vs val | usable as a proxy? |
|---|---|---|
| `pearson_delta` | **+0.941** | yes |
| `mae` ↓ | **−0.704** (negative is the good sign) | yes |
| `overlap_at_N` | **+0.444** | weak |
| `discrimination_score_l1` | **+0.198** | **no** |

So a val-only selection policy optimises `pearson_delta` and `mae` and is **blind to
`disc_l1`** — one of the four comparison metrics, and the one calibration wins. An arm chosen
this way is the best arm *on two of four metrics*, which is what the record must say.

**The val → cell-eval gap is systematically negative and tight:** mean **−0.0140**, range
**−0.0327 … −0.0019**, spread 0.0309. Validation overestimates by ~1–3 points of
`pearson_delta` every time. Useful: a val score of `v` predicts a cell-eval score of roughly
`v − 0.014`, and never above `v`.

### What this licenses, and what it does not

✅ **Licensed: evaluate only the arm that sets a new validation record.** The argmax test passed
(1 for 1), the full-range rho is +0.941, and the gap is one-sided.

❌ **Not licensed: reading the val ranking as a cell-eval ranking below the argmax.** Top-10 rho
is +0.491, so "2nd best on val" says little about 2nd best on cell-eval.

❌ **Not licensed: claiming the selected arm is best on `disc_l1` or `overlap_at_N`.** rho +0.198
and +0.444. If those matter for a decision, that arm must be scored directly.

## ⚠️ F-adj1 — THE VALIDATION LEADERS HAVE NOT BEEN SCORED YET

**Read the table below knowing this: the 94 arms with a cell-eval number are NOT the 94 best
arms on validation.** Eval was launched in grid order (`s000` upward) while the sweep was still
producing arms, so the scored set is biased toward LOW arm indices, and the sweep's best
validation arms live at HIGH indices.

Status 2026-09-17: **486 declared · 314 completed · 12 in flight · 0 failed.**
94 arms cell-eval'd uncalibrated, 82 calibrated.

| | arm | val | cell-eval `pearson_delta` |
|---|---|---|---|
| best VALIDATION arm | `s313` | **0.3769** | **not scored** |
| 2nd | `s145` | 0.3731 | not scored |
| 3rd | `s309` | 0.3705 | not scored |
| best SCORED arm | `s129` | 0.3676 | **0.3348** (5 splits) |

Validation spread over the 314 completed arms: **best 0.3769 · median 0.3170 · worst 0.2652.**

⭐ **Every one of the top 8 validation arms uses `decoder_lr = 1e-3`, and 7 of 8 use
`cosine_loss_weight` 0.5.** `s129`, the best SCORED arm, also has `decoder_lr = 1e-3`. The
decoder learning rate looks like the dominant axis in this sweep — but note the eval bias
above: that pattern is currently measured on validation, and only ONE `dlr 1e-3` arm has a
5-split cell-eval number.

| rank | arm | val | lr | decoder_lr | cos | film_mode |
|---|---|---|---|---|---|---|
| 1 | `s313` | 0.3769 | 1e-3 | **1e-3** | 0.5 | multiplicative_only |
| 2 | `s145` | 0.3731 | 3e-4 | **1e-3** | 0.5 | full |
| 3 | `s309` | 0.3705 | 1e-3 | **1e-3** | 0.5 | full |
| 4 | `s147` | 0.3704 | 3e-4 | **1e-3** | 0.5 | full |
| 5 | `s306` | 0.3699 | 1e-3 | **1e-3** | 0.5 | full |
| 6 | `s154` | 0.3679 | 3e-4 | **1e-3** | 0.5 | additive_only |
| 7 | `s129` | 0.3676 | 3e-4 | **1e-3** | 0.25 | full |
| 8 | `s136` | 0.3673 | 3e-4 | **1e-3** | 0.25 | additive_only |

**What this does and does not establish.** Validation predicts the uncalibrated cell-eval rank
well (Spearman **+0.938**, n=64→94), so the high-val arms are *likely* to score high. It is not
measured, and `s129`'s val 0.3676 → cell-eval 0.3348 is a **−0.033** drop, so the mapping is
monotone but not an identity. A decision between `s313` and `s129` needs `s313` scored.

⚠️ Calibration remains the wrong choice on `pearson_delta` and `mae`: the best calibrated arm
is 0.2925 with `mae` **0.0749** against the uncalibrated 0.3348 / **0.0210** — `mae` is
3.6x worse, and it is one of the four comparison metrics. It does buy `disc_l1`
(0.7410 vs 0.6479). Same split as F12.

## Rank correlation: validation vs cell-eval

Spearman rho over the arms scored so far. **Positive and high** = validation is a usable proxy. **Near zero** = validation is uninformative and arms must be chosen on cell-eval. **Negative** = validation is actively harmful.

| cell-eval metric | arm | Spearman rho vs val | n |
|---|---|---|---|
| `pearson_delta` | `model_nocal` | **+0.940** | 94 |
| `mae` | `model_nocal` | **-0.704** *(lower is better, so a NEGATIVE rho is the good sign)* | 94 |
| `overlap_at_N` | `model_nocal` | **+0.440** | 94 |
| `discrimination_score_l1` | `model_nocal` | **+0.206** | 94 |
| `pearson_delta` | `model_cal` | **+0.277** | 72 |
| `mae` | `model_cal` | **+0.134** *(lower is better, so a NEGATIVE rho is the good sign)* | 72 |
| `overlap_at_N` | `model_cal` | **+0.205** | 72 |
| `discrimination_score_l1` | `model_cal` | **-0.339** | 72 |

## Arms by cell-eval `pearson_delta` — `model_nocal`

| arm | val | `pearson_delta` | `mae` ↓ | `overlap_at_N` | `disc_l1` | splits | lr | dlr | cos | dpo | film_mode | fh | fmf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `s129` | 0.3676 | 0.3348 | 0.0210 | 0.0618 | 0.6479 | 5 | 3e-4 | 1e-3 | 0.25 | 0 | full | 256 | 1 |
| `s018` | 0.3367 | 0.3084 | 0.0215 | 0.0571 | 0.6384 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | full | 0 | 0 |
| `s084` | 0.3214 | 0.3062 | 0.0218 | 0.0546 | 0.6302 | 1 | 3e-4 | 3e-4 | 0.25 | 1 | full | 0 | 0 |
| `s031` | 0.3234 | 0.3054 | 0.0215 | 0.0548 | 0.6256 | 1 | 3e-4 | 1e-4 | 0.25 | 1 | full | 256 | 0 |
| `s028` | 0.3287 | 0.3053 | 0.0217 | 0.0565 | 0.6340 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | additive_only | 256 | 0 |
| `s019` | 0.3294 | 0.3048 | 0.0216 | 0.0566 | 0.6323 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | full | 0 | 1 |
| `s085` | 0.3223 | 0.3048 | 0.0215 | 0.0576 | 0.6316 | 1 | 3e-4 | 3e-4 | 0.25 | 1 | full | 256 | 0 |
| `s073` | 0.3267 | 0.3040 | 0.0219 | 0.0553 | 0.6408 | 1 | 3e-4 | 3e-4 | 0.25 | 0 | full | 0 | 1 |
| `s022` | 0.3228 | 0.3037 | 0.0216 | 0.0555 | 0.6263 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | multiplicative_only | 0 | 0 |
| `s099` | 0.3169 | 0.3036 | 0.0220 | 0.0584 | 0.6457 | 1 | 3e-4 | 3e-4 | 0.5 | 0 | additive_only | 0 | 1 |
| `s040` | 0.3142 | 0.3016 | 0.0216 | 0.0554 | 0.6263 | 1 | 3e-4 | 1e-4 | 0.5 | 0 | multiplicative_only | 0 | 0 |
| `s023` | 0.3209 | 0.3014 | 0.0215 | 0.0566 | 0.6285 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | multiplicative_only | 0 | 1 |
| `s027` | 0.3172 | 0.3008 | 0.0222 | 0.0538 | 0.6380 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | additive_only | 0 | 1 |
| `s049` | 0.3140 | 0.3004 | 0.0216 | 0.0525 | 0.6236 | 1 | 3e-4 | 1e-4 | 0.5 | 1 | full | 256 | 0 |
| `s096` | 0.3224 | 0.3003 | 0.0217 | 0.0522 | 0.6322 | 1 | 3e-4 | 3e-4 | 0.5 | 0 | multiplicative_only | 256 | 0 |
| `s021` | 0.3170 | 0.3000 | 0.0218 | 0.0549 | 0.6318 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | full | 256 | 1 |
| `s076` | 0.3159 | 0.2998 | 0.0218 | 0.0579 | 0.6360 | 1 | 3e-4 | 3e-4 | 0.25 | 0 | multiplicative_only | 0 | 0 |
| `s006` | 0.3184 | 0.2996 | 0.0217 | 0.0514 | 0.6219 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | multiplicative_only | 256 | 0 |
| `s055` | 0.3153 | 0.2996 | 0.0220 | 0.0573 | 0.6386 | 1 | 3e-4 | 3e-4 | 0.0 | 0 | full | 0 | 1 |
| `s035` | 0.3159 | 0.2994 | 0.0216 | 0.0550 | 0.6245 | 1 | 3e-4 | 1e-4 | 0.25 | 1 | additive_only | 256 | 0 |
| `s092` | 0.3193 | 0.2991 | 0.0215 | 0.0606 | 0.6356 | 1 | 3e-4 | 3e-4 | 0.5 | 0 | full | 256 | 0 |
| `s047` | 0.3149 | 0.2991 | 0.0217 | 0.0568 | 0.6309 | 1 | 3e-4 | 1e-4 | 0.5 | 0 | additive_only | 256 | 1 |
| `s038` | 0.3047 | 0.2989 | 0.0219 | 0.0568 | 0.6304 | 1 | 3e-4 | 1e-4 | 0.5 | 0 | full | 256 | 0 |
| `s074` | 0.3204 | 0.2986 | 0.0221 | 0.0578 | 0.6475 | 1 | 3e-4 | 3e-4 | 0.25 | 0 | full | 256 | 0 |
| `s056` | 0.3136 | 0.2982 | 0.0220 | 0.0565 | 0.6362 | 1 | 3e-4 | 3e-4 | 0.0 | 0 | full | 256 | 0 |
| `s024` | 0.3209 | 0.2981 | 0.0217 | 0.0549 | 0.6256 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | multiplicative_only | 256 | 0 |
| `s090` | 0.3131 | 0.2978 | 0.0218 | 0.0591 | 0.6396 | 1 | 3e-4 | 3e-4 | 0.5 | 0 | full | 0 | 0 |
| `s033` | 0.3113 | 0.2975 | 0.0218 | 0.0530 | 0.6158 | 1 | 3e-4 | 1e-4 | 0.25 | 1 | multiplicative_only | 256 | 0 |
| `s011` | 0.3125 | 0.2970 | 0.0221 | 0.0552 | 0.6338 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | additive_only | 256 | 1 |
| `s012` | 0.3121 | 0.2969 | 0.0219 | 0.0523 | 0.6222 | 1 | 3e-4 | 1e-4 | 0.0 | 1 | full | 0 | 0 |
| `s041` | 0.3152 | 0.2967 | 0.0216 | 0.0555 | 0.6304 | 1 | 3e-4 | 1e-4 | 0.5 | 0 | multiplicative_only | 0 | 1 |
| `s039` | 0.3142 | 0.2963 | 0.0219 | 0.0550 | 0.6291 | 1 | 3e-4 | 1e-4 | 0.5 | 0 | full | 256 | 1 |
| `s034` | 0.3078 | 0.2962 | 0.0216 | 0.0571 | 0.6256 | 1 | 3e-4 | 1e-4 | 0.25 | 1 | additive_only | 0 | 0 |
| `s005` | 0.3147 | 0.2960 | 0.0218 | 0.0536 | 0.6243 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | multiplicative_only | 0 | 1 |
| `s081` | 0.3087 | 0.2956 | 0.0225 | 0.0458 | 0.6359 | 1 | 3e-4 | 3e-4 | 0.25 | 0 | additive_only | 0 | 1 |
| `s079` | 0.3148 | 0.2952 | 0.0218 | 0.0545 | 0.6293 | 1 | 3e-4 | 3e-4 | 0.25 | 0 | multiplicative_only | 256 | 1 |
| `s052` | 0.3061 | 0.2952 | 0.0217 | 0.0533 | 0.6227 | 1 | 3e-4 | 1e-4 | 0.5 | 1 | additive_only | 0 | 0 |
| `s088` | 0.3072 | 0.2951 | 0.0217 | 0.0545 | 0.6283 | 1 | 3e-4 | 3e-4 | 0.25 | 1 | additive_only | 0 | 0 |
| `s048` | 0.3040 | 0.2951 | 0.0217 | 0.0551 | 0.6241 | 1 | 3e-4 | 1e-4 | 0.5 | 1 | full | 0 | 0 |
| `s009` | 0.3136 | 0.2950 | 0.0219 | 0.0550 | 0.6302 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | additive_only | 0 | 1 |
| `s004` | 0.3146 | 0.2945 | 0.0220 | 0.0519 | 0.6241 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | multiplicative_only | 0 | 0 |
| `s050` | 0.3030 | 0.2943 | 0.0218 | 0.0546 | 0.6185 | 1 | 3e-4 | 1e-4 | 0.5 | 1 | multiplicative_only | 0 | 0 |
| `s098` | 0.3102 | 0.2941 | 0.0221 | 0.0513 | 0.6404 | 1 | 3e-4 | 3e-4 | 0.5 | 0 | additive_only | 0 | 0 |
| `s063` | 0.3081 | 0.2941 | 0.0220 | 0.0582 | 0.6373 | 1 | 3e-4 | 3e-4 | 0.0 | 0 | additive_only | 0 | 1 |
| `s091` | 0.3072 | 0.2935 | 0.0225 | 0.0508 | 0.6389 | 1 | 3e-4 | 3e-4 | 0.5 | 0 | full | 0 | 1 |
| `s072` | 0.3108 | 0.2933 | 0.0226 | 0.0554 | 0.6436 | 1 | 3e-4 | 3e-4 | 0.25 | 0 | full | 0 | 0 |
| `s007` | 0.3089 | 0.2933 | 0.0219 | 0.0526 | 0.6280 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | multiplicative_only | 256 | 1 |
| `s030` | 0.3039 | 0.2931 | 0.0218 | 0.0544 | 0.6205 | 1 | 3e-4 | 1e-4 | 0.25 | 1 | full | 0 | 0 |
| `s080` | 0.3068 | 0.2931 | 0.0222 | 0.0566 | 0.6429 | 1 | 3e-4 | 3e-4 | 0.25 | 0 | additive_only | 0 | 0 |
| `s101` | 0.3136 | 0.2930 | 0.0224 | 0.0515 | 0.6474 | 1 | 3e-4 | 3e-4 | 0.5 | 0 | additive_only | 256 | 1 |
| `s053` | 0.3021 | 0.2928 | 0.0217 | 0.0553 | 0.6260 | 1 | 3e-4 | 1e-4 | 0.5 | 1 | additive_only | 256 | 0 |
| `s043` | 0.3055 | 0.2928 | 0.0224 | 0.0536 | 0.6315 | 1 | 3e-4 | 1e-4 | 0.5 | 0 | multiplicative_only | 256 | 1 |
| `s042` | 0.3186 | 0.2928 | 0.0217 | 0.0545 | 0.6238 | 1 | 3e-4 | 1e-4 | 0.5 | 0 | multiplicative_only | 256 | 0 |
| `s094` | 0.3064 | 0.2928 | 0.0220 | 0.0526 | 0.6390 | 1 | 3e-4 | 3e-4 | 0.5 | 0 | multiplicative_only | 0 | 0 |
| `s051` | 0.3048 | 0.2922 | 0.0218 | 0.0532 | 0.6196 | 1 | 3e-4 | 1e-4 | 0.5 | 1 | multiplicative_only | 256 | 0 |
| `s087` | 0.3004 | 0.2918 | 0.0220 | 0.0458 | 0.6201 | 1 | 3e-4 | 3e-4 | 0.25 | 1 | multiplicative_only | 256 | 0 |
| `s008` | 0.3107 | 0.2911 | 0.0220 | 0.0538 | 0.6260 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | additive_only | 0 | 0 |
| `s013` | 0.2978 | 0.2910 | 0.0220 | 0.0519 | 0.6190 | 1 | 3e-4 | 1e-4 | 0.0 | 1 | full | 256 | 0 |
| `s097` | 0.3036 | 0.2909 | 0.0225 | 0.0537 | 0.6372 | 1 | 3e-4 | 3e-4 | 0.5 | 0 | multiplicative_only | 256 | 1 |
| `s025` | 0.3046 | 0.2903 | 0.0224 | 0.0530 | 0.6375 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | multiplicative_only | 256 | 1 |
| `s010` | 0.3007 | 0.2890 | 0.0224 | 0.0524 | 0.6217 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | additive_only | 256 | 0 |
| `s095` | 0.2950 | 0.2889 | 0.0222 | 0.0528 | 0.6411 | 1 | 3e-4 | 3e-4 | 0.5 | 0 | multiplicative_only | 0 | 1 |
| `s105` | 0.3047 | 0.2880 | 0.0220 | 0.0525 | 0.6251 | 1 | 3e-4 | 3e-4 | 0.5 | 1 | multiplicative_only | 256 | 0 |
| `s016` | 0.2961 | 0.2877 | 0.0221 | 0.0502 | 0.6209 | 1 | 3e-4 | 1e-4 | 0.0 | 1 | additive_only | 0 | 0 |
| `s102` | 0.2976 | 0.2874 | 0.0219 | 0.0560 | 0.6272 | 1 | 3e-4 | 3e-4 | 0.5 | 1 | full | 0 | 0 |
| `s083` | 0.2938 | 0.2873 | 0.0224 | 0.0496 | 0.6370 | 1 | 3e-4 | 3e-4 | 0.25 | 0 | additive_only | 256 | 1 |
| `s077` | 0.2960 | 0.2864 | 0.0222 | 0.0564 | 0.6423 | 1 | 3e-4 | 3e-4 | 0.25 | 0 | multiplicative_only | 0 | 1 |
| `s054` | 0.2999 | 0.2862 | 0.0223 | 0.0582 | 0.6345 | 1 | 3e-4 | 3e-4 | 0.0 | 0 | full | 0 | 0 |
| `s060` | 0.3014 | 0.2857 | 0.0223 | 0.0511 | 0.6276 | 1 | 3e-4 | 3e-4 | 0.0 | 0 | multiplicative_only | 256 | 0 |
| `s058` | 0.2973 | 0.2852 | 0.0225 | 0.0506 | 0.6310 | 1 | 3e-4 | 3e-4 | 0.0 | 0 | multiplicative_only | 0 | 0 |
| `s032` | 0.2899 | 0.2850 | 0.0220 | 0.0533 | 0.6182 | 1 | 3e-4 | 1e-4 | 0.25 | 1 | multiplicative_only | 0 | 0 |
| `s062` | 0.2967 | 0.2850 | 0.0222 | 0.0543 | 0.6334 | 1 | 3e-4 | 3e-4 | 0.0 | 0 | additive_only | 0 | 0 |
| `s059` | 0.2954 | 0.2848 | 0.0223 | 0.0548 | 0.6299 | 1 | 3e-4 | 3e-4 | 0.0 | 0 | multiplicative_only | 0 | 1 |
| `s093` | 0.2986 | 0.2847 | 0.0225 | 0.0499 | 0.6427 | 1 | 3e-4 | 3e-4 | 0.5 | 0 | full | 256 | 1 |
| `s082` | 0.2939 | 0.2844 | 0.0219 | 0.0565 | 0.6354 | 1 | 3e-4 | 3e-4 | 0.25 | 0 | additive_only | 256 | 0 |
| `s089` | 0.2954 | 0.2844 | 0.0219 | 0.0528 | 0.6238 | 1 | 3e-4 | 3e-4 | 0.25 | 1 | additive_only | 256 | 0 |
| `s068` | 0.2935 | 0.2842 | 0.0226 | 0.0511 | 0.6222 | 1 | 3e-4 | 3e-4 | 0.0 | 1 | multiplicative_only | 0 | 0 |
| `s015` | 0.2922 | 0.2834 | 0.0221 | 0.0514 | 0.6162 | 1 | 3e-4 | 1e-4 | 0.0 | 1 | multiplicative_only | 256 | 0 |
| `s100` | 0.2981 | 0.2833 | 0.0223 | 0.0581 | 0.6366 | 1 | 3e-4 | 3e-4 | 0.5 | 0 | additive_only | 256 | 0 |
| `s017` | 0.2907 | 0.2832 | 0.0222 | 0.0515 | 0.6220 | 1 | 3e-4 | 1e-4 | 0.0 | 1 | additive_only | 256 | 0 |
| `s057` | 0.2955 | 0.2830 | 0.0225 | 0.0562 | 0.6358 | 1 | 3e-4 | 3e-4 | 0.0 | 0 | full | 256 | 1 |
| `s104` | 0.2934 | 0.2830 | 0.0218 | 0.0563 | 0.6214 | 1 | 3e-4 | 3e-4 | 0.5 | 1 | multiplicative_only | 0 | 0 |
| `s003` | 0.2919 | 0.2826 | 0.0227 | 0.0505 | 0.6277 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | full | 256 | 1 |
| `s075` | 0.2999 | 0.2820 | 0.0228 | 0.0477 | 0.6403 | 1 | 3e-4 | 3e-4 | 0.25 | 0 | full | 256 | 1 |
| `s065` | 0.2948 | 0.2818 | 0.0224 | 0.0548 | 0.6357 | 1 | 3e-4 | 3e-4 | 0.0 | 0 | additive_only | 256 | 1 |
| `s014` | 0.2883 | 0.2808 | 0.0223 | 0.0508 | 0.6079 | 1 | 3e-4 | 1e-4 | 0.0 | 1 | multiplicative_only | 0 | 0 |
| `s103` | 0.2821 | 0.2797 | 0.0225 | 0.0558 | 0.6399 | 1 | 3e-4 | 3e-4 | 0.5 | 1 | full | 256 | 0 |
| `s071` | 0.2893 | 0.2784 | 0.0225 | 0.0499 | 0.6264 | 1 | 3e-4 | 3e-4 | 0.0 | 1 | additive_only | 256 | 0 |
| `s061` | 0.2953 | 0.2780 | 0.0221 | 0.0520 | 0.6259 | 1 | 3e-4 | 3e-4 | 0.0 | 0 | multiplicative_only | 256 | 1 |
| `s067` | 0.2831 | 0.2767 | 0.0223 | 0.0507 | 0.6204 | 1 | 3e-4 | 3e-4 | 0.0 | 1 | full | 256 | 0 |
| `s064` | 0.2851 | 0.2755 | 0.0227 | 0.0506 | 0.6293 | 1 | 3e-4 | 3e-4 | 0.0 | 0 | additive_only | 256 | 0 |
| `s066` | 0.2773 | 0.2725 | 0.0231 | 0.0557 | 0.6385 | 1 | 3e-4 | 3e-4 | 0.0 | 1 | full | 0 | 0 |
| `s069` | 0.2796 | 0.2723 | 0.0232 | 0.0500 | 0.6267 | 1 | 3e-4 | 3e-4 | 0.0 | 1 | multiplicative_only | 256 | 0 |
| `s070` | 0.2682 | 0.2663 | 0.0231 | 0.0541 | 0.6338 | 1 | 3e-4 | 3e-4 | 0.0 | 1 | additive_only | 0 | 0 |

## Arms by cell-eval `pearson_delta` — `model_cal`

| arm | val | `pearson_delta` | `mae` ↓ | `overlap_at_N` | `disc_l1` | splits | lr | dlr | cos | dpo | film_mode | fh | fmf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `s055` | 0.3153 | 0.2925 | 0.0749 | 0.0303 | 0.7410 | 1 | 3e-4 | 3e-4 | 0.0 | 0 | full | 0 | 1 |
| `s059` | 0.2954 | 0.2914 | 0.0659 | 0.0246 | 0.7357 | 1 | 3e-4 | 3e-4 | 0.0 | 0 | multiplicative_only | 0 | 1 |
| `s005` | 0.3147 | 0.2912 | 0.0552 | 0.0372 | 0.7308 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | multiplicative_only | 0 | 1 |
| `s075` | 0.2999 | 0.2909 | 0.0737 | 0.0236 | 0.7353 | 1 | 3e-4 | 3e-4 | 0.25 | 0 | full | 256 | 1 |
| `s094` | 0.3064 | 0.2909 | 0.0677 | 0.0282 | 0.7542 | 1 | 3e-4 | 3e-4 | 0.5 | 0 | multiplicative_only | 0 | 0 |
| `s072` | 0.3108 | 0.2909 | 0.0756 | 0.0240 | 0.7670 | 1 | 3e-4 | 3e-4 | 0.25 | 0 | full | 0 | 0 |
| `s099` | 0.3169 | 0.2896 | 0.0615 | 0.0256 | 0.7455 | 1 | 3e-4 | 3e-4 | 0.5 | 0 | additive_only | 0 | 1 |
| `s090` | 0.3131 | 0.2890 | 0.0752 | 0.0307 | 0.7627 | 1 | 3e-4 | 3e-4 | 0.5 | 0 | full | 0 | 0 |
| `s129` | 0.3676 | 0.2883 | 0.0856 | 0.0250 | 0.7528 | 5 | 3e-4 | 1e-3 | 0.25 | 0 | full | 256 | 1 |
| `s077` | 0.2960 | 0.2865 | 0.0592 | 0.0284 | 0.7508 | 1 | 3e-4 | 3e-4 | 0.25 | 0 | multiplicative_only | 0 | 1 |
| `s101` | 0.3136 | 0.2863 | 0.0783 | 0.0236 | 0.7367 | 1 | 3e-4 | 3e-4 | 0.5 | 0 | additive_only | 256 | 1 |
| `s093` | 0.2986 | 0.2849 | 0.0773 | 0.0246 | 0.7349 | 1 | 3e-4 | 3e-4 | 0.5 | 0 | full | 256 | 1 |
| `s091` | 0.3072 | 0.2846 | 0.0621 | 0.0230 | 0.7553 | 1 | 3e-4 | 3e-4 | 0.5 | 0 | full | 0 | 1 |
| `s080` | 0.3068 | 0.2844 | 0.0668 | 0.0400 | 0.7724 | 1 | 3e-4 | 3e-4 | 0.25 | 0 | additive_only | 0 | 0 |
| `s056` | 0.3136 | 0.2842 | 0.0749 | 0.0319 | 0.7616 | 1 | 3e-4 | 3e-4 | 0.0 | 0 | full | 256 | 0 |
| `s063` | 0.3081 | 0.2835 | 0.0760 | 0.0294 | 0.7514 | 1 | 3e-4 | 3e-4 | 0.0 | 0 | additive_only | 0 | 1 |
| `s081` | 0.3087 | 0.2834 | 0.0722 | 0.0287 | 0.7495 | 1 | 3e-4 | 3e-4 | 0.25 | 0 | additive_only | 0 | 1 |
| `s027` | 0.3172 | 0.2832 | 0.0578 | 0.0268 | 0.7412 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | additive_only | 0 | 1 |
| `s073` | 0.3267 | 0.2827 | 0.0648 | 0.0307 | 0.7477 | 1 | 3e-4 | 3e-4 | 0.25 | 0 | full | 0 | 1 |
| `s097` | 0.3036 | 0.2827 | 0.0747 | 0.0233 | 0.7424 | 1 | 3e-4 | 3e-4 | 0.5 | 0 | multiplicative_only | 256 | 1 |
| `s096` | 0.3224 | 0.2819 | 0.0747 | 0.0293 | 0.7409 | 1 | 3e-4 | 3e-4 | 0.5 | 0 | multiplicative_only | 256 | 0 |
| `s057` | 0.2955 | 0.2807 | 0.0728 | 0.0339 | 0.7542 | 1 | 3e-4 | 3e-4 | 0.0 | 0 | full | 256 | 1 |
| `s060` | 0.3014 | 0.2804 | 0.0724 | 0.0317 | 0.7393 | 1 | 3e-4 | 3e-4 | 0.0 | 0 | multiplicative_only | 256 | 0 |
| `s082` | 0.2939 | 0.2796 | 0.0679 | 0.0301 | 0.7549 | 1 | 3e-4 | 3e-4 | 0.25 | 0 | additive_only | 256 | 0 |
| `s100` | 0.2981 | 0.2793 | 0.0694 | 0.0300 | 0.7504 | 1 | 3e-4 | 3e-4 | 0.5 | 0 | additive_only | 256 | 0 |
| `s092` | 0.3193 | 0.2788 | 0.0700 | 0.0330 | 0.7602 | 1 | 3e-4 | 3e-4 | 0.5 | 0 | full | 256 | 0 |
| `s058` | 0.2973 | 0.2786 | 0.0723 | 0.0324 | 0.7474 | 1 | 3e-4 | 3e-4 | 0.0 | 0 | multiplicative_only | 0 | 0 |
| `s079` | 0.3148 | 0.2781 | 0.0747 | 0.0302 | 0.7407 | 1 | 3e-4 | 3e-4 | 0.25 | 0 | multiplicative_only | 256 | 1 |
| `s009` | 0.3136 | 0.2780 | 0.0625 | 0.0418 | 0.7448 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | additive_only | 0 | 1 |
| `s007` | 0.3089 | 0.2776 | 0.0615 | 0.0453 | 0.7304 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | multiplicative_only | 256 | 1 |
| `s065` | 0.2948 | 0.2775 | 0.0734 | 0.0306 | 0.7561 | 1 | 3e-4 | 3e-4 | 0.0 | 0 | additive_only | 256 | 1 |
| `s022` | 0.3228 | 0.2770 | 0.0620 | 0.0343 | 0.7521 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | multiplicative_only | 0 | 0 |
| `s042` | 0.3186 | 0.2768 | 0.0579 | 0.0327 | 0.7457 | 1 | 3e-4 | 1e-4 | 0.5 | 0 | multiplicative_only | 256 | 0 |
| `s064` | 0.2851 | 0.2764 | 0.0735 | 0.0290 | 0.7637 | 1 | 3e-4 | 3e-4 | 0.0 | 0 | additive_only | 256 | 0 |
| `s004` | 0.3146 | 0.2756 | 0.0609 | 0.0444 | 0.7476 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | multiplicative_only | 0 | 0 |
| `s003` | 0.2919 | 0.2756 | 0.0670 | 0.0377 | 0.7273 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | full | 256 | 1 |
| `s024` | 0.3209 | 0.2697 | 0.0602 | 0.0299 | 0.7522 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | multiplicative_only | 256 | 0 |
| `s040` | 0.3142 | 0.2695 | 0.0603 | 0.0326 | 0.7524 | 1 | 3e-4 | 1e-4 | 0.5 | 0 | multiplicative_only | 0 | 0 |
| `s006` | 0.3184 | 0.2685 | 0.0573 | 0.0402 | 0.7504 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | multiplicative_only | 256 | 0 |
| `s047` | 0.3149 | 0.2680 | 0.0558 | 0.0341 | 0.7451 | 1 | 3e-4 | 1e-4 | 0.5 | 0 | additive_only | 256 | 1 |
| `s023` | 0.3209 | 0.2671 | 0.0655 | 0.0406 | 0.7281 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | multiplicative_only | 0 | 1 |
| `s008` | 0.3107 | 0.2671 | 0.0578 | 0.0359 | 0.7528 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | additive_only | 0 | 0 |
| `s041` | 0.3152 | 0.2634 | 0.0673 | 0.0358 | 0.7281 | 1 | 3e-4 | 1e-4 | 0.5 | 0 | multiplicative_only | 0 | 1 |
| `s028` | 0.3287 | 0.2622 | 0.0632 | 0.0337 | 0.7580 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | additive_only | 256 | 0 |
| `s010` | 0.3007 | 0.2603 | 0.0583 | 0.0401 | 0.7548 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | additive_only | 256 | 0 |
| `s039` | 0.3142 | 0.2546 | 0.0481 | 0.0192 | 0.7373 | 1 | 3e-4 | 1e-4 | 0.5 | 0 | full | 256 | 1 |
| `s021` | 0.3170 | 0.2406 | 0.0521 | 0.0170 | 0.7371 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | full | 256 | 1 |
| `s066` | 0.2773 | 0.2212 | 0.0611 | 0.0274 | 0.7879 | 1 | 3e-4 | 3e-4 | 0.0 | 1 | full | 0 | 0 |
| `s103` | 0.2821 | 0.2174 | 0.0572 | 0.0241 | 0.7813 | 1 | 3e-4 | 3e-4 | 0.5 | 1 | full | 256 | 0 |
| `s069` | 0.2796 | 0.2160 | 0.0589 | 0.0239 | 0.7861 | 1 | 3e-4 | 3e-4 | 0.0 | 1 | multiplicative_only | 256 | 0 |
| `s102` | 0.2976 | 0.2146 | 0.0581 | 0.0242 | 0.7795 | 1 | 3e-4 | 3e-4 | 0.5 | 1 | full | 0 | 0 |
| `s084` | 0.3214 | 0.2139 | 0.0576 | 0.0279 | 0.7784 | 1 | 3e-4 | 3e-4 | 0.25 | 1 | full | 0 | 0 |
| `s088` | 0.3072 | 0.2129 | 0.0557 | 0.0247 | 0.7779 | 1 | 3e-4 | 3e-4 | 0.25 | 1 | additive_only | 0 | 0 |
| `s048` | 0.3040 | 0.2115 | 0.0508 | 0.0338 | 0.7631 | 1 | 3e-4 | 1e-4 | 0.5 | 1 | full | 0 | 0 |
| `s071` | 0.2893 | 0.2108 | 0.0562 | 0.0243 | 0.7723 | 1 | 3e-4 | 3e-4 | 0.0 | 1 | additive_only | 256 | 0 |
| `s031` | 0.3234 | 0.2101 | 0.0519 | 0.0300 | 0.7689 | 1 | 3e-4 | 1e-4 | 0.25 | 1 | full | 256 | 0 |
| `s012` | 0.3121 | 0.2097 | 0.0512 | 0.0367 | 0.7633 | 1 | 3e-4 | 1e-4 | 0.0 | 1 | full | 0 | 0 |
| `s049` | 0.3140 | 0.2093 | 0.0470 | 0.0321 | 0.7581 | 1 | 3e-4 | 1e-4 | 0.5 | 1 | full | 256 | 0 |
| `s104` | 0.2934 | 0.2089 | 0.0547 | 0.0260 | 0.7689 | 1 | 3e-4 | 3e-4 | 0.5 | 1 | multiplicative_only | 0 | 0 |
| `s089` | 0.2954 | 0.2086 | 0.0570 | 0.0213 | 0.7784 | 1 | 3e-4 | 3e-4 | 0.25 | 1 | additive_only | 256 | 0 |
| `s087` | 0.3004 | 0.2084 | 0.0561 | 0.0206 | 0.7714 | 1 | 3e-4 | 3e-4 | 0.25 | 1 | multiplicative_only | 256 | 0 |
| `s030` | 0.3039 | 0.2079 | 0.0493 | 0.0328 | 0.7610 | 1 | 3e-4 | 1e-4 | 0.25 | 1 | full | 0 | 0 |
| `s032` | 0.2899 | 0.2076 | 0.0528 | 0.0293 | 0.7663 | 1 | 3e-4 | 1e-4 | 0.25 | 1 | multiplicative_only | 0 | 0 |
| `s067` | 0.2831 | 0.2072 | 0.0544 | 0.0310 | 0.7664 | 1 | 3e-4 | 3e-4 | 0.0 | 1 | full | 256 | 0 |
| `s035` | 0.3159 | 0.2071 | 0.0503 | 0.0261 | 0.7633 | 1 | 3e-4 | 1e-4 | 0.25 | 1 | additive_only | 256 | 0 |
| `s033` | 0.3113 | 0.2070 | 0.0505 | 0.0276 | 0.7636 | 1 | 3e-4 | 1e-4 | 0.25 | 1 | multiplicative_only | 256 | 0 |
| `s051` | 0.3048 | 0.2061 | 0.0502 | 0.0270 | 0.7618 | 1 | 3e-4 | 1e-4 | 0.5 | 1 | multiplicative_only | 256 | 0 |
| `s016` | 0.2961 | 0.2043 | 0.0511 | 0.0311 | 0.7630 | 1 | 3e-4 | 1e-4 | 0.0 | 1 | additive_only | 0 | 0 |
| `s015` | 0.2922 | 0.2039 | 0.0489 | 0.0312 | 0.7606 | 1 | 3e-4 | 1e-4 | 0.0 | 1 | multiplicative_only | 256 | 0 |
| `s017` | 0.2907 | 0.2008 | 0.0510 | 0.0318 | 0.7553 | 1 | 3e-4 | 1e-4 | 0.0 | 1 | additive_only | 256 | 0 |
| `s013` | 0.2978 | 0.1995 | 0.0507 | 0.0294 | 0.7570 | 1 | 3e-4 | 1e-4 | 0.0 | 1 | full | 256 | 0 |
| `s014` | 0.2883 | 0.1995 | 0.0468 | 0.0339 | 0.7454 | 1 | 3e-4 | 1e-4 | 0.0 | 1 | multiplicative_only | 0 | 0 |

