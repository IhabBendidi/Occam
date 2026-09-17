# `rhj_fewshot` adjacency-map sweep — validation vs cell-eval

**The question:** does the validation metric used for checkpoint selection predict the benchmark result? One GNN arm showed val 0.3368 against cell-eval 0.0020 (`docs/rhj_splithalf.md` F11), so they are not interchangeable.

Comparison metrics (decision 2026-09-17): `pearson_delta`, `mae` ↓, `overlap_at_N`, `discrimination_score_l1` *(chance 0.5)*. Both calibration settings are shown because they win different metrics of the four (F12).

`val` is group-mean `pearson_delta` on the 103 HepG2 **validation** perturbations — the checkpoint-selection metric, read at the epoch that scored best. The cell-eval columns are the 611 held-out **test** perturbations under the split-half protocol.

33 of 212 completed arms scored so far.

## Baselines, all 5 splits

| arm | `pearson_delta` | `mae` ↓ | `overlap_at_N` | `disc_l1` |
|---|---|---|---|---|
| `ctrlcell` | -0.0018 | 0.0296 | 0.0000 | 0.5038 |
| `ctrlmean` | 0.0016 | 0.0231 | 0.0081 | 0.5014 |
| `meanpert` | 0.1324 | 0.0290 | 0.0108 | 0.5037 |
| `transfer` | 0.2323 | 0.0299 | 0.0351 | 0.6404 |
| `experimental` | 0.3722 | 0.0257 | 0.1942 | 0.7821 |

## Rank correlation: validation vs cell-eval

Spearman rho over the arms scored so far. **Positive and high** = validation is a usable proxy. **Near zero** = validation is uninformative and arms must be chosen on cell-eval. **Negative** = validation is actively harmful.

| cell-eval metric | arm | Spearman rho vs val | n |
|---|---|---|---|
| `pearson_delta` | `model_nocal` | **+0.940** | 33 |
| `mae` | `model_nocal` | **-0.775** *(lower is better, so a NEGATIVE rho is the good sign)* | 33 |
| `overlap_at_N` | `model_nocal` | **+0.658** | 33 |
| `discrimination_score_l1` | `model_nocal` | **+0.624** | 33 |
| `pearson_delta` | `model_cal` | **+0.578** | 26 |
| `mae` | `model_cal` | **+0.503** *(lower is better, so a NEGATIVE rho is the good sign)* | 26 |
| `overlap_at_N` | `model_cal` | **-0.099** | 26 |
| `discrimination_score_l1` | `model_cal` | **-0.124** | 26 |

## Arms by cell-eval `pearson_delta` — `model_nocal`

| arm | val | `pearson_delta` | `mae` ↓ | `overlap_at_N` | `disc_l1` | splits | lr | dlr | cos | dpo | film_mode | fh | fmf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `s129` | 0.3676 | 0.3348 | 0.0210 | 0.0618 | 0.6479 | 5 | 3e-4 | 1e-3 | 0.25 | 0 | full | 256 | 1 |
| `s018` | 0.3367 | 0.3084 | 0.0215 | 0.0571 | 0.6384 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | full | 0 | 0 |
| `s031` | 0.3234 | 0.3054 | 0.0215 | 0.0548 | 0.6256 | 1 | 3e-4 | 1e-4 | 0.25 | 1 | full | 256 | 0 |
| `s028` | 0.3287 | 0.3053 | 0.0217 | 0.0565 | 0.6340 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | additive_only | 256 | 0 |
| `s019` | 0.3294 | 0.3048 | 0.0216 | 0.0566 | 0.6323 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | full | 0 | 1 |
| `s022` | 0.3228 | 0.3037 | 0.0216 | 0.0555 | 0.6263 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | multiplicative_only | 0 | 0 |
| `s023` | 0.3209 | 0.3014 | 0.0215 | 0.0566 | 0.6285 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | multiplicative_only | 0 | 1 |
| `s027` | 0.3172 | 0.3008 | 0.0222 | 0.0538 | 0.6380 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | additive_only | 0 | 1 |
| `s021` | 0.3170 | 0.3000 | 0.0218 | 0.0549 | 0.6318 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | full | 256 | 1 |
| `s006` | 0.3184 | 0.2996 | 0.0217 | 0.0514 | 0.6219 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | multiplicative_only | 256 | 0 |
| `s035` | 0.3159 | 0.2994 | 0.0216 | 0.0550 | 0.6245 | 1 | 3e-4 | 1e-4 | 0.25 | 1 | additive_only | 256 | 0 |
| `s038` | 0.3047 | 0.2989 | 0.0219 | 0.0568 | 0.6304 | 1 | 3e-4 | 1e-4 | 0.5 | 0 | full | 256 | 0 |
| `s024` | 0.3209 | 0.2981 | 0.0217 | 0.0549 | 0.6256 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | multiplicative_only | 256 | 0 |
| `s033` | 0.3113 | 0.2975 | 0.0218 | 0.0530 | 0.6158 | 1 | 3e-4 | 1e-4 | 0.25 | 1 | multiplicative_only | 256 | 0 |
| `s011` | 0.3125 | 0.2970 | 0.0221 | 0.0552 | 0.6338 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | additive_only | 256 | 1 |
| `s012` | 0.3121 | 0.2969 | 0.0219 | 0.0523 | 0.6222 | 1 | 3e-4 | 1e-4 | 0.0 | 1 | full | 0 | 0 |
| `s039` | 0.3142 | 0.2963 | 0.0219 | 0.0550 | 0.6291 | 1 | 3e-4 | 1e-4 | 0.5 | 0 | full | 256 | 1 |
| `s034` | 0.3078 | 0.2962 | 0.0216 | 0.0571 | 0.6256 | 1 | 3e-4 | 1e-4 | 0.25 | 1 | additive_only | 0 | 0 |
| `s005` | 0.3147 | 0.2960 | 0.0218 | 0.0536 | 0.6243 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | multiplicative_only | 0 | 1 |
| `s009` | 0.3136 | 0.2950 | 0.0219 | 0.0550 | 0.6302 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | additive_only | 0 | 1 |
| `s004` | 0.3146 | 0.2945 | 0.0220 | 0.0519 | 0.6241 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | multiplicative_only | 0 | 0 |
| `s007` | 0.3089 | 0.2933 | 0.0219 | 0.0526 | 0.6280 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | multiplicative_only | 256 | 1 |
| `s030` | 0.3039 | 0.2931 | 0.0218 | 0.0544 | 0.6205 | 1 | 3e-4 | 1e-4 | 0.25 | 1 | full | 0 | 0 |
| `s008` | 0.3107 | 0.2911 | 0.0220 | 0.0538 | 0.6260 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | additive_only | 0 | 0 |
| `s013` | 0.2978 | 0.2910 | 0.0220 | 0.0519 | 0.6190 | 1 | 3e-4 | 1e-4 | 0.0 | 1 | full | 256 | 0 |
| `s025` | 0.3046 | 0.2903 | 0.0224 | 0.0530 | 0.6375 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | multiplicative_only | 256 | 1 |
| `s010` | 0.3007 | 0.2890 | 0.0224 | 0.0524 | 0.6217 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | additive_only | 256 | 0 |
| `s016` | 0.2961 | 0.2877 | 0.0221 | 0.0502 | 0.6209 | 1 | 3e-4 | 1e-4 | 0.0 | 1 | additive_only | 0 | 0 |
| `s032` | 0.2899 | 0.2850 | 0.0220 | 0.0533 | 0.6182 | 1 | 3e-4 | 1e-4 | 0.25 | 1 | multiplicative_only | 0 | 0 |
| `s015` | 0.2922 | 0.2834 | 0.0221 | 0.0514 | 0.6162 | 1 | 3e-4 | 1e-4 | 0.0 | 1 | multiplicative_only | 256 | 0 |
| `s017` | 0.2907 | 0.2832 | 0.0222 | 0.0515 | 0.6220 | 1 | 3e-4 | 1e-4 | 0.0 | 1 | additive_only | 256 | 0 |
| `s003` | 0.2919 | 0.2826 | 0.0227 | 0.0505 | 0.6277 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | full | 256 | 1 |
| `s014` | 0.2883 | 0.2808 | 0.0223 | 0.0508 | 0.6079 | 1 | 3e-4 | 1e-4 | 0.0 | 1 | multiplicative_only | 0 | 0 |

## Arms by cell-eval `pearson_delta` — `model_cal`

| arm | val | `pearson_delta` | `mae` ↓ | `overlap_at_N` | `disc_l1` | splits | lr | dlr | cos | dpo | film_mode | fh | fmf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `s005` | 0.3147 | 0.2912 | 0.0552 | 0.0372 | 0.7308 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | multiplicative_only | 0 | 1 |
| `s129` | 0.3676 | 0.2883 | 0.0856 | 0.0250 | 0.7528 | 5 | 3e-4 | 1e-3 | 0.25 | 0 | full | 256 | 1 |
| `s027` | 0.3172 | 0.2832 | 0.0578 | 0.0268 | 0.7412 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | additive_only | 0 | 1 |
| `s009` | 0.3136 | 0.2780 | 0.0625 | 0.0418 | 0.7448 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | additive_only | 0 | 1 |
| `s007` | 0.3089 | 0.2776 | 0.0615 | 0.0453 | 0.7304 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | multiplicative_only | 256 | 1 |
| `s022` | 0.3228 | 0.2770 | 0.0620 | 0.0343 | 0.7521 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | multiplicative_only | 0 | 0 |
| `s004` | 0.3146 | 0.2756 | 0.0609 | 0.0444 | 0.7476 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | multiplicative_only | 0 | 0 |
| `s003` | 0.2919 | 0.2756 | 0.0670 | 0.0377 | 0.7273 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | full | 256 | 1 |
| `s024` | 0.3209 | 0.2697 | 0.0602 | 0.0299 | 0.7522 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | multiplicative_only | 256 | 0 |
| `s006` | 0.3184 | 0.2685 | 0.0573 | 0.0402 | 0.7504 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | multiplicative_only | 256 | 0 |
| `s023` | 0.3209 | 0.2671 | 0.0655 | 0.0406 | 0.7281 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | multiplicative_only | 0 | 1 |
| `s008` | 0.3107 | 0.2671 | 0.0578 | 0.0359 | 0.7528 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | additive_only | 0 | 0 |
| `s028` | 0.3287 | 0.2622 | 0.0632 | 0.0337 | 0.7580 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | additive_only | 256 | 0 |
| `s010` | 0.3007 | 0.2603 | 0.0583 | 0.0401 | 0.7548 | 1 | 3e-4 | 1e-4 | 0.0 | 0 | additive_only | 256 | 0 |
| `s021` | 0.3170 | 0.2406 | 0.0521 | 0.0170 | 0.7371 | 1 | 3e-4 | 1e-4 | 0.25 | 0 | full | 256 | 1 |
| `s031` | 0.3234 | 0.2101 | 0.0519 | 0.0300 | 0.7689 | 1 | 3e-4 | 1e-4 | 0.25 | 1 | full | 256 | 0 |
| `s012` | 0.3121 | 0.2097 | 0.0512 | 0.0367 | 0.7633 | 1 | 3e-4 | 1e-4 | 0.0 | 1 | full | 0 | 0 |
| `s030` | 0.3039 | 0.2079 | 0.0493 | 0.0328 | 0.7610 | 1 | 3e-4 | 1e-4 | 0.25 | 1 | full | 0 | 0 |
| `s032` | 0.2899 | 0.2076 | 0.0528 | 0.0293 | 0.7663 | 1 | 3e-4 | 1e-4 | 0.25 | 1 | multiplicative_only | 0 | 0 |
| `s035` | 0.3159 | 0.2071 | 0.0503 | 0.0261 | 0.7633 | 1 | 3e-4 | 1e-4 | 0.25 | 1 | additive_only | 256 | 0 |
| `s033` | 0.3113 | 0.2070 | 0.0505 | 0.0276 | 0.7636 | 1 | 3e-4 | 1e-4 | 0.25 | 1 | multiplicative_only | 256 | 0 |
| `s016` | 0.2961 | 0.2043 | 0.0511 | 0.0311 | 0.7630 | 1 | 3e-4 | 1e-4 | 0.0 | 1 | additive_only | 0 | 0 |
| `s015` | 0.2922 | 0.2039 | 0.0489 | 0.0312 | 0.7606 | 1 | 3e-4 | 1e-4 | 0.0 | 1 | multiplicative_only | 256 | 0 |
| `s017` | 0.2907 | 0.2008 | 0.0510 | 0.0318 | 0.7553 | 1 | 3e-4 | 1e-4 | 0.0 | 1 | additive_only | 256 | 0 |
| `s013` | 0.2978 | 0.1995 | 0.0507 | 0.0294 | 0.7570 | 1 | 3e-4 | 1e-4 | 0.0 | 1 | full | 256 | 0 |
| `s014` | 0.2883 | 0.1995 | 0.0468 | 0.0339 | 0.7454 | 1 | 3e-4 | 1e-4 | 0.0 | 1 | multiplicative_only | 0 | 0 |

