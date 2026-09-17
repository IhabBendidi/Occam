# rhj_fewshot ablation grid — results

27/27 completed · 0 failed · 0 missing.

⚠️ Read `pearson_delta` and `discrimination_score_l2` from the split-half protocol, not `best_metric` below: `best_metric` is the group-mean validation `pearson_delta` used for CHECKPOINT SELECTION on the 103 HepG2 validation perturbations. It is not the benchmark result and is not comparable to any published number.

| arm | kind | film_mode | fh | fmf | loss | status | val pearson_delta (group-mean) | epochs |
|---|---|---|---|---|---|---|---|---|
| `a00` | grid | `full` | 0 | 0 | `smooth_l1` | completed | 0.3368 | 30 |
| `a01` | grid | `full` | 0 | 0 | `nb` | completed | 0.2846 | 30 |
| `a02` | grid | `full` | 0 | 1 | `smooth_l1` | completed | 0.3397 | 30 |
| `a03` | grid | `full` | 0 | 1 | `nb` | completed | 0.2893 | 30 |
| `a04` | grid | `full` | 256 | 0 | `smooth_l1` | completed | 0.3379 | 30 |
| `a05` | grid | `full` | 256 | 0 | `nb` | completed | 0.2833 | 30 |
| `a06` | grid | `full` | 256 | 1 | `smooth_l1` | completed | 0.3454 | 30 |
| `a07` | grid | `full` | 256 | 1 | `nb` | completed | 0.2935 | 30 |
| `a08` | grid | `multiplicative_only` | 0 | 0 | `smooth_l1` | completed | 0.3427 | 30 |
| `a09` | grid | `multiplicative_only` | 0 | 0 | `nb` | completed | 0.2979 | 30 |
| `a10` | grid | `multiplicative_only` | 0 | 1 | `smooth_l1` | completed | 0.3435 | 30 |
| `a11` | grid | `multiplicative_only` | 0 | 1 | `nb` | completed | 0.2998 | 30 |
| `a12` | grid | `multiplicative_only` | 256 | 0 | `smooth_l1` | completed | 0.3378 | 30 |
| `a13` | grid | `multiplicative_only` | 256 | 0 | `nb` | completed | 0.2912 | 30 |
| `a14` | grid | `multiplicative_only` | 256 | 1 | `smooth_l1` | completed | 0.3313 | 30 |
| `a15` | grid | `multiplicative_only` | 256 | 1 | `nb` | completed | 0.2765 | 30 |
| `a16` | grid | `additive_only` | 0 | 0 | `smooth_l1` | completed | 0.3543 | 30 |
| `a17` | grid | `additive_only` | 0 | 0 | `nb` | completed | 0.3019 | 30 |
| `a18` | grid | `additive_only` | 0 | 1 | `smooth_l1` | completed | 0.3479 | 30 |
| `a19` | grid | `additive_only` | 0 | 1 | `nb` | completed | 0.2940 | 30 |
| `a20` | grid | `additive_only` | 256 | 0 | `smooth_l1` | completed | 0.3276 | 30 |
| `a21` | grid | `additive_only` | 256 | 0 | `nb` | completed | 0.2937 | 30 |
| `a22` | grid | `additive_only` | 256 | 1 | `smooth_l1` | completed | 0.3355 | 30 |
| `a23` | grid | `additive_only` | 256 | 1 | `nb` | completed | 0.2972 | 30 |
| `a24` | anchor | `full` | 0 | 0 | `smooth_l1` | completed | 0.3222 | 30 |
| `a25` | anchor | `full` | 0 | 0 | `nb` | completed | 0.3006 | 30 |
| `a26` | baseline | `full` | 0 | 0 | `smooth_l1` | completed | 0.3222 | 30 |
