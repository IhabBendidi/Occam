# Current tables — snapshot 2026-09-23 ~09:00 (partial; final tables = docs/FINAL_TABLES_PLAN.md)

Mean ± SD over all (seed, split) rows present; "seeds × splits" says how many rows each mean
rests on. nocal. `mae` is an ERROR (↓). Occam: one model per family — **Y06** for rhj_fewshot,
h1_seen, k562_seen, k562_seen_h1set (the best rhj arm scored on every panel; AA03 is +0.0002 on
rhj, a tie) and **NF31** for norman_combo (best fully scored norman arm, `decode_seq_mean`).
Competitors: only those with scores so far. CellFlow / STATE norman are being RETRAINED (their
previous numbers are archived as defective), so they are absent from the norman tables.
h1_seen / k562_* inputs are as built (absent genes zero-filled; row-sum over panel ∩ HepG2 —
docs/audits_2026-09-23.md); STATE's h1_seen output is unrelated to its input (audit).
Built with `scripts/export_split_metrics.py --scope all` + `scripts/final_table.py`.

### rhj_fewshot (few-shot, HepG2; 611 perts)

| model | seeds × splits | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ |
|---|---|---|---|---|---|
| Occam Y06 | 1 × 5 | 0.4043 ± 0.0019 | 0.0215 ± 0.0000 | 0.0946 ± 0.0011 | 0.7288 ± 0.0010 |
| cellflow | 5 × 5 | 0.2135 ± 0.0085 | 0.0249 ± 0.0006 | 0.0316 ± 0.0031 | 0.5467 ± 0.0093 |
| perturbdiff_genept | 5 × 5 | 0.2491 ± 0.0079 | 0.0265 ± 0.0011 | 0.0644 ± 0.0058 | 0.6435 ± 0.0141 |
| state | 1 × 5 | 0.2362 ± 0.0019 | 0.0291 ± 0.0001 | 0.0849 ± 0.0008 | 0.6639 ± 0.0030 |
| txpert_single | 1 × 5 | 0.3463 ± 0.0035 | 0.0227 ± 0.0000 | 0.0582 ± 0.0010 | 0.6704 ± 0.0010 |
| baseline: ctrlcell | 5 seeds, 23 splits | -0.0000 ± 0.0034 | 0.0296 ± 0.0001 | 0.0000 ± 0.0000 | 0.5045 ± 0.0014 |
| baseline: ctrlmean | 5 seeds, 23 splits | 0.0013 ± 0.0022 | 0.0231 ± 0.0000 | 0.0080 ± 0.0008 | 0.5015 ± 0.0000 |
| baseline: meanpert | 5 seeds, 23 splits | 0.1336 ± 0.0031 | 0.0290 ± 0.0001 | 0.0106 ± 0.0006 | 0.5046 ± 0.0018 |
| baseline: transfer | 5 seeds, 23 splits | 0.2332 ± 0.0021 | 0.0299 ± 0.0000 | 0.0354 ± 0.0009 | 0.6414 ± 0.0028 |
| baseline: experimental | 5 seeds, 23 splits | 0.3721 ± 0.0009 | 0.0257 ± 0.0000 | 0.1945 ± 0.0018 | 0.7820 ± 0.0005 |

### h1_seen (unseen assay + cell line, H1; 80 perts)

| model | seeds × splits | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ |
|---|---|---|---|---|---|
| Occam Y06 | 1 × 5 | 0.0925 ± 0.0020 | 0.1377 ± 0.0000 | 0.3212 ± 0.0012 | 0.6032 ± 0.0021 |
| cellflow | 1 × 5 | 0.0427 ± 0.0020 | 0.1632 ± 0.0001 | 0.2746 ± 0.0012 | 0.5067 ± 0.0024 |
| perturbdiff_genept | 1 × 5 | 0.0835 ± 0.0018 | 0.1420 ± 0.0001 | 0.3274 ± 0.0012 | 0.5242 ± 0.0036 |
| state | 1 × 5 | 0.0455 ± 0.0016 | 0.2329 ± 0.0002 | 0.3013 ± 0.0007 | 0.5054 ± 0.0018 |
| baseline: ctrlcell | 5 × 5 | 0.0007 ± 0.0061 | 0.0147 ± 0.0000 | 0.0000 ± 0.0000 | 0.5091 ± 0.0013 |
| baseline: ctrlmean | 5 × 5 | 0.0037 ± 0.0020 | 0.0137 ± 0.0000 | 0.1701 ± 0.0018 | 0.5082 ± 0.0005 |
| baseline: meanpert | 5 × 5 | 0.0846 ± 0.0034 | 0.0162 ± 0.0000 | 0.1882 ± 0.0010 | 0.5097 ± 0.0015 |
| baseline: transfer | 5 × 5 | 0.2324 ± 0.0026 | 0.0175 ± 0.0000 | 0.2542 ± 0.0009 | 0.5695 ± 0.0015 |
| baseline: experimental | 5 × 5 | 0.8881 ± 0.0006 | 0.0065 ± 0.0000 | 0.6054 ± 0.0023 | 0.9870 ± 0.0012 |

### k562_seen (seen assay, unseen cell line, K562; 528 RHJ test perts)

| model | seeds × splits | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ |
|---|---|---|---|---|---|
| Occam Y06 | 1 × 1 | 0.1720 ± 0.0000 | 0.0494 ± 0.0000 | 0.0466 ± 0.0000 | 0.6797 ± 0.0000 |
| perturbdiff_genept | 1 × 5 | 0.1289 ± 0.0029 | 0.0634 ± 0.0001 | 0.0390 ± 0.0004 | 0.5905 ± 0.0045 |
| state | 1 × 5 | 0.1366 ± 0.0018 | 0.0531 ± 0.0001 | 0.0332 ± 0.0003 | 0.5972 ± 0.0023 |
| baseline: ctrlcell | 2 seeds, 5 splits | 0.0008 ± 0.0019 | 0.0233 ± 0.0000 | 0.0000 ± 0.0000 | 0.5042 ± 0.0006 |
| baseline: ctrlmean | 2 seeds, 5 splits | 0.0069 ± 0.0056 | 0.0184 ± 0.0000 | 0.0155 ± 0.0005 | 0.5018 ± 0.0001 |
| baseline: meanpert | 2 seeds, 5 splits | 0.0877 ± 0.0023 | 0.0238 ± 0.0000 | 0.0183 ± 0.0005 | 0.5046 ± 0.0006 |
| baseline: transfer | 2 seeds, 5 splits | 0.2720 ± 0.0011 | 0.0245 ± 0.0000 | 0.0535 ± 0.0017 | 0.7095 ± 0.0012 |
| baseline: experimental | 2 seeds, 5 splits | 0.4012 ± 0.0003 | 0.0199 ± 0.0000 | 0.2226 ± 0.0030 | 0.8214 ± 0.0026 |

### k562_seen_h1set (K562; the same 80 perts as h1_seen)

| model | seeds × splits | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ |
|---|---|---|---|---|---|
| cellflow | 1 × 5 | 0.0846 ± 0.0095 | 0.0591 ± 0.0002 | 0.0325 ± 0.0009 | 0.5219 ± 0.0056 |
| perturbdiff_genept | 1 × 5 | 0.1085 ± 0.0069 | 0.0616 ± 0.0002 | 0.0342 ± 0.0030 | 0.5497 ± 0.0038 |
| state | 1 × 5 | 0.1133 ± 0.0063 | 0.0511 ± 0.0001 | 0.0316 ± 0.0006 | 0.5461 ± 0.0064 |
| baseline: ctrlcell | 5 seeds, 21 splits | 0.0013 ± 0.0066 | 0.0216 ± 0.0001 | 0.0000 ± 0.0000 | 0.5114 ± 0.0037 |
| baseline: ctrlmean | 5 seeds, 21 splits | 0.0044 ± 0.0080 | 0.0168 ± 0.0001 | 0.0174 ± 0.0018 | 0.5077 ± 0.0007 |
| baseline: meanpert | 5 seeds, 21 splits | 0.0769 ± 0.0062 | 0.0223 ± 0.0001 | 0.0182 ± 0.0020 | 0.5109 ± 0.0034 |
| baseline: transfer | 5 seeds, 21 splits | 0.2819 ± 0.0041 | 0.0224 ± 0.0001 | 0.0603 ± 0.0031 | 0.6632 ± 0.0030 |
| baseline: experimental | 5 seeds, 21 splits | 0.4151 ± 0.0024 | 0.0186 ± 0.0000 | 0.2643 ± 0.0052 | 0.8147 ± 0.0044 |

### norman_combo (K562 pairs) — panel **additive**

| model | seeds × splits | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ |
|---|---|---|---|---|---|
| Occam NF31 | 1 × 5 | 0.9096 ± 0.0014 | 0.0112 ± 0.0001 | 0.3153 ± 0.0033 | 0.8878 ± 0.0042 |
| perturbdiff | 1 × 5 | 0.7269 ± 0.0059 | 0.0157 ± 0.0001 | 0.1711 ± 0.0045 | 0.7613 ± 0.0085 |
| txpert | 5 × 5 | 0.8513 ± 0.0077 | 0.0136 ± 0.0002 | 0.3236 ± 0.0029 | 0.9747 ± 0.0035 |
| baseline: ctrlcell | 3 seeds, 11 splits | -0.0007 ± 0.0136 | 0.0195 ± 0.0001 | 0.0000 ± 0.0000 | 0.5413 ± 0.0031 |
| baseline: ctrlmean | 3 seeds, 11 splits | 0.0051 ± 0.0176 | 0.0179 ± 0.0001 | 0.1170 ± 0.0013 | 0.5424 ± 0.0023 |
| baseline: meanpert | 3 seeds, 11 splits | 0.4628 ± 0.0052 | 0.0171 ± 0.0001 | 0.1351 ± 0.0018 | 0.5431 ± 0.0049 |
| baseline: sum_of_singles | 3 seeds, 11 splits | 0.8820 ± 0.0015 | 0.0126 ± 0.0001 | 0.2880 ± 0.0036 | 0.9885 ± 0.0026 |
| baseline: experimental | 3 seeds, 11 splits | 0.9163 ± 0.0010 | 0.0111 ± 0.0001 | 0.5231 ± 0.0034 | 0.9978 ± 0.0004 |

### norman_combo (K562 pairs) — panel **nonadditive**

| model | seeds × splits | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ |
|---|---|---|---|---|---|
| Occam NF31 | 1 × 5 | 0.7982 ± 0.0048 | 0.0133 ± 0.0001 | 0.3000 ± 0.0020 | 0.7644 ± 0.0033 |
| perturbdiff | 1 × 5 | 0.6463 ± 0.0073 | 0.0176 ± 0.0001 | 0.1802 ± 0.0019 | 0.7265 ± 0.0039 |
| txpert | 5 × 5 | 0.7069 ± 0.0082 | 0.0192 ± 0.0003 | 0.3009 ± 0.0031 | 0.7156 ± 0.0093 |
| baseline: ctrlcell | 3 seeds, 11 splits | -0.0095 ± 0.0153 | 0.0209 ± 0.0001 | 0.0000 ± 0.0000 | 0.5167 ± 0.0058 |
| baseline: ctrlmean | 3 seeds, 11 splits | 0.0052 ± 0.0053 | 0.0189 ± 0.0001 | 0.1062 ± 0.0011 | 0.5143 ± 0.0039 |
| baseline: meanpert | 3 seeds, 11 splits | 0.4002 ± 0.0074 | 0.0188 ± 0.0001 | 0.1216 ± 0.0021 | 0.4995 ± 0.0084 |
| baseline: sum_of_singles | 3 seeds, 11 splits | 0.7755 ± 0.0050 | 0.0168 ± 0.0001 | 0.2747 ± 0.0029 | 0.8562 ± 0.0068 |
| baseline: experimental | 3 seeds, 11 splits | 0.8374 ± 0.0010 | 0.0129 ± 0.0001 | 0.4698 ± 0.0062 | 0.9629 ± 0.0029 |
