# ablation/ — Occam ablations, one row per (benchmark, panel, arm, seed, split)

`scores_by_seed_split.csv`: every cell-eval metric for every ablation arm (columns `axis`,
`change`, `base`) plus the reference runs, on rhj_fewshot, h1_seen, k562_seen, k562_seen_h1set;
nocal. `summary.csv`: per panel × arm, mean over seeds of the per-seed split-mean and the sd
across seeds for pΔ, mae, overlap_at_N, discrimination_score_l1. Arm definitions:
`scripts/ablation/arms.tsv`; plan, readings and job ids: `docs/ABLATION_PLAN.md`. Regenerate:
`python scripts/ablation/ablate.py table` (after `scripts/final_table.py`).
