#!/usr/bin/env python
"""Freeze the K562 "seen perturbation, SEEN assay, UNSEEN cell line" panels.

Same builder as h1_seen (scripts/build_h1_seen_panel.py) with `--dataset k562`; everything
downstream of reading the source -- normalisation, cohort, 5 split-halves, meanpert,
transfer, manifest, self-checks, --verify-against -- is shared code.

Source: Replogle 2022 K562 ESSENTIAL CRISPRi screen, raw counts (310,385 x 8,563). Same screen
family as the rhj_fewshot RPE1 file, so the ASSAY is seen; K562 itself is not in rhj_fewshot.

Two panels, both from ONE local copy `k562_seen_raw.h5ad` (RHJ test perts + the H1 cohort +
all 10,691 controls):

  k562_seen          the 528 RHJ TEST perturbations with >= 10 cells  -> compare with rhj_fewshot
  k562_seen_h1set    the 80 perturbations in BOTH the H1 cohort and K562 (>= 10 cells each),
                     frozen to perts.json                               -> compare with h1_seen_k562set

USAGE
    python scripts/build_k562_seen_panel.py                                  # k562_seen
    python scripts/build_k562_seen_panel.py --from-raw data/benchmarks/k562_seen/k562_seen_raw.h5ad \\
        --eligible h1set --bench-id k562_seen_h1set                          # the 80-set
    # the H1 panel on the same 80:
    python scripts/build_h1_seen_panel.py --from-raw data/benchmarks/h1_seen/h1_seen_raw.h5ad \\
        --eligible file:data/benchmarks/k562_seen_h1set/perts.json --bench-id h1_seen_k562set
"""
import os
import sys

sys.argv[1:1] = ["--dataset", "k562"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_h1_seen_panel import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
