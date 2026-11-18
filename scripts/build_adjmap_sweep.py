#!/usr/bin/env python
"""Emit the adjacency-map sweep: the optimisation knobs x the FiLM axes, at batch 2048.

WHAT THIS ANSWERS
-----------------
Overall objective: predict a perturbation's effect in a context where that perturbation was
never measured, from a single control cell.

This sweep's question: with the **simple adjacency map** as the perturbation model (the
standing decision, 2026-09-17) and batch size fixed at the research job's 2048, which
optimisation setting and which FiLM configuration actually work on this benchmark?

Varied, fully crossed:
    lr                  {3e-4, 1e-3, 3e-3}
    decoder_lr          {1e-4, 3e-4, 1e-3}
    cosine_loss_weight  {0.0, 0.25, 0.5}
    data_perts_only     {false, true}
    film_mode           {full, multiplicative_only, additive_only}
    film_hidden_dim     {0, 256}
    film_mean_free      {false, true}

= 648 cells, of which **162 are invalid and excluded** (see below) -> **486 arms**.

HELD FIXED, and why each
------------------------
    pert_model_type   adjacency_map   the standing decision; also 5.89x faster than the GNN
                                      at batch 256 (docs/rhj_fewshot.md F8)
    batch_size        2048            the research value, 6.46x faster than 256. Fixed
                                      rather than swept BECAUSE it changes the effective
                                      learning rate -- crossing it with lr would make
                                      neither axis interpretable (F8)
    loss_type         smooth_l1       the only F6 finding that survived F7: smooth_l1 beat
                                      nb 12/12 with no range overlap
    film_scale        1.0             standing decision, not an axis
    epochs            30              unchanged, so arms stay comparable to the old grid
    seed              42              replication is a separate sweep (build_seed_grid.py)
    benchmark         rhj_fewshot     the frozen artifact, hash-asserted

THE EXCLUDED CELL, and why it is not a gap
------------------------------------------
`data_perts_only=true` x `film_mean_free=true` is **confounded by construction**.
`film_mean_free` subtracts `interaction(z, ebar)` where `ebar` is the mean over the WHOLE
graph node table; `data_perts_only` restricts that table. So the two together silently
change the reference being subtracted, and the arm would not measure either axis. The
config file carries the same warning at `data_perts_only`. All three other combinations of
the pair are kept, so both axes are still fully measured.

⚠️ WHAT THIS SWEEP CANNOT SETTLE
--------------------------------
F7: the validation metric swings +/-0.20 between consecutive epochs, so `best_metric`
selects the peak of an oscillation. If that persists here, **no arm ordering in this sweep
is interpretable either** -- and one of its two candidate causes (learning rate) is an axis
here, which is the point. Read the per-epoch curve, not just `best_metric`:
`scripts/collect_rhj_results.py` reports the metric; the oscillation is visible only in the
slurm logs. If a low-lr arm's curve is smooth while a high-lr arm's alternates, F7 is
answered as a learning-rate artefact.

    python scripts/build_adjmap_sweep.py
    sbatch --qos=hooke-predict --wckey=hooke-predict --array=0-485%27 \
        scripts/slurm/rhj_adjmap_sweep.sbatch
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LRS = ("3e-4", "1e-3", "3e-3")
DLRS = ("1e-4", "3e-4", "1e-3")
COSINES = ("0.0", "0.25", "0.5")
DPOS = ("0", "1")
MODES = ("full", "multiplicative_only", "additive_only")
FHIDS = ("0", "256")
FMFS = ("0", "1")

MODE_ABBR = {"full": "full", "multiplicative_only": "mult", "additive_only": "add"}
HEADER = ("arm_id", "kind", "tag", "lr", "decoder_lr", "cosine_loss_weight",
          "data_perts_only", "film_mode", "film_hidden_dim", "film_mean_free", "seed")

# Fixed for every arm. Written into the file's header so the record is complete.
FIXED = {
    "pert_model_type": "adjacency_map",
    "batch_size": "2048",
    "loss_type": "smooth_l1",
    "film_scale": "1.0",
    "interaction_type": "film",
    "epochs": "30",
    "seed": "42",
}


def _tag(lr, dlr, cos, dpo, mode, fh, fmf, seed):
    return (f"lr{lr}_dlr{dlr}_cos{cos}_dpo{dpo}"
            f"_fmode-{MODE_ABBR[mode]}_fh{fh}_fmf{fmf}_s{seed}")


def build(seed="42"):
    rows, skipped = [], 0
    for lr, dlr, cos, dpo, mode, fh, fmf in itertools.product(
            LRS, DLRS, COSINES, DPOS, MODES, FHIDS, FMFS):
        if dpo == "1" and fmf == "1":
            # confounded by construction -- see the module docstring
            skipped += 1
            continue
        rows.append({
            "arm_id": f"s{len(rows):03d}", "kind": "adjmap",
            "tag": _tag(lr, dlr, cos, dpo, mode, fh, fmf, seed),
            "lr": lr, "decoder_lr": dlr, "cosine_loss_weight": cos,
            "data_perts_only": dpo, "film_mode": mode, "film_hidden_dim": fh,
            "film_mean_free": fmf, "seed": seed,
        })
    return rows, skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="configs/rhj_adjmap_sweep.tsv")
    ap.add_argument("--seed", default="42")
    args = ap.parse_args(argv)

    rows, skipped = build(args.seed)
    total = (len(LRS) * len(DLRS) * len(COSINES) * len(DPOS)
             * len(MODES) * len(FHIDS) * len(FMFS))
    assert len(rows) + skipped == total, (len(rows), skipped, total)

    tags = [r["tag"] for r in rows]
    dupes = sorted({t for t in tags if tags.count(t) > 1})
    if dupes:
        raise SystemExit(f"duplicate tags: {dupes[:5]}")
    too_long = [r for r in rows if len(f"rhj_adjmap_{r['arm_id']}__{r['tag']}") > 120]
    if too_long:
        raise SystemExit(f"{len(too_long)} run-dir name(s) exceed 120 chars, e.g. "
                         f"rhj_adjmap_{too_long[0]['arm_id']}__{too_long[0]['tag']}")

    out = os.path.join(REPO, args.out) if not os.path.isabs(args.out) else args.out
    with open(out, "w") as fh:
        fh.write("# rhj_fewshot ADJACENCY-MAP sweep. Generated by "
                 "scripts/build_adjmap_sweep.py -- do not hand-edit.\n")
        fh.write(f"# {total} cells crossed, {skipped} excluded as confounded "
                 f"(data_perts_only=1 x film_mean_free=1), {len(rows)} arms.\n")
        fh.write("# HELD FIXED: "
                 + ", ".join(f"{k}={v}" for k, v in FIXED.items()) + "\n")
        fh.write("# batch_size is FIXED, not swept: it changes the effective learning "
                 "rate, so crossing it with lr would make neither axis readable (F8).\n")
        fh.write("# ⚠️ If the F7 oscillation (+/-0.20 between epochs) persists here, no arm "
                 "ordering is interpretable. lr is an axis precisely to test that.\n")
        fh.write("# Launch:\n")
        fh.write(f"#   sbatch --qos=hooke-predict --wckey=hooke-predict "
                 f"--array=0-{len(rows) - 1}%27 scripts/slurm/rhj_adjmap_sweep.sbatch\n")
        fh.write("\t".join(HEADER) + "\n")
        for r in rows:
            fh.write("\t".join(r[k] for k in HEADER) + "\n")

    print(f"[adjmap] {total} cells crossed")
    print(f"[adjmap] {skipped} excluded as confounded (data_perts_only x film_mean_free)")
    print(f"[adjmap] {len(rows)} arms -> {args.out}")
    print(f"[adjmap] held fixed: "
          + ", ".join(f"{k}={v}" for k, v in FIXED.items()))
    longest = max(len("rhj_adjmap_" + r["arm_id"] + "__" + r["tag"]) for r in rows)
    print(f"[adjmap] longest run-dir name: {longest} chars (limit 120)")
    print(f"\n[adjmap] launch with:")
    print(f"    sbatch --qos=hooke-predict --wckey=hooke-predict "
          f"--array=0-{len(rows) - 1}%27 scripts/slurm/rhj_adjmap_sweep.sbatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
