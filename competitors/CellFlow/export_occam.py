"""Phase A for CellFlow: predict our panel's split from the drawn half-B control cells.

WHAT THIS ANSWERS
-----------------
Overall goal: predict a perturbation's effect in a context where that perturbation was never
measured, from a single control cell.

This script's job: run CellFlow's selected state over one split of our frozen panel and write
`(18575, 6983)` float32 in OUR evaluation space, which `scripts/eval_competitor.py` then scores
through OUR cell-eval 0.6.0. It measures nothing itself; it is the transport.

HOW IT PREDICTS, AND THE ONE STRUCTURAL DIFFERENCE FROM THE OTHER THREE ARMS
----------------------------------------------------------------------------
CellFlow is a flow in a **256-component PCA space**, not a gene-space head:

    control cells -> centre and project through the TRAIN-FIT basis -> flow -> PCA coords
                  -> `reconstruct_from_pca` -> gene space

`cf.predict(adata_ctrl, covariate_data=cov, ...)` returns `{pert_str: (n_ctrl_cells, n_comps)}`
-- one predicted cell PER CONTROL CELL, per condition. That shape is what makes our protocol
expressible at all: our panel gives row `i` exactly one control cell `c_i` and one perturbation
`p_i`, so the rows are GROUPED BY PERTURBATION, predicted with that group's own control cells,
and scattered back to their original positions.

⚠️ THE ALTERNATIVE WOULD BE 611x THE COMPUTE AND THE SAME ANSWER. Calling `predict` once with
every condition against every control cell and then reading the diagonal gives the same rows;
grouping is not an approximation.

⚠️ ROW ORDER WITHIN A GROUP IS ASSUMED PRESERVED by `cf.predict` -- it applies a flow per cell,
so output row `j` is input row `j`. The shape is asserted; if a future version reorders, the
prediction would be silently permuted WITHIN a perturbation. That is the one assumption here
that the shape check cannot catch, and it is why `scripts/eval_competitor.py`'s label-order
check plus the plan's one-arm-one-split dry run both exist.

⚠️ THE PCA BASIS IS THE TRAIN-FIT ONE, loaded from the run directory, never refitted. Refitting
on the panel would leak evaluation data into the representation, and `pearson_delta` would
improve for the wrong reason.

EMISSION SPACE
--------------
`reconstruct_from_pca` returns gene space in the space the model was trained on:
`log1p(counts / rowsum_FULL_PANEL * 10_000)`, clipped at 0 below. Declared to `export_split` as
`from_target_sum=10_000, from_log1p=True`; `assert_representable` refuses only what cannot be SCORED; library size and density are
recorded as diagnostics, because refusing on them would exclude weak arms.

⚠️ `clip_max` IS LEFT UNSET ON PURPOSE. Their own scorer clips to `[0, 14]`
(`CELL_EVAL_GENE_MAX`) before scoring, which is part of their pipeline, not ours; our arms are
not clipped, and clipping only the competitor would be an unrecorded asymmetry. The lower clip
at 0 is kept because negative expression is not a prediction, it is a reconstruction artifact.

⚠️ UNEXERCISED UNTIL A CHECKPOINT EXISTS. Per the plan's own rule, run ONE arm, ONE seed, ONE
split end to end before any 4x5 sweep: the last two 135-task launches died on code that had
never executed once on the real path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

import numpy as np

ARM = "cellflow"
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, os.path.join(_REPO, "competitors", "shared")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


def _bench_tag(bench_dir: str) -> str:
    """`""` for rhj_fewshot, `"__<bench>"` otherwise -- the run-directory suffix.

    ⚠️ The training legs tag their run directory by benchmark, and this export did NOT,
    so it defaulted to `occam_s42` -- the rhj_fewshot run -- and loaded a 6,983-component
    PCA basis to export an 8,000-gene panel. The emission-space guard caught it ("the gene
    axis is not ours: 6983 genes vs the frozen 8000"), which is the only reason it was not
    a plausible-looking wrong number. Measured: job 18328366, 5/5 splits lost.
    """
    name = os.path.basename(os.path.normpath(str(bench_dir)))
    return "" if name == "rhj_fewshot" else f"__{name}"


def _resolve_run_dir(seed: int, explicit: str, bench_dir: str = "") -> tuple[str, dict]:
    """The run directory whose pickle holds the state CellFlow selected on VALIDATION.

    ⚠️ Unlike the other three arms there is no `best_*.ckpt`: CellFlow keeps the best
    parameter set in memory and restores it before saving. `run_meta.json`'s `selection` block
    is the only record of what was chosen. If it is absent the pickle holds the FINAL state,
    and that must be said beside the number rather than assumed away -- so this prints which
    it is instead of failing.
    """
    d = explicit or os.path.join(_REPO, "runs", "competitors", "cellflow",
                                 f"occam_s{seed}{_bench_tag(bench_dir)}")
    if not os.path.isdir(d):
        raise SystemExit(
            f"no run directory at {d}. Either training has not finished for seed {seed}, or "
            f"it wrote somewhere else.")
    meta_p = os.path.join(d, "run_meta.json")
    record: dict = {"selected": False}
    if os.path.isfile(meta_p):
        sel_block = json.load(open(meta_p)).get("selection") or {}
        sel = sel_block.get("best")
        if sel:
            # The pickle IS the selected state: `SelectionCallback.on_train_end` restores the
            # best params into the solver before `train_cellflow.py` calls `cf.save()`.
            record = {"selected": True, "tag": sel.get("tag"),
                      "final": bool(sel.get("final")),
                      "selection_score": sel.get("selection_score"),
                      "n_scored_states": sel_block.get("n_scored_states")}
            print(f"[{ARM}] selected state {sel.get('tag')} "
                  f"selection_score={sel.get('selection_score')} "
                  f"({sel_block.get('n_scored_states')} validation passes scored)")
        else:
            print(f"[{ARM}] ⚠️ no selection record in run_meta.json: the pickle holds the "
                  f"FINAL state, so this arm is NOT selected-on-validation. Say so beside "
                  f"its number.")
    else:
        print(f"[{ARM}] ⚠️ no run_meta.json at {meta_p}; selection is UNRECORDED")
    return d, record


def _evaluated_line(bench_dir: str) -> str:
    """The cell line whose perturbations are held out and scored, from the artifact."""
    import json as _json
    import os as _os
    b = _json.load(open(_os.path.join(bench_dir, "benchmark.json")))
    line = b.get("cell_line") or ""
    if not line:
        for rec in (b.get("sources") or {}).values():
            if rec.get("cell_line"):
                line = rec["cell_line"]
                break
    if not line:
        ecl = b.get("eval_cell_lines") or []
        line = ecl[0] if ecl else ""
    if not line:
        raise SystemExit(
            f"{bench_dir}/benchmark.json records no `cell_line` or `eval_cell_lines`, so "
            f"the evaluated line cannot be resolved. Pass --cell-type explicitly.")
    return str(line)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--panel-dir", required=True)
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--checkpoint", default="", help="the run directory, not a file")
    ap.add_argument("--cell-type", default="",
                    help="the EVALUATED cell line. Empty (the default) resolves it from "
                         "--bench-dir, which is the only place that knows: it was pinned "
                         "to rhj_fewshot's held-out line, and every other benchmark then "
                         "asked for a line its artifact does not have.")
    ap.add_argument("--batch-size", type=int, default=2048)
    args = ap.parse_args(argv)
    if not args.cell_type:
        args.cell_type = _evaluated_line(args.bench_dir)
        print(f"[export] evaluated cell line from the benchmark: "
              f"{args.cell_type!r}")

    import anndata as ad
    import pandas as pd
    from cellflow.model import CellFlow

    from export_predictions import TRAIN_TARGET_SUM, export_split, load_panel_split
    from occam_bridge import load_pca_artifacts, reconstruct_from_pca
    from occam_bridge.occam import assert_axis_order, pert_components
    from occam_bridge.schema import NULL_TOKEN

    run_dir, selection = _resolve_run_dir(args.seed, args.checkpoint, args.bench_dir)
    split = load_panel_split(args.panel_dir, args.split, target_sum=TRAIN_TARGET_SUM)

    pca = load_pca_artifacts(os.path.join(run_dir, "pca"))
    cf = CellFlow.load(run_dir)
    print(f"[{ARM}] loaded: train_n={cf.adata.n_obs:,} · n_comps={pca['n_comps']}")

    # The panel must sit on the same axis the model was trained on, in the same order: the
    # PCA basis is a positional (6983 x n_comps) matrix, so a reordered axis would project
    # the wrong genes onto the wrong components and nothing would look odd.
    import occam_data as od

    bench = od.load_occam_benchmark(args.bench_dir, verify_sources=False)
    assert_axis_order(cf.adata.var_names, bench)

    means = np.asarray(pca["means"], dtype=np.float32).reshape(-1)
    pcs = np.asarray(pca["pcs"], dtype=np.float32)
    if pcs.shape[0] != bench.n_genes:
        raise SystemExit(
            f"the PCA basis has {pcs.shape[0]} genes, the benchmark axis has "
            f"{bench.n_genes}. The basis is positional; refusing to project.")

    genes = [str(g) for g in bench.axis_ensembl]
    ct_col = "cell_type"

    def predict(ctrl: np.ndarray, pert_names: Sequence[str]) -> np.ndarray:
        X = np.asarray(ctrl, dtype=np.float32)
        n, g = X.shape
        names = np.asarray([str(p) for p in pert_names], dtype=str)
        out = np.empty((n, g), dtype=np.float32)

        # centre and project through the TRAIN-FIT basis, never a refitted one
        x_pca_all = (X - means) @ pcs

        # How many perturbation slots the TRAINED model has, read off its own obs columns.
        _NULL_TOKEN = NULL_TOKEN
        _slot_cols = sorted(c for c in cf.adata.obs.columns
                            if str(c).startswith("gene_") and str(c)[5:].isdigit())
        _n_slots = len(_slot_cols) or 1
        print(f"[{ARM}] condition slots from the trained object: {_n_slots} "
              f"({_slot_cols or ['gene_1 (assumed)']})")

        for pert in pd.unique(names):
            rows = np.flatnonzero(names == pert)
            block = ad.AnnData(
                X=X[rows],
                # ⚠️ `is_control` MUST BE TRUE ON EVERY ROW OF THE PASSED AnnData.
                # `CellFlow.predict` refuses otherwise (`model/_cellflow.py:619-622`, and
                # again in `data/_datamanager.py:794-805`):
                #   "all samples in `adata` must be control samples, and thus
                #    adata.obs[`is_control`] must be set to `True` everywhere."
                # These rows ARE controls -- they are the drawn half-B control cells, i.e.
                # the OT source population the flow is integrated from. `False` here (what
                # this adapter used to pass) made the export raise before emitting a single
                # row, which is why no split_* directory has ever existed for this arm.
                # The covariate frame below is where `is_control=False` belongs: that
                # describes the TARGET condition being predicted, not the source cells.
                # Their own working path does exactly this split
                # (`predict_cellflow.py:215-222` passes `cf.adata[ctrl_mask]`, and `:316`
                # sets False only on the covariate frame).
                obs=pd.DataFrame({ct_col: np.full(len(rows), args.cell_type),
                                  "pert_str": np.full(len(rows), pert),
                                  "is_control": np.ones(len(rows), dtype=bool)},
                                 index=[str(i) for i in range(len(rows))]),
                var=pd.DataFrame(index=pd.Index(genes, name="ensembl_gene_id")))
            block.obsm["X_pca"] = np.ascontiguousarray(x_pca_all[rows])
            # ⚠️ CARRY `uns` FROM THE TRAINED OBJECT. `predict` resolves the condition through
            # `rep_dict = adata.uns` (`data/_datamanager.py:245`) and then indexes
            # `rep_dict['gene_embeddings']` / `['cell_type_embeddings']` (`:401-403`,
            # `:484-486`). A bare block has an empty `uns`, so this raised a KeyError --
            # NOT the "Representation ... not found" the guard below tells the reader to
            # interpret as a vocabulary gap. Their path gets this for free by slicing the
            # pickled training AnnData.
            block.uns.update(cf.adata.uns)

            # ⚠️ ONE SLOT PER `max_combination_length`, built exactly as
            # occam_bridge/schema.py builds it at TRAINING and VALIDATION time (through the
            # same `occam.pert_components` the bridge uses): the component ids
            # SORTED (so `A+B` and `B+A` are one condition), then padded with the same
            # NULL token. A single `gene_1` column is only right when
            # max_combination_length == 1; against a model trained at 2 the predict call
            # raised `KeyError: "['gene_2'] not in index"` -- after training had already
            # completed. Measured: job 18363662.
            #
            # The slot count comes from the TRAINED object's own condition vocabulary, not
            # from a constant here, so this adapter follows whatever the run was trained
            # at instead of asserting a number.
            parts = pert_components(str(pert))
            padded = parts + [_NULL_TOKEN] * max(0, _n_slots - len(parts))
            if len(parts) > _n_slots:
                raise RuntimeError(
                    f"{pert!r} has {len(parts)} genes but the trained model has "
                    f"{_n_slots} slot(s). Retrain with max_combination_length >= "
                    f"{len(parts)}.")
            cov = pd.DataFrame({"pert_str": [pert], ct_col: [args.cell_type],
                                "is_control": [False],
                                **{f"gene_{i + 1}": [padded[i]]
                                   for i in range(_n_slots)}})
            preds = cf.predict(block, covariate_data=cov, sample_rep="X_pca",
                               condition_id_key="pert_str", key_added_prefix=None)
            if not preds or pert not in {str(k) for k in preds}:
                raise RuntimeError(
                    f"CellFlow.predict returned nothing for {pert!r}. If the message is "
                    f"'Representation ... not found', the one-hot vocabulary was built "
                    f"without it -- see occam_bridge/occam.py::occam_pert_tokens.")
            x_pca = np.asarray(next(v for k, v in preds.items() if str(k) == pert),
                               dtype=np.float32)
            if x_pca.ndim == 1:
                x_pca = x_pca[None, :]
            if x_pca.shape[0] != len(rows):
                raise RuntimeError(
                    f"{pert!r}: predict returned {x_pca.shape[0]} cells for {len(rows)} "
                    f"control cells. One predicted cell per control cell is what makes the "
                    f"paired protocol expressible; refusing to broadcast or truncate.")
            out[rows] = reconstruct_from_pca(x_pca, means=means, pcs=pcs,
                                             clip_min=0.0, clip_max=None)
        return out

    export_split(predict, split, args.out_dir, arm=ARM,
                 from_target_sum=TRAIN_TARGET_SUM, from_log1p=True,
                 batch_size=args.batch_size,
                 extra_manifest={
                     "run_dir": run_dir,
                     "seed": args.seed,
                     # which state the pickle holds: the validation-selected one, or FINAL
                     "selection": selection,
                     "n_comps": int(pca["n_comps"]),
                     # NOT "emission": that key holds the STRUCTURED, machine-read
                     # {from_target_sum, from_log1p, to_target_sum, map} dict written by
                     # export_split. Overwriting it with prose is what broke scoring.
                     "emission_note": "reconstruct_from_pca through the TRAIN-FIT basis, "
                                      "clip_min=0.0, clip_max=None",
                     "clip_note": "their own scorer clips to [0, 14]; ours does not, and "
                                  "clipping only the competitor would be an unrecorded "
                                  "asymmetry",
                     "control_pairing": "rows grouped by perturbation, each predicted from "
                                        "its own drawn half-B control cell",
                 })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
