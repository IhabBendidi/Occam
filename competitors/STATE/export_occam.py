"""Phase A for STATE: predict our panel's split from the drawn half-B control cells.

WHAT THIS ANSWERS
-----------------
Overall goal: predict a perturbation's effect in a context where that perturbation was never
measured, from a single control cell.

This script's job: run STATE's selected checkpoint over one split of our frozen panel and write
`(18575, 6983)` float32 in OUR evaluation space, which `scripts/eval_competitor.py` then scores
through OUR cell-eval 0.6.0. It measures nothing itself; it is the transport.

HOW IT PREDICTS
---------------
STATE is a SET-level model: `cell_set_len=64`, so `predict_step` consumes a batch of
"sentences" shaped `(B, S, G)` and returns `preds` the same way. The batch dict is built here
to `collate_fn`'s exact contract (`zarr_perturbation.py:764`) --

    pert_cell_emb · ctrl_cell_emb · pert_emb · pert_name · cell_type ·
    cell_type_onehot · batch · batch_name · dataset_name

-- using the datamodule's OWN one-hot maps, so a perturbation's vector is the one the
checkpoint was trained with. Building the maps independently would silently permute the
condition space.

⚠️ `pert_cell_emb` IS NOT AN INPUT. It is the target during training and is ignored at
predict time; it is filled with the control cells here so the tensor exists with the right
shape. Do not read the prediction as conditioned on it.

⚠️ ONE DRAWN CONTROL CELL PER PREDICTED ROW, which is ours and must be. STATE's own loader
pairs a perturbed cell with a control drawn from its (cell_type, batch) pool. Our split-half
protocol conditions each predicted cell on ONE SPECIFIC drawn control cell -- the same cell our
own arms and all five baselines see (`eval_control="single"`, `ctrl_avg_k=1`,
`rng = default_rng(42 + 1000 * split)`). The paired design is what buys the 0.001-0.003
split-to-split std on `pearson_delta`; feeding STATE a differently drawn control while every
other arm gets the panel's own would stop the comparison being paired. Recorded beside STATE's
numbers in `docs/rhj_competitors.md`.

⚠️ THE SENTENCE IS PADDED, NOT WRAPPED. The last sentence of a split is padded with repeats of
its final row and the padding is DISCARDED after prediction. Wrapping to the front would make
the first cells' predictions depend on how many cells the split happens to have.

EMISSION SPACE
--------------
`output_space: gene`, so STATE emits gene space in the space it was trained on:
`log1p(counts / rowsum_FULL_PANEL * 10_000)`. Declared to `export_split` as
`from_target_sum=10_000, from_log1p=True`; `assert_representable` refuses only what cannot be SCORED; library size and density are
recorded as diagnostics, because refusing on them would exclude weak arms.

⚠️ UNEXERCISED UNTIL A CHECKPOINT EXISTS. Per the plan's own rule, run ONE arm, ONE seed, ONE
split end to end before any 4x5 sweep: the last two 135-task launches died on code that had
never executed once on the real path.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

import numpy as np

ARM = "state"
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_HERE, "state", "src"),
           os.path.join(_REPO, "competitors", "shared")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


def _resolve_checkpoint(seed: int, explicit: str) -> str:
    """The checkpoint STATE itself selected on VALIDATION, never the last step.

    Its training leg writes `best_celleval.ckpt`, chosen on
    selection_score = mean(pearson_delta, discrimination_score_l1), mode=max. Refuse rather
    than fall back to `final.ckpt`: silently scoring the last step would be incomparable to
    every other arm and nothing about the output would say so.
    """
    if explicit:
        if not os.path.isfile(explicit):
            raise SystemExit(f"no checkpoint at {explicit}")
        return explicit
    d = os.path.join(_REPO, "runs", "competitors", "state", f"occam_s{seed}", "checkpoints")
    p = os.path.join(d, "best_celleval.ckpt")
    if not os.path.isfile(p):
        have = sorted(os.listdir(d)) if os.path.isdir(d) else []
        raise SystemExit(
            f"no validation-selected checkpoint at {p}. Present: {have[:6]}. Either training "
            f"has not finished for seed {seed}, or it finished without selecting one. "
            f"Refusing to fall back to the final step, which would be silently "
            f"incomparable to every other arm.")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--panel-dir", required=True)
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--cell-type", default="HEPG2")
    ap.add_argument("--sentence-len", type=int, default=64)
    ap.add_argument("--sentences-per-batch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    import torch

    from export_predictions import TRAIN_TARGET_SUM, export_split, load_panel_split
    from state.tx.data import get_datamodule

    ckpt = _resolve_checkpoint(args.seed, args.checkpoint)
    print(f"[{ARM}] checkpoint {ckpt}")

    split = load_panel_split(args.panel_dir, args.split, target_sum=TRAIN_TARGET_SUM)

    # The datamodule is instantiated for its ONE-HOT MAPS and its var dims, not for its
    # loaders: a perturbation's vector must be the one the checkpoint was trained with.
    # No `toml_config_path`: in occam mode the benchmark directory IS the source manifest
    # (`ZarrExperimentConfig.from_occam`), and the datamodule refuses both at once so the
    # source list has exactly one owner. See C23.
    dm = get_datamodule("ZarrPerturbationDataModule", dict(
        toml_config_path=None,
        source_format="occam", occam_bench_dir=args.bench_dir,
        occam_verify_sources=False,
        feature_loading="ram", target_sum=None, log1p=False,
        embed_key="X_hvg", output_space="gene", batch_col="batch_center",
        cell_type_key="cell_type", pert_col="gene_pert", control_pert="empty",
        num_workers=0, basal_mapping_strategy="random",
        cell_sentence_len=args.sentence_len,
    ), batch_size=args.sentences_per_batch, cell_sentence_len=args.sentence_len)
    dm.setup()

    missing = sorted({str(p) for p in split.labels} - set(dm.pert_onehot_map))
    if missing:
        raise SystemExit(
            f"{len(missing)} cohort perturbation(s) are absent from the checkpoint's one-hot "
            f"map, e.g. {missing[:5]}. The frozen 611-perturbation cohort is not negotiable; "
            f"a partial arm would be scored on a different panel.")
    if args.cell_type not in dm.cell_type_onehot_map:
        raise SystemExit(
            f"cell type {args.cell_type!r} is not in the checkpoint's map "
            f"({sorted(dm.cell_type_onehot_map)}). The panel is HepG2.")

    from state.tx.models.state_transition import StateTransitionPerturbationModel as _M

    # Their `_predict.py` passes the dims explicitly rather than trusting the checkpoint's
    # saved hyperparameters, so this does the same: `var_dims` comes from the datamodule, and
    # everything else from the checkpoint's own `hyper_parameters`. Trusting only the
    # checkpoint risks a silent dim mismatch against OUR 6,983-gene axis.
    var_dims = dm.get_var_dims()
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    hp = dict(saved.get("hyper_parameters", {}))
    for k in ("input_dim", "gene_dim", "hvg_dim", "output_dim", "pert_dim"):
        if k in var_dims:
            hp[k] = var_dims[k]
    del saved
    model = _M.load_from_checkpoint(ckpt, weights_only=False, **hp).to(args.device).eval()
    print(f"[{ARM}] {len(dm.pert_onehot_map):,} perturbations · "
          f"gene_dim {var_dims.get('gene_dim')} · output_dim {var_dims.get('output_dim')}")

    ct_onehot = dm.cell_type_onehot_map[args.cell_type]

    # ⚠️ PER-CELL BATCH, not a constant. `batch_encoder=True` in this checkpoint
    # (`batch_dim=167`) and the embedding is added to EVERY token, so pinning the covariate to
    # one label applies that batch's learned offset to the whole panel. Training reads the
    # PERTURBED cell's own batch (`zarr_perturbation.py:689`), and a real 30-cell sentence
    # spans ~22 distinct batches. Measured cost of the constant: 0.37 relative change, with
    # the perturbation-specific delta correlating only 0.73-0.86 with the faithful prediction.
    #
    # The panel names batches `hepg2-nad-49`; STATE's map is keyed `HEPG2__hepg2-nad-49`. The
    # mapping is BUILT and ASSERTED here rather than assumed -- an unknown key would index
    # past the embedding table, and a silently-wrong one would not.
    if split.batches is None:
        raise SystemExit(
            "the panel carries no per-cell `batches`, so this arm would have to condition on "
            "a single constant batch. Rebuild the panel, or pass --allow-constant-batch to "
            "accept the measured 0.37 relative distortion.")

    def _batch_key(raw: str) -> str:
        for cand in (f"{args.cell_type}__{raw}", raw, f"{args.cell_type}_{raw}"):
            if cand in dm.batch_onehot_map:
                return cand
        raise SystemExit(
            f"batch {raw!r} from the panel is not in the checkpoint's batch_onehot_map "
            f"(tried {args.cell_type}__{raw}, {raw}, {args.cell_type}_{raw}). "
            f"Known examples: {sorted(dm.batch_onehot_map)[:3]}")

    _uniq = sorted(set(str(b) for b in split.batches))
    _keymap = {b: _batch_key(b) for b in _uniq}
    batch_labels = np.asarray([_keymap[str(b)] for b in split.batches], dtype=object)
    batch_onehot = torch.stack([dm.batch_onehot_map[k] for k in batch_labels])
    print(f"[{ARM}] batch covariate: {len(_uniq)} distinct, per cell "
          f"(e.g. {_uniq[0]!r} -> {_keymap[_uniq[0]]!r})")

    S = int(args.sentence_len)

    def predict(ctrl: np.ndarray, pert_names: Sequence[str]) -> np.ndarray:
        """One predicted cell per control row, in PURE cell sentences.

        ⚠️ STATE DOES NOT PREDICT PER CELL. It predicts per SENTENCE of `cell_set_len`
        cells, and its backbone is a `LlamaBidirectionalModel` -- bidirectional attention
        across the whole sentence -- so every cell's prediction depends on its companions.
        `SentenceBatchSampler._create_sentences` groups training sentences by
        `(cell_type, perturbation)`, i.e. **one perturbation per sentence, always**.

        This function used to pad the whole matrix to a multiple of S and reshape
        sequentially. On split 0 that produced only 39 PURE sentences out of 290: the median
        sentence mixed 3 perturbations and some mixed 6, so for ~87% of the panel STATE was
        predicting each cell while attending over cells of two to five OTHER perturbations --
        a regime absent from training. The result collapsed to the `meanpert` baseline
        (`pearson_delta` 0.1306 vs 0.1324) while the same checkpoint scored 0.2049 on its own
        validation pass, which uses the sampler and therefore pure sentences. See C27.

        Grouping happens inside this function rather than over the whole split because
        `export_split` calls it on chunks. A perturbation whose cells straddle a chunk
        boundary simply yields two pure sentences instead of one -- exactly what training
        does for any perturbation with more than `cell_set_len` cells.
        """
        X = np.asarray(ctrl, dtype=np.float32)
        n, g = X.shape
        names = np.asarray([str(p) for p in pert_names], dtype=object)
        onehot = torch.stack([dm.pert_onehot_map[p] for p in names])
        # `batch_onehot` / `batch_labels` are indexed by the FULL split, so this function must
        # receive the whole split -- see the `batch_size=len(split)` note at the call site.
        if n != len(batch_labels):
            raise RuntimeError(
                f"predict got {n} rows but the batch covariate has {len(batch_labels)}. "
                f"export_split must hand this adapter the whole split in one call, or every "
                f"row is conditioned on another row's batch.")

        # PURE sentences: partition rows by perturbation, chunk each group into at most S.
        #
        # ⚠️ NO PADDING. STATE's own inference runs a short sentence at its true length --
        # `_infer.py:810` takes `end = min(start + cell_set_len, len(idxs))`, and the test
        # sampler skips the replacement-fill explicitly
        # (`zarr_perturbation.py:765`, `if len(sentence) < cell_sentence_len and not
        # self.test`). Padding is a TRAINING convention only.
        #
        # An earlier version of this function tiled a 30-cell sentence up to 64 with
        # `np.resize`, on the theory that it was a deterministic stand-in for the sampler's
        # `rng.choice(..., replace=True)`. Measured against the true 30-cell sentence that
        # changed the prediction by 7.2%, because `np.resize(idx_30, 64)` repeats cells 0-3
        # three times and 4-29 twice -- a non-uniform reweighting. Exact 2x duplication is
        # attention-invariant (measured 0.0007%), so the entire 7.2% was the remainder.
        # Not padding is both faithful and ~2.3x cheaper.
        plan: list[np.ndarray] = []
        # first-appearance order, without pulling in pandas: np.unique sorts, so recover the
        # original order from the returned first indices. Order only affects sentence
        # locality, never correctness -- but a stable order keeps two runs byte-comparable.
        _, _first = np.unique(names, return_index=True)
        for lbl in names[np.sort(_first)]:
            idx = np.flatnonzero(names == lbl)
            for i in range(0, len(idx), S):
                plan.append(idx[i:i + S])

        covered = np.concatenate(plan)
        if len(covered) != n or len(np.unique(covered)) != n:
            raise RuntimeError(
                f"the sentence plan covers {len(np.unique(covered))} of {n} rows "
                f"({len(covered)} slots). Every row must be predicted exactly once or the "
                f"output matrix is silently part-stale.")
        for sent in plan:
            if len(set(names[sent])) != 1:
                raise RuntimeError(
                    f"a sentence carries {len(set(names[sent]))} perturbations. Every "
                    f"sentence must hold exactly one -- the backbone attends across the "
                    f"whole sentence, so a mixed one predicts each cell in the company of "
                    f"the wrong perturbations. See C27.")

        out = np.full((n, g), np.nan, dtype=np.float32)

        # ⚠️ ONE SENTENCE PER `predict_step`, ALWAYS. `padded=False` does NOT mean "this batch
        # is unpadded" -- it means "this whole batch IS one sentence":
        #
        #   state_transition.py:409-415
        #     if padded:  pert = batch["pert_emb"].reshape(-1, self.cell_sentence_len, ...)
        #     else:       pert = batch["pert_emb"].reshape(1, -1, ...)
        #
        # Submitting 4 pure sentences with `padded=False` therefore flattened them into ONE
        # 256-cell sentence spanning ~8-9 perturbations, and the backbone is `NoRoPE` so there
        # is no length error and no warning. Measured: 52% change per cell, and the
        # per-perturbation prediction correlated only 0.12-0.25 with the faithful one. That,
        # not the windowing fixed in C27, was the bulk of the 0.2049 -> 0.1306 collapse.
        #
        # Both of STATE's own inference paths submit exactly one sentence:
        #   * `_infer.py:810,835`  -- iterates by perturbation, one `predict_step` per window,
        #                             "All tokens in this batch share the same perturbation."
        #   * `_predict.py`        -- via `test_dataloader()` -> `test=True, batch_size=1`
        # and so does our own in-training validation, which is why the same checkpoint scored
        # 0.2049 there. See C27/S1.
        with torch.no_grad():
            for si, sent in enumerate(plan):
                t = len(sent)                       # true sentence length; NOT padded to S
                ctrl_b = torch.from_numpy(X[sent]).reshape(1, t, g).to(args.device)
                pert_b = onehot[sent].reshape(1, t, -1).to(args.device)
                row_names = [str(v) for v in names[sent]]
                batch = {
                    # ⚠️ pert_cell_emb is the TARGET, ignored at predict time. Filled with
                    # the control cells so the tensor has the right shape.
                    "pert_cell_emb": ctrl_b,
                    "ctrl_cell_emb": ctrl_b,
                    "pert_emb": pert_b,
                    "pert_name": row_names,
                    "cell_type": [args.cell_type] * t,
                    "cell_type_onehot": ct_onehot.unsqueeze(0).expand(t, -1).to(args.device),
                    "batch": batch_onehot[sent].to(args.device),
                    "batch_name": [str(v) for v in batch_labels[sent]],
                    "dataset_name": ["occam_panel"] * t,
                }
                preds = model.predict_step(batch, si, padded=False)
                p = preds["preds"] if isinstance(preds, dict) else preds
                p = np.asarray(p.reshape(-1, g).float().cpu(), dtype=np.float32)
                if p.shape[0] != t:
                    raise RuntimeError(
                        f"predict_step returned {p.shape[0]} rows for {t} input cells; "
                        f"a positional prediction matrix would then be misaligned")
                out[sent] = p

        if np.isnan(out).any():
            raise RuntimeError(
                f"{int(np.isnan(out).any(1).sum())} row(s) were never written by any "
                f"sentence; refusing to emit a partly-NaN prediction.")
        return out

    # ⚠️ THE WHOLE SPLIT IN ONE `predict` CALL. `export_split` otherwise chunks the panel and
    # calls `predict` per chunk, which breaks this adapter two ways:
    #   1. the per-cell batch covariate is indexed by the FULL split, while `sent` inside
    #      `predict` is chunk-local -- every row would get another row's batch;
    #   2. the per-perturbation sentence plan is built inside `predict`, so a chunk boundary
    #      would cut a perturbation into two short sentences (~36 of 611 at the old size).
    # `predict` already loops one sentence at a time internally, so this costs no extra
    # memory beyond holding the split's control matrix (~519 MB).
    export_split(predict, split, args.out_dir, arm=ARM,
                 from_target_sum=TRAIN_TARGET_SUM, from_log1p=True,
                 batch_size=len(split),
                 extra_manifest={
                     "checkpoint": ckpt,
                     "seed": args.seed,
                     "cell_set_len": S,
                     "batch_covariate": batch_label,
                     "control_pairing": "one drawn half-B control cell per predicted row; "
                                        "OUR protocol, not their (cell_type, batch) pool",
                     "sentences": "PURE -- one perturbation per sentence, matching "
                                  "SentenceBatchSampler._create_sentences. See C27: mixed "
                                  "sentences collapsed this arm to the meanpert baseline.",
                     "padding": "a short sentence is tiled from that perturbation's OWN "
                                "cells; padding rows are discarded after prediction",
                 })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
