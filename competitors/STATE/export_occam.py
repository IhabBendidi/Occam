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
    dm = get_datamodule("ZarrPerturbationDataModule", dict(
        toml_config_path=os.path.join(_HERE, "state", "occam_rhj_fewshot.toml"),
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
    # Any batch label is a label the checkpoint knows; the panel is not one of them, so the
    # first HepG2 batch is used. STATE's batch_encoder consumes it as a covariate, and a
    # label it has never seen would index past the embedding table.
    batch_label = sorted(b for b in dm.batch_onehot_map if b.startswith("HEPG2"))
    if not batch_label:
        batch_label = sorted(dm.batch_onehot_map)
    batch_label = batch_label[0]
    batch_onehot = dm.batch_onehot_map[batch_label]
    print(f"[{ARM}] batch covariate {batch_label!r}")

    S = int(args.sentence_len)

    def predict(ctrl: np.ndarray, pert_names: Sequence[str]) -> np.ndarray:
        X = torch.from_numpy(np.asarray(ctrl, dtype=np.float32))
        n, g = X.shape
        names = [str(p) for p in pert_names]
        pert = torch.stack([dm.pert_onehot_map[p] for p in names])

        # pad to a whole number of sentences with repeats of the LAST row; the padding is
        # discarded below. Wrapping to the front would make the first cells' predictions
        # depend on the split's length.
        pad = (-n) % S
        if pad:
            X = torch.cat([X, X[-1:].expand(pad, g)], 0)
            pert = torch.cat([pert, pert[-1:].expand(pad, pert.shape[1])], 0)
            names = names + [names[-1]] * pad
        n_sent = X.shape[0] // S

        out = np.empty((X.shape[0], g), dtype=np.float32)
        B = int(args.sentences_per_batch)
        with torch.no_grad():
            for s0 in range(0, n_sent, B):
                s1 = min(s0 + B, n_sent)
                sl = slice(s0 * S, s1 * S)
                b = s1 - s0
                ctrl_b = X[sl].reshape(b, S, g).to(args.device)
                pert_b = pert[sl].reshape(b, S, -1).to(args.device)
                batch = {
                    # ⚠️ pert_cell_emb is the TARGET, ignored at predict time. Filled with
                    # the control cells so the tensor has the right shape.
                    "pert_cell_emb": ctrl_b,
                    "ctrl_cell_emb": ctrl_b,
                    "pert_emb": pert_b,
                    "pert_name": [names[i] for i in range(sl.start, sl.stop)],
                    "cell_type": [args.cell_type] * (b * S),
                    "cell_type_onehot": ct_onehot.unsqueeze(0).expand(b * S, -1).to(
                        args.device),
                    "batch": batch_onehot.unsqueeze(0).expand(b * S, -1).to(args.device),
                    "batch_name": [batch_label] * (b * S),
                    "dataset_name": ["occam_panel"] * (b * S),
                }
                preds = model.predict_step(batch, s0 // B, padded=False)
                p = preds["preds"] if isinstance(preds, dict) else preds
                p = np.asarray(p.reshape(-1, g).float().cpu(), dtype=np.float32)
                if p.shape[0] != b * S:
                    raise RuntimeError(
                        f"predict_step returned {p.shape[0]} rows for {b * S} input cells; "
                        f"a positional prediction matrix would then be misaligned")
                out[sl] = p
        return out[:n]

    export_split(predict, split, args.out_dir, arm=ARM,
                 from_target_sum=TRAIN_TARGET_SUM, from_log1p=True,
                 batch_size=S * int(args.sentences_per_batch) * 8,
                 extra_manifest={
                     "checkpoint": ckpt,
                     "seed": args.seed,
                     "cell_set_len": S,
                     "batch_covariate": batch_label,
                     "control_pairing": "one drawn half-B control cell per predicted row; "
                                        "OUR protocol, not their (cell_type, batch) pool",
                     "padding": "last sentence padded with repeats of its final row, "
                                "discarded after prediction",
                 })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
