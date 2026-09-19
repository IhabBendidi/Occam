"""Phase A for PerturbDiff: predict our panel's split from the drawn half-B control cells.

WHAT THIS ANSWERS
-----------------
Overall goal: predict a perturbation's effect in a context where that perturbation was never
measured, from a single control cell.

This script's job: run PerturbDiff's selected checkpoint over one split of our frozen panel and
write `(18575, 6983)` float32 in OUR evaluation space, which `scripts/eval_competitor.py` then
scores through OUR cell-eval 0.6.0. It measures nothing itself; it is the transport.

HOW IT PREDICTS
---------------
PerturbDiff is a DIFFUSION model in gene space. Its sampling call is

    sample_fn(model.model, (B, S, G), self_condition={batch_emb, cont_emb, gene_emb, ds_name},
              clip_denoised=..., device=..., **sampling_kwargs)

where `S = data.use_cell_set` (16 on the official recipe) and the CONTROL enters entirely
through `self_condition["cont_emb"]`. The batch dict is built here to `collate_fn`'s exact
output contract (`dataset_core.py:113`) --

    pert_emb · cont_emb · cov_pert · cov_mapped_pert · cov_celltype · cov_batch ·
    is_padded_list · ds_name · col_genes

-- and then pushed through the SAME three calls the sampling app makes:
`model._encode_covariates`, `build_gene_embedding_cache`, `build_self_condition`. Reproducing
those rather than writing a shorter path is deliberate: the covariate embedding and the gene
embedding cache are where a hand-built batch would silently diverge from what the model was
trained on.

⚠️ `pert_emb` IS THE TARGET, not an input. At sampling time only its shape and the padding mask
are read (`(current_batch_size, cell_set_number, resolved_input_dim)`), so it is filled with the
control cells here. Do not read the prediction as conditioned on it.

⚠️ ONE DRAWN CONTROL CELL PER PREDICTED ROW, which is ours and must be. `mapping_cells` pairs
each perturbed cell with a control drawn at RANDOM from its (cell_type, batch) group -- their
convention. Our split-half protocol conditions each predicted cell on ONE SPECIFIC drawn
control cell, the same cell our own arms and all five baselines see (`eval_control="single"`,
`ctrl_avg_k=1`, `rng = default_rng(42 + 1000 * split)`). The paired design is what buys the
0.001-0.003 split-to-split std on `pearson_delta`. Recorded beside PerturbDiff's numbers in
`docs/rhj_competitors.md`.

⚠️ CELL SETS ARE PADDED, NOT WRAPPED. The last set is padded with repeats of its final row,
`is_padded_list` marks the padding, and the padded rows are DISCARDED after sampling. Wrapping
to the front would make the first cells' predictions depend on the split's length.

⚠️ COVARIATE CODES ARE THE GLOBAL DICTIONARY'S, not a per-file cache's. `cov_pert` /
`cov_celltype` / `cov_batch` index `datamodule.pert_dict` / `cell_type_dict` / `batch_dict`, and
the batch label is namespaced `f"{ds_name}_{batch}"` exactly as `occam_setup` builds it. A code
from the wrong space would point at a different perturbation and nothing would look odd.

EMISSION SPACE
--------------
`output_fn: relu`, so PerturbDiff emits non-negative gene space in the space it was trained on:
`log1p(counts / rowsum_FULL_PANEL * 10_000)`. Declared to `export_split` as
`from_target_sum=10_000, from_log1p=True`; `assert_representable` refuses only what cannot be SCORED (negatives, non-finite, wrong
axis, impossible range); library size and density are recorded as diagnostics because they
are confounded with model quality. A
relu head is NOT a guarantee of log1p shape, which is exactly why the declaration is explicit
and checked rather than inferred.

⚠️ UNEXERCISED UNTIL A CHECKPOINT EXISTS. Per the plan's own rule, run ONE arm, ONE seed, ONE
split end to end before any 4x5 sweep: the last two 135-task launches died on code that had
never executed once on the real path.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from typing import Sequence

import numpy as np

ARM = "perturbdiff"
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, os.path.join(_REPO, "competitors", "shared")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


def _resolve_checkpoint(seed: int, explicit: str, penc: str = "") -> str:
    """The checkpoint PerturbDiff itself selected on VALIDATION, never the last step.

    Its training leg runs `ModelCheckpoint(monitor="validation/celleval/selection_score",
    mode=max, save_top_k=1)`, so exactly one file survives per run directory. Refuse if there
    is more than one: `save_top_k=1` only prunes files it tracked itself, which is how a smoke
    checkpoint once ended up sitting beside a real one in the same directory.
    """
    if explicit:
        if not os.path.isfile(explicit):
            raise SystemExit(f"no checkpoint at {explicit}")
        return explicit
    root = os.path.join(_REPO, "runs", "competitors", "perturbdiff", "ckpt")
    # ⚠️ `penc` NARROWS THE GLOB. Two encodings of the same seed (genept and onehot) both match
    # `pd_occam_*_s{seed}`, and the >1 guard below would then refuse both. The run name carries
    # the encoding (`pd_occam_<group>_lr<lr>_<penc>_cs<n>_s<seed>`), so pass --penc to pick one.
    pat = f"pd_occam_*_{penc}_*_s{seed}" if penc else f"pd_occam_*_s{seed}"
    hits = sorted(glob.glob(os.path.join(root, pat, "*.ckpt")))
    if not hits:
        raise SystemExit(
            f"no checkpoint under {root}/{pat}/. Either training has not finished for seed "
            f"{seed}, or it wrote elsewhere, or --penc {penc!r} does not match the run name.")
    if len(hits) > 1:
        raise SystemExit(
            f"{len(hits)} checkpoints under {root}/{pat}/: {hits}. "
            f"save_top_k=1 only prunes files it tracked itself, so a smoke run's checkpoint "
            f"can sit beside a real one. Pass --checkpoint explicitly.")
    return hits[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--panel-dir", required=True)
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--penc", default="",
                    help="perturbation encoding in the run name (genept | onehot). Narrows the "
                         "checkpoint glob when several encodings share a seed.")
    ap.add_argument("--config-name", default="rawdata_diffusion_sampling")
    ap.add_argument("--cell-type", default="HEPG2")
    ap.add_argument("--sets-per-batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    from export_predictions import TRAIN_TARGET_SUM, export_split, load_panel_split
    from src.apps.sampling.sampling_generation_helpers import (
        build_gene_embedding_cache, build_self_condition, resolve_sampling_runner,
    )
    from src.apps.sampling.sampling_setup import load_sampling_model
    from src.data.data_module import OccamPerturbationDataModule

    logger = logging.getLogger(ARM)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    ckpt = _resolve_checkpoint(args.seed, args.checkpoint, args.penc)
    print(f"[{ARM}] checkpoint {ckpt}")

    split = load_panel_split(args.panel_dir, args.split, target_sum=TRAIN_TARGET_SUM)

    with initialize_config_dir(config_dir=os.path.join(_HERE, "configs"), version_base=None):
        cfg = compose(config_name=args.config_name, overrides=[
            "data=occam_h5ad", "path=occam_path", "cov_encoding=occam_h5ad",
            f"path.occam_bench_dir={args.bench_dir}",
            "model.input_dim=6983", "model.hidden_num=[6983,512]",
            f"optimization.seed={args.seed}",
        ] + ([f"cov_encoding.replogle_gene_encoding={args.penc}"] if args.penc else []))
    with open_dict(cfg):
        cfg.model_checkpoint_path = ckpt
        cfg.data.occam_verify_sources = False
        cfg.data.num_workers = 0

    # The datamodule is built for its GLOBAL COVARIATE DICTIONARIES and its metadata cache,
    # not for its loaders: `cov_*` codes must index the same dictionaries the checkpoint was
    # trained with, or they point at a different perturbation.
    dm = OccamPerturbationDataModule(seed=args.seed,
                                     micro_batch_size=cfg.optimization.micro_batch_size,
                                     data_args=cfg.data, py_logger=logger)
    dm.setup()
    dm.setup_dataset()

    from src.apps.sampling.sampling_setup import populate_covariate_cfg

    populate_covariate_cfg(cfg, dm)
    model = load_sampling_model(cfg, logger, dm).to(args.device).eval()

    missing = sorted({str(p) for p in split.labels} - set(dm.pert_dict))
    if missing:
        raise SystemExit(
            f"{len(missing)} cohort perturbation(s) are absent from the checkpoint's "
            f"pert_dict, e.g. {missing[:5]}. The frozen 611-perturbation cohort is not "
            f"negotiable; a partial arm would be scored on a different panel.")

    # ⚠️ The dataset name is NOT the cell line. `occam_setup` names datasets
    # `replogle_nadig_<line>`, because `get_short_dsname` must recognise a family or the loss
    # asserts (see the note there). Resolve by suffix rather than hardcoding either form.
    matches = [n for n in dm.dataset_name_list if n.endswith(args.cell_type)]
    if len(matches) != 1:
        raise SystemExit(
            f"cell line {args.cell_type!r} matches {matches} of {dm.dataset_name_list}; "
            f"expected exactly one")
    ds_name = matches[0]
    print(f"[{ARM}] dataset {ds_name!r} for cell line {args.cell_type!r}")
    # Any batch label the checkpoint knows; the panel is not one of them, so the first HepG2
    # batch is used. Namespaced exactly as occam_setup builds batch_dict.
    # batch_dict keys are namespaced `f"{dataset_name}_{batch}"` by occam_setup.
    batch_keys = sorted(b for b in dm.batch_dict if b.startswith(f"{ds_name}_"))
    if not batch_keys:
        raise SystemExit(f"no batch label for {ds_name!r} in batch_dict")
    batch_code = int(dm.batch_dict[batch_keys[0]])
    ct_code = int(dm.cell_type_dict[args.cell_type])
    ctrl_code = int(dm.pert_dict[str(cfg.data.control_pert)])
    print(f"[{ARM}] batch covariate {batch_keys[0]!r} · cell type code {ct_code} · "
          f"control code {ctrl_code}")

    S = int(cfg.data.use_cell_set)
    use_ddim = bool(cfg.sampling.get("use_ddim", False))
    clip_denoised = bool(cfg.sampling.get("clip_denoised", False))
    train_ds = dm.train_dataset
    col_genes = train_ds.gene_vars[ds_name]

    def predict(ctrl: np.ndarray, pert_names: Sequence[str]) -> np.ndarray:
        """One predicted cell per control row, in PURE cell sets.

        ⚠️ PERTURBDIFF IS SET-BASED. It samples `(b, S, g)` at once with `S =
        data.use_cell_set` (16 on the official recipe), and training groups its sets by
        `data.cell_set_group_by=pert_celltype` -- i.e. **one perturbation per set**.

        This function used to pad the whole matrix to a multiple of S and reshape
        sequentially, which put 46.1% of the panel's cells in a set spanning two to four
        DIFFERENT perturbations (measured on split 0: 625 of 1,160 sets pure). That is the
        same defect that collapsed STATE to the `meanpert` baseline -- see C27 -- and it is
        caught here BEFORE this arm's first export rather than after.

        Sets are built per perturbation. A short set is filled from that perturbation's own
        cells and the filler slots are flagged in `is_padded_list`, which is the mask the
        model already consumes for exactly this purpose.
        """
        X = torch.from_numpy(np.asarray(ctrl, dtype=np.float32))
        n, g = X.shape
        names = np.asarray([str(p) for p in pert_names], dtype=object)
        codes = torch.tensor([dm.pert_dict[v] for v in names], dtype=torch.long)

        # PURE sets: partition rows by perturbation, then chunk each group into S.
        # `np.resize` keeps the originals first, so a set's real rows are its leading slots.
        plan: list[tuple[np.ndarray, np.ndarray]] = []
        _, _first = np.unique(names, return_index=True)
        for lbl in names[np.sort(_first)]:
            idx = np.flatnonzero(names == lbl)
            for i in range(0, len(idx), S):
                sett = idx[i:i + S]
                plan.append((sett, sett if len(sett) == S else np.resize(sett, S)))

        covered = np.concatenate([sett for sett, _ in plan])
        if len(covered) != n or len(np.unique(covered)) != n:
            raise RuntimeError(
                f"the set plan covers {len(np.unique(covered))} of {n} rows "
                f"({len(covered)} slots). Every row must be predicted exactly once or the "
                f"output matrix is silently part-stale.")

        out = np.full((n, g), np.nan, dtype=np.float32)
        B = int(args.sets_per_batch)
        with torch.no_grad():
            for s0 in range(0, len(plan), B):
                chunk = plan[s0:s0 + B]
                b = len(chunk)
                rows = np.concatenate([padded for _, padded in chunk])
                # 1 marks a filler slot, per their own padding convention
                pad_mask = np.concatenate([
                    np.r_[np.zeros(len(sett), dtype=np.int64),
                          np.ones(S - len(sett), dtype=np.int64)]
                    for sett, _ in chunk])
                is_pad = torch.from_numpy(pad_mask)
                cont = X[rows].reshape(b, S, g).to(args.device)
                cov_pert = codes[rows].reshape(b, S).to(args.device)
                batch_data = {
                    # ⚠️ pert_emb is the TARGET. Only its shape and the padding mask are read.
                    "pert_emb": cont,
                    "cont_emb": cont,
                    "cov_pert": cov_pert,
                    # the control token for the conditioning branch, per their own convention
                    "cov_mapped_pert": torch.full((b, S), ctrl_code, dtype=torch.long,
                                                  device=args.device),
                    "cov_celltype": torch.full((b, S), ct_code, dtype=torch.long,
                                               device=args.device),
                    "cov_batch": torch.full((b, S), batch_code, dtype=torch.long,
                                            device=args.device),
                    "is_padded_list": is_pad.reshape(b, S).to(args.device),
                    "ds_name": [ds_name] * b,
                    "col_genes": [col_genes] * b,
                }
                batch_data["batch_emb"] = model._encode_covariates(batch_data)
                gene_emb = build_gene_embedding_cache(model, batch_data, args.device)
                self_condition = build_self_condition(cfg, model, batch_data, gene_emb)
                sample_fn, sampling_kwargs = resolve_sampling_runner(
                    cfg, model.diffusion, use_ddim)

                sample, _ = sample_fn(
                    model.model, (b, S, g),
                    self_condition=self_condition,
                    clip_denoised=clip_denoised,
                    device=args.device, progress=False, **sampling_kwargs)
                p = np.asarray(sample.reshape(-1, g).float().cpu(), dtype=np.float32)
                if p.shape[0] != b * S:
                    raise RuntimeError(
                        f"sampling returned {p.shape[0]} rows for {b * S} input cells; a "
                        f"positional prediction matrix would then be misaligned")
                for k, (sett, _) in enumerate(chunk):
                    out[sett] = p[k * S: k * S + len(sett)]

        if np.isnan(out).any():
            raise RuntimeError(
                f"{int(np.isnan(out).any(1).sum())} row(s) were never written by any set; "
                f"refusing to emit a partly-NaN prediction.")
        return out

    export_split(predict, split, args.out_dir, arm=ARM,
                 from_target_sum=TRAIN_TARGET_SUM, from_log1p=True,
                 batch_size=S * int(args.sets_per_batch) * 4,
                 extra_manifest={
                     "checkpoint": ckpt,
                     "seed": args.seed,
                     "use_cell_set": S,
                     "use_ddim": use_ddim,
                     "batch_covariate": batch_keys[0],
                     "control_pairing": "one drawn half-B control cell per predicted row via "
                                        "self_condition['cont_emb']; OUR protocol, not their "
                                        "random (cell_type, batch) draw",
                     "padding": "last cell set padded with repeats of its final row, marked "
                                "in is_padded_list and discarded after sampling",
                 })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
