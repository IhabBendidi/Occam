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

⚠️ CELL SETS ARE PURE AND PADDED WITHIN ONE PERTURBATION. Rows are partitioned by
perturbation first (training groups sets by `cell_set_group_by=pert_celltype`, i.e. one
perturbation per set), then chunked into `use_cell_set`; a short set is filled from THAT
perturbation's own cells and the filler is flagged in `is_padded_list` and discarded after
sampling. An earlier version reshaped the panel sequentially, which put 46.1% of cells in a
set spanning two to four different perturbations -- the defect that collapsed the sibling arm
to the `meanpert` baseline (C27).

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
import pathlib
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
    # ⚠️ RESOLVE FROM THE MARKER THE TRAINING LEG WROTE, not by globbing.
    #
    # `ModelCheckpoint.best_model_path` is recorded by `training_pipeline.py:72-77` at
    # `runs/competitors/perturbdiff/runs/<run_name>.best_ckpt_path`, and PerturbDiff's own
    # eval script reads exactly that and refuses to run without it:
    #
    #   launch.real.eval.sh:71-80
    #     if [ ! -f "${MARKER}" ]; then
    #       echo "Refusing to score the final weights instead -- that would report a model
    #             that was never selected." >&2; exit 1; fi
    #
    # Globbing could not work here and had three independent faults:
    #   1. `save_last: True` (lightning/base.yaml:27) puts `last.ckpt` beside the selected
    #      file, so the >1 guard always fired;
    #   2. the launcher appends TAG_SUFFIX *after* `_s${SEED}` (`..._cs16_s42_genept`), which
    #      NO variant of `pd_occam_*_s{seed}` matches -- so it found nothing at all;
    #   3. `--penc` is never passed by `competitor_export.sbatch`.
    # The natural workaround, `--checkpoint .../last.ckpt`, reports final-step weights that
    # `validation/celleval/selection_score` never chose -- here step 64000 of 80000.
    runs_dir = os.path.join(_REPO, "runs", "competitors", "perturbdiff", "runs")
    markers = sorted(glob.glob(os.path.join(runs_dir, f"*_s{seed}*.best_ckpt_path")))
    if penc:
        markers = [m for m in markers if f"_{penc}_" in os.path.basename(m)
                   or os.path.basename(m).rstrip(".best_ckpt_path").endswith(f"_{penc}")]
    if not markers:
        raise SystemExit(
            f"no best-checkpoint marker matching *_s{seed}* in {runs_dir} "
            f"(penc={penc!r}). Training either has not finished or did not record its "
            f"selected checkpoint. REFUSING to glob for a .ckpt: that risks scoring the "
            f"final weights, which selection never chose.")
    if len(markers) > 1:
        raise SystemExit(
            f"{len(markers)} markers match *_s{seed}* (penc={penc!r}): "
            f"{[os.path.basename(m) for m in markers]}. Pass --penc to disambiguate.")
    path = pathlib.Path(markers[0]).read_text().strip()
    if not path or not os.path.isfile(path):
        raise SystemExit(
            f"the marker {markers[0]} points at {path!r}, which is not a file.")
    if os.path.basename(path) == "last.ckpt":
        raise SystemExit(
            f"the marker points at last.ckpt, i.e. the FINAL weights rather than the ones "
            f"`validation/celleval/selection_score` selected. Refusing, as their own eval "
            f"script does.")
    print(f"[{ARM}] checkpoint from marker {os.path.basename(markers[0])}")
    return path


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
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--penc", default="",
                    help="perturbation encoding in the run name (genept | onehot). Narrows the "
                         "checkpoint glob when several encodings share a seed.")
    ap.add_argument("--config-name", default="rawdata_diffusion_sampling")
    ap.add_argument("--cell-type", default="",
                    help="the EVALUATED cell line. Empty (the default) resolves it from "
                         "--bench-dir, which is the only place that knows: it was pinned "
                         "to rhj_fewshot's held-out line, and every other benchmark then "
                         "asked for a line its artifact does not have.")
    ap.add_argument("--sets-per-batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    # ⛔ HOW A PAIR IS ANSWERED. This arm has no multi-perturbation mechanism of its own,
    # so on a combinatorial panel (norman_combo) the composition rule is applied HERE, at
    # the export boundary, by feeding the first pass's output back in as the second pass's
    # control. `none` leaves single-perturbation panels bit-identical to before, which is
    # why it is the default. See competitors/shared/export_predictions.py and
    # docs/competitors_on_doubles.md.
    ap.add_argument("--compose", default="none",
                    choices=["none", "decode_seq_ab", "decode_seq_ba",
                             "decode_seq_mean"],
                    help="sequential composition for `A+B` labels. Required on "
                         "norman_combo; a no-op on a single-perturbation panel, where "
                         "every label passes straight through.")
    ap.add_argument("--proxy-batch-draw", action="store_true",
                    help="OOD-context panels only (h1_seen): the panel's batches are not "
                         "in this checkpoint's vocabulary, so each cell's batch covariate "
                         "is a seeded uniform draw from the proxy cell line's batches. "
                         "Recorded in the manifest as `context_proxy`.")
    args = ap.parse_args(argv)
    if not args.cell_type:
        args.cell_type = _evaluated_line(args.bench_dir)
        print(f"[export] evaluated cell line from the benchmark: "
              f"{args.cell_type!r}")

    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    from export_predictions import (TRAIN_TARGET_SUM, compose_sequential, export_split,
                                 load_panel_split)
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
    # ⚠️ PER-CELL BATCH, not a constant. `cov_batch` feeds a TRAINED
    # `nn.Embedding(num_batch, 128)` whose output is concatenated into the conditioning vector
    # that adaLN-modulates every block, so pinning it applies one batch's offset to the whole
    # panel. Training reads the PERTURBED cell's own batch (`dataset_core.py:189-196`
    # `get_covarates`), which is why `PanelSplit.batches` is indexed by `a_rows`, not by the
    # drawn control. Measured on the sibling arm (STATE, same defect): 0.37 relative change.
    #
    # The panel names batches `hepg2-nad-49`; `dm.batch_dict` is keyed
    # `<ds_name>_<batch_center>`. The mapping is BUILT AND ASSERTED here -- a key that is
    # merely plausible would index a different batch with no error.
    def _batch_code(raw: str) -> int:
        for cand in (f"{ds_name}_{raw}",
                     f"{ds_name}_{args.cell_type}__{raw}",
                     f"{ds_name}_{args.cell_type}_{raw}", raw):
            if cand in dm.batch_dict:
                return int(dm.batch_dict[cand])
        raise SystemExit(
            f"batch {raw!r} from the panel is not in dm.batch_dict under any expected key "
            f"form. Known examples: {sorted(dm.batch_dict)[:3]}")

    if split.batches is None:
        raise SystemExit(
            "the panel carries no per-cell `batches`, so this arm would condition every row "
            "on one constant batch against a trained batch embedding. Rebuild the panel.")
    if split.manifest.get("ood_context"):
        if not args.proxy_batch_draw:
            raise SystemExit(
                f"panel is an OOD context ({split.manifest.get('cell_line')!r}); its batches "
                f"are not in this checkpoint's batch_dict. Pass --proxy-batch-draw.")
        from export_predictions import draw_proxy_batches
        draw_proxy_batches(split, [b for b in dm.batch_dict if str(b).startswith(f"{ds_name}_")],
                           seed=args.seed, why=f"OOD panel; proxy dataset {ds_name}")
    _uniq_b = sorted(set(str(b) for b in split.batches))
    _bmap = {b: _batch_code(b) for b in _uniq_b}
    batch_codes_all = torch.tensor([_bmap[str(b)] for b in split.batches], dtype=torch.long)
    print(f"[{ARM}] batch covariate: {len(_uniq_b)} distinct, per cell "
          f"(e.g. {_uniq_b[0]!r} -> code {_bmap[_uniq_b[0]]})")

    batch_keys = sorted(b for b in dm.batch_dict if b.startswith(f"{ds_name}_"))
    if not batch_keys:
        raise SystemExit(f"no batch label for {ds_name!r} in batch_dict")
    ct_code = int(dm.cell_type_dict[args.cell_type])
    ctrl_code = int(dm.pert_dict[str(cfg.data.control_pert)])
    print(f"[{ARM}] cell type code {ct_code} · control code {ctrl_code}")

    S = int(cfg.data.use_cell_set)
    use_ddim = bool(cfg.sampling.get("use_ddim", False))
    clip_denoised = bool(cfg.sampling.get("clip_denoised", False))
    train_ds = dm.train_dataset
    col_genes = train_ds.gene_vars[ds_name]

    # ⚠️ SEED THE SAMPLER. Their eval entry point does this
    # (`rawdata_diffusion_sampling.py:37`, `pl.seed_everything(cfg.optimization.seed)`) and
    # `launch.real.eval.sh:108` passes the seed in specifically so sampling is reproducible.
    # It matters here more than usual: `ddim_sample_loop_progressive` starts from
    # `th.randn(*shape)` and with `eta: 0.0` the per-step noise is scaled by sigma=0, so the
    # ENTIRE prediction is a deterministic function of that single draw. Unseeded, two export
    # runs of the same checkpoint and split disagree, and `--seed` -- which only selects the
    # checkpoint -- does not control it.
    import pytorch_lightning as _pl
    _pl.seed_everything(int(args.seed) + 1000 * int(args.split), workers=True)
    print(f"[{ARM}] sampler seeded with {int(args.seed) + 1000 * int(args.split)} "
          f"(seed {args.seed} + 1000*split {args.split}) -- per split, so the five splits "
          f"are independent but reproducible")

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
        cells, and the filler slots are flagged in `is_padded_list`.

        ⚠️ `is_padded_list` IS NOT SEEN BY THE MODEL. An earlier version of this docstring
        claimed it was "the mask the model already consumes for exactly this purpose"; that is
        false. It is used only by `generate_samples` AFTER sampling, to drop the filler rows
        (`sampling_generation.py:188`), and attention runs across the whole cell-set axis with
        no key-padding mask (`cross_dit_blocks.py:73-75`). So filler rows DO influence the
        real rows' predictions, and what we fill with matters. Theirs draws with replacement
        from the whole `(pert, cell_type)` group (`src/data/sampler.py:119-129`); this tiles
        the short set's own rows. Both stay within one perturbation and coincide when a group
        has a single cell, so the gap is small -- but it is a real difference, not a no-op.
        """
        X = torch.from_numpy(np.asarray(ctrl, dtype=np.float32))
        n, g = X.shape
        names = np.asarray([str(p) for p in pert_names], dtype=object)
        # `batch_codes_all` is indexed by the FULL split -- see `batch_size=len(split)`.
        if n != len(batch_codes_all):
            raise RuntimeError(
                f"predict got {n} rows but the batch covariate has "
                f"{len(batch_codes_all)}. export_split must hand this adapter the whole "
                f"split in one call, or every row is conditioned on another row's batch.")
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
                    # ⚠️ CONSUMED BY NOTHING. A repo-wide grep finds `cov_mapped_pert`
                    # produced in `dataset_core.py:118` and read nowhere. An earlier comment
                    # here called it "the control token for the conditioning branch, per
                    # their own convention", which is wrong -- there is no such branch. The
                    # value is still the semantically right one (in training it is the
                    # companion control cell's pert code), so it is kept for shape parity.
                    "cov_mapped_pert": torch.full((b, S), ctrl_code, dtype=torch.long,
                                                  device=args.device),
                    "cov_celltype": torch.full((b, S), ct_code, dtype=torch.long,
                                               device=args.device),
                    "cov_batch": batch_codes_all[rows].reshape(b, S).to(args.device),
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

    # ⚠️ THE WHOLE SPLIT IN ONE `predict` CALL. `export_split` otherwise chunks at
    # S*sets_per_batch*4 = 512 rows, which breaks this adapter two ways:
    #   1. `batch_codes_all` is indexed by the FULL split while `rows` inside `predict` is
    #      chunk-local, so every cell would get another cell's batch;
    #   2. the pure-set partition is built inside `predict`, so each of ~36 chunk boundaries
    #      would cut one perturbation into two short sets.
    # Their sampler groups over the whole split (`src/data/sampler.py:89-129`) and pads only
    # the last set of each group, once. `predict` already batches internally at
    # `--sets-per-batch`, so this costs no extra GPU memory.
        # Wrap AFTER the arm's own `predict` is fully built, so the composition sees the
    # real model and nothing inside it changes.
    if args.compose != "none":
        predict = compose_sequential(predict, order=args.compose.rsplit("_", 1)[1])
        print(f"[export] composition: {args.compose} (sequential, at the export "
              f"boundary -- this arm has no native multi-perturbation mechanism)")

    export_split(predict, split, args.out_dir, arm=ARM,
                 from_target_sum=TRAIN_TARGET_SUM, from_log1p=True,
                 batch_size=len(split),
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
