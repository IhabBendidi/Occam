#!/usr/bin/env python
"""Score a trained virtual-cell checkpoint on a panel, with cell-eval v1.

    python scripts/eval_panel.py \
        --checkpoint runs/mine/vcell/checkpoints/best_model.pth \
        --panel-dir data/panels/h1_hesc \
        --out runs/mine/panel

Single cell in, single cell out: one predicted cell per real perturbed cell,
each from one real control cell drawn from a pool the scoring reference does not
contain.

Two zero-signal anchors are always scored alongside the model and cannot be
disabled. On this evaluation the anchor beats every model arm on
`pearson_delta`, so a table without it invites the wrong conclusion outright.
See docs/evaluation.md.

⚠️ `--output-activation` is REQUIRED unless the checkpoint records one. Rebuilding
a softplus-trained decoder under the `rectified_sigmoid` default loads cleanly
with `strict=True` and emits ~99% exact zeros -- a dead model that looks fine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _ensure_src_on_path() -> None:
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


_ensure_src_on_path()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="virtual-cell checkpoint")
    ap.add_argument("--panel-dir", required=True,
                    help="built by scripts/build_panel.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--panels", default="all",
                    help="comma-separated: all[,seen][,unseen]. 'all' is the headline "
                         "and is always included.")
    ap.add_argument("--calibrate", default="both", choices=["none", "quantile", "both"],
                    help="control-fit quantile calibration. Worth ~0.37 pearson_delta "
                         "against ~0.02 for the loss/head choice, so 'none' alone "
                         "measures very little.")
    ap.add_argument("--shrink", default="1.0",
                    help="comma-separated alphas for the delta shrink")
    ap.add_argument("--anchors", default="ctrlmean,ctrlcell",
                    help="zero-signal arms; may not be empty")
    ap.add_argument("--output-activation", default=None,
                    choices=["rectified_sigmoid", "softplus"],
                    help="required unless the checkpoint records it")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=43200)
    args = ap.parse_args(argv)

    import torch

    if not os.path.isfile(args.checkpoint):
        ap.error(f"checkpoint not found: {args.checkpoint}")
    for name in ("panel_real.h5ad", "panel_controls_model.h5ad"):
        p = os.path.join(args.panel_dir, name)
        if not os.path.isfile(p):
            ap.error(f"panel is missing {name}: {p}\n"
                     f"Build one with scripts/build_panel.py.")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    recorded = (ckpt.get("decoder_config") or {}).get("output_activation")
    if args.output_activation is None and recorded is None:
        ap.error(
            "this checkpoint does not record `output_activation`, and none was "
            "given.\nRebuilding would default to 'rectified_sigmoid'; if the "
            "decoder was trained with softplus that produces ~99% exact zeros -- "
            "a dead model that still loads cleanly.\nPass --output-activation "
            "explicitly."
        )
    activation = args.output_activation or recorded
    print(f"[eval] output activation: {activation}"
          f"{' (from the checkpoint)' if recorded else ' (from --output-activation)'}")

    train_perts = list((ckpt.get("split_info") or {}).get("train", ()))
    wanted = [p.strip() for p in args.panels.split(",") if p.strip()]
    if {"seen", "unseen"} & set(wanted) and not train_perts:
        ap.error("--panels seen/unseen need the training perturbation set, which "
                 "this checkpoint does not record in split_info['train'].")

    from occam.vcell.panel import (
        load_panel, predict_panel, run_panel_evaluation,
    )
    from occam.vcell.panel.calibrate import apply_quantile_map, fit_control_quantile_map
    from occam.vcell.pipeline import VirtualCellTrainer

    trainer = VirtualCellTrainer.from_vcell_checkpoint(
        args.checkpoint, device=args.device, output_activation=activation)
    config = trainer.config
    config.panel_dir = args.panel_dir
    config.panel_celleval_threads = args.threads
    config.panel_celleval_timeout = args.timeout

    panel = load_panel(config, trainer.encoder_gene_names,
                       trainer.skip_normalize, trainer.skip_log1p)
    pred, labels, ctrl_used, cin_idx = predict_panel(
        trainer.model, panel, trainer.pert2id, trainer.device,
        eval_control=str(getattr(config, "panel_eval_control", "single")),
        batch_size=int(getattr(config, "panel_batch_size", 512)),
        seed=int(getattr(config, "seed", 42)),
        ctrl_avg_k=int(getattr(config, "panel_ctrl_avg_k", 1) or 1))

    calibrated = None
    if args.calibrate in ("quantile", "both"):
        src_q, ref_q = fit_control_quantile_map(
            trainer.model, panel, trainer.pert2id, trainer.device,
            max_cells=getattr(config, "panel_calib_max_cells", 20000),
            seed=getattr(config, "seed", 42),
            ctrl_avg_k=int(getattr(config, "panel_ctrl_avg_k", 1) or 1))
        calibrated = apply_quantile_map(pred, src_q, ref_q, trainer.device)

    results = run_panel_evaluation(
        panel=panel, pred=pred, labels=labels, ctrl_used=ctrl_used,
        out_dir=args.out,
        train_pert_names=train_perts or None,
        panels=wanted,
        calibrate=args.calibrate,
        calibrated=calibrated,
        shrinks=[float(x) for x in args.shrink.split(",") if x],
        anchors=[a.strip() for a in args.anchors.split(",") if a.strip()],
        celleval_kwargs={"num_threads": args.threads, "timeout": args.timeout},
    )

    print(f"\n[eval] wrote {args.out}/{{metrics.json,metrics.csv,report.md}}")
    print(f"[eval] read report.md -- every number belongs next to its anchor")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
