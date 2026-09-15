"""Score a decoder checkpoint on one eval file with ARC cell-eval v1.

    occam-celleval --checkpoint checkpoints/release/occam_small_encoder_decoder.pth \
                   --eval-data data/eval/pert/nadig_2024_jurkat.h5ad \
                   --output-dir runs/celleval/released_jurkat \
                   --pert-col gene_name --calibrate quantile \
                   --output-activation rectified_sigmoid

The decoder reconstructs every cell of the eval file from its own embedding,
and cell-eval scores the per-cell reconstruction against the same cells
preprocessed exactly as encoder input. Matched single cells on both sides, so
this does not hit the pseudobulk-variance inflation that pseudobulk predictions
suffer under a Wilcoxon test.

This is the **eval-only** path: the identical measurement the decoder pipeline
makes at the end of training, reachable for a checkpoint that already exists.
It calls :meth:`DecoderTrainer.run_cell_eval_on_file`, the same method the
trainer calls, so an existing checkpoint and a freshly trained one are
comparable without an "is it the same code?" caveat.

Architecture comes from the checkpoint's own ``decoder_config`` -- never from a
flag and never from a default -- so a geometry mismatch is impossible rather
than merely unlikely.

``--output-activation`` is the exception, and it is REQUIRED when the
checkpoint does not record one. That is not pedantry: a softplus-trained
decoder loads ``strict=True`` under ``rectified_sigmoid`` and then emits ~99%
exact zeros, which looks like a working model and scores like a broken one.
``occam-verify`` determines a checkpoint's activation empirically from a
forward pass; pass what it reports.

Modes, both scored against the same real target:

``injctrl`` (the main number)
    Control rows in the prediction are replaced by the real preprocessed
    controls, so perturbation deltas are read against a shared real-control
    baseline. This is the virtual-cell convention and the harder metric.
``reconctrl``
    The prediction keeps the decoder's own reconstructed controls. Inflated --
    reconstructed controls share the decoder's bias with reconstructed
    perturbations, which cancels in the delta.

Requires cell-eval 0.6.0, which on this cluster is reachable through the
``jupyter`` pyenv; :func:`occam.decoder_train.metrics.run_cell_eval` sets
``PYENV_VERSION`` itself.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

# Metrics promoted to the printed table, in reading order. pearson_delta and
# discrimination_score_l2 carry the conclusion; overlap_at_N is reported but
# must not be ranked on -- a zero-signal predictor beat the real model on it
# (TREK 0.3360 vs 0.2030) at chance-level discrimination.
REPORTED = (
    "pearson_delta",
    "discrimination_score_l2",
    "discrimination_score_l1",
    "overlap_at_N",
    "de_direction_match",
    "mse",
    "mae",
)

# decoder_config key -> DecoderTrainConfig field. Geometry is read from the
# checkpoint so it cannot disagree with the weights.
_ARCH_FIELDS = {
    "hidden_dim": "decoder_hidden_dim",
    "num_blocks": "decoder_num_blocks",
    "dropout_rate": "decoder_dropout",
    "se_reduction": "decoder_se_reduction",
    "weight_tie": "decoder_weight_tie",
    "bottleneck_k": "decoder_bottleneck_k",
    "monarch_blocks": "decoder_monarch_blocks",
    "monarch_mup": "decoder_monarch_mup",
    "kron_rank": "decoder_kron_rank",
    "glu_mode": "decoder_glu",
    "glu_ratio": "decoder_glu_ratio",
    "sparsity_24": "decoder_sparsity_24",
    "sparsity_24_decay": "decoder_sparsity_24_decay",
}


def build_config(checkpoint: Dict[str, Any], args) -> Any:
    """Build a :class:`DecoderTrainConfig` that matches the checkpoint's decoder.

    Raises ``ValueError`` if the checkpoint records no output activation and the
    caller did not pass one.
    """
    from occam.decoder_train.config import DecoderTrainConfig

    if "decoder_config" not in checkpoint:
        raise ValueError(
            "checkpoint has no 'decoder_config'; this command needs a unified "
            "encoder+decoder checkpoint (is_unified_checkpoint=True)")
    dcfg = dict(checkpoint["decoder_config"])
    tcfg = dict(checkpoint.get("training_config", {}) or {})

    recorded = dcfg.get("output_activation")
    if recorded and args.output_activation and args.output_activation != recorded:
        raise ValueError(
            f"--output-activation {args.output_activation!r} contradicts the "
            f"{recorded!r} recorded in this checkpoint's decoder_config.\n"
            f"  The checkpoint is the authority: its weights were trained under "
            f"{recorded!r}. Scoring them under the other head is not a variant "
            f"of the experiment, it is a broken model -- rectified_sigmoid on "
            f"softplus-trained weights emits ~99% exact zeros.\n"
            f"  Drop the flag.")
    activation = recorded or args.output_activation
    if not activation:
        raise ValueError(
            "this checkpoint does not record 'output_activation' in its "
            "decoder_config, so it cannot be determined from the file.\n"
            "  Run:  occam-verify --checkpoint <ckpt>\n"
            "  which determines it from a forward pass (rectified_sigmoid is "
            "exactly 0 for every non-positive pre-activation; softplus is "
            "strictly positive), then pass --output-activation with what it "
            "reports.\n"
            "  Do NOT guess: a softplus decoder loads cleanly under "
            "rectified_sigmoid and emits ~99% exact zeros.")

    config = DecoderTrainConfig()
    config.encoder_checkpoint = args.checkpoint
    config.freeze_encoder = True
    for src, dst in _ARCH_FIELDS.items():
        if src in dcfg:
            setattr(config, dst, dcfg[src])
    config.decoder_output_activation = activation

    # The loss is not used to score anything here, but NB/ZINB keep their
    # per-gene dispersion on the loss module, and load_decoder_weights restores
    # it into that module -- so the module has to be the right class.
    config.loss_type = args.loss_type or tcfg.get("loss_type", "smooth_l1")

    config.celleval_pert_col = args.pert_col
    config.celleval_control_label = args.control_label
    config.celleval_calibrate = args.calibrate
    config.celleval_modes = args.modes
    config.celleval_num_threads = args.num_threads
    config.celleval_timeout = args.timeout
    config.celleval_de_batch_size = args.de_batch_size
    config.celleval_de_real_cache = args.de_real_cache
    config.celleval_present_genes_only = not args.all_genes
    return config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="occam-celleval", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="unified encoder+decoder checkpoint")
    ap.add_argument("--eval-data", required=True, help="eval .h5ad")
    ap.add_argument("--output-dir", required=True,
                    help="written here: metrics.json, celleval/, cell_eval_*/")
    ap.add_argument("--pert-col", default="gene_name",
                    help="obs column naming the perturbation (default: gene_name)")
    ap.add_argument("--control-label", default="non-targeting",
                    help="value of --pert-col marking controls; renamed to "
                         "'non-targeting' for cell-eval")
    ap.add_argument("--eval-gene-col", default="ensembl_gene_id",
                    help="var column matching the encoder's gene vocabulary")
    ap.add_argument("--output-activation", default=None,
                    choices=["rectified_sigmoid", "softplus"],
                    help="required only when the checkpoint does not record one")
    ap.add_argument("--loss-type", default=None,
                    help="override the checkpoint's recorded loss; only affects "
                         "which module holds NB/ZINB dispersion")
    ap.add_argument("--calibrate", default="none", choices=["none", "quantile"],
                    help="per-gene quantile calibration fit on CONTROL cells only "
                         "(default: none). It is the dominant term in this "
                         "measurement -- report which setting produced a number.")
    ap.add_argument("--modes", default="injctrl",
                    help="comma-separated: injctrl (main), reconctrl (inflated)")
    ap.add_argument("--all-genes", action="store_true",
                    help="score every encoder gene, including those absent from "
                         "the eval file and therefore zero-filled. Off by "
                         "default: absent genes can never be significant in the "
                         "ground truth but can still enter the prediction's "
                         "top-N ranking and dilute overlap@N.")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--num-threads", type=int, default=16)
    ap.add_argument("--de-batch-size", type=int, default=None,
                    help="cell-eval --batch-size: (perturbation, gene) pairs "
                         "per DE task. cell-eval's default of 100 turns a "
                         "2,394-perturbation panel into ~212,000 pool tasks "
                         "and trips a shared-memory race in its DE backend. "
                         "Raise it for large panels; 2000 is a good start.")
    ap.add_argument("--de-real-cache", default=None,
                    help="directory of reusable ground-truth DE tables, keyed "
                         "by the target file's sha256. The real-side DE depends "
                         "only on the target, so arms scored against the same "
                         "one share it -- ~1.5 h of a ~2.5 h run on jurkat. A "
                         "target built from different data or a different gene "
                         "subset hashes differently and misses the cache.")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="seconds for one cell-eval subprocess. A panel with "
                         "thousands of perturbations needs far more than the "
                         "default; cell-eval is killed, not paused, at this limit.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    for label, path in (("checkpoint", args.checkpoint), ("eval data", args.eval_data)):
        if not os.path.isfile(path):
            ap.error(f"{label} not found: {path}")

    import torch

    from occam.decoder_train.trainer import DecoderTrainer

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    try:
        config = build_config(checkpoint, args)
    except ValueError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2
    del checkpoint

    print(f"decoder      : {config.decoder_hidden_dim} x {config.decoder_num_blocks}, "
          f"{config.decoder_output_activation}, loss {config.loss_type}")
    print(f"eval data    : {args.eval_data}")
    print(f"perturbation : obs['{args.pert_col}'], controls '{args.control_label}'")
    print(f"calibration  : {args.calibrate}"
          + ("  <- fit on control cells; declare this next to any number"
             if args.calibrate != "none" else ""))

    os.makedirs(args.output_dir, exist_ok=True)
    trainer = DecoderTrainer(config, device=args.device)
    trainer.load_decoder_weights(args.checkpoint)

    results = trainer.run_cell_eval_on_file(
        eval_path=args.eval_data,
        output_dir=args.output_dir,
        gene_col=args.eval_gene_col,
        pert_col=args.pert_col,
        control_label=args.control_label,
        batch_size=args.batch_size,
    )

    payload = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "eval_data": os.path.abspath(args.eval_data),
        "settings": {
            "decoder_hidden_dim": config.decoder_hidden_dim,
            "decoder_num_blocks": config.decoder_num_blocks,
            "output_activation": config.decoder_output_activation,
            "loss_type": config.loss_type,
            "pert_col": args.pert_col,
            "control_label": args.control_label,
            "calibrate": args.calibrate,
            "num_threads": args.num_threads,
            "de_batch_size": args.de_batch_size,
            "de_real_cache": args.de_real_cache,
            "present_genes_only": not args.all_genes,
            "epoch": trainer.current_epoch,
        },
        "metrics": results,
    }
    out_path = os.path.join(args.output_dir, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    if not results:
        print("\ncell-eval returned no metrics — see the messages above.",
              file=sys.stderr)
        return 1

    modes = [m for m in results if results[m]]
    width = max((len(m) for m in REPORTED), default=20)
    print(f"\n{'metric'.ljust(width)}  " + "  ".join(f"{m:>12s}" for m in modes))
    for metric in REPORTED:
        if not any(metric in results[m] for m in modes):
            continue
        cells = []
        for m in modes:
            v = results[m].get(metric)
            cells.append("           -" if v is None else f"{v:12.4f}")
        print(f"{metric.ljust(width)}  " + "  ".join(cells))
    print(f"\nwrote {out_path}")
    print("Read pearson_delta and discrimination_score_l2. Do NOT rank on "
          "overlap_at_N.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
