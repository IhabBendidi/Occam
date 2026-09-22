#!/usr/bin/env python
"""Predict perturbed expression from control cells with a published Occam checkpoint.

WHAT THIS IS FOR
    The evaluation path (`scripts/eval_splithalf.py`) only ever predicts on a FROZEN
    benchmark panel. This script is the same model applied to an arbitrary `.h5ad` of
    control cells and an arbitrary list of perturbations, which is what anyone reusing the
    checkpoint actually needs.

WHAT IT DOES
    For every (control cell x requested perturbation) it emits one predicted profile.
    A label of the form "A+B" is a PAIR and is composed with `--compose`
    (default `decode_seq_mean`, the adopted default -- see docs/default_recipe.md).

    Output `.X` is in the model's own space: library-normalised to the checkpoint's
    `target_sum` and log1p'd. `--raw-counts` inverts that back to counts (expm1, then
    renormalise each cell to its original library size).

WHAT IT REFUSES TO GUESS
    * the gene axis -- taken from the checkpoint, never from the input file. Genes the
      encoder does not know are dropped; genes it knows that the file lacks are
      zero-filled. Both counts are printed, and `--max-zero-fill` aborts when the
      zero-filled fraction is implausible, because a wrong `--gene-col` produces a
      near-empty input that otherwise predicts happily and silently.
    * `target_sum` -- read from the checkpoint. A mismatch with training is silent and
      ruins the prediction.
    * the output activation -- read from the checkpoint; `load_model` refuses when absent
      rather than defaulting (a softplus decoder rebuilt as rectified_sigmoid is ~99%
      exact zeros and still loads cleanly).

USAGE
    python scripts/repro/occam_predict.py \
        --checkpoint runs/.../best_model.pth \
        --controls my_controls.h5ad \
        --perts NRF1 CBL "CBL+CNN1" \
        --out predictions.h5ad

    # every perturbation the checkpoint knows that is also listed in a file
    python scripts/repro/occam_predict.py ... --perts-file perts.txt

⚠️ The control cells must be RAW COUNTS unless the checkpoint's encoder was trained on
   already-transformed data (`skip_normalize` / `skip_log1p` in its config, which this
   script reads and honours). Passing log1p data to a raw-count encoder is silent.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="a vcell best_model.pth")
    ap.add_argument("--controls", required=True,
                    help=".h5ad of CONTROL cells to predict from")
    ap.add_argument("--perts", nargs="*", default=[],
                    help="perturbation labels; 'A+B' is a pair")
    ap.add_argument("--perts-file", default=None,
                    help="one label per line; merged with --perts")
    ap.add_argument("--out", required=True, help="output .h5ad")
    ap.add_argument("--compose", default="decode_seq_mean",
                    help="composition mode for PAIRS; no effect on single perturbations, "
                         "where every reconstruction-based mode is identical")
    ap.add_argument("--gene-col", default=None,
                    help="column in .var holding gene ids; default = var_names. Use "
                         "ensembl_gene_id when the encoder vocabulary is Ensembl (it is, "
                         "for both published checkpoints)")
    ap.add_argument("--symbol-map", default="",
                    help="optional symbol->Ensembl csv (columns gene_symbol, "
                         "ensembl_gene_id) for a symbol-keyed input file")
    ap.add_argument("--max-cells", type=int, default=0,
                    help="use at most N control cells (0 = all). One prediction per "
                         "cell x perturbation, so this bounds the output size")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--raw-counts", action="store_true",
                    help="invert the log1p/library normalisation on the output")
    ap.add_argument("--max-zero-fill", type=float, default=0.95,
                    help="abort if more than this fraction of encoder genes had to be "
                         "zero-filled; 0 disables the check")
    ap.add_argument("--output-activation", default=None,
                    help="only needed for checkpoints that do not record it")
    args = ap.parse_args()

    import anndata as ad
    import numpy as np
    import torch

    from eval_splithalf import load_model  # noqa: E402  (scripts/ is on sys.path)
    from occam.vcell.dataset import resolve_pert_indices, set_combo_decomposition
    from occam.vcell.model import VirtualCellModel
    from occam.vcell.panel.panel_io import _align_to_encoder, _preprocess

    if args.compose not in VirtualCellModel.COMPOSE_MODES:
        raise SystemExit(f"--compose {args.compose!r} is not a mode; expected "
                         f"{list(VirtualCellModel.COMPOSE_MODES)}")

    labels = list(args.perts)
    if args.perts_file:
        with open(args.perts_file) as fh:
            labels += [ln.strip() for ln in fh if ln.strip()
                       and not ln.startswith("#")]
    if not labels:
        raise SystemExit("no perturbations requested; pass --perts and/or --perts-file")

    trainer, ckpt = load_model(args.checkpoint, args.device, args.output_activation)
    model, pert2id, device = trainer.model, trainer.pert2id, trainer.device
    encoder_genes = list(ckpt["gene_names"])
    target_sum = float(ckpt["target_sum"])
    tcfg = ckpt.get("training_config") or {}
    skip_norm = bool(tcfg.get("skip_normalize", False))
    skip_log = bool(tcfg.get("skip_log1p", False))
    print(f"[predict] encoder vocabulary {len(encoder_genes):,} genes, "
          f"target_sum {target_sum:g}, skip_normalize={skip_norm}, skip_log1p={skip_log}")

    # `"A+B"` is not a graph node; without this every pair resolves to None and is
    # silently dropped -- the failure that emptied a whole validation split once.
    set_combo_decomposition(True)
    resolved, unknown = [], []
    for lab in labels:
        idx = resolve_pert_indices(lab, pert2id)
        (resolved if idx is not None else unknown).append((lab, idx))
    if unknown:
        print(f"[predict] REFUSING: {len(unknown)} label(s) are not in this checkpoint's "
              f"vocabulary of {len(pert2id):,}:", file=sys.stderr)
        for lab, _ in unknown[:20]:
            missing = [p for p in str(lab).split("+") if p and p not in pert2id]
            print(f"    {lab}   (unresolved component(s): {missing or [lab]})",
                  file=sys.stderr)
        return 2
    n_pairs = sum(1 for _, idx in resolved if len(idx) > 1)
    print(f"[predict] {len(resolved)} perturbation(s): {len(resolved) - n_pairs} single, "
          f"{n_pairs} pair(s). Composition mode for pairs: {args.compose}")

    adata = ad.read_h5ad(args.controls)
    if args.gene_col:
        if args.gene_col not in adata.var.columns:
            raise SystemExit(f"--gene-col {args.gene_col!r} not in .var "
                             f"({list(adata.var.columns)[:10]})")
        file_genes = [str(g) for g in adata.var[args.gene_col]]
    else:
        file_genes = [str(g) for g in adata.var_names]

    enc_cols, file_cols = _align_to_encoder(file_genes, encoder_genes, args.symbol_map)
    n_zero_fill = len(encoder_genes) - len(enc_cols)
    frac = n_zero_fill / len(encoder_genes)
    print(f"[predict] gene alignment: {len(enc_cols):,} matched, "
          f"{len(file_genes) - len(file_cols):,} input genes dropped, "
          f"{n_zero_fill:,} encoder genes zero-filled ({frac:.1%})")
    if args.max_zero_fill and frac > args.max_zero_fill:
        raise SystemExit(
            f"{frac:.1%} of the encoder's genes had to be zero-filled, above "
            f"--max-zero-fill {args.max_zero_fill:.0%}. The input is being read on the "
            f"wrong axis -- check --gene-col (the published checkpoints use Ensembl ids) "
            f"or pass --symbol-map. Override with --max-zero-fill 0 if this is intended.")

    X = adata.X
    n_cells = X.shape[0] if not args.max_cells else min(args.max_cells, X.shape[0])
    raw = X[:n_cells]
    raw = np.asarray(raw.todense() if hasattr(raw, "todense") else raw, dtype=np.float32)
    libsize = raw[:, file_cols].sum(axis=1, keepdims=True)

    # Preprocess on the FILE's own columns first, then scatter into encoder order. The
    # other order would divide by a library size computed over zero-filled columns.
    proc = _preprocess(raw[:, file_cols], target_sum, skip_norm, skip_log)
    ctrl = np.zeros((n_cells, len(encoder_genes)), dtype=np.float32)
    ctrl[:, enc_cols] = proc
    print(f"[predict] {n_cells:,} control cell(s) x {len(resolved)} perturbation(s) "
          f"= {n_cells * len(resolved):,} predictions")

    model.eval()
    preds, obs_pert, obs_cell = [], [], []
    with torch.no_grad():
        for lab, idx in resolved:
            for start in range(0, n_cells, args.batch_size):
                chunk = ctrl[start:start + args.batch_size]
                ct = torch.from_numpy(chunk).to(device)
                out = model.forward_composed(ct, [list(idx)] * chunk.shape[0],
                                             args.compose)
                preds.append(out["prediction"].float().cpu().numpy())
                obs_pert += [lab] * chunk.shape[0]
                obs_cell += list(range(start, start + chunk.shape[0]))
    P = np.concatenate(preds, axis=0)

    if args.raw_counts:
        if skip_log:
            print("[predict] --raw-counts: encoder is skip_log1p, nothing to invert")
        else:
            P = np.expm1(P)
        if not skip_norm:
            s = P.sum(axis=1, keepdims=True)
            s = np.where(s > 0, s, 1.0)
            lib = np.tile(libsize, (len(resolved), 1))
            P = P / s * lib
        print("[predict] output space: RAW COUNTS (expm1 + original library size)")
    else:
        print(f"[predict] output space: log1p at target_sum {target_sum:g}")

    import pandas as pd
    out = ad.AnnData(
        X=P,
        obs=pd.DataFrame({"target_gene": obs_pert, "control_row": obs_cell},
                         index=[f"{p}__{c}" for p, c in zip(obs_pert, obs_cell)]),
        var=pd.DataFrame(index=encoder_genes),
    )
    out.uns["occam_predict"] = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "compose": args.compose,
        "target_sum": target_sum,
        "space": "raw_counts" if args.raw_counts else "log1p",
        "n_zero_filled_genes": int(n_zero_fill),
        "benchmark": (ckpt.get("benchmark") or {}).get("id", ""),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out.write_h5ad(args.out)
    print(f"[predict] wrote {args.out}  {out.shape[0]:,} x {out.shape[1]:,}")
    print("[predict] obs['target_gene'] is the perturbation; .uns['occam_predict'] "
          "records the checkpoint, mode and output space")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
