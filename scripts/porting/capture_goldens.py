#!/usr/bin/env python
"""Capture numerical-equivalence fixtures using the ORIGINAL pre-rename code.

RUN THIS FROM THE SOURCE REPO (`scMAE-main`), BEFORE copying or renaming
anything. It is the blocking prerequisite for the whole port: these fixtures are
the only thing that makes "preserve numerical behaviour exactly" a testable
claim rather than an assertion.

    cd <research checkout>
    PYTHONHASHSEED=0 python /path/to/Occam/scripts/capture_goldens.py \
        --out /path/to/Occam/golden

Two fixtures are written:

`release_forward.npz`
    A deterministic forward pass through the released unified encoder+decoder,
    loaded via the original `external_decoder.load_decoder` path. Stores the
    exact input alongside the outputs, so the replay in
    `scripts/verify_checkpoint.py` needs no shared RNG assumptions.

`release_meta.json`
    Everything about the released checkpoint that Occam asserts: config dicts,
    parameter counts, key inventories, gene-name digest. Recorded from the file
    itself rather than transcribed, because the surrounding `run_config.json`
    belongs to a different SLURM array task (see docs/reproducing_chain_a.md).

Everything runs on CPU in fp32 under `torch.no_grad()` and `model.eval()`, so
the fixture does not depend on GPU nondeterminism or cuDNN algorithm selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

# The released decoder. Geometry 2048x5 -- NOT the code default of 4.
DEFAULT_RELEASE = (
    "decoder_results/ihabnet_decoder_20260501_172850/checkpoints/best_decoder.pth"
)

# Fixed forward-pass input. 8 cells x 44,728 genes of U[0,10), which spans the
# range of log1p(normalised counts) the decoder actually sees (upper bound
# log1p(2000) ~= 7.6) plus some headroom above it.
N_CELLS = 8
INPUT_SCALE = 10.0
INPUT_SEED = 0


def _sha256_list(items) -> str:
    h = hashlib.sha256()
    for item in items:
        h.update(str(item).encode())
        h.update(b"\0")
    return h.hexdigest()


def _file_sha256(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _count_params(state_dict) -> int:
    return int(sum(int(np.prod(tuple(v.shape))) for v in state_dict.values()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-repo", default=".",
                    help="scMAE-main checkout to import the original modules from")
    ap.add_argument("--checkpoint", default=DEFAULT_RELEASE,
                    help="released unified checkpoint, relative to --source-repo")
    ap.add_argument("--out", required=True, help="output directory (Occam/golden)")
    ap.add_argument("--skip-file-hash", action="store_true",
                    help="skip the 928 MB sha256 pass (faster; leaves the field null)")
    args = ap.parse_args()

    repo = os.path.abspath(args.source_repo)
    if not os.path.isfile(os.path.join(repo, "external_decoder.py")):
        print(f"ERROR: {repo} does not look like scMAE-main "
              f"(no external_decoder.py). Pass --source-repo.", file=sys.stderr)
        return 2
    sys.path.insert(0, repo)

    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(repo, ckpt_path)
    if not os.path.isfile(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)

    # Import the ORIGINAL modules. Any rename applied to these invalidates the
    # fixture, which is exactly why this script must run before the copy.
    import external_decoder as ext  # noqa: E402
    from model import AutoEncoder  # noqa: E402

    print(f"[goldens] source repo : {repo}")
    print(f"[goldens] checkpoint  : {ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    enc_cfg = dict(raw["encoder_config"])
    dec_cfg = dict(raw["decoder_config"])
    gene_names = list(raw["gene_names"])
    n_genes = len(gene_names)

    print(f"[goldens] encoder_config: {enc_cfg}")
    print(f"[goldens] decoder_config: {dec_cfg}")
    print(f"[goldens] genes: {n_genes}")

    # ---- decoder, via the original loader ---------------------------------
    decoder, loaded_genes, gene_to_index, model_config = ext.load_decoder(
        ckpt_path, device="cpu"
    )
    decoder.eval()
    assert list(loaded_genes) == gene_names, "load_decoder returned different gene_names"

    # ---- encoder ----------------------------------------------------------
    # The unified checkpoint stores only the encoder half, so build the full
    # AutoEncoder and load non-strictly, exactly as the research tree's
    # DecoderTrainer._load_encoder does, then keep .encoder.
    ae = AutoEncoder(
        num_genes=enc_cfg["num_genes"],
        hidden_size=enc_cfg["hidden_size"],
        model_size=enc_cfg["model_size"],
        dropout=0.0,
    )
    enc_sd = raw["encoder_state_dict"]
    missing, unexpected = ae.load_state_dict(enc_sd, strict=False)
    print(f"[goldens] encoder load: {len(missing)} missing, {len(unexpected)} unexpected")
    assert not unexpected, f"unexpected encoder keys: {sorted(unexpected)[:5]}"
    ae.eval()

    # ---- deterministic forward pass ---------------------------------------
    gen = torch.Generator().manual_seed(INPUT_SEED)
    x = torch.rand(N_CELLS, n_genes, generator=gen, dtype=torch.float32) * INPUT_SCALE

    with torch.no_grad():
        latent = ae.encoder(x)
        recon = decoder(latent)

    latent_np = latent.detach().cpu().numpy()
    recon_np = recon.detach().cpu().numpy() if torch.is_tensor(recon) else None
    assert recon_np is not None, "decoder returned a non-tensor; hurdle/pi_head mode is not expected here"

    print(f"[goldens] latent {latent_np.shape} "
          f"[{latent_np.min():.6f}, {latent_np.max():.6f}] mean {latent_np.mean():.6f}")
    print(f"[goldens] recon  {recon_np.shape} "
          f"[{recon_np.min():.6f}, {recon_np.max():.6f}] mean {recon_np.mean():.6f}")

    exact_zero_frac = float((recon_np == 0.0).mean())
    const_genes = int((recon_np.std(axis=0) == 0.0).sum())
    print(f"[goldens] recon exact-zero fraction : {exact_zero_frac:.6f}")
    print(f"[goldens] recon zero-variance genes : {const_genes} / {n_genes}")

    npz_path = os.path.join(args.out, "release_forward.npz")
    np.savez_compressed(
        npz_path,
        x=x.numpy(),
        latent=latent_np,
        recon=recon_np,
        input_seed=np.int64(INPUT_SEED),
        input_scale=np.float32(INPUT_SCALE),
        n_cells=np.int64(N_CELLS),
        n_genes=np.int64(n_genes),
    )
    print(f"[goldens] wrote {npz_path} ({os.path.getsize(npz_path) / 1e6:.1f} MB)")

    # ---- metadata ---------------------------------------------------------
    train_cfg = raw.get("training_config")
    train_cfg = dict(train_cfg) if isinstance(train_cfg, dict) else None
    if train_cfg:
        # A checkpoint's own config records paths on the machine that trained it.
        # Reduce them to basenames; the fixture is tracked.
        for key, value in list(train_cfg.items()):
            if isinstance(value, str) and value.startswith("/"):
                train_cfg[key] = os.path.basename(value)

    meta = {
        # Basenames only. Where the source happened to live is not provenance
        # anyone else can use, and no absolute path belongs in a tracked file.
        "source_checkpoint": os.path.basename(ckpt_path),
        "source_checkpoint_bytes": os.path.getsize(ckpt_path),
        "source_checkpoint_sha256": None if args.skip_file_hash else _file_sha256(ckpt_path),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "top_level_keys": sorted(raw.keys()),
        "is_unified_checkpoint": bool(raw.get("is_unified_checkpoint", False)),
        "encoder_config": enc_cfg,
        "decoder_config": dec_cfg,
        "target_sum": raw.get("target_sum"),
        "epoch": raw.get("epoch"),
        "best_val_loss": raw.get("best_val_loss"),
        "n_genes": n_genes,
        "gene_names_sha256": _sha256_list(gene_names),
        "gene_names_head": gene_names[:5],
        "gene_names_tail": gene_names[-5:],
        "encoder_state_dict": {
            "n_tensors": len(raw["encoder_state_dict"]),
            "n_params": _count_params(raw["encoder_state_dict"]),
            "keys": sorted(raw["encoder_state_dict"].keys()),
        },
        "decoder_state_dict": {
            "n_tensors": len(raw["decoder_state_dict"]),
            "n_params": _count_params(raw["decoder_state_dict"]),
        },
        # The research tree aliased model_state_dict to the DECODER, not the
        # whole model. Recorded so the trap is documented, not rediscovered.
        "model_state_dict_is_decoder": (
            "model_state_dict" in raw
            and len(raw["model_state_dict"]) == len(raw["decoder_state_dict"])
            and _count_params(raw["model_state_dict"]) == _count_params(raw["decoder_state_dict"])
        ),
        "encoder_load_missing_keys": sorted(missing),
        "forward": {
            "input_seed": INPUT_SEED,
            "input_scale": INPUT_SCALE,
            "n_cells": N_CELLS,
            "latent_shape": list(latent_np.shape),
            "recon_shape": list(recon_np.shape),
            "latent_sum": float(latent_np.sum()),
            "recon_sum": float(recon_np.sum()),
            "recon_exact_zero_fraction": exact_zero_frac,
            "recon_zero_variance_genes": const_genes,
        },
        # The activation is NOT recorded in the checkpoint. This is what
        # load_decoder resolved it to, which is the code default -- not
        # evidence. verify_checkpoint.py determines it empirically.
        "output_activation_as_loaded": model_config.get("output_activation", "rectified_sigmoid"),
        "output_activation_recorded_in_checkpoint": dec_cfg.get("output_activation"),
        "training_config": train_cfg,
    }

    meta_path = os.path.join(args.out, "release_meta.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)
    print(f"[goldens] wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
