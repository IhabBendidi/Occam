#!/usr/bin/env python
"""Record checksums, sizes and provenance for the bootstrapped binary assets.

Written by ``scripts/bootstrap_assets.sh``; also runnable standalone to refresh
the manifest after a re-copy. Reads the checkpoints' embedded configs so the
manifest carries the *verified* recipe rather than the documented one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

ASSETS = [
    ("stage1_pretrain", "checkpoints/reference/occam_small_stage1_pretrain.pth"),
    ("stage2_finetune", "checkpoints/reference/occam_small_stage2_finetune.pth"),
    ("stage3_release", "checkpoints/release/occam_small_encoder_decoder.pth"),
    # Superseded 2026-09-16, retained because the golden fixtures pin it and it
    # is therefore the port-equivalence evidence. Not published.
    ("stage3_legacy", "checkpoints/legacy/occam_small_encoder_decoder_rectified_sigmoid.pth"),
]

# Fields worth pulling out of each checkpoint's embedded config, per stage.
CONFIG_FIELDS = {
    "stage1_pretrain": (
        "reconstruction_loss", "learning_rate", "weight_decay", "epochs", "batch_size",
        "model_size", "hidden_size", "mask_probability", "mask_loss_weight",
        "masked_data_weight", "mlw_schedule_start", "mlw_schedule_end",
        "mlw_schedule_warmup", "mlw_schedule_cooldown", "warmup_epochs",
        "lr_scheduler", "target_sum", "clamp_max_log1p", "data_mode", "val_split",
        "wandb_run_name",
    ),
    "stage2_finetune": (
        "recon_loss", "learning_rate", "epochs", "batch_size", "mask_probability",
        "mask_loss_weight", "masked_data_weight", "mlw_schedule_start",
        "mlw_schedule_end", "unmasking_mode", "unmasking_zero_at", "wandb_run_name",
    ),
    "stage3_release": (
        "loss_type", "lr", "eta_min", "weight_decay", "epochs", "batch_size",
        "decoder_hidden_dim", "decoder_num_blocks", "decoder_dropout",
        "decoder_se_reduction", "decoder_output_activation", "gradient_clip",
        "smooth_l1_beta", "freeze_encoder", "seed", "wandb_run_name",
    ),
}
CONFIG_FIELDS["stage3_legacy"] = CONFIG_FIELDS["stage3_release"]


def sha256(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def n_params(state_dict) -> int:
    total = 0
    for v in state_dict.values():
        n = 1
        for d in tuple(v.shape):
            n *= int(d)
        total += n
    return total


def describe(path: str, name: str) -> dict:
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    out: dict = {"top_level_keys": sorted(ck.keys())}

    for key in ("encoder_config", "decoder_config"):
        if key in ck:
            out[key] = dict(ck[key])
    for key in ("target_sum", "epoch", "current_epoch", "best_val_loss",
                "is_unified_checkpoint"):
        if key in ck:
            out[key] = ck[key]

    if "gene_names" in ck:
        out["n_genes"] = len(ck["gene_names"])

    for key in ("encoder_state_dict", "decoder_state_dict", "model_state_dict"):
        if key in ck:
            out[key] = {"n_tensors": len(ck[key]), "n_params": n_params(ck[key])}

    # The research tree aliased model_state_dict to the decoder in unified
    # checkpoints. Record it so the trap is documented, not rediscovered.
    if "decoder_state_dict" in ck and "model_state_dict" in ck:
        out["model_state_dict_is_decoder"] = (
            out["model_state_dict"] == out["decoder_state_dict"]
        )

    cfg = ck.get("config") or ck.get("training_config") or {}
    if not isinstance(cfg, dict):
        cfg = getattr(cfg, "__dict__", {}) or {}
    wanted = CONFIG_FIELDS.get(name, ())
    out["verified_config"] = {k: cfg[k] for k in wanted if k in cfg}
    # Paths inside a checkpoint's own config point at the machine that trained
    # it. Reduce them to basenames so no absolute path lands in a tracked file.
    for key, value in list(out["verified_config"].items()):
        if isinstance(value, str) and value.startswith("/"):
            out["verified_config"][key] = os.path.basename(value)

    # Stage 2 records its finetuning overlay separately.
    ft = ck.get("finetuning_config")
    if isinstance(ft, dict):
        out["verified_finetuning_config"] = {
            k: (os.path.basename(v) if isinstance(v, str) and v.startswith("/") else v)
            for k, v in ft.items() if v is not None
        }

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--skip-hash", action="store_true")
    args = ap.parse_args()

    manifest = {
        "schema": 1,
        "generated": datetime.now(timezone.utc).isoformat(),

        "provenance_warning": (
            "Stage 1 and stage 3 were written by colliding SLURM array tasks. The "
            "'verified_config' blocks below come from each checkpoint's own embedded "
            "config, which is the only primary record. Do not trust "
            "outputs/experiments/*/metadata.json or decoder_results/*/run_config.json "
            "in the source tree. See docs/reproducing_chain_a.md."
        ),
        "assets": {},
    }

    for name, rel in ASSETS:
        path = os.path.join(args.repo, rel)
        if not os.path.isfile(path):
            print(f"   {name}: MISSING ({rel}) -- skipped")
            continue
        entry = {"path": rel, "bytes": os.path.getsize(path)}  # repo-relative
        if not args.skip_hash:
            print(f"   {name}: hashing...")
            entry["sha256"] = sha256(path)
        entry.update(describe(path, name))
        manifest["assets"][name] = entry
        print(f"   {name}: {entry['bytes'] / 1e9:.2f} GB")

    bmdb = os.path.join(args.repo, "assets", "bmdb_database")
    if os.path.isdir(bmdb):
        manifest["bmdb_database"] = {
            fn: {
                "bytes": os.path.getsize(os.path.join(bmdb, fn)),
                "sha256": None if args.skip_hash else sha256(os.path.join(bmdb, fn)),
            }
            for fn in sorted(os.listdir(bmdb))
        }

    out = os.path.join(args.repo, "checkpoints", "MANIFEST.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, default=str)
    print(f"   wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
