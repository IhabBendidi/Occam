#!/usr/bin/env python
"""Prepare a stage-2 MAE checkpoint as the starting point of a further finetune.

    python scripts/finetune/make_finetune_init.py \
        --parent checkpoints/reference/occam_small_stage2_finetune.pth \
        --out runs/finetune_init/occam_small_stage2_mask0.1.pth

WHY THIS EXISTS
---------------
Under `unmasking.mode: gradual` the mask probability ramps down from the PARENT
checkpoint's `config.mask_probability`, never from the finetune config
(`occam.mae.trainer`, `UnmaskingSchedule(training_mask_prob=...)`). A stage-2 checkpoint
records 0.0 -- it ended fully unmasked -- so finetuning it again with the stage-2 recipe
silently applies NO masking at all.

This writes a copy whose `config.mask_probability` is `--mask-start` (default 0.1, the
value stage 2 itself started from), records why in `finetune_init_note`, sets
`parent_checkpoint_path`, and verifies every tensor of `model_state_dict` is unchanged.
The weights are not touched.
"""

from __future__ import annotations

import argparse
import os
import sys


def make_init(parent: str, out: str, mask_start: float = 0.1) -> dict:
    import torch

    ck = torch.load(parent, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ck or "config" not in ck:
        raise ValueError(f"{parent} is not a plain MAE checkpoint (needs model_state_dict + config)")
    if ck.get("is_unified_checkpoint"):
        raise ValueError(f"{parent} is a unified encoder+decoder checkpoint; its model_state_dict "
                         f"is the external decoder. Use the stage-2 MAE checkpoint instead.")
    cfg = ck["config"] if isinstance(ck["config"], dict) else dict(vars(ck["config"]))
    before = cfg.get("mask_probability")
    cfg["mask_probability"] = float(mask_start)
    ck["config"] = cfg
    ck["parent_checkpoint_path"] = os.path.abspath(parent)
    ck["finetune_init_note"] = (
        f"copy of {os.path.basename(parent)}; config.mask_probability {before} -> {mask_start} so "
        f"that gradual unmasking starts from it. Weights unchanged.")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save(ck, out)

    src = torch.load(parent, map_location="cpu", weights_only=False)["model_state_dict"]
    dst = torch.load(out, map_location="cpu", weights_only=False)["model_state_dict"]
    same = set(src) == set(dst) and all(torch.equal(src[k], dst[k]) for k in src)
    if not same:
        raise RuntimeError(f"{out}: weights differ from {parent}")
    return {"mask_probability_before": before, "mask_probability_after": float(mask_start),
            "n_tensors": len(dst)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parent", required=True, help="stage-2 MAE checkpoint (plain, not unified)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask-start", type=float, default=0.1)
    args = ap.parse_args(argv)
    info = make_init(args.parent, args.out, args.mask_start)
    print(f"wrote {args.out}: mask_probability {info['mask_probability_before']} -> "
          f"{info['mask_probability_after']}, {info['n_tensors']} tensors identical to the parent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
