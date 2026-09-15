#!/usr/bin/env python
"""Copy a file out of the research tree, applying the Occam rename.

Used by scripts/port_all.sh to perform rename passes A and B mechanically and
reproducibly, so the port is auditable rather than a pile of hand edits.

    python scripts/port_file.py SRC DST [--check-only]

Rules are applied longest-pattern-first, so `train_ihabnet_decoder` is rewritten
before `train_ihabnet`, which is rewritten before the bare `ihabnet`. Identifier
rules use word boundaries; prose and string rules do not.

`--check-only` reports what would change without writing, and exits non-zero if
any banned token would survive.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

# (pattern, replacement, word_boundary)
# ORDER MATTERS. Longest / most specific first.
RULES = [
    # ---- Pass A: identifiers -------------------------------------------
    ("OptimizedIhabNetDataset", "OccamChunkedDataset", True),
    ("IhabNetFeatureExtractor", "OccamFeatureExtractor", True),
    ("IhabNetReconstruction", "OccamReconstructor", True),
    ("IhabNetAdataDataset", "OccamAdataDataset", True),
    ("IhabNetPreprocessor", "OccamPreprocessor", True),
    ("IhabNetInference", "OccamEmbedder", True),
    ("IhabNetTransform", "OccamTransform", True),
    ("IhabNetTrainer", "OccamTrainer", True),

    ("train_ihabnet_decoder", "run_decoder", True),
    ("create_ihabnet_model", "create_occam_model", True),
    ("finetune_ihabnet", "run_finetune", True),
    ("train_ihabnet", "run_pretrain", True),

    ("DecoderTrainingConfig", "DecoderTrainConfig", True),
    ("FinetuningConfig", "FinetuneConfig", True),
    ("TrainingConfig", "PretrainConfig", True),

    # ---- Pass C: the persisted key --------------------------------------
    # _compat.migrate_config_dict reads the old names off disk; the code uses
    # the new ones.
    ("ihabnet_checkpoint_path", "parent_checkpoint_path", True),
    ("ihabnet_checkpoint", "encoder_checkpoint", True),

    # ---- Pass B: env vars, module names, strings ------------------------
    ("IHABNET_BMDB_DIR", "OCCAM_BMDB_DIR", False),
    ("IHABNET_EVAL_H5AD", "OCCAM_EVAL_H5AD", False),
    ("IHABNET_EVAL_FREQ", "OCCAM_EVAL_FREQ", False),
    ("SCMAE_BMDB_DIR", "OCCAM_BMDB_DIR", False),
    ("SCMAE_EVAL_H5AD", "OCCAM_EVAL_H5AD", False),
    ("SCMAE_EVAL_FREQ", "OCCAM_EVAL_FREQ", False),

    ("reconstruct_ihabnet", "occam_reconstruct", False),
    ("infer_ihabnet", "occam_embed", False),

    ("X_ihabnet", "X_occam", False),

    ("ihabnet_finetuning", "occam_finetune", False),
    ("ihabnet_training", "occam_pretrain", False),
    ("ihabnet_decoder", "occam_decoder", False),
    ("ihabnet_vcell", "occam_vcell", False),

    # ---- Prose / bare tokens, last --------------------------------------
    ("IhabNet", "Occam", False),
    ("ihabnet", "occam", False),
    ("IHABNET", "OCCAM", False),
    ("scMAE", "Occam", False),
    ("ScMAE", "Occam", False),
    ("scmae", "occam", False),
    ("SCMAE", "OCCAM", False),
]

# Flat top-level modules in the research tree -> their Occam package paths.
# Applied BEFORE the rename rules, anchored to import statements only, so a
# variable or string that happens to be called `datasets` is untouched.
IMPORT_RULES = {
    "losses": "occam.losses",
    "gpl_loss": "occam.gpl_loss",
    "util": "occam.util",
    "transforms": "occam.transforms",
    "model": "occam.model",
    "datasets": "occam.data.datasets",
    "optimized_chunked": "occam.data.chunked",
    "external_decoder": "occam.decoder.external",
    "decoder_discriminator": "occam.decoder.discriminator",
    "output_manager": "occam.run_dir",
    "evaluation_utils": "occam.evaluation.perturbation",
    "evaluation": "occam.evaluation",
    "training_pipeline": "occam.mae.legacy_pipeline",
    "decoder_training_pipeline": "occam.decoder_train.legacy_pipeline",
    "inference_pipeline": "occam.inference.embed",
    "reconstruction_pipeline": "occam.inference.reconstruct",
    "virtual_cell": "occam.vcell",
}


def apply_import_rules(text: str) -> str:
    """Rewrite `from X import ...` / `import X` for the flat research-tree modules."""
    for old, new in sorted(IMPORT_RULES.items(), key=lambda kv: -len(kv[0])):
        # from X import ...   /   from X.sub import ...
        text = re.sub(
            r"^(\s*from\s+)" + re.escape(old) + r"(\s|\.)",
            lambda m, new=new: m.group(1) + new + m.group(2),
            text,
            flags=re.MULTILINE,
        )
        # import X   /   import X.sub   (but not `import X_other`)
        text = re.sub(
            r"^(\s*import\s+)" + re.escape(old) + r"(\s|\.|$)",
            lambda m, new=new: m.group(1) + new + m.group(2),
            text,
            flags=re.MULTILINE,
        )
    return text


BANNED = re.compile(r"ihabnet|scmae", re.IGNORECASE)

# Lines carrying this marker are allowed to keep a banned token (the compat
# shim's lookup tables, and provenance documentation).
ALLOW_MARKER = "noqa: legacy-name"


def apply_rules(text: str) -> str:
    text = apply_import_rules(text)
    for pattern, replacement, word in RULES:
        if word:
            text = re.sub(r"\b" + re.escape(pattern) + r"\b", replacement, text)
        else:
            text = text.replace(pattern, replacement)
    return text


def surviving_banned(text: str) -> list[tuple[int, str]]:
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if ALLOW_MARKER in line:
            continue
        if BANNED.search(line):
            out.append((i, line.rstrip()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    with open(args.src, encoding="utf-8") as fh:
        original = fh.read()
    ported = apply_rules(original)

    before = len(BANNED.findall(original))
    after = len(BANNED.findall(ported))
    leftovers = surviving_banned(ported)

    if not args.quiet:
        print(f"{os.path.basename(args.src)}: {before} banned tokens -> {after}")
    for lineno, line in leftovers:
        print(f"  LEFTOVER {args.dst}:{lineno}: {line[:120]}", file=sys.stderr)

    if not args.check_only:
        os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
        with open(args.dst, "w", encoding="utf-8") as fh:
            fh.write(ported)

    return 1 if leftovers else 0


if __name__ == "__main__":
    raise SystemExit(main())
