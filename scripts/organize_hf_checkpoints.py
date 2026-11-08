#!/usr/bin/env python3
"""Add original Occam experiment names to the downloaded Hub mirrors.

The complete private and public Hub repositories stay in ``private_repo`` and
``public_repo``.  Canonical Occam paths are hard links, so checkpoints are not
copied or counted twice on disk.
"""

import argparse
import json
import os
from pathlib import Path


RHJ_VARIANTS = {
    "gostring": "ABL_A_DC",
    "all_sources": "ABL_N_ALL",
}
NORMAN_VARIANTS = {
    "gostring_dcos0.0255": "NF32DCA",
    "gostring_dcos0.32977": "NF32DC",
    "all_sources_dcos0.0255": "NF32ALLA",
    "all_sources_dcos0.32977": "NF32ALL",
}
SEEDS = range(42, 47)


def link_checkpoint(root, source, destination, records):
    source_path = root / source
    destination_path = root / destination
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        if not os.path.samefile(source_path, destination_path):
            raise FileExistsError(destination_path)
    else:
        os.link(source_path, destination_path)
    records.append(
        {
            "source": str(source),
            "original_name": str(destination),
            "size_bytes": source_path.stat().st_size,
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("checkpoints/local_artifacts"))
    args = parser.parse_args()
    root = args.root
    records = []

    for hub_name, original_name in RHJ_VARIANTS.items():
        for seed in SEEDS:
            source_dir = Path("private_repo/checkpoints/occam_seeds/rhj_fewshot") / hub_name / f"seed_{seed}"
            target_dir = Path("occam_variants/rhj_fewshot") / original_name / f"seed_{seed}"
            for filename in ("best_model.pth", "resolved_config.yaml"):
                link_checkpoint(root, source_dir / filename, target_dir / filename, records)

    for hub_name, original_name in NORMAN_VARIANTS.items():
        for seed in SEEDS:
            if original_name == "NF32ALLA":
                source = (Path("private_repo/checkpoints/vcell/norman_combo_NF32ALLA") /
                          f"seed_{seed}" / "best_model.pth")
            else:
                source = (Path("public_repo/ablations/occam/norman_combo") / original_name /
                          f"seed_{seed}" / "best_model.pth")
            if not (root / source).is_file():
                source = (Path("public_repo/ablations/occam/norman_combo") / original_name /
                          f"seed_{seed}" / "best_model.pth")
                if not (root / source).is_file():
                    # Existing mirrors downloaded before the Hub relocation.
                    source = (Path("public_repo/razor/norman_combo") / hub_name /
                              f"seed_{seed}" / "checkpoint.pth")
            target = Path("occam_variants/norman_combo") / original_name / f"seed_{seed}" / "best_model.pth"
            link_checkpoint(root, source, target, records)

    manifest = {
        "mirrors": {
            "private_repo": "bendidiihab/Occam",
            "public_repo": "bendidiihab/Occam-competitor-checkpoints",
        },
        "occam_variants": records,
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "README.md").write_text(
        "# Local Hugging Face archives\n\n"
        "`private_repo/` is the complete private Occam mirror, including benchmark and "
        "evaluation data. `public_repo/` is the complete public competitor checkpoint "
        "mirror. Competitor checkpoints keep their Hub layout under `public_repo/`.\n\n"
        "`occam_variants/` exposes the original experiment names for the ten RHJ and "
        "twenty Norman Occam seed checkpoints. These files are hard links to the mirrors, "
        "so they use no additional checkpoint storage. See `MANIFEST.json` for every "
        "source-to-original-name mapping.\n"
    )
    print(f"Created or verified {len(records)} original-name links under {root / 'occam_variants'}")


if __name__ == "__main__":
    main()
