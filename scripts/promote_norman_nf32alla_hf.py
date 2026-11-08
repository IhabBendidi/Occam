#!/usr/bin/env python3
"""Promote NF32ALLA and file the previous Norman checkpoints under ablations.

Dry run by default. ``--apply`` uploads all five official NF32ALLA seeds to the
private Occam repo, then uses server-side copies and atomic deletes to move the
old NQ02 release and the other public ``razor`` Occam seeds into ablation paths.
The source bytes are checked before upload and destination LFS hashes are
checked before the next move. Re-running after an interruption is safe.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import (
    CommitOperationCopy,
    CommitOperationDelete,
    HfApi,
)


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = "bendidiihab/Occam"
PUBLIC = "bendidiihab/Occam-competitor-checkpoints"
SOURCE_DIR = ROOT / "checkpoints/local_artifacts/occam_variants/norman_combo/NF32ALLA"
CARD = ROOT / "checkpoints/cards/norman_combo_NF32ALLA_CARD.md"
HISTORICAL_CARD = ROOT / "checkpoints/cards/norman_combo_NQ02_CARD.md"
SCORES = ROOT / "data/release/norman_allsource"
OFFICIAL = "checkpoints/vcell/norman_combo_NF32ALLA"
SHA256 = {
    42: "2c73b717bb5391cd4d1dec4f76fc22fbd680a147a124442db354ff8c87597af8",
    43: "e18fff6e6c676f9daad9b7ba0828e747b39e929b77e791b0ce2e5429bb25022d",
    44: "c181809206fa1e1b06c9f5837696e6a11363e71fbf726e3da6c85953c353660a",
    45: "f54ab574a80f725171fe6935ae5a889da27ed2a845327fa30551f8f8a268c0fd",
    46: "b37bb66244378ea32709e3f7424c05af1b397e804762ca0e81483a42fcba0720",
}
VARIANTS = {
    "gostring_dcos0.0255": "NF32DCA",
    "gostring_dcos0.32977": "NF32DC",
    "all_sources_dcos0.32977": "NF32ALL",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def files(api: HfApi, repo: str) -> set[str]:
    return set(api.list_repo_files(repo, repo_type="model"))


def lfs_sha(api: HfApi, repo: str, path: str) -> str | None:
    info = api.get_paths_info(repo, path, repo_type="model", expand=True)
    if not info:
        return None
    lfs = info[0].lfs
    return lfs.get("sha256") if lfs else None


def move(api: HfApi, repo: str, pairs: list[tuple[str, str]], label: str) -> None:
    current = files(api, repo)
    operations = []
    for source, target in pairs:
        if source in current:
            if target in current:
                raise RuntimeError(f"Both source and destination exist: {source} -> {target}")
            operations.extend((CommitOperationCopy(source, target), CommitOperationDelete(source)))
        elif target not in current:
            raise RuntimeError(f"Neither source nor destination exists: {source} -> {target}")
    if operations:
        api.create_commit(repo, operations=operations, repo_type="model",
                          commit_message=f"File Norman Occam {label} checkpoints under ablations")
    after = files(api, repo)
    for source, target in pairs:
        if source in after or target not in after:
            raise RuntimeError(f"Move did not finish: {source} -> {target}")
    print(f"Verified {label}: {len(pairs)} paths under ablations")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="perform the Hub changes")
    args = parser.parse_args()
    if not CARD.is_file():
        raise FileNotFoundError("Official checkpoint card is missing")
    for seed, expected in SHA256.items():
        source = SOURCE_DIR / f"seed_{seed}/best_model.pth"
        if not source.is_file():
            raise FileNotFoundError(source)
        got = digest(source)
        if got != expected:
            raise RuntimeError(f"NF32ALLA seed {seed} SHA-256 mismatch: {got}")
        print(f"Verified NF32ALLA seed {seed} local SHA-256: {got}")

    api = HfApi()
    private_info = api.repo_info(PRIVATE, repo_type="model")
    public_info = api.repo_info(PUBLIC, repo_type="model")
    if not private_info.private or public_info.private:
        raise RuntimeError("Unexpected Hub repository visibility")
    private_files, public_files = files(api, PRIVATE), files(api, PUBLIC)
    old = [(f"checkpoints/vcell/norman_combo_NQ02/{name}",
            f"ablations/occam/norman_combo/NQ02/seed_42/{name}")
           for name in ("best_model.pth", "CARD.md")]
    public_moves = {}
    for source_variant, variant in VARIANTS.items():
        public_moves[variant] = [
            (f"razor/norman_combo/{source_variant}/seed_{seed}/checkpoint.pth",
             f"ablations/occam/norman_combo/{variant}/seed_{seed}/best_model.pth")
            for seed in range(42, 47)
        ]
    for source, target in old:
        if source not in private_files and target not in private_files:
            raise RuntimeError(f"Missing historical checkpoint: {source}")
    for pairs in public_moves.values():
        for source, target in pairs:
            if source not in public_files and target not in public_files:
                raise RuntimeError(f"Missing public Occam checkpoint: {source}")

    print(f"Official: NF32ALLA seeds 42–46 -> {PRIVATE}/{OFFICIAL}/seed_<seed>/best_model.pth")
    print(f"Ablations: {PRIVATE}/NQ02 (2 files), {PUBLIC}/razor (15 other-variant files)"
          " -> ablations/occam/norman_combo/<original arm>/seed_<seed>/")
    print("After all five private checks pass, remove the duplicate public NF32ALLA seed files")
    if not args.apply:
        print("Dry run: no remote changes")
        return

    checkpoint_path = f"{OFFICIAL}/best_model.pth"
    card_path = f"{OFFICIAL}/CARD.md"
    if checkpoint_path not in private_files:
        api.upload_file(path_or_fileobj=str(SOURCE_DIR / "seed_42/best_model.pth"), path_in_repo=checkpoint_path,
                        repo_id=PRIVATE, repo_type="model",
                        commit_message="Publish NF32ALLA as official Norman virtual cell")
    if lfs_sha(api, PRIVATE, checkpoint_path) != SHA256[42]:
        raise RuntimeError("Published NF32ALLA checkpoint LFS hash differs from local checkpoint")
    print("Verified seed-42 convenience path on private Hub")
    for seed, expected in SHA256.items():
        path = f"{OFFICIAL}/seed_{seed}/best_model.pth"
        if path not in files(api, PRIVATE):
            api.upload_file(path_or_fileobj=str(SOURCE_DIR / f"seed_{seed}/best_model.pth"),
                            path_in_repo=path, repo_id=PRIVATE, repo_type="model",
                            commit_message=f"Publish official NF32ALLA seed {seed}")
        if lfs_sha(api, PRIVATE, path) != expected:
            raise RuntimeError(f"Official NF32ALLA seed {seed} LFS hash mismatch")
        print(f"Verified official NF32ALLA seed {seed} on private Hub")
    api.upload_file(path_or_fileobj=str(CARD), path_in_repo=card_path,
                    repo_id=PRIVATE, repo_type="model",
                    commit_message="Document five official NF32ALLA Norman seeds")
    if "final/norman_allsource/coverage.csv" not in files(api, PRIVATE):
        api.upload_folder(folder_path=str(SCORES), path_in_repo="final/norman_allsource",
                          repo_id=PRIVATE, repo_type="model",
                          commit_message="Publish complete Norman all-source five-seed scores")

    move(api, PRIVATE, old, "NQ02")
    api.upload_file(path_or_fileobj=str(HISTORICAL_CARD),
                    path_in_repo="ablations/occam/norman_combo/NQ02/seed_42/CARD.md",
                    repo_id=PRIVATE, repo_type="model",
                    commit_message="Mark NQ02 Norman checkpoint as historical ablation")
    for variant, pairs in public_moves.items():
        move(api, PUBLIC, pairs, variant)
    public_nf32alla = [
        f"ablations/occam/norman_combo/NF32ALLA/seed_{seed}/best_model.pth"
        for seed in SHA256
    ]
    present = files(api, PUBLIC)
    deletions = [CommitOperationDelete(path) for path in public_nf32alla if path in present]
    if deletions:
        api.create_commit(PUBLIC, operations=deletions, repo_type="model",
                          commit_message="Keep official NF32ALLA seeds only in private Occam release")
    if any(path in files(api, PUBLIC) for path in public_nf32alla):
        raise RuntimeError("NF32ALLA remains in public ablations")
    final = files(api, PRIVATE)
    official_norman = {p for p in final if p.startswith("checkpoints/vcell/norman_combo_")}
    expected_official = {checkpoint_path, card_path} | {
        f"{OFFICIAL}/seed_{seed}/best_model.pth" for seed in SHA256
    }
    if official_norman != expected_official:
        raise RuntimeError(f"Unexpected official Norman files: {sorted(official_norman)}")
    print("Complete: all five NF32ALLA seeds are official; other Norman variants are ablations")


if __name__ == "__main__":
    main()
