#!/usr/bin/env python
"""Publish the official all-source Occam release.

Dry-run lists selected artifacts without authentication or downloads. Uploads
verify checkpoint hashes against checkpoints/official_manifest.json and retain
the existing repository visibility. Retired experiments are never staged.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The manifest pins the exact weights used by the paper, for all five seeds.
import json

with open(os.path.join(REPO, "checkpoints/official_manifest.json")) as stream:
    OFFICIAL = json.load(stream)
RHJ_SEED_CKPTS = {m["seed"]: m["local"] for m in OFFICIAL["virtual_cell"] if m["benchmark"] == "rhj"}
NORMAN_SEED_CKPTS = {m["seed"]: m["local"] for m in OFFICIAL["virtual_cell"] if m["benchmark"] == "norman"}
RHJ_CKPT = RHJ_SEED_CKPTS[42]
NORMAN_CKPT = NORMAN_SEED_CKPTS[42]
CKPT_SHA256 = {m["local"]: m["sha256"] for m in OFFICIAL["virtual_cell"]}
CKPT_SHA256[OFFICIAL["encoder_decoder"]["local"]] = OFFICIAL["encoder_decoder"]["sha256"]

ARTIFACTS = [
    (OFFICIAL["encoder_decoder"]["local"], OFFICIAL["encoder_decoder"]["remote"], "pretrained encoder and decoder"),
    ("checkpoints/official_manifest.json", "checkpoints/official_manifest.json", "official checkpoint hashes and paths"),
    ("checkpoints/MANIFEST.json", "checkpoints/MANIFEST.json", "encoder provenance"),
    ("MODEL_CARD.md", "checkpoints/MODEL_CARD.md", "encoder model card"),
    ("docs/huggingface_model_card.md", "README.md", "release overview and downloads"),
    ("data/eval", "eval", "expression datasets"),
    *[("data/benchmarks/" + name, "benchmarks/" + name, "frozen benchmark definition and data")
      for name in ("rhj_fewshot", "norman_combo", "h1_seen", "k562_seen", "k562_seen_h1set", "shared")],
    ("data/graphs", "graphs", "GO and STRING graphs"),
    ("data/assets", "assets", "GenePT and ESM2 embeddings and metadata"),
    *[(m["local"], m["remote"], f"official {m['arm']} seed {m['seed']}") for m in OFFICIAL["virtual_cell"]],
    (NORMAN_CKPT, "checkpoints/vcell/norman_combo_NF32ALLA/best_model.pth", "Norman seed-42 alias"),
    ("checkpoints/cards/rhj_all_sources_CARD.md", "checkpoints/occam_seeds/rhj_fewshot/all_sources/CARD.md", "RHJ all-source model card"),
    ("checkpoints/cards/norman_combo_NF32ALLA_CARD.md", "checkpoints/vcell/norman_combo_NF32ALLA/CARD.md", "Norman all-source model card"),
    ("data/release/official_comparisons", "official_comparisons", "official tables and complete seed/split scores"),
    ("data/release/final", "final", "published models, competitors and baselines only"),
    ("data/release/norman_allsource", "final/norman_allsource", "official Norman NF32ALLA scores"),
    ("data/release/metrics", "metrics", "all published cell-eval metrics"),
    ("data/release/eval_outputs", "eval_outputs", "raw competitor and baseline scorer outputs"),
    ("scripts/repro", "repro", "reproduction commands"),
    ("configs/rhj_fewshot_official.yaml", "repro/configs/rhj_fewshot_official.yaml", "RHJ official training recipe"),
    ("configs/norman_combo_official.yaml", "repro/configs/norman_combo_official.yaml", "Norman official training recipe"),
    ("scripts/build_h1_seen_panel.py", "repro/build_h1_seen_panel.py", "H1/K562 benchmark builder"),
    ("scripts/build_splithalf_panel.py", "repro/build_splithalf_panel.py", "split-half benchmark builder"),
    ("scripts/norman/norman_stratify_panels.py", "repro/norman_stratify_panels.py", "Norman evaluation subsets"),
]

# Never stage these, wherever they appear.
EXCLUDE_SUFFIXES = (".UNTRUSTED", ".pyc")
EXCLUDE_DIRS = ("__pycache__", ".cache", ".git")


def _iter_files(root: str):
    """Yield (absolute path, path relative to root) for every file to upload."""
    if os.path.isfile(root):
        yield root, os.path.basename(root)
        return
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for fn in sorted(files):
            if fn.endswith(EXCLUDE_SUFFIXES):
                continue
            src = os.path.join(dirpath, fn)
            yield src, os.path.relpath(src, root)


def stage(stage_dir: str, wanted) -> tuple:
    """Hardlink every wanted artifact into `stage_dir` under its repo path.

    Returns (n_files, n_bytes). Falls back to a copy if the link fails, which
    happens across GPFS filesets.
    """
    n_files = 0
    n_bytes = 0
    for local, remote, _desc in wanted:
        src_root = os.path.join(REPO, local)
        if not os.path.exists(src_root):
            continue
        for src, rel in _iter_files(src_root):
            dst = os.path.join(stage_dir, remote if os.path.isfile(src_root)
                               else os.path.join(remote, rel))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            size = os.path.getsize(src)
            if os.path.exists(dst):
                # Re-staging: replace only if the source changed.
                if os.path.getsize(dst) == size and os.path.samefile(dst, src):
                    n_files += 1
                    n_bytes += size
                    continue
                os.remove(dst)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            n_files += 1
            n_bytes += size
    return n_files, n_bytes


def prune(stage_dir: str) -> int:
    """Drop staged files whose source is gone, so the mirror stays a mirror.

    Deliberately checked against every artifact, not against `--only`: a
    restricted run stages a subset but must not un-stage the rest, because the
    upload always mirrors the whole staging tree.
    """
    live = set()
    for local, remote, _desc in ARTIFACTS:
        src_root = os.path.join(REPO, local)
        if not os.path.exists(src_root):
            continue
        for src, rel in _iter_files(src_root):
            live.add(remote if os.path.isfile(src_root)
                     else os.path.join(remote, rel))
    removed = 0
    for dirpath, dirs, files in os.walk(stage_dir):
        dirs[:] = [d for d in dirs if d != ".cache"]
        for fn in files:
            p = os.path.join(dirpath, fn)
            if os.path.relpath(p, stage_dir) not in live:
                os.remove(p)
                removed += 1
    return removed


def _size(path: str) -> int:
    return sum(os.path.getsize(s) for s, _ in _iter_files(path))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="e.g. <user>/Occam")
    ap.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    ap.add_argument("--stage-dir",
                    default=os.path.join(os.path.dirname(REPO), "occam_hf_stage"),
                    help="hardlink mirror of the repo layout; safe to delete")
    ap.add_argument("--only", action="append", default=[],
                    help="stage just this artifact's repo path; repeatable")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    wanted = [a for a in ARTIFACTS if not args.only or a[1] in args.only]
    missing = [a for a in wanted if not os.path.exists(os.path.join(REPO, a[0]))]

    print(f"[hf] target: {args.repo} ({args.repo_type})\n")
    total = 0
    for local, remote, desc in wanted:
        path = os.path.join(REPO, local)
        if not os.path.exists(path):
            print(f"  {remote:44s} MISSING          {desc}")
            continue
        n = _size(path)
        total += n
        kind = "dir " if os.path.isdir(path) else "file"
        print(f"  {remote:44s} {kind} {n / 1e9:7.2f} GB  {desc}")
    print(f"\n  total: {total / 1e9:.2f} GB")

    if missing:
        print(f"\n  {len(missing)} artifact(s) not built yet — re-run this script\n"
              f"  once they are; the upload is incremental:", file=sys.stderr)
        for local, remote, _ in missing:
            print(f"    {remote}  (expected at {local})", file=sys.stderr)

    # Verify the checkpoints ARE the published weights before anything is uploaded.
    # The path selects them, and a re-run of the same arm overwrites that path; the
    # sha256 is the only thing that ties a file to a published number.
    staged_ckpts = [a[0] for a in wanted if a[0] in CKPT_SHA256]
    if staged_ckpts and not args.dry_run:
        import hashlib
        for rel in staged_ckpts:
            path = os.path.join(REPO, rel)
            if not os.path.isfile(path):
                continue
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for block in iter(lambda: fh.read(8 << 20), b""):
                    h.update(block)
            got, want = h.hexdigest(), CKPT_SHA256[rel]
            if got != want:
                print(f"\nREFUSING: {rel}\n  sha256 {got}\n  expected {want}\n"
                      f"  This is not the checkpoint the published numbers came from. "
                      f"Either the run was re-executed and overwrote it, or the wrong "
                      f"arm is being published. Update CKPT_SHA256 deliberately, with "
                      f"the card and the docs, or point at the right file.",
                      file=sys.stderr)
                return 4
            print(f"[hf] sha256 verified: {os.path.basename(rel)} "
                  f"({os.path.dirname(rel).split('/')[0]})")

    if args.dry_run:
        print("\n[hf] --dry-run: nothing staged, nothing uploaded")
        return 0

    from huggingface_hub import HfApi
    api = HfApi()
    api.repo_info(args.repo, repo_type=args.repo_type)

    os.makedirs(args.stage_dir, exist_ok=True)
    n_files, n_bytes = stage(args.stage_dir, wanted)
    n_pruned = prune(args.stage_dir)
    print(f"[hf] staged {n_files} files ({n_bytes / 1e9:.2f} GB) at {args.stage_dir}"
          + (f", pruned {n_pruned}" if n_pruned else ""))

    api.upload_large_folder(
        repo_id=args.repo, repo_type=args.repo_type, folder_path=args.stage_dir,
        num_workers=args.workers, print_report=True,
    )

    print(f"\n[hf] https://huggingface.co/{args.repo}")
    if missing:
        print(f"[hf] {len(missing)} artifact(s) still pending — re-run to add them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
