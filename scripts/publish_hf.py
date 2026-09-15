#!/usr/bin/env python
"""Upload the released artifacts to a single PRIVATE HuggingFace repo.

    python scripts/publish_hf.py --repo <user>/Occam --dry-run
    python scripts/publish_hf.py --repo <user>/Occam

Layout created in the repo:

    checkpoints/    the released unified encoder+decoder, and its model card
    eval/           the evaluation bundle (6 datasets) and its card
    diverserna/     the sharded pretraining corpus and its card

⚠️ PRIVATE ONLY. The repo is created with `private=True` and this script refuses
to run against a public repo. That is not a formality: neither dataset has had
its licensing resolved. The pretraining corpus has 7 of 9 constituent datasets
unattributed and the two that are identified (GTEx, TCGA) carry redistribution
conditions; the evaluation bundle is six third-party datasets with no licence
recorded. `scripts/check_upload_gate.py` gates PUBLIC release and still refuses.
Private storage for the people who already have these files is a different
thing, which is why this script exists and that gate still says no.

Only the final checkpoint is uploaded. The stage-1 and stage-2 files are
training intermediates, and the finetuned encoder they hold is already inside
the released unified checkpoint as `encoder_state_dict`.

How the upload works: `upload_large_folder` is resumable, parallel and retries
per file, which is what ~14 GB over a shared filesystem needs — but it has no
`path_in_repo`, it mirrors the folder it is given. So the folder it is given is
a staging tree of HARDLINKS (`--stage-dir`, default a sibling of the repo, same
GPFS fileset) laid out exactly as the repo should look. Hardlinks cost no space
and no copy time. Re-running is incremental: already-uploaded files are skipped,
so this can be run now for the checkpoint and the eval bundle and again once
the corpus export finishes.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (local path, path in the repo, description). Directories are staged whole.
ARTIFACTS = [
    ("checkpoints/release/occam_small_encoder_decoder.pth",
     "checkpoints/occam_small_encoder_decoder.pth",
     "the released model: unified encoder + external decoder"),
    ("checkpoints/MANIFEST.json", "checkpoints/MANIFEST.json",
     "checksums, parameter counts and the verified recipe"),
    ("MODEL_CARD.md", "checkpoints/MODEL_CARD.md", "model card"),
    ("data/eval", "eval", "evaluation bundle: 6 datasets"),
    ("data/diverserna", "diverserna", "pretraining corpus shards"),
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

    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    api = HfApi()
    try:
        who = api.whoami()["name"]
    except Exception as exc:  # noqa: BLE001
        print(f"not logged in: {type(exc).__name__}: {exc}\n  run: hf auth login",
              file=sys.stderr)
        return 2
    print(f"[hf] authenticated as {who}")

    wanted = [a for a in ARTIFACTS if not args.only or a[1] in args.only]
    missing = [a for a in wanted if not os.path.exists(os.path.join(REPO, a[0]))]

    print(f"[hf] target: {args.repo} ({args.repo_type}, PRIVATE)\n")
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

    if args.dry_run:
        print("\n[hf] --dry-run: nothing staged, nothing uploaded")
        return 0

    # Create private, or verify an existing repo really is private.
    try:
        info = api.repo_info(args.repo, repo_type=args.repo_type)
        if not info.private:
            print(f"\nREFUSING: {args.repo} is PUBLIC.\n"
                  f"  Neither dataset has had its licensing resolved; see the module\n"
                  f"  docstring and scripts/check_upload_gate.py. Make the repo private\n"
                  f"  or pass a different --repo.", file=sys.stderr)
            return 3
        print("[hf] repo exists and is private")
    except HfHubHTTPError:
        api.create_repo(args.repo, repo_type=args.repo_type, private=True,
                        exist_ok=True)
        print(f"[hf] created private repo {args.repo}")

    os.makedirs(args.stage_dir, exist_ok=True)
    n_files, n_bytes = stage(args.stage_dir, wanted)
    n_pruned = prune(args.stage_dir)
    print(f"[hf] staged {n_files} files ({n_bytes / 1e9:.2f} GB) at {args.stage_dir}"
          + (f", pruned {n_pruned}" if n_pruned else ""))

    api.upload_large_folder(
        repo_id=args.repo, repo_type=args.repo_type, folder_path=args.stage_dir,
        num_workers=args.workers, print_report=True,
    )

    print(f"\n[hf] https://huggingface.co/{args.repo}  (private)")
    if missing:
        print(f"[hf] {len(missing)} artifact(s) still pending — re-run to add them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
