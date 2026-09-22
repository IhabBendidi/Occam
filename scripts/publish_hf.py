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

# The two trained vcell checkpoints. Spelled out here rather than globbed: the run
# directory name IS the configuration, and a glob would happily pick up a different arm
# after a re-run and publish it under the published arm's name.
RHJ_CKPT = ("runs/rhj_fewshot_Y06__smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_"
            "lmse0.0044_e150_em3e-4_s42/vcell/checkpoints/best_model.pth")
NORMAN_CKPT = ("runs/norman_combo_NQ02__smooth_l1_nbw0_nbs1.0_dcos0.32977_mag0.0400866_"
               "lmse0.00198826_e150_em3e-4_s42/vcell/checkpoints/best_model.pth")

# sha256 of each, asserted before upload: these are 2.2 GB files selected by a path that
# a re-run can overwrite, and publishing the wrong weights under a published number is
# not recoverable once someone has downloaded it.
CKPT_SHA256 = {
    RHJ_CKPT: "fe93eeac08d345f3ec53a411879d6331985b4ef9df2b5c856cfc8f2b50fce219",
    NORMAN_CKPT: "91aed55e3db04afb49e225e6d49193c14f5bdf72b74b3815111c7da8cff26d4a",
}

# (local path, path in the repo, description). Directories are staged whole.
ARTIFACTS = [
    ("checkpoints/release/occam_small_encoder_decoder.pth",
     "checkpoints/occam_small_encoder_decoder.pth",
     "the released model: unified encoder + external decoder"),
    ("checkpoints/MANIFEST.json", "checkpoints/MANIFEST.json",
     "checksums, parameter counts and the verified recipe"),
    ("MODEL_CARD.md", "checkpoints/MODEL_CARD.md", "model card"),
    ("data/eval", "eval", "evaluation bundle: 6 datasets + the Norman-2019 combo source"),
    # The frozen split-half benchmarks. Without these the .h5ad files are just data: the
    # cohort, the seeded splits, the axis and the baselines are what make a number
    # comparable between two people.
    ("data/benchmarks/norman_combo", "benchmarks/norman_combo",
     "the Norman-2019 double-perturbation benchmark: axis, cohort, 5 splits, baselines"),
    # ⚠️ CORRECTION (2026-09-22): this used to say rhj_fewshot could not be published
    # because its `files/*.json` carry absolute source paths. Checked -- those files hold
    # `path`, `sha256`, `mtime`, `bytes`, `n_obs` and friends, NOTHING under src/ reads
    # them, and the absolute `path` is provenance about the machine that froze the
    # benchmark. Withholding the benchmark meant withholding REQUESTED.json, i.e. the
    # split definition itself, which made the headline result unreproducible. Published.
    ("data/benchmarks/rhj_fewshot", "benchmarks/rhj_fewshot",
     "the RPE1/HepG2/Jurkat few-shot benchmark: axis, cohort, the requested val/test "
     "split, 5 split-halves, baselines"),
    # The perturbation graph. Required to TRAIN (the adjacency-map perturbation model is
    # an MLP over each gene's adjacency row) and to BUILD either benchmark (the cohort is
    # intersected with `master_gene_set_sorted.csv`). Without it neither is reproducible.
    ("data/graphs", "graphs",
     "GO + STRING edge tables and the 18,216-gene master set the pert model indexes"),
    ("data/diverserna", "diverserna", "pretraining corpus shards"),
    # The two trained vcell checkpoints and their cards. These are DELIVERABLES, unlike
    # the stage-1/stage-2 MAE files: they are the models the reported numbers come from,
    # and each is self-contained (encoder + decoder + perturbation model + FiLM +
    # gene_names + pert2id + target_sum), so nothing has to resolve the absolute
    # `training_config["encoder_checkpoint"]` path they happen to record.
    (RHJ_CKPT, "checkpoints/vcell/rhj_fewshot_Y06/best_model.pth",
     "rhj_fewshot main checkpoint, arm Y06 (pearson_delta 0.4043)"),
    ("checkpoints/cards/rhj_fewshot_Y06_CARD.md",
     "checkpoints/vcell/rhj_fewshot_Y06/CARD.md", "its model card"),
    (NORMAN_CKPT, "checkpoints/vcell/norman_combo_NQ02/best_model.pth",
     "norman_combo main checkpoint, arm NQ02 (0.8169 additive / 0.7380 nonadditive)"),
    ("checkpoints/cards/norman_combo_NQ02_CARD.md",
     "checkpoints/vcell/norman_combo_NQ02/CARD.md", "its model card"),
    # Every cell-eval metric, every arm, every split -- so a reader can re-derive any
    # reported number instead of trusting a four-metric mean. Regenerate with
    # `python scripts/export_split_metrics.py --out data/release/metrics`.
    ("data/release/metrics", "metrics",
     "PUBLISHED scope: all 35 cell-eval metrics for our published arms, every baseline "
     "and every competitor, per split and panel. Also tracked in git."),
    ("data/release/metrics_all", "metrics_all",
     "FULL scope: the same, for all 422 eval roots -- the whole rhj_fewshot sweep and "
     "every norman base. Not in git."),
    # The four-command reproduction path: train, evaluate, read the metrics, predict.
    ("scripts/repro", "repro", "reproducibility scripts (train / eval / predict)"),
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

    # Verify the checkpoints ARE the published weights before anything is uploaded.
    # The path selects them, and a re-run of the same arm overwrites that path; the
    # sha256 is the only thing that ties a file to a published number.
    staged_ckpts = [a[0] for a in wanted if a[0] in CKPT_SHA256]
    if staged_ckpts:
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
