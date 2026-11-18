"""The vendored competitor trees: integrity, hygiene, and the bash footgun.

These four arms are **independent code in this repository**. They began as a port from another
tree, but that tree is no longer a dependency, a reference or a source of truth: the copier was
deleted, every reference to its name is gone, and CellFlow's checkpoint selection scores
through our own cell-eval rather than a sibling repo's scorer.

What is still worth asserting:

1. **Payload must not enter git.** These trees carry code and configs; checkpoints, h5ads,
   venvs and run products must stay out, and the tracked tree must stay small.
2. **Vendored code must not change silently.** `competitors/MANIFEST.json` records a sha256 per
   file as the port's own integrity baseline. An intended edit is recorded with
   `scripts/refresh_competitor_manifest.py`; an unintended one shows up here.
3. **No arm's own scorer may be wired into a launcher.** Every reported number comes from
   `scripts/eval_competitor.py` and our cell-eval 0.6.0. The arms' original scorers clip to
   [0,14], skip metrics and pin different cell-eval versions.
4. **`#` inside a backslash-continued bash command.** A comment TERMINATES the continuation.
   Upstream this dropped 29 of 32 Hydra overrides on STATE, which fell back to `name=debug`,
   auto-resumed an unrelated checkpoint, and **exited 0**.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPET = os.path.join(REPO, "competitors")
MANIFEST = os.path.join(COMPET, "MANIFEST.json")

ARMS = ("STATE", "TxPert", "CellFlow", "PerturbDiff")


# ⚠️ These directory names carry the 92 GB. If one is ever TRACKED, the guard failed.
FORBIDDEN_DIRS = ("processed_data", "test_norman", "test_replogle", "debugging",
                  "cell-eval-outdir", ".venv", "wandb", "lightning_logs", "outputs",
                  "checkpoints")
FORBIDDEN_EXTS = (".ckpt", ".pkl", ".pt", ".pth", ".h5ad", ".h5", ".npy", ".npz",
                  ".out", ".err", ".log", ".sif", ".parquet")

MAX_TRACKED_BYTES = 40 * 1024 * 1024


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _tracked():
    """Paths git actually tracks under competitors/, relative to that directory."""
    out = subprocess.run(["git", "-C", REPO, "ls-files", "competitors"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    return [p[len("competitors/"):] for p in out.stdout.split("\n")
            if p.startswith("competitors/")]


@pytest.mark.skipif(not os.path.isfile(MANIFEST), reason="competitors not copied yet")
def test_the_manifest_records_every_arm():
    man = json.load(open(MANIFEST))
    assert man.get("schema") == 2
    for arm in ARMS:
        rec = man["arms"].get(arm)
        assert rec, f"{arm} is not in MANIFEST.json, so its tree has no integrity baseline"
        assert rec["n_files"] > 0 and rec["bytes"] > 0
        assert rec["files"], f"{arm} records no per-file sha256"


@pytest.mark.skipif(not os.path.isfile(MANIFEST), reason="competitors not copied yet")
def test_every_recorded_file_still_exists_and_matches_its_recorded_hash():
    """Port integrity against the manifest's OWN baseline, not against upstream.

    The hashes are THIS CODE's own baseline. They were briefly an upstream comparison, which
    stopped meaning anything once these trees became independent -- that check would now be
    either permanently red or exempting most of the tree, and neither detects a thing.

    What it catches is the real risk with vendored code: a file edited without the record
    being updated. Run `scripts/refresh_competitor_manifest.py` to record a deliberate edit.
    """
    man = json.load(open(MANIFEST))
    missing, changed = [], []
    for arm, rec in man["arms"].items():
        for rel, meta in rec["files"].items():
            p = os.path.join(COMPET, arm, rel)
            if not os.path.isfile(p):
                missing.append(f"{arm}/{rel}")
                continue
            if meta.get("sha256") and _sha256(p) != meta["sha256"]:
                changed.append(f"{arm}/{rel}")
    assert not missing, f"recorded but absent: {missing[:6]}"
    assert not changed, (
        f"{len(changed)} vendored file(s) modified without refreshing the manifest: "
        f"{changed[:6]}\nRun scripts/refresh_competitor_manifest.py if the edit was "
        f"intended. The manifest is this code's own integrity baseline.")


@pytest.mark.skipif(not os.path.isdir(COMPET), reason="competitors not copied yet")
def test_no_payload_is_tracked_under_competitors():
    tracked = _tracked()
    if not tracked:
        pytest.skip("nothing tracked under competitors/ yet")

    bad_dir = [p for p in tracked
               if any(seg.startswith(d) for seg in p.split("/") for d in FORBIDDEN_DIRS)]
    bad_ext = [p for p in tracked if os.path.splitext(p)[1] in FORBIDDEN_EXTS]
    assert not bad_dir, f"payload directories are TRACKED: {bad_dir[:6]}"
    assert not bad_ext, f"payload files are TRACKED: {bad_ext[:6]}"


@pytest.mark.skipif(not os.path.isdir(COMPET), reason="competitors not copied yet")
def test_the_tracked_competitor_tree_stays_small():
    """Code is 1-2 MB per arm. Double-digit MB means the denylist missed something."""
    tracked = _tracked()
    if not tracked:
        pytest.skip("nothing tracked under competitors/ yet")
    total = sum(os.path.getsize(os.path.join(COMPET, p))
                for p in tracked if os.path.isfile(os.path.join(COMPET, p)))
    assert total < MAX_TRACKED_BYTES, (
        f"{total / 1e6:.1f} MB tracked under competitors/, over the "
        f"{MAX_TRACKED_BYTES / 1e6:.0f} MB budget. Upstream has 92 GB of "
        f"untracked-and-unignored payload; check what slipped in.")


@pytest.mark.skipif(not os.path.isdir(COMPET), reason="competitors not copied yet")
def test_no_arms_original_scorer_is_wired_into_our_launchers():
    """Their scorers are kept as REFERENCE and never run.

    They clip to [0,14], skip `pearson_edistance` and `clustering_agreement`, label controls
    `"empty"`, and run cell-eval 0.7.2 / pdex 0.2.2-0.3.0 against our 0.6.0 / pdex 0.1.24.
    A launcher that calls one produces a number that is not comparable to any other row in
    the table, and nothing about the output says so.
    """
    sl = os.path.join(REPO, "scripts", "slurm")
    offenders = []
    for f in sorted(os.listdir(sl)):
        if not f.startswith("competitor"):
            continue
        txt = open(os.path.join(sl, f)).read()
        body = "\n".join(ln for ln in txt.split("\n") if not ln.lstrip().startswith("#"))
        for pat in ("cell-eval run", "predict_state_fewshot", "predict_cellflow",
                    "cell_eval_cli", "celleval_score"):
            if pat in body:
                offenders.append(f"{f}: {pat}")
    assert not offenders, (
        f"a competitor launcher invokes an arm's own scorer: {offenders}. "
        f"Every reported number must come from scripts/eval_competitor.py.")


def test_no_launcher_puts_a_comment_inside_a_continued_command():
    """⛔ The most expensive bug in this port, and it exits 0.

    A `#` comment inside a backslash continuation terminates the command. Only a continuation
    that STARTED from a real command line counts -- a block of `#`-prefixed usage examples,
    where the backslash is itself inside a comment, is fine and is the normal way these
    headers are written.
    """
    sl = os.path.join(REPO, "scripts", "slurm")
    violations = []
    for f in sorted(os.listdir(sl)):
        if not (f.endswith(".sbatch") or f.endswith(".sh")):
            continue
        in_command_continuation = False
        for i, raw in enumerate(open(os.path.join(sl, f)), 1):
            line = raw.rstrip("\n")
            stripped = line.lstrip()
            if in_command_continuation and stripped.startswith("#"):
                violations.append(f"{f}:{i}: {stripped[:60]}")
            # a continuation is only a COMMAND continuation if the line is not a comment
            in_command_continuation = (line.rstrip().endswith("\\")
                                       and not stripped.startswith("#"))
    assert not violations, (
        "a `#` inside a backslash-continued command silently drops every remaining "
        f"argument:\n  " + "\n  ".join(violations))


def test_every_competitor_launcher_names_its_arm_and_its_anchors():
    """RULE 0c, enforced. A launcher whose header does not say what it is testing costs a
    re-derivation every time someone reads a log six weeks later."""
    sl = os.path.join(REPO, "scripts", "slurm")
    launchers = [f for f in sorted(os.listdir(sl)) if f.startswith("competitor_train_")]
    assert len(launchers) == 4, f"expected 4 per-arm train launchers, found {launchers}"
    for f in launchers:
        txt = open(os.path.join(sl, f)).read()
        head = txt[:txt.index("#SBATCH")] if "#SBATCH" in txt else txt
        assert "WHAT THIS ANSWERS" in head, f"{f}: no objective/question block"
        assert "VARIED" in head.upper() and "HELD FIXED" in head.upper(), (
            f"{f}: does not say what is varied and what is held fixed")
        assert "WHAT EACH OUTCOME MEANS" in head, f"{f}: outcomes are not pre-registered"
        for anchor in ("0.2323", "0.1324", "0.3722"):
            assert anchor in head, f"{f}: missing the {anchor} anchor"
