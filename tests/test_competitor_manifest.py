"""The port's hygiene: what was copied, what must never be, and the bash footgun.

Three separate failures are guarded here, and all three are the kind that pass CI and cost a
day:

1. **Payload entering the repo.** The upstream tree is 141 GB, ~92 GB of it untracked *and*
   unignored there (`STATE/state/processed_data*/`, `test_norman*/`, `test_replogle*/`,
   `debugging/`, `cell-eval-outdir/`). `scripts/copy_competitor_code.py` copies an explicit
   allowlist and refuses a denylist; this asserts the result.

2. **Drift between the copy and its record.** `competitors/MANIFEST.json` holds a sha256 per
   copied file. A file edited in place with no manifest update makes the port
   unreproducible -- and four of these files ARE edited deliberately (the occam wirings), so
   "the sha differs" is not automatically a bug. What is a bug is an unrecorded file, or a
   recorded file that has vanished.

3. **`#` inside a backslash-continued bash command.** A comment TERMINATES the continuation.
   Upstream this dropped 29 of 32 Hydra overrides on STATE, which then fell back to
   `name=debug`, auto-resumed an unrelated checkpoint, and **exited 0**. Every competitor
   launcher is linted for it here, because nothing at runtime will tell you.
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

# Files we adapt on purpose. Their sha256 is EXPECTED to differ from the upstream record --
# that is the port. Everything else must still match.
ADAPTED = (
    # TxPert (step 4a)
    "gspp/data/occam_loader.py",
    "gspp/data/datamodule.py",
    "configs/config-txpert-occam.yaml",
    # STATE (step 4b) -- the third source_format and its three call sites
    "tx/data/occam_source.py",
    "tx/data/zarr_perturbation.py",
    "configs/data/occam_h5ad.yaml",
    # CellFlow (step 4c) -- the occam bridge and the branch that selects it
    "occam_bridge/occam.py",
    "train_cellflow.py",
    "cellflow_selection.py",
    "configs/occam_rhj_fewshot.yaml",
    # PerturbDiff (step 4d) -- the third store, cache, split and datamodule
    "src/data/occam_bridge.py",
    "src/data/metadata_cache.py",
    "src/data/dataset/dataset_core.py",
    "src/data/data_module/data_module.py",
    "src/data/data_module/data_module_setup.py",
    "src/data/data_module/__init__.py",
    "src/apps/training/training_datamodule_builder.py",
    # every arm's Phase-A adapter
    "export_occam.py",
)

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
def test_the_manifest_records_every_arm_with_its_upstream_commit():
    man = json.load(open(MANIFEST))
    assert man.get("schema") == 2
    for arm in ARMS:
        rec = man["arms"].get(arm)
        assert rec, f"{arm} is not in MANIFEST.json; the port is not auditable without it"
        assert rec["upstream_commit"] and rec["upstream_commit"] != "unknown", (
            f"{arm} has no upstream commit recorded, so the copy cannot be reproduced")
        assert rec["n_files"] > 0 and rec["bytes"] > 0
        assert rec["files"], f"{arm} records no per-file sha256"


@pytest.mark.skipif(not os.path.isfile(MANIFEST), reason="competitors not copied yet")
def test_every_recorded_file_still_exists_and_matches_its_recorded_hash():
    """Port integrity against the manifest's OWN baseline, not against upstream.

    ⚠️ THE SEMANTICS CHANGED ON 2026-09-18 (manifest schema 1 -> 2). Schema 1 recorded each
    file's UPSTREAM sha256 and this test asserted every file still matched it except an
    `ADAPTED` allowlist. That became meaningless when the port deliberately diverged from
    upstream across the whole tree -- every reference to the upstream benchmark's name was
    removed, which touched ~70 vendored files. Keeping the old check would have meant either a
    permanently red test or an `ADAPTED` list naming most of the tree, and neither detects
    anything.

    So the hashes are now the port's own baseline, and this catches what the check is actually
    for: a vendored file modified without the manifest being refreshed.
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
        f"{changed[:6]}\nRefresh it if the edit was intended -- the manifest is the port's "
        f"integrity baseline now, not an upstream comparison.")


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


@pytest.mark.skipif(not os.path.isdir(COMPET), reason="competitors not copied yet")
def test_every_adapted_file_is_protected_from_being_re_copied():
    """⛔ The bug this test exists for reverted the ENTIRE port in one command, silently.

    `scripts/copy_competitor_code.py` copies an allowlist of upstream paths. Nine of those
    paths are files we deliberately ADAPTED so each arm reads our data. Re-running the script
    overwrote all nine with the pristine upstream versions -- and
    `test_every_recorded_file_still_exists_and_only_the_adapted_ones_differ` then PASSED,
    because every file matched its upstream sha256 again. MEASURED 2026-09-17.

    The sha256 test structurally cannot catch this: a reverted file is indistinguishable from
    a file that was never adapted. Only the PROTECTED list can, so it is checked here against
    the ADAPTED list above -- the two must not drift apart.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "copy_competitor_code",
        os.path.join(REPO, "scripts", "copy_competitor_code.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    protected = {arm: set(paths) for arm, paths in mod.PROTECTED.items()}
    allow = {arm: list(paths) for arm, paths in mod.ALLOWLIST.items()}

    # Every adapted file that the allowlist would re-copy must be protected.
    unguarded = []
    for arm, paths in protected.items():
        for rel in paths:
            f = os.path.join(COMPET, arm, rel)
            if not os.path.isfile(f):
                unguarded.append(f"{arm}/{rel}: protected but absent")
    assert not unguarded, unguarded

    # And every file we call adapted, which lives under an allowlisted tree, must be in
    # PROTECTED -- otherwise the next copy silently reverts it.
    missing_guard = []
    for arm in ARMS:
        arm_dir = os.path.join(COMPET, arm)
        if not os.path.isdir(arm_dir):
            continue
        for dirpath, dirnames, filenames in os.walk(arm_dir):
            dirnames[:] = [d for d in dirnames if d not in {".venv", "__pycache__", ".git"}]
            for fn in filenames:
                rel = os.path.relpath(os.path.join(dirpath, fn), arm_dir).replace(os.sep, "/")
                if not any(rel.endswith(a) for a in ADAPTED):
                    continue
                # A NEW file (occam_source.py, occam_bridge.py, export_occam.py, ...) is not
                # in any allowlisted upstream tree, so a copy cannot touch it. Only files
                # that EXIST upstream need guarding.
                upstream = os.path.join(mod.UPSTREAM, arm, rel)
                if not os.path.isfile(upstream):
                    continue
                if rel in protected.get(arm, set()):
                    continue
                # A file the ALLOWLIST cannot reach is equally safe -- the copier only walks
                # allowlisted paths. CellFlow's adapted files were removed from its allowlist
                # outright (2026-09-18), which is stronger than protecting them: there is
                # nothing to overwrite rather than something skipped.
                reachable = any(rel == e or rel.startswith(e.rstrip("/") + "/")
                                for e in allow.get(arm, []))
                if reachable:
                    missing_guard.append(f"{arm}/{rel}")
    assert not missing_guard, (
        f"adapted file(s) that a re-copy would silently REVERT: {missing_guard}. "
        f"Add them to PROTECTED in scripts/copy_competitor_code.py.")


@pytest.mark.skipif(not os.path.isdir(COMPET), reason="competitors not copied yet")
def test_the_upstream_benchmarks_name_is_gone_from_the_codebase():
    """⛔ 2026-09-18, at the user's explicit instruction: "remove even mentions of the word
    hooke from the codebase".

    537 references were removed across 4 arms, our scripts, our docs and the launcher headers.
    Two classes are deliberately KEPT and are the only allowed hits:

    1. **`hooke-predict`** -- a live Slurm QOS and wckey. It is a CLUSTER resource, not our
       naming; every launcher submits to it and four chains were running on it at the time.
       Removing it would have broken them.
    2. **`scripts/copy_competitor_code.py`** -- its `UPSTREAM` constant is a real filesystem
       path in someone else's repo, and its denylist globs match filenames on disk there. Both
       are DATA, not prose. The sweep rewrote both and broke them once each: the copier could
       not find upstream, and `"*hooke*"` became `"*upstream*"` and silently stopped denying
       anything, which let a tree of reference files come back on the next copy.

    Anything else is a regression -- most likely a re-copy from upstream, which is why
    `copy_competitor_code.py` now refuses to run while `MANIFEST.json` exists.
    """
    import subprocess

    # ⚠️ EXCLUDE VENVS AND USE A WORD-ISH PATTERN. A plain `grep -rin hooke` matches
    # `UnhookEvent` inside pygments' SourceMod builtins under
    # competitors/CellFlow/cellflow/.venv, which is third-party code we neither wrote nor
    # vendored. `hooke\b` also stops `unhook`/`hooked` from counting.
    out = subprocess.run(
        ["grep", "-rinE", r"\bhooke", "--exclude-dir=.venv", "--exclude-dir=site-packages",
         "--exclude-dir=__pycache__", "--exclude-dir=outputs", "--exclude-dir=wandb",
         "competitors", "scripts", "docs", "tests", "configs", "src"],
        cwd=REPO, capture_output=True, text=True)
    hits = []
    for line in out.stdout.split("\n"):
        if not line.strip():
            continue
        path = line.split(":", 1)[0]
        low = line.lower()
        if "hooke-predict" in low:                       # the QOS
            continue
        if path.startswith("scripts/copy_competitor_code.py"):   # paths + denylist globs
            continue
        if path.startswith("competitors/MANIFEST.json"):         # records those globs
            continue
        if path.startswith("tests/test_competitor_manifest.py"):  # this guard's own prose
            continue
        # The same `noqa: legacy-name` marker the repo already uses for the pre-rename
        # interpreter name. A doc that DOCUMENTS the rename has to name the old identifiers,
        # and without this the guard flags the record of its own enforcement.
        if "legacy-name" in line:
            continue
        if "/outputs/" in path or "/wandb/" in path:             # run products
            continue
        hits.append(line[:160])
    assert not hits, (
        f"{len(hits)} reference(s) to the upstream benchmark's name came back:\n  "
        + "\n  ".join(hits[:10]))
