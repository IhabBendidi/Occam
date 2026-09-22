"""The release surface is pinned: the published checkpoints, their cards, the repro path.

WHY THIS FILE EXISTS
    `scripts/publish_hf.py` selects 2.2 GB checkpoints by a `runs/<arm>__<tag>/...` path
    that a re-run of the same arm silently overwrites. The sha256 is the only thing that
    ties a file to a published number, and a typo in either the path or the digest is
    invisible until someone downloads the wrong weights. These tests check the wiring --
    that the entries exist, that the digests are recorded, and that the scripts a reader
    is told to run are actually there and reference real things.

    They deliberately do NOT hash the checkpoints (2.2 GB x 2 per test run);
    `publish_hf.py` does that immediately before uploading.
"""
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _publish_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_publish_hf", os.path.join(REPO, "scripts/publish_hf.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_published_artifact_has_a_distinct_repo_path():
    m = _publish_module()
    remotes = [a[1] for a in m.ARTIFACTS]
    assert len(remotes) == len(set(remotes)), \
        "two artifacts map to one repo path; the second would overwrite the first"
    for local, remote, desc in m.ARTIFACTS:
        assert not remote.startswith("/"), remote
        assert desc.strip(), f"{remote} has no description"


def test_both_vcell_checkpoints_are_published_with_a_recorded_digest():
    m = _publish_module()
    remotes = {a[1]: a[0] for a in m.ARTIFACTS}
    for remote in ("checkpoints/vcell/rhj_fewshot_Y06/best_model.pth",
                   "checkpoints/vcell/norman_combo_NQ02/best_model.pth"):
        assert remote in remotes, f"{remote} is not in ARTIFACTS"
        local = remotes[remote]
        assert local in m.CKPT_SHA256, f"{local} has no recorded sha256"
        assert re.fullmatch(r"[0-9a-f]{64}", m.CKPT_SHA256[local]), \
            f"{local}: sha256 is not 64 hex chars"
    # Every digest must belong to an artifact actually being published, or the gate
    # silently never runs on it.
    for local in m.CKPT_SHA256:
        assert local in {a[0] for a in m.ARTIFACTS}, \
            f"{local} has a digest but is not in ARTIFACTS"


def test_each_published_checkpoint_ships_its_card():
    m = _publish_module()
    remotes = {a[1] for a in m.ARTIFACTS}
    for arm in ("rhj_fewshot_Y06", "norman_combo_NQ02"):
        assert f"checkpoints/vcell/{arm}/CARD.md" in remotes, arm
        card = os.path.join(REPO, "checkpoints/cards", f"{arm}_CARD.md")
        assert os.path.isfile(card), card
        text = open(card).read()
        # A card without its anchors is a number with no scale beside it.
        assert "sha256" in text and "Limitations" in text, arm
        for anchor in ("meanpert", "experimental"):
            assert anchor in text, f"{arm} card does not name the {anchor} anchor"


def test_the_repro_path_exists_and_is_executable():
    for rel in ("scripts/repro/README.md",
                "scripts/repro/rhj_train.sh", "scripts/repro/rhj_eval.sh",
                "scripts/repro/norman_train.sh", "scripts/repro/norman_eval.sh",
                "scripts/repro/build_benchmark_rhj.sh",
                "scripts/repro/build_benchmark_norman.sh",
                "scripts/repro/baselines.sh",
                "scripts/repro/competitors_rhj.sh",
                "scripts/repro/occam_predict.py",
                "scripts/export_split_metrics.py"):
        p = os.path.join(REPO, rel)
        assert os.path.isfile(p), rel
        if rel.endswith((".sh", ".py")) and "repro" in rel:
            assert os.access(p, os.X_OK), f"{rel} is not executable"


def test_the_eval_scripts_keep_the_load_bearing_throttle_and_nocal():
    """%6 is the pdex /dev/shm race limit, and every reported number is `nocal`."""
    for rel in ("scripts/repro/rhj_eval.sh", "scripts/repro/norman_eval.sh"):
        src = open(os.path.join(REPO, rel)).read()
        assert "--array=0-4%6" in src, f"{rel} lost the %6 throttle"
        assert "--calibrate none" in src, f"{rel} is not scoring nocal"


def test_the_norman_eval_script_defaults_to_the_default_composition_mode():
    src = open(os.path.join(REPO, "scripts/repro/norman_eval.sh")).read()
    assert '${COMPOSE:-decode_seq_mean}' in src
    rhj = open(os.path.join(REPO, "scripts/repro/rhj_eval.sh")).read()
    # Single-gene panel: passing a mode would imply it means something here.
    assert "--compose" not in rhj


def test_the_train_scripts_pass_no_loss_or_schedule_overrides():
    """The config IS the recipe. A --set here would mean the config is not the default."""
    for rel in ("scripts/repro/rhj_train.sh", "scripts/repro/norman_train.sh"):
        src = open(os.path.join(REPO, rel)).read()
        sets = re.findall(r'--set "([^"=]+)=', src)
        assert sets == ["seed"], f"{rel} overrides {sets}; expected only seed"


def test_the_rebuild_scripts_assert_the_published_digests():
    """A rebuild that does not verify is a description, not a reproduction."""
    expect = {
        "scripts/repro/build_benchmark_rhj.sh": (
            "d934211e5eb1e37e28c893fb8c5ff4211d607a1d77b0103d5c2b79221d21677f",
            "ef64f2ba7ff7816ed947796066cd8a68014884c6176619d750ef79462c78d4c0"),
        "scripts/repro/build_benchmark_norman.sh": (
            "8ac7da0d5adb3ae49e3a3dfe084494f4dce051286d219e346b4ea11c467cb3cf",
            "6efa91af1af5c8b88fa5b024c683a53962ab9b7aa0c02b82173babb45966c77d"),
    }
    for rel, (bench_sha, axis_sha) in expect.items():
        src = open(os.path.join(REPO, rel)).read()
        assert bench_sha in src, f"{rel} does not pin the benchmark sha256"
        assert axis_sha in src, f"{rel} does not pin the gene-axis sha256"
        # It must FAIL on a mismatch, not print a warning and carry on.
        assert "exit 3" in src, f"{rel} does not exit non-zero on a digest mismatch"


def test_the_benchmark_digests_match_the_configs_they_are_asserted_against():
    """The rebuild scripts and the configs must pin the SAME benchmark."""
    import yaml

    class _L(yaml.SafeLoader):
        pass

    _L.add_constructor("!include", lambda loader, node: {})
    pairs = (("configs/rhj_fewshot.yaml", "scripts/repro/build_benchmark_rhj.sh"),
             ("configs/norman_combo.yaml", "scripts/repro/build_benchmark_norman.sh"))
    for cfg_rel, sh_rel in pairs:
        with open(os.path.join(REPO, cfg_rel)) as fh:
            sha = yaml.load(fh, Loader=_L)["stages"]["vcell"]["benchmark"]["sha256"]
        src = open(os.path.join(REPO, sh_rel)).read()
        assert sha in src, f"{sh_rel} pins a different benchmark than {cfg_rel}"


def test_the_competitor_script_refuses_an_unknown_arm_and_names_the_deviations():
    src = open(os.path.join(REPO, "scripts/repro/competitors_rhj.sh")).read()
    for arm in ("txpert", "state", "cellflow", "perturbdiff"):
        assert arm in src, arm
    assert "unknown ARM" in src, "it does not refuse an unknown arm"
    # A competitor number without its deviations is a misattribution.
    assert "WITHDRAWN" in src, "the withdrawn TxPert number is not flagged"
    assert "single-cell controls" in src.lower() or "SINGLE-CELL" in src


def test_the_baseline_script_covers_both_benchmarks_and_states_the_expected_values():
    src = open(os.path.join(REPO, "scripts/repro/baselines.sh")).read()
    assert "rhj_fewshot|norman_combo" in src, "it does not validate BENCHMARK"
    assert "--mode reference" in src
    for anchor in ("0.1324", "0.2323", "0.3722", "0.8824", "0.7757"):
        assert anchor in src, f"the {anchor} anchor is not stated in the header"


def test_the_metric_export_defaults_to_the_published_scope():
    """Git tracks our PUBLISHED arms plus every baseline and competitor -- not the sweep."""
    src = open(os.path.join(REPO, "scripts/export_split_metrics.py")).read()
    assert 'ap.add_argument("--scope", default="published"' in src
    assert 'PUBLISHED_ARMS = {"Y06", "NQ02"}' in src
    # Only OUR arms may be filtered by scope. Filtering on the eval-root name instead
    # dropped every baseline silently, because the reference arms live in roots called
    # `reference`, which matches neither PUBLISHED_ARMS nor the competitor pattern.
    assert 'and kind == "occam"' in src, \
        "the scope filter is not restricted to our own arms"
