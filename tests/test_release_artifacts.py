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
