"""End-to-end check that the three-stage pipeline actually runs.

Trains a real (tiny) model on a real (tiny) synthetic corpus, on CPU, in about
a minute. Slow by unit-test standards and worth it: this is the only test that
exercises the chunked reader, both loss modules, the schedules, the stage
fence, the ledger, artifact hand-off and the unified-checkpoint writer
together. Every bug the static checks missed showed up here.

Skipped unless the smoke artifacts already exist. Produce them with:

    python scripts/make_smoke_corpus.py --out /tmp/occam_smoke.h5ad
    python -m occam.train --config configs/occam_small_smoke.yaml \
        --set data.train=/tmp/occam_smoke.h5ad
"""

from __future__ import annotations

import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(REPO, "runs", "occam_small_smoke")
DECODER_CKPT = os.path.join(RUN, "decoder", "checkpoints", "best_decoder.pth")

_HAVE_RUN = os.path.isfile(os.path.join(RUN, "ledger.json"))
_reason = f"no smoke run at {RUN}; see this module's docstring"


@pytest.mark.skipif(not _HAVE_RUN, reason=_reason)
def test_all_three_stages_completed():
    from occam.pipeline.ledger import Ledger, StageStatus

    led = Ledger.load(RUN)
    for stage in ("pretrain", "finetune", "decoder"):
        assert led.status(stage) == StageStatus.COMPLETED, \
            f"stage {stage} is {led.status(stage)}"
        assert led.artifacts(stage), f"stage {stage} recorded no artifacts"
        assert not led.missing_artifacts(stage), \
            f"stage {stage} artifacts missing from disk: {led.missing_artifacts(stage)}"


@pytest.mark.skipif(not _HAVE_RUN, reason=_reason)
def test_artifacts_land_in_the_deterministic_run_dir():
    """Not in a globally-numbered exp_<NNN> directory.

    The research tree's output manager allocated `exp_<NNN>` from a shared
    registry, and the counter raced between concurrent array tasks. Two of the
    three released checkpoints became unattributable as a result.
    """
    from occam.pipeline.ledger import Ledger

    led = Ledger.load(RUN)
    for stage in ("pretrain", "finetune", "decoder"):
        for name, path in led.artifacts(stage).items():
            assert path.startswith(RUN), \
                f"{stage}/{name} escaped the run dir: {path}"
            assert "exp_" not in os.path.relpath(path, RUN), \
                f"{stage}/{name} went through the global experiment registry: {path}"


@pytest.mark.skipif(not _HAVE_RUN, reason=_reason)
def test_stage_two_consumed_stage_one_and_stage_three_consumed_stage_two():
    from occam.pipeline.ledger import Ledger

    led = Ledger.load(RUN)
    assert led.entry("finetune").get("init_from") == led.artifact("pretrain", "final_model")
    assert led.entry("decoder").get("init_from") == led.artifact("finetune", "final_model")


@pytest.mark.skipif(not _HAVE_RUN, reason=_reason)
def test_overlay_audit_records_the_gradual_unmasking_source():
    """Stage 2's effective mask probability comes from the PARENT checkpoint."""
    import json

    path = os.path.join(RUN, "finetune", "config", "overlay_audit.json")
    assert os.path.isfile(path), f"overlay audit missing: {path}"
    with open(path, encoding="utf-8") as fh:
        audit = json.load(fh)

    g = audit["gradual_unmasking"]
    assert g["mode"] == "gradual"
    assert g["config_mask_probability"] == 0.0, "the config value should be the inert 0.0"
    assert g["effective_mask_probability"] == 0.1, \
        "the effective value must come from the parent checkpoint, not the config"
    assert g["source"] == "parent_checkpoint.mask_probability"

    # And the loss really does differ between the stages, exercising the
    # strict=False hand-off that drops the NB dispersion parameter.
    overlay = audit["overlay"]
    assert overlay["recon_loss"]["inherited_from_parent"] == "nb_softplus"
    assert overlay["recon_loss"]["effective"] == "poisson"


@pytest.mark.skipif(not os.path.isfile(DECODER_CKPT), reason=_reason)
def test_produced_checkpoint_is_a_loadable_unified_checkpoint():
    """The smoke run must produce the same shape of artifact as the release."""
    torch = pytest.importorskip("torch")

    from occam.decoder.external import HybridNormGlobalSkipDecoder
    from occam.model import OccamEncoder

    ck = torch.load(DECODER_CKPT, map_location="cpu", weights_only=False)
    assert ck["is_unified_checkpoint"] is True
    for key in ("encoder_state_dict", "decoder_state_dict", "encoder_config",
                "decoder_config", "gene_names", "gene_to_index", "target_sum"):
        assert key in ck, f"missing checkpoint key: {key}"

    # The defect the released checkpoint has and this one must not: the output
    # activation is recorded, so a reloader cannot silently pick the wrong one.
    assert ck["decoder_config"].get("output_activation"), \
        ("output_activation is not recorded. A softplus decoder reloaded under "
         "the rectified_sigmoid default loads strict=True and emits ~99% exact "
         "zeros -- a dead model that looks fine.")

    # Provenance is intrinsic to the artifact, not a sidecar file.
    assert ck.get("parent_checkpoint_path"), "checkpoint does not record its parent"

    enc, info = OccamEncoder.from_checkpoint(ck, strict=True)
    assert info["missing"] == [] and info["unexpected"] == []

    d = dict(ck["decoder_config"])
    dec = HybridNormGlobalSkipDecoder(
        input_dim=d["input_dim"], output_dim=d["output_dim"],
        hidden_dim=d["hidden_dim"], num_blocks=d["num_blocks"],
        dropout_rate=d["dropout_rate"], se_reduction=d["se_reduction"],
        output_activation=d["output_activation"],
    )
    missing, unexpected = dec.load_state_dict(ck["decoder_state_dict"], strict=True)
    assert list(missing) == [] and list(unexpected) == []

    enc.eval()
    dec.eval()
    with torch.no_grad():
        out = dec(enc(torch.rand(4, ck["encoder_config"]["num_genes"]) * 5.0))
    assert out.shape == (4, ck["decoder_config"]["output_dim"])
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not _HAVE_RUN, reason=_reason)
def test_rerunning_reuses_every_stage():
    """Same config, same data, same parent -> nothing retrains.

    The train path is read back from the run's own ``resolved_config.yaml``
    rather than from the checked-in config, because the smoke run is normally
    launched with ``--set data.train=<tmp path>``. Using the config default
    here would change the data and correctly invalidate every stage, which
    would test the wrong thing.
    """
    yaml = pytest.importorskip("yaml")

    from occam.config.loader import load_config
    from occam.pipeline.ledger import Ledger
    from occam.pipeline.orchestrator import Orchestrator

    resolved_path = os.path.join(RUN, "resolved_config.yaml")
    assert os.path.isfile(resolved_path), f"missing {resolved_path}"
    with open(resolved_path, encoding="utf-8") as fh:
        resolved = yaml.safe_load(fh)
    train = resolved["data"]["train"]

    led = Ledger.load(RUN)
    cfg = load_config(
        os.path.join(REPO, "configs", "occam_small_smoke.yaml"),
        overrides=[f"data.train={train}"],
        ledger=led.data,
    )
    plan = dict(Orchestrator(cfg).plan())
    assert plan == {"pretrain": "reuse", "finetune": "reuse", "decoder": "reuse"}, plan


@pytest.mark.skipif(not _HAVE_RUN, reason=_reason)
def test_changing_a_hyperparameter_blocks_the_affected_stage_only():
    """Editing stage 3 must not invalidate stages 1 and 2."""
    yaml = pytest.importorskip("yaml")

    from occam.config.loader import load_config
    from occam.pipeline.ledger import Ledger
    from occam.pipeline.orchestrator import Orchestrator

    with open(os.path.join(RUN, "resolved_config.yaml"), encoding="utf-8") as fh:
        train = yaml.safe_load(fh)["data"]["train"]

    led = Ledger.load(RUN)
    cfg = load_config(
        os.path.join(REPO, "configs", "occam_small_smoke.yaml"),
        overrides=[f"data.train={train}", "stages.decoder.optim.lr=1.0e-05"],
        ledger=led.data,
    )
    plan = dict(Orchestrator(cfg).plan())
    assert plan["pretrain"] == "reuse"
    assert plan["finetune"] == "reuse"
    assert plan["decoder"] == "blocked", (
        "a changed learning rate must block rather than reuse weights trained "
        "under the old one"
    )
