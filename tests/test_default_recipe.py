"""The historical Q02 recipe and the official Norman NF32ALLA recipe are pinned here.

WHY THIS FILE EXISTS. On 2026-09-22 the user adopted one default for every vcell
benchmark: the **Q02 loss base at 150 epochs with eta_min 3e-4**, and
**`decode_seq_mean`** as the multi-perturbation composition mode. Both were chosen from
measured results (docs/rhj_splithalf.md F-sched3/F-sched4, docs/norman_results.md
F-nor5..F-nor9), and both live in places that are easy to edit by accident: a default
argument, an argparse default, and two YAML blocks.

⚠️ The LOSS WEIGHTS are deliberately NOT asserted to be equal across datasets. `dcos`,
`delta_mag` and the latent MSE are scale-matched to each corpus's own measured term
magnitudes (RHJ 0.5 / 0.01363 / 0.0044 vs Norman 0.32977 / 0.0400866 / 0.00198826), so a
shared literal would be a bug, not consistency. What IS shared is the SHAPE of the base:
all three terms on, latent_loss_type delta_z.
"""
import os

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = ("configs/rhj_fewshot.yaml", "configs/norman_combo.yaml")


def _vcell(rel):
    class _L(yaml.SafeLoader):
        pass

    # `paths: !include paths.yaml` -- irrelevant to the optim/loss blocks under test.
    _L.add_constructor("!include", lambda loader, node: {})
    with open(os.path.join(REPO, rel)) as fh:
        return yaml.load(fh, Loader=_L)["stages"]["vcell"]


def test_the_default_schedule_is_150_epochs_with_eta_min_3e_4():
    for rel in CONFIGS:
        optim = _vcell(rel)["optim"]
        assert optim["epochs"] == 150, rel
        if "norman" in rel:   # NB10, adopted 2026-09-23 (docs/norman_results.md F-sweep1)
            assert optim["eta_min"] == pytest.approx(1e-3), rel
            assert optim["batch_size"] == 256, rel
        else:
            assert optim["eta_min"] == pytest.approx(3e-4), rel
            assert optim["batch_size"] == 2048, rel
        assert optim["lr"] == pytest.approx(1e-3), rel
        assert optim["decoder_lr"] == pytest.approx(1e-3), rel
        assert optim["warmup_epochs"] == 5, rel


def test_the_default_loss_base_is_Q02_shaped():
    for rel in CONFIGS:
        loss = _vcell(rel)["loss"]
        assert loss["loss_type"] == "smooth_l1", rel
        # Q02 = M06 (dcos + delta_mag) + the latent MSE. All three must be ON and the
        # latent type must not be "none", or the weight is silently ignored (F-loss1).
        assert loss["delta_cos_weight"] > 0, rel
        assert loss["delta_mag_weight"] > 0, rel
        assert loss["latent_loss_weight"] > 0, rel
        assert loss["latent_loss_type"] == "delta_z", rel
        # M03/M06/Q02 differ only in those three; nothing else creeps into the base.
        assert loss["vicreg_var_weight"] == 0.0, rel
        assert loss["vicreg_cov_weight"] == 0.0, rel
        assert loss["latent_cos_weight"] == 0.0, rel
        assert loss["recon_split"] is False, rel
        assert loss.get("dcos_conc_weight", 0.0) == 0.0, rel


def test_the_loss_weights_are_NOT_shared_between_datasets():
    """Scale-matching is per corpus; identical weights would mean someone copied them."""
    a, b = (_vcell(r)["loss"] for r in CONFIGS)
    assert a["delta_cos_weight"] != b["delta_cos_weight"]
    assert a["delta_mag_weight"] != b["delta_mag_weight"]
    assert a["latent_loss_weight"] != b["latent_loss_weight"]


def test_official_norman_recipe_is_nf32alla():
    cfg = _vcell("configs/norman_combo_official.yaml")
    optim, loss, graph = cfg["optim"], cfg["loss"], cfg["graph"]
    assert optim["batch_size"] == 32
    assert optim["epochs"] == 150
    assert optim["eta_min"] == pytest.approx(3e-4)
    assert cfg["data"]["panel_compose"] == "decode_seq_mean"
    assert loss["loss_type"] == "smooth_l1"
    assert loss["nb_weight"] == 0
    assert loss["delta_cos_weight"] == pytest.approx(0.0255)
    assert loss["cosine_loss_weight"] == 0
    assert loss["delta_mag_weight"] == 0
    assert loss["latent_loss_weight"] == 0
    assert loss["latent_loss_type"] == "none"
    assert graph["pert_embedding_paths"] == [
        "data/assets/genept_full_occam_rhj_fewshot.pkl",
        "data/assets/esm2_t33_650M_UR50D_occam_rhj_fewshot.pkl",
    ]


def test_forward_composed_defaults_to_decode_seq_mean():
    import inspect

    from occam.vcell.model import VirtualCellModel

    sig = inspect.signature(VirtualCellModel.forward_composed)
    assert sig.parameters["mode"].default == "decode_seq_mean"
    assert "decode_seq_mean" in VirtualCellModel.RECON_BASE_MODES


def test_eval_splithalf_defaults_to_decode_seq_mean():
    src = open(os.path.join(REPO, "scripts/eval_splithalf.py")).read()
    assert 'ap.add_argument("--compose", default="decode_seq_mean"' in src


def test_the_validation_metric_composes_pairs_the_way_the_report_does():
    """Selecting on `sum` while reporting `decode_seq_mean` is a silent mismatch.

    `predict_panel`'s default predict_fn is `model(ctrl, pert_idx)` -- the `sum` mode --
    so before `panel_compose` existed, checkpoint selection on norman_combo optimised a
    different composition than every reported number used. The two differ by 0.04-0.08
    `pearson_delta` on the 72-cell matrix, i.e. more than most effects being selected
    between.
    """
    from occam.vcell.model import VirtualCellModel

    cfg = _vcell("configs/norman_combo.yaml")["data"]
    mode = cfg.get("panel_compose")
    assert mode == "decode_seq_mean", \
        f"norman_combo selects on {mode!r} but reports decode_seq_mean"
    assert mode in VirtualCellModel.COMPOSE_MODES

    # rhj_fewshot is single-gene: every recon mode is identical there, so the key is
    # deliberately absent rather than set to a value that would imply it matters.
    rhj = _vcell("configs/rhj_fewshot.yaml")["data"]
    assert not rhj.get("panel_compose"), \
        "rhj_fewshot is single-gene; panel_compose would be a no-op that implies otherwise"

    # All three config layers, or the key is silently ignored (the F-loss1 failure).
    import occam.config.schema as schema
    src = open(schema.__file__).read()
    assert '"panel_compose": "panel_compose"' in src, "not reachable via --set"
    from occam.vcell.config import VirtualCellConfig
    assert hasattr(VirtualCellConfig(), "panel_compose"), "not on the dataclass"
