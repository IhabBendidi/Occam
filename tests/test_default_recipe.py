"""The adopted default recipe is pinned here, so it cannot drift silently.

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


def test_forward_composed_defaults_to_decode_seq_mean():
    import inspect

    from occam.vcell.model import VirtualCellModel

    sig = inspect.signature(VirtualCellModel.forward_composed)
    assert sig.parameters["mode"].default == "decode_seq_mean"
    assert "decode_seq_mean" in VirtualCellModel.RECON_BASE_MODES


def test_eval_splithalf_defaults_to_decode_seq_mean():
    src = open(os.path.join(REPO, "scripts/eval_splithalf.py")).read()
    assert 'ap.add_argument("--compose", default="decode_seq_mean"' in src
