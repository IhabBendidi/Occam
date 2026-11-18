"""Pin the training recipes used by the official all-source checkpoints."""
import os

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = ("configs/rhj_fewshot_official.yaml", "configs/norman_combo_official.yaml")


def _vcell(rel):
    class Loader(yaml.SafeLoader):
        pass
    Loader.add_constructor("!include", lambda loader, node: {})
    with open(os.path.join(REPO, rel)) as stream:
        return yaml.load(stream, Loader=Loader)["stages"]["vcell"]


@pytest.mark.parametrize("rel,epochs,batch_size,eta_min,delta_cos", [
    (CONFIGS[0], 180, 2048, 7e-4, 0.5),
    (CONFIGS[1], 150, 32, 3e-4, 0.0255),
])
def test_official_all_source_recipes(rel, epochs, batch_size, eta_min, delta_cos):
    cfg = _vcell(rel)
    optim, loss, graph = cfg["optim"], cfg["loss"], cfg["graph"]
    assert optim["epochs"] == epochs
    assert optim["batch_size"] == batch_size
    assert optim["eta_min"] == pytest.approx(eta_min)
    assert optim["lr"] == optim["decoder_lr"] == pytest.approx(1e-3)
    assert optim["warmup_epochs"] == 5
    assert cfg["encoder"]["freeze_encoder"] is True
    assert cfg["encoder"]["freeze_decoder"] is False
    assert graph["graph_types"] == ["go", "string"]
    assert graph["pert_model_type"] == "adjacency_map"
    assert graph["pert_embedding_paths"] == [
        "data/assets/genept_full_occam_rhj_fewshot.pkl",
        "data/assets/esm2_t33_650M_UR50D_occam_rhj_fewshot.pkl",
    ]
    assert loss["loss_type"] == "smooth_l1"
    assert loss["delta_cos_weight"] == pytest.approx(delta_cos)
    for key in ("nb_weight", "cosine_loss_weight", "delta_mag_weight", "latent_loss_weight",
                "latent_cos_weight", "vicreg_var_weight", "vicreg_cov_weight"):
        assert loss[key] == 0, key
    assert loss["latent_loss_type"] == "none"
    assert loss["recon_split"] is False
    assert cfg["init_from"] == "checkpoints/release/occam_small_encoder_decoder.pth"


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

    cfg = _vcell("configs/norman_combo_official.yaml")["data"]
    mode = cfg.get("panel_compose")
    assert mode == "decode_seq_mean", \
        f"norman_combo selects on {mode!r} but reports decode_seq_mean"
    assert mode in VirtualCellModel.COMPOSE_MODES

    # rhj_fewshot is single-gene: every recon mode is identical there, so the key is
    # deliberately absent rather than set to a value that would imply it matters.
    rhj = _vcell("configs/rhj_fewshot_official.yaml")["data"]
    assert not rhj.get("panel_compose"), \
        "rhj_fewshot is single-gene; panel_compose would be a no-op that implies otherwise"

    # All three config layers, or the key is silently ignored (the F-loss1 failure).
    import occam.config.schema as schema
    src = open(schema.__file__).read()
    assert '"panel_compose": "panel_compose"' in src, "not reachable via --set"
    from occam.vcell.config import VirtualCellConfig
    assert hasattr(VirtualCellConfig(), "panel_compose"), "not on the dataclass"
