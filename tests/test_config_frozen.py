"""The published recipe must match the released checkpoints, field by field.

`configs/occam_small.yaml` is the paper's recipe. If it drifts from the weights
it claims to describe, the repo is lying about how the model was trained -- the
exact failure this whole port exists to fix.

Two properties:

1. Every hyperparameter in the YAML equals the value embedded in the
   corresponding checkpoint.
2. Every numeric field parses as a *number*. PyYAML resolves `3e-06` as a
   string, so an lr written without a dot in the mantissa would silently become
   `"3e-06"` and fail far from its cause.
"""

from __future__ import annotations

import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(REPO, "configs", "occam_small.yaml")

CHECKPOINTS = {
    "pretrain": "checkpoints/reference/occam_small_stage1_pretrain.pth",
    "finetune": "checkpoints/reference/occam_small_stage2_finetune.pth",
    # configs/occam_small.yaml records the recipe that produced the LEGACY
    # decoder, so that is what its stage-3 block must agree with. The released
    # decoder came from configs/decoder_softplus_nb.yaml; see
    # test_retrain_recipe_agrees_with_the_released_decoder below.
    "decoder": "checkpoints/legacy/occam_small_encoder_decoder_rectified_sigmoid.pth",
}

# The values that define the released model. Verified 2026-09-10 from the
# checkpoints' own embedded configs; see docs/reproducing_chain_a.md.
EXPECTED = {
    "pretrain": {
        "model.model_size": "small",
        "optim.epochs": 100,
        "optim.batch_size": 1024,
        "optim.learning_rate": 0.0004,
        "optim.weight_decay": 1.81890136e-09,
        "optim.lr_scheduler": "cosine",
        "optim.warmup_epochs": 15,
        "masking.mask_probability": 0.1,
        "masking.corruption_mode": "swap",
        "loss.reconstruction_loss": "nb_softplus",     # NOT poisson
        "loss.masked_data_weight": 0.75,
        "loss.mask_loss_weight": 0.7,
        "loss.mlw_schedule_start": 0.7,                # NOT 0.0
        "loss.mlw_schedule_end": 0.9,                  # NOT 0.7
        "loss.clamp_max_log1p": 5.5,
        "data.data_mode": "optim",
        "data.target_sum": 2000.0,
        "data.val_split": 0.1,
    },
    "finetune": {
        "unmasking.mode": "gradual",
        "unmasking.zero_at": 0.9,
        "optim.epochs": 100,
        "optim.learning_rate": 0.0006,
        "optim.warmup_epochs": 0,
        "masking.mask_probability": 0.0,               # inert under gradual
        "loss.recon_loss": "poisson",
        "loss.masked_data_weight": 0.0,
        "loss.mlw_schedule_start": 0.7,
        "loss.mlw_schedule_end": 0.0,
        "loss.mlw_schedule_cooldown": 0.1,
    },
    "decoder": {
        "architecture.decoder_hidden_dim": 2048,
        "architecture.decoder_num_blocks": 5,          # code default is 4
        "architecture.decoder_dropout": 0.3,
        "architecture.decoder_se_reduction": 16,
        "architecture.decoder_output_activation": "rectified_sigmoid",
        "optim.lr": 3.0e-06,                           # NOT 1e-05
        "optim.eta_min": 1.0e-07,
        "optim.weight_decay": 0.01,
        "optim.gradient_clip": 0.5,
        "optim.batch_size": 1024,
        "optim.epochs": 200,
        "optim.loss_type": "smooth_l1",
        "optim.smooth_l1_beta": 2.0,
        "encoder.freeze_encoder": True,
    },
}

NUMERIC_KEY_HINTS = (
    "lr", "learning_rate", "weight_decay", "eta_min", "epochs", "batch_size",
    "gradient_clip", "beta", "probability", "weight", "sigma", "dropout",
    "split", "target_sum", "warmup", "cooldown", "reduction", "hidden_dim",
    "num_blocks", "clamp", "factor", "freq", "patience", "interval",
)


def _load_yaml_without_include():
    """Parse the config, tolerating the `!include` tag the real loader handles."""
    yaml = pytest.importorskip("yaml")

    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_constructor(
        "!include", lambda loader, node: {"__include__": loader.construct_scalar(node)}
    )
    with open(CONFIG, encoding="utf-8") as fh:
        return yaml.load(fh, Loader=_Loader)


def _dig(mapping, dotted):
    node = mapping
    for part in dotted.split("."):
        assert part in node, f"missing key: {dotted} (at {part!r})"
        node = node[part]
    return node


def test_config_exists():
    assert os.path.isfile(CONFIG), (
        f"{CONFIG} missing. Regenerate with "
        f"the frozen reference settings"
    )


def test_recipe_matches_the_released_checkpoints():
    cfg = _load_yaml_without_include()
    stages = cfg["stages"]
    problems = []
    for stage, expected in EXPECTED.items():
        for dotted, want in expected.items():
            got = _dig(stages[stage], dotted)
            if isinstance(want, float) and isinstance(got, (int, float)):
                same = abs(got - want) <= 1e-12 * max(1.0, abs(want))
            else:
                same = got == want
            if not same:
                problems.append(f"stages.{stage}.{dotted}: expected {want!r}, got {got!r}")
    assert not problems, "the published recipe drifted:\n  " + "\n  ".join(problems)


def test_every_numeric_field_parsed_as_a_number():
    """Guard against the YAML 1.1 exponent trap (`3e-06` resolves to a string)."""
    cfg = _load_yaml_without_include()
    offenders = []

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, str):
            leaf = path.rsplit(".", 1)[-1]
            if any(h in leaf for h in NUMERIC_KEY_HINTS):
                # A numeric-looking string here means the exponent trap bit.
                try:
                    float(node)
                except (TypeError, ValueError):
                    return
                offenders.append(f"{path} = {node!r} parsed as str, not a number")

    walk(cfg)
    assert not offenders, (
        "numeric fields parsed as strings (YAML 1.1 needs a dot in the "
        "mantissa, e.g. 3.0e-06 not 3e-06):\n  " + "\n  ".join(offenders)
    )


def test_stage3_records_the_collision_caveats():
    """The recipe must carry the provenance warnings, not just the values."""
    with open(CONFIG, encoding="utf-8") as fh:
        text = fh.read()
    for needle in (
        "colliding SLURM array tasks",
        "docs/reproducing_chain_a.md",
        "code default is 4",
        "inert under mode=gradual",
    ):
        assert needle in text, f"the recipe no longer explains: {needle!r}"


@pytest.mark.skipif(
    not all(os.path.isfile(os.path.join(REPO, p)) for p in CHECKPOINTS.values()),
    reason="checkpoints not bootstrapped; run scripts/bootstrap_assets.sh",
)
def test_recipe_agrees_with_checkpoints_on_disk():
    """Strongest form: read the checkpoints and compare, not just the frozen dict."""
    torch = pytest.importorskip("torch")
    cfg = _load_yaml_without_include()

    ck1 = torch.load(os.path.join(REPO, CHECKPOINTS["pretrain"]),
                     map_location="cpu", weights_only=False)
    c1 = ck1["config"]
    c1 = c1 if isinstance(c1, dict) else vars(c1)
    p1 = cfg["stages"]["pretrain"]
    assert p1["loss"]["reconstruction_loss"] == c1["reconstruction_loss"]
    assert p1["loss"]["mlw_schedule_start"] == c1["mlw_schedule_start"]
    assert p1["loss"]["mlw_schedule_end"] == c1["mlw_schedule_end"]
    assert p1["optim"]["learning_rate"] == c1["learning_rate"]

    ck3 = torch.load(os.path.join(REPO, CHECKPOINTS["decoder"]),
                     map_location="cpu", weights_only=False)
    c3 = ck3["training_config"]
    d3 = ck3["decoder_config"]
    p3 = cfg["stages"]["decoder"]
    assert p3["optim"]["lr"] == c3["lr"], (
        f"recipe lr {p3['optim']['lr']} != checkpoint lr {c3['lr']}"
    )
    assert p3["architecture"]["decoder_num_blocks"] == d3["num_blocks"]
    assert p3["architecture"]["decoder_hidden_dim"] == d3["hidden_dim"]


RELEASE = "checkpoints/release/occam_small_encoder_decoder.pth"


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(REPO, RELEASE)),
    reason="checkpoints not bootstrapped; run scripts/bootstrap_assets.sh",
)
def test_retrain_recipe_agrees_with_the_released_decoder():
    """The published config must describe the weights actually shipped.

    The released decoder was produced by configs/decoder_softplus_nb.yaml, not
    by the stage-3 block of configs/occam_small.yaml -- that one describes the
    superseded rectified_sigmoid decoder now under checkpoints/legacy/. A
    recipe that disagrees with its own checkpoint is the exact defect this
    repository was built to remove.
    """
    torch = pytest.importorskip("torch")

    from occam.config.loader import load_config

    cfg = load_config(os.path.join(REPO, "configs", "decoder_softplus_nb.yaml"))
    dc, _meta = cfg.build_decoder_config()

    ck = torch.load(os.path.join(REPO, RELEASE), map_location="cpu", weights_only=False)
    d = ck["decoder_config"]
    t = ck["training_config"]

    assert d["output_activation"] == dc.decoder_output_activation == "softplus"
    assert t["loss_type"] == dc.loss_type == "nb"
    assert d["hidden_dim"] == dc.decoder_hidden_dim == 2048
    assert d["num_blocks"] == dc.decoder_num_blocks == 4
    assert t["lr"] == dc.lr
    assert ck["target_sum"] == 2000.0
    assert len(ck["gene_names"]) == 44728
