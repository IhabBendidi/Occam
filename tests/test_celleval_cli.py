"""`occam-celleval` must take the decoder's shape from the checkpoint, not flags.

Two failures this guards against, both silent:

* A geometry flag that disagrees with the weights. Reading ``decoder_config``
  makes the disagreement impossible rather than unlikely.
* A guessed output activation. A softplus-trained decoder loads ``strict=True``
  under ``rectified_sigmoid`` and then emits ~99% exact zeros -- a dead model
  that looks healthy and scores like a broken one. The released checkpoint does
  not record the field, so the command must refuse rather than default.
"""

from __future__ import annotations

import argparse
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE = os.path.join(REPO, "checkpoints", "release", "occam_small_encoder_decoder.pth")


def _args(**overrides):
    """The CLI's parsed-args surface, with the defaults build_config reads."""
    base = dict(
        checkpoint="/nonexistent.pth", output_activation=None, loss_type=None,
        pert_col="gene_name", control_label="non-targeting", calibrate="none",
        modes="injctrl", num_threads=16, timeout=3600, all_genes=False,
        de_batch_size=None, de_real_cache=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _checkpoint(activation=None, **decoder_config):
    dcfg = dict(input_dim=512, output_dim=44728, hidden_dim=2048, num_blocks=5,
                dropout_rate=0.3, se_reduction=16)
    dcfg.update(decoder_config)
    if activation is not None:
        dcfg["output_activation"] = activation
    return {"decoder_config": dcfg, "training_config": {"loss_type": "smooth_l1"}}


def test_geometry_comes_from_the_checkpoint():
    pytest.importorskip("torch")
    from occam.cli.celleval import build_config

    cfg = build_config(_checkpoint(activation="softplus", hidden_dim=1024,
                                   num_blocks=7, dropout_rate=0.1), _args())
    assert cfg.decoder_hidden_dim == 1024
    assert cfg.decoder_num_blocks == 7
    assert cfg.decoder_dropout == 0.1
    assert cfg.decoder_output_activation == "softplus"
    # The loss module has to be the right class for NB/ZINB dispersion to load.
    assert cfg.loss_type == "smooth_l1"


def test_missing_output_activation_is_refused_not_defaulted():
    pytest.importorskip("torch")
    from occam.cli.celleval import build_config

    with pytest.raises(ValueError) as exc:
        build_config(_checkpoint(activation=None), _args())
    message = str(exc.value)
    assert "occam-verify" in message
    assert "--output-activation" in message


def test_an_explicit_activation_is_accepted_when_the_checkpoint_omits_it():
    pytest.importorskip("torch")
    from occam.cli.celleval import build_config

    cfg = build_config(_checkpoint(activation=None),
                       _args(output_activation="rectified_sigmoid"))
    assert cfg.decoder_output_activation == "rectified_sigmoid"


def test_celleval_settings_are_passed_through():
    pytest.importorskip("torch")
    from occam.cli.celleval import build_config

    cfg = build_config(
        _checkpoint(activation="softplus"),
        _args(pert_col="gene_name", calibrate="quantile", modes="injctrl",
              num_threads=16, timeout=36000, all_genes=False,
              de_batch_size=2000))
    assert cfg.celleval_pert_col == "gene_name"
    assert cfg.celleval_calibrate == "quantile"
    assert cfg.celleval_modes == "injctrl"
    assert cfg.celleval_num_threads == 16
    assert cfg.celleval_timeout == 36000
    # cell-eval's default DE batch size trips a shared-memory race on panels
    # with thousands of perturbations, so this has to reach the runner.
    assert cfg.celleval_de_batch_size == 2000
    assert cfg.celleval_present_genes_only is True
    assert cfg.freeze_encoder is True


def test_a_non_unified_checkpoint_is_rejected():
    pytest.importorskip("torch")
    from occam.cli.celleval import build_config

    with pytest.raises(ValueError) as exc:
        build_config({"model_state_dict": {}}, _args())
    assert "decoder_config" in str(exc.value)


@pytest.mark.skipif(not os.path.isfile(RELEASE),
                    reason="checkpoints not bootstrapped; run scripts/bootstrap_assets.sh")
def test_the_released_checkpoint_really_does_omit_the_activation():
    """The refusal above is only meaningful if the release actually omits it."""
    torch = pytest.importorskip("torch")

    ck = torch.load(RELEASE, map_location="cpu", weights_only=False)
    assert "output_activation" not in ck["decoder_config"]
    assert ck["decoder_config"]["num_blocks"] == 5, "released decoder is 2048 x 5"


def test_de_real_cache_is_keyed_by_the_target_content():
    """Reuse must be impossible across different targets, not merely discouraged.

    The ground-truth DE is reused across arms. Keying it on the target file's
    own hash is what makes that safe: a target built from different data,
    different transforms or a different gene subset lands on a different path
    and misses the cache, rather than being silently scored against another
    arm's ground truth.
    """
    pytest.importorskip("torch")
    import tempfile

    from occam.decoder_train.metrics import real_de_cache_path

    with tempfile.TemporaryDirectory() as tmp:
        a = os.path.join(tmp, "target_a.h5ad")
        b = os.path.join(tmp, "target_b.h5ad")
        c = os.path.join(tmp, "target_a_copy.h5ad")
        with open(a, "wb") as fh:
            fh.write(b"cells x genes A")
        with open(b, "wb") as fh:
            fh.write(b"cells x genes B")
        with open(c, "wb") as fh:
            fh.write(b"cells x genes A")

        path_a, digest_a = real_de_cache_path(tmp, a)
        path_b, digest_b = real_de_cache_path(tmp, b)
        path_c, digest_c = real_de_cache_path(tmp, c)

        assert digest_a != digest_b and path_a != path_b
        # Same bytes anywhere on disk -> same cache entry. That is the reuse.
        assert digest_a == digest_c and path_a == path_c
        assert os.path.dirname(path_a) == tmp
        assert path_a.endswith(".csv")


def test_de_real_cache_reaches_the_config():
    pytest.importorskip("torch")
    from occam.cli.celleval import build_config

    cfg = build_config(_checkpoint(activation="softplus"),
                       _args(de_real_cache="/tmp/cache"))
    assert cfg.celleval_de_real_cache == "/tmp/cache"


def test_an_explicit_activation_may_not_contradict_the_checkpoint():
    """The checkpoint is the authority when it records one.

    A launcher carrying `ACT=rectified_sigmoid` as a default, pointed at a
    softplus checkpoint, would otherwise score a dead model: rectified_sigmoid
    on softplus-trained weights emits ~99% exact zeros and loads strict=True
    without complaint.
    """
    pytest.importorskip("torch")
    from occam.cli.celleval import build_config

    with pytest.raises(ValueError) as exc:
        build_config(_checkpoint(activation="softplus"),
                     _args(output_activation="rectified_sigmoid"))
    assert "contradicts" in str(exc.value)

    # Agreeing with the record is fine.
    cfg = build_config(_checkpoint(activation="softplus"),
                       _args(output_activation="softplus"))
    assert cfg.decoder_output_activation == "softplus"
