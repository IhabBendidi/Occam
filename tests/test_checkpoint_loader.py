"""The shared checkpoint loader must handle both checkpoint shapes.

Occam ships plain MAE checkpoints (stages 1-2) and unified encoder+decoder
checkpoints (stage 3, the released artifact). The unified one's
``model_state_dict`` is byte-identical to its ``decoder_state_dict``, so code
that treats it as the whole model gets a silent encoder-shaped hole. These
tests pin the distinction.
"""

from __future__ import annotations

import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE = os.path.join(REPO, "checkpoints", "release", "occam_small_encoder_decoder.pth")
STAGE1 = os.path.join(REPO, "checkpoints", "reference", "occam_small_stage1_pretrain.pth")

_reason = "checkpoints not bootstrapped; run scripts/bootstrap_assets.sh"


@pytest.mark.skipif(not os.path.isfile(RELEASE), reason=_reason)
def test_unified_checkpoint_is_detected_and_loads_the_encoder_half():
    torch = pytest.importorskip("torch")

    from occam.inference.loader import checkpoint_kind, load_encoder_wrapper

    ck = torch.load(RELEASE, map_location="cpu", weights_only=False)
    assert checkpoint_kind(ck) == "unified"

    w = load_encoder_wrapper(RELEASE, device="cpu")
    assert len(w._gene_names) == 44728
    assert w.target_sum == 2000.0
    # The released encoder was trained on raw counts, so both transforms apply.
    assert [type(t).__name__ for t in w._transforms] == ["NormalizePerCell", "Log1p"]

    x = torch.rand(4, 44728) * 5.0
    with torch.no_grad():
        emb = w.model.feature(x)
    assert emb.shape == (4, 512)
    assert torch.isfinite(emb).all()


@pytest.mark.skipif(not os.path.isfile(RELEASE), reason=_reason)
def test_unified_model_state_dict_is_the_decoder_not_the_model():
    """Pins the trap, so nobody 'simplifies' the loader back into it."""
    torch = pytest.importorskip("torch")

    ck = torch.load(RELEASE, map_location="cpu", weights_only=False)
    assert len(ck["model_state_dict"]) == len(ck["decoder_state_dict"])
    assert set(ck["model_state_dict"]) == set(ck["decoder_state_dict"])
    # And it is emphatically NOT the encoder.
    assert set(ck["model_state_dict"]) != set(ck["encoder_state_dict"])


@pytest.mark.skipif(not os.path.isfile(STAGE1), reason=_reason)
def test_plain_mae_checkpoint_loads_the_full_autoencoder():
    torch = pytest.importorskip("torch")

    from occam.inference.loader import checkpoint_kind, load_encoder_wrapper

    ck = torch.load(STAGE1, map_location="cpu", weights_only=False)
    assert checkpoint_kind(ck) == "mae"

    w = load_encoder_wrapper(STAGE1, device="cpu")
    x = torch.rand(4, 44728) * 5.0
    with torch.no_grad():
        emb = w.model.feature(x)
    assert emb.shape == (4, 512)


@pytest.mark.skipif(not os.path.isfile(RELEASE), reason=_reason)
def test_both_shapes_agree_on_the_encoder_geometry():
    from occam.inference.loader import load_encoder_wrapper

    a = load_encoder_wrapper(RELEASE, device="cpu")
    b = load_encoder_wrapper(STAGE1, device="cpu")
    assert a.config.num_genes == b.config.num_genes == 44728
    assert a.config.model_size == b.config.model_size == "small"
    assert a._gene_names == b._gene_names, (
        "the released decoder and its grandparent encoder must share a gene "
        "vocabulary; if they do not, the chain of custody is broken"
    )


def test_missing_file_raises_a_clear_error():
    from occam.inference.loader import load_encoder_wrapper

    with pytest.raises(FileNotFoundError):
        load_encoder_wrapper("/nonexistent/checkpoint.pth", device="cpu")
