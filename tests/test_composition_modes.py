"""The 8 double-perturbation composition modes: identical on singles, correct on doubles.

WHY THIS FILE EXISTS
    A composition mode is an INFERENCE strategy, and a wrong one fails silently: it returns
    a plausible profile and simply scores a bit worse, which reads as "this composition is
    not as good" rather than "this composition is broken". Every property below is therefore
    pinned rather than inspected once by eye.

    The load-bearing one is the FIRST test. All 8 modes must be algebraically identical on a
    SINGLE perturbation, because that is what licenses training once on singles and
    evaluating under every mode without retraining. If they are not, the comparison is
    confounded by whatever else the mode changed.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

N_GENES, N_PERTS, HID, PLAT = 32, 7, 8, 8


class _Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Linear(N_GENES, HID)

    def forward(self, x):
        return torch.tanh(self.f(x))


class _Dec(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Linear(HID, N_GENES)

    def forward(self, z):
        return self.f(z)


class _Gnn(nn.Module):
    """Returns a fixed (N_PERTS, PLAT) table; the graph itself is irrelevant here."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(N_PERTS, PLAT))

    def forward(self, edge_index, edge_weight):
        return self.emb


def _model(interaction="additive", mean_free=False):
    from occam.vcell.model import VirtualCellModel
    torch.manual_seed(0)
    m = VirtualCellModel(
        encoder=_Enc(), gnn=_Gnn(), decoder=_Dec(),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_weight=torch.ones(2),
        encoder_hidden_size=HID, pert_latent_dim=PLAT,
        freeze_encoder=False, freeze_decoder=False,
        interaction_type=interaction, film_mean_free=mean_free,
    )
    m.eval()
    return m


def _ctrl(n=4):
    torch.manual_seed(1)
    return torch.randn(n, N_GENES)


# The two emission families. Every mode is identical to the others IN ITS OWN FAMILY on a
# single perturbation; the families differ from each other by the decoder's reconstruction
# error, which is larger than the effect being predicted (F-recon1).
RECON_MODES = ("sum", "latent_seq_ab", "latent_seq_ba", "latent_seq_mean",
               "decode_seq_ab", "decode_seq_ba", "decode_seq_mean", "delta_sum")
REAL_MODES = ("delta_sum_realctrl", "resid_seq_ab", "resid_seq_ba", "resid_seq_mean")
MODES = RECON_MODES + REAL_MODES


def test_every_RECON_mode_is_identical_on_a_SINGLE_perturbation():
    """The property that licenses train-on-singles / evaluate-under-every-mode.

    A double is the only thing these modes may disagree about. If they differed on a single
    perturbation then the comparison across them would be measuring something other than
    composition, and every number from it would be confounded.
    """
    m = _model()
    c = _ctrl()
    idx = [[2], [5], [0], [3]]
    with torch.no_grad():
        ref = m.forward_composed(c, idx, "sum")["prediction"]
        for mode in RECON_MODES:
            got = m.forward_composed(c, idx, mode)["prediction"]
            assert torch.allclose(ref, got, atol=1e-5), \
                f"{mode} differs from `sum` on a single perturbation (max |d| " \
                f"{(ref - got).abs().max():.3e})"


def test_the_REAL_ctrl_family_agrees_with_ITSELF_on_a_single_perturbation():
    """`c + delta` for every member, and it is NOT `pred` -- by exactly the recon error.

    This is the fact that makes `resid_seq_*` incomparable to `sum` on its own: it changes
    the EMISSION as well as the composition. `delta_sum_realctrl` is the symmetric control
    that holds the emission fixed, which is why it exists.
    """
    m = _model()
    c = _ctrl()
    idx = [[2], [5], [0], [3]]
    with torch.no_grad():
        ref = m.forward_composed(c, idx, "delta_sum_realctrl")["prediction"]
        for mode in REAL_MODES:
            got = m.forward_composed(c, idx, mode)["prediction"]
            assert torch.allclose(ref, got, atol=1e-5), \
                f"{mode} differs from the rest of its own family on a single perturbation"
        pred = m.forward_composed(c, idx, "sum")["prediction"]
        zg = m._pert_embeddings()
        recon = m._zero_pert_recon(m._encode_ctrl(c), zg, c.device)
        # the families differ by exactly (c - recon): the reconstruction error, nothing else
        assert torch.allclose(ref - pred, c - recon, atol=1e-5), \
            "the two emission families differ by something other than the recon error"
        gap = (c - recon).norm(dim=1).mean()
        eff = (pred - recon).norm(dim=1).mean()
        print(f"    recon error {gap:.3f} vs effect size {eff:.3f} "
              f"(ratio {gap / max(eff, 1e-9):.2f}x)")


def test_mode5_is_the_sequential_counterpart_of_delta_sum_realctrl():
    """Both add decoded deltas to the real control; only mode 5 conditions delta_B on A.

    If they came out equal, mode 5's second step is not seeing the first -- i.e. it has
    silently degenerated into the summed version.
    """
    m = _model(interaction="film")
    c = _ctrl()
    idx = [[1, 4], [2, 6], [0, 3], [5, 1]]
    with torch.no_grad():
        summed = m.forward_composed(c, idx, "delta_sum_realctrl")["prediction"]
        seq = m.forward_composed(c, idx, "resid_seq_ab")["prediction"]
    assert not torch.allclose(summed, seq, atol=1e-4), \
        "resid_seq collapsed onto delta_sum_realctrl; delta_B is not conditioned on A"


def test_mode5_equals_c_plus_both_deltas_exactly():
    """Pin the arithmetic of the chain, step by step, against a hand-rolled oracle."""
    m = _model(interaction="film")
    c = _ctrl(2)
    idx = [[1, 4], [0, 3]]
    with torch.no_grad():
        zg = m._pert_embeddings()
        # step 1, from the REAL control
        z0 = m._encode_ctrl(c)
        recon0 = m._zero_pert_recon(z0, zg, c.device)
        e0, m0 = m._slot_embed(zg, [p[0] for p in idx], c.device)
        d0 = m.decoder(z0 + m._interact(z0, e0, zg, m0))
        d0 = d0["prediction"] if isinstance(d0, dict) else d0
        x1 = c + (d0 - recon0)
        # step 2, from x1
        z1 = m._encode_ctrl(x1)
        recon1 = m._zero_pert_recon(z1, zg, c.device)
        e1, m1 = m._slot_embed(zg, [p[1] for p in idx], c.device)
        d1 = m.decoder(z1 + m._interact(z1, e1, zg, m1))
        d1 = d1["prediction"] if isinstance(d1, dict) else d1
        want = x1 + (d1 - recon1)
        got = m.forward_composed(c, idx, "resid_seq_ab")["prediction"]
    assert torch.allclose(got, want, atol=1e-6)


def test_control_rows_are_untouched_by_every_mode():
    """A row with no perturbation must decode its own reconstruction under every mode."""
    m = _model()
    c = _ctrl(3)
    idx = [[-1], [-1], [-1]]
    with torch.no_grad():
        ref = m.forward_composed(c, idx, "sum")["prediction"]
        for mode in RECON_MODES:
            got = m.forward_composed(c, idx, mode)["prediction"]
            assert torch.allclose(ref, got, atol=1e-5), f"{mode} moved a control row"
        # the real-ctrl family emits the control cell ITSELF for an unperturbed row, which is
        # the right answer in that convention and is not the same number as D(z + f(z,0))
        for mode in REAL_MODES:
            got = m.forward_composed(c, idx, mode)["prediction"]
            assert torch.allclose(got, c, atol=1e-5), \
                f"{mode} should return the real control verbatim for an unperturbed row"


def test_where_the_order_dependence_actually_comes_from():
    """Two DIFFERENT sources of order dependence, and they are worth separating.

    * `latent_seq` under `additive` is order-INVARIANT: f(z, e) never reads z, so
      z + f(z, e_A) + f(z', e_B) = z + e_A + e_B regardless of order.
    * `decode_seq` under the SAME interaction is order-DEPENDENT anyway, because it
      re-encodes between steps and E(D(.)) is nonlinear. Its asymmetry is a property of the
      round trip, not of the interaction.

    So a `decode_seq` AB-vs-BA gap is NOT evidence that the model learned an interaction;
    part of it is encoder/decoder noise. Pinned here so that gap is never read as biology.
    """
    m = _model(interaction="additive")
    c = _ctrl()
    ab = [[1, 4], [2, 6], [0, 3], [5, 1]]
    ba = [list(reversed(p)) for p in ab]
    with torch.no_grad():
        x = m.forward_composed(c, ab, "latent_seq_ab")["prediction"]
        y = m.forward_composed(c, ba, "latent_seq_ab")["prediction"]
        assert torch.allclose(x, y, atol=1e-5), \
            "latent_seq must be order-invariant when the interaction ignores z"
        x = m.forward_composed(c, ab, "decode_seq_ab")["prediction"]
        y = m.forward_composed(c, ba, "decode_seq_ab")["prediction"]
        assert not torch.allclose(x, y, atol=1e-4), \
            "decode_seq should be order-dependent even here, via the E(D(.)) round trip"


def test_delta_sum_is_symmetric_and_the_sequential_modes_are_not():
    """`delta_sum` and `sum` commute by construction; `*_ab` and `*_ba` must not.

    ⚠️ Tested under `film`, which is what every trained arm uses (arm s313 =
    multiplicative_only). Under `additive` the chain is order-invariant by construction --
    f(z, e) never reads z -- so asserting asymmetry there would be asserting a falsehood.
    If a sequential mode came out symmetric under film it would mean the chain is not
    conditioning step 2 on step 1, i.e. the mode is secretly `sum`.
    """
    m = _model(interaction="film")
    c = _ctrl()
    ab = [[1, 4], [2, 6], [0, 3], [5, 1]]
    ba = [list(reversed(p)) for p in ab]
    with torch.no_grad():
        for mode in ("sum", "delta_sum", "delta_sum_realctrl"):
            x = m.forward_composed(c, ab, mode)["prediction"]
            y = m.forward_composed(c, ba, mode)["prediction"]
            assert torch.allclose(x, y, atol=1e-6), f"{mode} should be order-invariant"
        for mode in ("latent_seq_ab", "decode_seq_ab", "resid_seq_ab"):
            x = m.forward_composed(c, ab, mode)["prediction"]
            y = m.forward_composed(c, ba, mode)["prediction"]
            assert not torch.allclose(x, y, atol=1e-4), \
                f"{mode} came out order-INVARIANT; the chain is not conditioning on step 1"


def test_ab_and_ba_are_the_two_chains_and_mean_is_exactly_their_average():
    m = _model(interaction="film")
    c = _ctrl()
    ab = [[1, 4], [2, 6], [0, 3], [5, 1]]
    with torch.no_grad():
        for fam in ("latent_seq", "decode_seq", "resid_seq"):
            a = m.forward_composed(c, ab, f"{fam}_ab")["prediction"]
            b = m.forward_composed(c, ab, f"{fam}_ba")["prediction"]
            mean = m.forward_composed(c, ab, f"{fam}_mean")["prediction"]
            assert torch.allclose(mean, 0.5 * (a + b), atol=1e-5), \
                f"{fam}_mean is not the average of its two orders"
            # and `_ba` must equal `_ab` on the reversed index list, or the reversal is
            # happening somewhere other than where it is documented
            rev = m.forward_composed(c, [list(reversed(p)) for p in ab],
                                     f"{fam}_ab")["prediction"]
            assert torch.allclose(b, rev, atol=1e-5), \
                f"{fam}_ba != {fam}_ab on reversed indices"


def test_delta_sum_cancels_the_decoder_reconstruction_term():
    """Its defining property: the prediction is recon + sum of decoded deltas, exactly.

    This is what makes it different from `decode_seq`, which feeds the reconstruction into
    its own second round rather than subtracting it.
    """
    m = _model()
    c = _ctrl(2)
    idx = [[1, 4], [0, 3]]
    with torch.no_grad():
        z = m._encode_ctrl(c)
        zg = m._pert_embeddings()
        # the reference is the ZERO-perturbation forward, matching forward_delta_sum
        e0, m0 = m._slot_embed(zg, [-1] * c.shape[0], c.device)
        recon = m.decoder(z + m._interact(z, e0, zg, m0))
        recon = recon["prediction"] if isinstance(recon, dict) else recon
        want = recon.clone()
        for k in (0, 1):
            e, msk = m._slot_embed(zg, [p[k] for p in idx], c.device)
            d = m._interact(z, e, zg, msk)
            pk = m.decoder(z + d)
            pk = pk["prediction"] if isinstance(pk, dict) else pk
            want = want + (pk - recon)
        got = m.forward_composed(c, idx, "delta_sum")["prediction"]
    assert torch.allclose(got, want, atol=1e-6)


def test_mixed_arity_batch_is_handled_row_by_row():
    """Controls, singles and doubles in ONE call: each row must match its own pure batch."""
    m = _model()
    c = _ctrl(3)
    mixed = [[-1], [2], [1, 4]]
    with torch.no_grad():
        for mode in MODES:
            got = m.forward_composed(c, mixed, mode)["prediction"]
            for i, idx in enumerate(mixed):
                alone = m.forward_composed(c[i:i + 1], [idx], mode)["prediction"]
                assert torch.allclose(got[i:i + 1], alone, atol=1e-5), \
                    f"{mode}: row {i} ({idx}) depends on the rest of the batch"


def test_every_mode_returns_the_keys_the_pipeline_reads():
    m = _model()
    c = _ctrl()
    idx = [[1, 4], [2, 6], [0, 3], [5, 1]]
    with torch.no_grad():
        for mode in MODES:
            out = m.forward_composed(c, idx, mode)
            for k in ("prediction", "delta_z", "z_hat", "z_ctrl"):
                assert k in out, f"{mode} is missing {k!r}"
                assert torch.isfinite(out[k]).all(), f"{mode}: {k} has non-finite values"
            assert out["prediction"].shape == (4, N_GENES)
            assert out["z_hat"].shape == (4, HID)


def test_unknown_mode_raises_rather_than_silently_falling_back():
    m = _model()
    with pytest.raises(ValueError):
        m.forward_composed(_ctrl(1), [[1]], "sum_of_singles")


def test_modes_hold_under_the_mean_free_interaction():
    """`film_mean_free` subtracts a reference interaction; the single-pert identity must
    survive it, since every trained arm in the campaign has it ON (arm s313)."""
    m = _model(interaction="film", mean_free=True)
    c = _ctrl()
    idx = [[2], [5], [0], [3]]
    with torch.no_grad():
        ref = m.forward_composed(c, idx, "sum")["prediction"]
        for mode in RECON_MODES:
            got = m.forward_composed(c, idx, mode)["prediction"]
            assert torch.allclose(ref, got, atol=1e-5), f"{mode} breaks under film_mean_free"
        # and the real-ctrl family stays internally consistent -- it is NOT expected to match
        # `sum`, by construction (different emission), so comparing it to `sum` here would be
        # asserting a falsehood.
        ref2 = m.forward_composed(c, idx, "delta_sum_realctrl")["prediction"]
        for mode in REAL_MODES:
            got = m.forward_composed(c, idx, mode)["prediction"]
            assert torch.allclose(ref2, got, atol=1e-5), \
                f"{mode} breaks its own family's identity under film_mean_free"
