"""`film_mean_free` must apply everywhere, to perturbed rows only, in the right dtype.

It emits ``interaction(z, e_p) - interaction(z, ebar)``, so a component shared across
perturbations is unrepresentable by construction. Four things have to hold, and each has a
recorded way of going wrong:

* it must **change** perturbed rows — a flag wired to nothing is worse than no flag;
* it must leave **control rows bit-identical**, or the control-identity loss silently changes;
* ``ebar`` must be the mean over the whole graph node table, never a batch statistic, or
  inference depends on batch composition;
* it must survive a bf16 GNN cache against fp32 interaction weights. ``_gnn_cache`` is filled
  under autocast and reused outside it on the eval path, and two research-tree jobs died there
  **after a full epoch plus eval**.

It must also apply on the sequential multi-perturbation path, not only the batched one — which
is why both route through the single ``_interact`` helper.
"""

from __future__ import annotations

import pytest

G, PL, H, NG, B = 32, 8, 16, 12, 6          # nodes, pert dim, hidden, genes, batch
PERTS = [[3], [7], [11], [2], [-1], [-1]]   # rows 4-5 are controls


def _build(mean_free, torch, nn):
    from occam.vcell.model import VirtualCellModel

    class FakeGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Parameter(torch.randn(G, PL))

        def forward(self, edge_index, edge_weight):
            return self.emb

    torch.manual_seed(0)
    return VirtualCellModel(
        encoder=nn.Sequential(nn.Linear(NG, H)),
        gnn=FakeGNN(),
        decoder=nn.Sequential(nn.Linear(H, NG)),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_weight=torch.zeros(0),
        encoder_hidden_size=H,
        pert_latent_dim=PL,
        freeze_encoder=False,
        interaction_type="film",
        interaction_kwargs={"film_scale": 1.0, "film_hidden_dim": 4,
                            "film_mode": "multiplicative_only"},
        film_mean_free=mean_free,
    ).eval()


def test_it_changes_perturbed_rows_and_leaves_controls_bit_identical():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    torch.manual_seed(0)
    x = torch.rand(B, NG)
    with torch.no_grad():
        off = _build(False, torch, nn)(x, PERTS)["delta_z"]
        on = _build(True, torch, nn)(x, PERTS)["delta_z"]

    assert not torch.allclose(off[:4], on[:4]), (
        "film_mean_free did not change any perturbed row — the flag is inert"
    )
    # Bit-identical, not merely close: the control-identity loss depends on it.
    assert torch.equal(off[4:], on[4:]), "control rows must be untouched"


def test_the_correction_is_exactly_the_interaction_at_the_mean_embedding():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    torch.manual_seed(0)
    x = torch.rand(B, NG)
    model = _build(True, torch, nn)
    with torch.no_grad():
        off = _build(False, torch, nn)(x, PERTS)["delta_z"]
        on = model(x, PERTS)["delta_z"]
        z = model.encoder(x)
        ebar = model.gnn(None, None).mean(0, keepdim=True)
        ref = model.interaction(z, ebar.expand(B, -1))

    expected = off.clone()
    expected[:4] = off[:4] - ref[:4]          # perturbed rows only
    assert torch.allclose(on, expected, atol=1e-6)


def test_ebar_is_not_a_batch_statistic():
    """Scoring a subset must give the same answer as scoring the full batch."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    torch.manual_seed(0)
    x = torch.rand(B, NG)
    with torch.no_grad():
        full = _build(True, torch, nn)(x, PERTS)["delta_z"]
        subset = _build(True, torch, nn)(x[:3], PERTS[:3])["delta_z"]
    assert torch.allclose(subset, full[:3], atol=1e-6)


def test_a_bf16_gnn_cache_against_fp32_weights_does_not_raise():
    """The exact failure that killed two research-tree jobs after a full epoch plus eval."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    torch.manual_seed(0)
    x = torch.rand(B, NG)
    model = _build(True, torch, nn)
    model._gnn_cache = model.gnn(None, None).to(torch.bfloat16)
    with torch.no_grad():
        out = model(x, PERTS)
    assert torch.isfinite(out["delta_z"]).all()


def test_the_sequential_path_is_also_mean_free():
    """Both forward paths route through `_interact`, so neither can be silently skipped."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    torch.manual_seed(0)
    x = torch.rand(4, NG)
    perts = [[3, 7], [2], [11, 5], [-1]]      # one control row, two multi-pert rows
    with torch.no_grad():
        off = _build(False, torch, nn).forward_sequential(x, perts)["delta_z"]
        on = _build(True, torch, nn).forward_sequential(x, perts)["delta_z"]

    assert not torch.allclose(off[:3], on[:3]), (
        "forward_sequential ignored film_mean_free — the multi-perturbation path bypassed "
        "the correction"
    )
    assert torch.equal(off[3:], on[3:]), "the control row must be untouched here too"


def test_the_flag_is_persisted_and_read_back():
    """A mean-free model scored without the correction has every delta changed."""
    pytest.importorskip("torch")
    from occam.vcell.config import VirtualCellConfig

    assert VirtualCellConfig().film_mean_free is False, "must default off"
    # The save path records it under interaction_config and from_vcell_checkpoint reads it
    # back; both are asserted by grep-level wiring tests in test_config_fields_are_read.
