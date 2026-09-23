"""FrozenEmbeddingModel: a frozen per-gene table (GenePT / ESM2) -> the adjacency-map MLP.

Ablation arms A4 (GenePT) and A5 (ESM2) of docs/ABLATION_PLAN.md change ONLY the source of the
per-gene input vector relative to Y06. These tests pin what that requires:

* same output interface as SimpleAdjacencyMapModel: ``forward(edge_index, edge_weight)`` ->
  ``[num_genes, out_dim]``;
* the table is frozen (a buffer: no grad, not in the optimiser), while the MLP and the single
  shared ``missing_embedding`` train;
* the missing-gene policy: missing genes get a zero row + one shared learned vector, so they are
  identical to each other, and coverage is counted;
* a checkpoint reloads bit-exact through ``VirtualCellTrainer.from_vcell_checkpoint`` WITHOUT the
  table file, with the table path + sha256 recorded;
* the config route is opt-in: the default is "", and the schema routes the new keys.
"""

from __future__ import annotations

import os
import pickle

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VOCAB = ["G%d" % i for i in range(10)]
PRESENT = ["G0", "G2", "G3", "G7", "G9"]         # 5/10 present
DIM = 6


def _write_table(tmp_path, np):
    rng = np.random.RandomState(0)
    table = {g: list(rng.randn(DIM) * 3.0 + 1.0) for g in PRESENT}
    table["NOT_IN_VOCAB"] = list(rng.randn(DIM))
    p = os.path.join(str(tmp_path), "tbl.pkl")
    with open(p, "wb") as fh:
        pickle.dump(table, fh)
    return p, table


def _pert2id():
    return {g: i for i, g in enumerate(VOCAB)}


def test_matrix_alignment_coverage_and_zscore(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from occam.vcell.adjacency_map import build_pert_embedding_matrix

    path, table = _write_table(tmp_path, np)
    mat, present, info = build_pert_embedding_matrix(path, _pert2id(), norm="none")
    assert tuple(mat.shape) == (10, DIM)
    assert info["n_present"] == 5 and info["n_missing"] == 5 and info["n_vocab"] == 10
    assert info["n_table_keys"] == 6 and len(info["sha256"]) == 64
    for g in PRESENT:
        assert np.allclose(mat[VOCAB.index(g)].numpy(), np.asarray(table[g], dtype=np.float32))
    assert float(mat[~present].abs().sum()) == 0.0

    mz, pz, _ = build_pert_embedding_matrix(path, _pert2id(), norm="zscore")
    assert np.allclose(mz[pz].numpy().mean(0), 0.0, atol=1e-5)
    assert np.allclose(mz[pz].numpy().std(0), 1.0, atol=1e-4)
    assert float(mz[~pz].abs().sum()) == 0.0

    with pytest.raises(ValueError):
        build_pert_embedding_matrix(path, {"X": 0, "Y": 1}, norm="none")


def test_npz_format_matches_pkl(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from occam.vcell.adjacency_map import build_pert_embedding_matrix

    path, table = _write_table(tmp_path, np)
    npz = os.path.join(str(tmp_path), "tbl.npz")
    keys = sorted(table)
    np.savez(npz, genes=np.asarray(keys), embeddings=np.stack([table[k] for k in keys]))
    a, _, _ = build_pert_embedding_matrix(path, _pert2id())
    b, _, _ = build_pert_embedding_matrix(npz, _pert2id())
    assert np.allclose(a.numpy(), b.numpy(), atol=1e-6)


def test_shapes_frozen_table_and_missing_policy(tmp_path):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    from occam.vcell.adjacency_map import FrozenEmbeddingModel, build_pert_embedding_matrix

    path, _ = _write_table(tmp_path, np)
    mat, present, _ = build_pert_embedding_matrix(path, _pert2id())
    torch.manual_seed(0)
    m = FrozenEmbeddingModel(mat, present, hidden_dim=16, out_dim=8, dropout=0.0)
    out = m(torch.zeros(2, 0, dtype=torch.long), None)
    assert tuple(out.shape) == (10, 8)

    # the table is a buffer, not a parameter
    names = {n for n, _ in m.named_parameters()}
    assert "embedding_table" not in names and not m.embedding_table.requires_grad
    assert "missing_embedding" in names

    out.pow(2).sum().backward()
    assert m.embedding_table.grad is None
    assert m.missing_embedding.grad is not None and m.missing_embedding.grad.abs().sum() > 0
    assert all(p.grad is not None for p in m.map_to_latent.parameters())

    # missing genes share one representation; present genes do not
    miss = (~present).nonzero().flatten().tolist()
    for i in miss[1:]:
        assert torch.equal(out[miss[0]], out[i])
    pres = present.nonzero().flatten().tolist()
    assert not torch.equal(out[pres[0]], out[pres[1]])

    # zero-init missing vector: at step 0 a missing gene's input IS the zero row
    assert torch.equal(m.node_features()[miss[0]], torch.zeros(DIM))
    with torch.no_grad():
        m.missing_embedding.fill_(1.0)
    assert torch.equal(m.node_features()[miss[0]], torch.ones(DIM))
    assert torch.equal(m.node_features()[pres[0]], m.embedding_table[pres[0]])

    # same MLP construction as SimpleAdjacencyMapModel (only the input source changes)
    from occam.vcell.adjacency_map import SimpleAdjacencyMapModel
    adj = SimpleAdjacencyMapModel(num_genes=10, edge_index=torch.zeros(2, 0, dtype=torch.long),
                                  hidden_dim=16, out_dim=8)
    assert [type(x).__name__ for x in adj.map_to_latent] == \
        [type(x).__name__ for x in m.map_to_latent]


def test_state_dict_round_trip_is_bit_exact(tmp_path):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    from occam.vcell.adjacency_map import FrozenEmbeddingModel, build_pert_embedding_matrix

    path, _ = _write_table(tmp_path, np)
    mat, present, _ = build_pert_embedding_matrix(path, _pert2id())
    torch.manual_seed(1)
    a = FrozenEmbeddingModel(mat, present, hidden_dim=16, out_dim=8).eval()
    with torch.no_grad():
        a.missing_embedding.normal_()
    sd = a.state_dict()
    b = FrozenEmbeddingModel.empty_like_state(sd, hidden_dim=16, out_dim=8).eval()
    b.load_state_dict(sd)
    assert torch.equal(a(None), b(None))


def test_from_vcell_checkpoint_round_trip_without_the_table_file(tmp_path):
    """The real reload path (pipeline.py from_vcell_checkpoint) rebuilds the frozen model."""
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    from occam.vcell.adjacency_map import FrozenEmbeddingModel, build_pert_embedding_matrix
    from occam.vcell.model import VirtualCellModel
    from occam.vcell.pipeline import VirtualCellTrainer
    from occam.model import AutoEncoder
    from occam.decoder.external import HybridNormGlobalSkipDecoder

    path, _ = _write_table(tmp_path, np)
    mat, present, info = build_pert_embedding_matrix(path, _pert2id())
    NG, H, PL = 12, 512, 8
    torch.manual_seed(2)
    ae = AutoEncoder(num_genes=NG, model_size="small", hidden_size=H)
    dec = HybridNormGlobalSkipDecoder(input_dim=H, output_dim=NG, hidden_dim=32, num_blocks=1)
    pm = FrozenEmbeddingModel(mat, present, hidden_dim=16, out_dim=PL, dropout=0.2)
    with torch.no_grad():
        pm.missing_embedding.normal_()
    vc = VirtualCellModel(encoder=ae.encoder, gnn=pm, decoder=dec,
                          edge_index=torch.zeros(2, 0, dtype=torch.long),
                          edge_weight=torch.zeros(0), encoder_hidden_size=H,
                          pert_latent_dim=PL, interaction_type="additive")
    ckpt = {
        "is_virtual_cell_checkpoint": True,
        "encoder_state_dict": ae.state_dict(),
        "gnn_state_dict": vc.gnn.state_dict(),
        "decoder_state_dict": vc.decoder.state_dict(),
        "interaction_state_dict": vc.interaction.state_dict(),
        "encoder_config": {"model_size": "small", "hidden_size": H, "num_genes": NG},
        "gnn_config": {"pert_model_type": "adjacency_map", "num_genes": 10, "out_dim": PL,
                       "no_gnn": False, "adj_hidden_dim": 16, "adj_dropout": 0.2,
                       "adj_activation": "relu", "adj_use_ffn": True,
                       "pert_embedding": info},
        "interaction_config": {"interaction_type": "additive"},
        "decoder_config": {"input_dim": H, "output_dim": NG, "hidden_dim": 32, "num_blocks": 1,
                           "output_activation": "rectified_sigmoid"},
        "training_config": {"pert_model_type": "adjacency_map",
                            "pert_embedding_path": info["path"],
                            "pert_embedding_sha256": info["sha256"]},
        "gene_names": ["g%d" % i for i in range(NG)],
        "pert2id": _pert2id(),
        "target_sum": 1e4,
    }
    cp = os.path.join(str(tmp_path), "vc.pth")
    torch.save(ckpt, cp)
    os.remove(path)            # the reload must not need the table file

    tr, _ = VirtualCellTrainer.from_vcell_checkpoint(cp, device="cpu")
    g = tr.model.gnn
    assert isinstance(g, FrozenEmbeddingModel)
    vc.gnn.eval()
    g.eval()
    assert torch.equal(vc.gnn(None), g(None))
    assert torch.equal(g.present_mask, present)


def test_config_route_is_opt_in():
    pytest.importorskip("torch")
    from occam.config.schema import flatten_stage
    from occam.vcell.config import VirtualCellConfig

    assert VirtualCellConfig().pert_embedding_path == ""
    attrs, _ = flatten_stage("vcell", {"graph": {
        "pert_embedding_path": "data/assets/x.pkl", "pert_embedding_norm": "zscore"}})
    assert attrs == {"pert_embedding_path": "data/assets/x.pkl", "pert_embedding_norm": "zscore"}


def test_pertemb_config_is_rhj_fewshot_plus_exactly_two_keys():
    """configs/rhj_fewshot_pertemb.yaml must not drift from the Y06 reference config."""
    pytest.importorskip("torch")
    from occam.config.loader import load_config

    ref = os.path.join(REPO, "configs", "rhj_fewshot.yaml")
    der = os.path.join(REPO, "configs", "rhj_fewshot_pertemb.yaml")
    if not (os.path.isfile(ref) and os.path.isfile(der)):
        pytest.skip("configs missing")
    a, b = load_config(ref), load_config(der)
    ga = dict(a.raw["stages"]["vcell"]["graph"])
    gb = dict(b.raw["stages"]["vcell"]["graph"])
    assert gb.pop("pert_embedding_path") == ""
    assert gb.pop("pert_embedding_norm") == "zscore"
    assert ga == gb
    ra, rb = dict(a.raw), dict(b.raw)
    sa, sb = dict(ra.pop("stages")["vcell"]), dict(rb.pop("stages")["vcell"])
    sa.pop("graph"), sb.pop("graph")
    assert ra == rb and sa == sb

    # the Y06 reference config is untouched and still resolves to the off state
    vc, _ = a.build_vcell_config()
    assert vc.pert_embedding_path == "" and vc.pert_model_type == "adjacency_map"

    # the derived config accepts the A4 override
    c = load_config(der, overrides=[
        "stages.vcell.graph.pert_embedding_path=data/assets/genept_occam_rhj_fewshot.pkl"])
    vc, _ = c.build_vcell_config()
    assert vc.pert_embedding_path.endswith("genept_occam_rhj_fewshot.pkl")
