"""FrozenEmbeddingModel: a frozen per-gene table (GenePT / ESM2) -> the adjacency-map MLP.

GenePT and ESM2 embedding configurations change only the source of the
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


def test_pertemb_config_is_rhj_fewshot_plus_exactly_three_keys():
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
    assert gb.pop("pert_embedding_paths") == []
    assert ga == gb
    ra, rb = dict(a.raw), dict(b.raw)
    sa, sb = dict(ra.pop("stages")["vcell"]), dict(rb.pop("stages")["vcell"])
    sa.pop("graph"), sb.pop("graph")
    assert ra == rb and sa == sb

    # the Y06 reference config is untouched and still resolves to the off state
    vc, _ = a.build_vcell_config()
    assert vc.pert_embedding_path == "" and vc.pert_model_type == "adjacency_map"
    assert vc.pert_embedding_paths == []
    vcb, _ = b.build_vcell_config()
    assert vcb.pert_embedding_paths == [] and vcb.pert_embedding_path == ""

    # the derived config accepts the A4 override
    c = load_config(der, overrides=[
        "stages.vcell.graph.pert_embedding_path=data/assets/genept_occam_rhj_fewshot.pkl"])
    vc, _ = c.build_vcell_config()
    assert vc.pert_embedding_path.endswith("genept_occam_rhj_fewshot.pkl")


# ---------------------------------------------------------------------------
# ConcatKnowledgeModel: [adjacency row || table_1 || table_2] -> the same MLP ("ALL" arm)
# ---------------------------------------------------------------------------

PRESENT_B = ["G1", "G2", "G5", "G9"]           # a second table, different coverage
DIM_B = 4


def _write_table_b(tmp_path, np):
    rng = np.random.RandomState(1)
    table = {g: list(rng.randn(DIM_B) * 0.01) for g in PRESENT_B}
    p = os.path.join(str(tmp_path), "tbl_b.npz")
    keys = sorted(table)
    np.savez(p, genes=np.asarray(keys), embeddings=np.stack([table[k] for k in keys]))
    return p


def _edges(torch):
    # G8 has no edge at all -> all-zero adjacency row (default behaviour, kept)
    return torch.tensor([[0, 1, 2, 3, 4, 5, 6, 0], [1, 2, 3, 4, 5, 6, 7, 9]], dtype=torch.long)


def _concat_model(tmp_path, np, torch, **kw):
    from occam.vcell.adjacency_map import ConcatKnowledgeModel, build_pert_embedding_matrix
    pa, _ = _write_table(tmp_path, np)
    pb = _write_table_b(tmp_path, np)
    ta = build_pert_embedding_matrix(pa, _pert2id())
    tb = build_pert_embedding_matrix(pb, _pert2id())
    m = ConcatKnowledgeModel(num_genes=10, edge_index=_edges(torch),
                             tables=[ta[:2], tb[:2]], hidden_dim=16, out_dim=8, **kw)
    return m, ta, tb


def test_concat_shapes_adjacency_identical_and_equals_linear_on_cat(tmp_path):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    from occam.vcell.adjacency_map import SimpleAdjacencyMapModel

    torch.manual_seed(0)
    m, _, _ = _concat_model(tmp_path, np, torch, dropout=0.0)
    assert m.adj_width == 10 and m.table_dims == [DIM, DIM_B]
    assert m.in_dim == 10 + DIM + DIM_B
    assert tuple(m.map_to_latent[0].weight.shape) == (16, m.in_dim)
    out = m(torch.zeros(2, 0, dtype=torch.long), None)
    assert tuple(out.shape) == (10, 8)

    # the adjacency block is exactly what SimpleAdjacencyMapModel builds
    ref = SimpleAdjacencyMapModel(num_genes=10, edge_index=_edges(torch), hidden_dim=16, out_dim=8)
    assert torch.equal(m.adj_matrix, ref.adj_matrix)
    assert torch.equal(m.node_features()[:, :10], ref.adj_matrix)
    assert [type(x).__name__ for x in ref.map_to_latent] == \
        [type(x).__name__ for x in m.map_to_latent]

    # block-wise first layer == Linear(cat(blocks))
    m.eval()
    assert torch.allclose(m(None), m.map_to_latent(m.node_features()), atol=1e-5)

    # no-FFN variant
    m2, _, _ = _concat_model(tmp_path, np, torch, use_ffn=False)
    m2.eval()
    assert torch.allclose(m2(None), m2.map_to_latent(m2.node_features()), atol=1e-5)


def test_concat_energy_matching(tmp_path):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    m, (ma, pa_, _), (mb, pb_, _) = _concat_model(tmp_path, np, torch)
    deg = m.adj_matrix.pow(2).sum(1)
    e_adj = float(deg[deg > 0].mean())
    assert m.adj_energy == pytest.approx(e_adj, rel=1e-6)
    assert m.adj_n_present == 9                # G8 has no edge
    for k, (mat, pres) in enumerate([(ma, pa_), (mb, pb_)]):
        tbl = getattr(m, "emb_table_%d" % k)
        e_tab = float(tbl[pres].pow(2).sum(1).mean())
        assert e_tab == pytest.approx(e_adj, rel=1e-5)
        # one scalar per table: direction of every row is preserved
        assert torch.allclose(tbl, mat * m.table_scales[k], atol=1e-6)


def test_concat_frozen_buffers_and_per_table_missing_policy(tmp_path):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    m, (_, pa_, _), (_, pb_, _) = _concat_model(tmp_path, np, torch, dropout=0.0)
    names = {n for n, _ in m.named_parameters()}
    assert names == {"missing_embeddings.0", "missing_embeddings.1", "map_to_latent.0.weight",
                     "map_to_latent.0.bias", "map_to_latent.3.weight", "map_to_latent.3.bias"}
    bufs = {n for n, _ in m.named_buffers()}
    assert {"adj_matrix", "emb_table_0", "emb_table_1", "emb_present_0", "emb_present_1"} <= bufs

    m(None).pow(2).sum().backward()
    for b in ("adj_matrix", "emb_table_0", "emb_table_1"):
        assert getattr(m, b).grad is None and not getattr(m, b).requires_grad
    for k in (0, 1):
        assert m.missing_embeddings[k].grad is not None
        assert m.missing_embeddings[k].grad.abs().sum() > 0
    assert all(p.grad is not None for p in m.map_to_latent.parameters())

    # zero-init: at step 0 a missing gene's block is exactly zero, per table
    feats = m.node_features()
    sl = [slice(10, 10 + DIM), slice(10 + DIM, 10 + DIM + DIM_B)]
    for k, pres in enumerate([pa_, pb_]):
        for i in (~pres).nonzero().flatten().tolist():
            assert torch.equal(feats[i, sl[k]], torch.zeros(sl[k].stop - sl[k].start))
    # each table's missing vector only touches ITS block, only for ITS missing genes
    with torch.no_grad():
        m.missing_embeddings[1].fill_(1.0)
    f2 = m.node_features()
    for i in range(10):
        assert torch.equal(f2[i, :10 + DIM], feats[i, :10 + DIM])
        if pb_[i]:
            assert torch.equal(f2[i, sl[1]], feats[i, sl[1]])
        else:
            assert torch.equal(f2[i, sl[1]], torch.ones(DIM_B))


def test_concat_from_vcell_checkpoint_round_trip_without_table_files(tmp_path):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    from occam.vcell.adjacency_map import ConcatKnowledgeModel
    from occam.vcell.model import VirtualCellModel
    from occam.vcell.pipeline import VirtualCellTrainer
    from occam.model import AutoEncoder
    from occam.decoder.external import HybridNormGlobalSkipDecoder

    NG, H, PL = 12, 512, 8
    torch.manual_seed(3)
    pm, (_, _, ia), (_, _, ib) = _concat_model(tmp_path, np, torch, dropout=0.2)
    with torch.no_grad():
        for p_ in pm.missing_embeddings:
            p_.normal_()
    ia["block_scale"], ib["block_scale"] = pm.table_scales
    ae = AutoEncoder(num_genes=NG, model_size="small", hidden_size=H)
    dec = HybridNormGlobalSkipDecoder(input_dim=H, output_dim=NG, hidden_dim=32, num_blocks=1)
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
                       "adj_activation": "relu", "adj_use_ffn": True, "adj_deg": 1,
                       "adj_normalize": "none", "adj_only_reverse": True,
                       "adj_make_undirected": True,
                       "pert_embedding_concat": {"tables": [ia, ib], "norm": "zscore"}},
        "interaction_config": {"interaction_type": "additive"},
        "decoder_config": {"input_dim": H, "output_dim": NG, "hidden_dim": 32, "num_blocks": 1,
                           "output_activation": "rectified_sigmoid"},
        "training_config": {"pert_model_type": "adjacency_map",
                            "pert_embedding_paths": [ia["path"], ib["path"]],
                            "pert_embedding_sha256s": [ia["sha256"], ib["sha256"]]},
        "gene_names": ["g%d" % i for i in range(NG)],
        "pert2id": _pert2id(),
        "target_sum": 1e4,
    }
    cp = os.path.join(str(tmp_path), "vc.pth")
    torch.save(ckpt, cp)
    for f in (ia["path"], ib["path"]):
        os.remove(f)             # the reload must not need the table files

    tr, _ = VirtualCellTrainer.from_vcell_checkpoint(cp, device="cpu")
    g = tr.model.gnn
    assert isinstance(g, ConcatKnowledgeModel)
    assert g.table_scales == pm.table_scales and g.in_dim == pm.in_dim
    vc.gnn.eval()
    g.eval()
    assert torch.equal(vc.gnn(None), g(None))
    assert torch.equal(g.adj_matrix, pm.adj_matrix)


def test_concat_config_route_and_default_configs_unchanged():
    pytest.importorskip("torch")
    from occam.config.schema import flatten_stage
    from occam.config.loader import load_config
    from occam.vcell.config import VirtualCellConfig

    assert VirtualCellConfig().pert_embedding_paths == []
    assert VirtualCellConfig().pert_embedding_paths is not VirtualCellConfig().pert_embedding_paths
    attrs, _ = flatten_stage("vcell", {"graph": {"pert_embedding_paths": ["a.pkl", "b.pkl"]}})
    assert attrs == {"pert_embedding_paths": ["a.pkl", "b.pkl"]}

    for name in ("rhj_fewshot", "rhj_fewshot_pertemb", "norman_combo", "rhj_fewshot_encoder"):
        cp = os.path.join(REPO, "configs", name + ".yaml")
        if not os.path.isfile(cp):
            continue
        vc, _ = load_config(cp).build_vcell_config()
        assert vc.pert_embedding_paths == [] and vc.pert_embedding_path == "", name

    der = os.path.join(REPO, "configs", "rhj_fewshot_pertemb.yaml")
    if os.path.isfile(der):
        # the exact EXTRA_SETS token: a YAML flow list, no spaces
        c = load_config(der, overrides=[
            "stages.vcell.graph.pert_embedding_paths=[data/assets/genept_full_occam_rhj_fewshot.pkl,"
            "data/assets/esm2_t33_650M_UR50D_occam_rhj_fewshot.pkl]"])
        vc, _ = c.build_vcell_config()
        assert vc.pert_embedding_paths == ["data/assets/genept_full_occam_rhj_fewshot.pkl",
                                           "data/assets/esm2_t33_650M_UR50D_occam_rhj_fewshot.pkl"]
        assert vc.pert_embedding_path == "" and vc.graph_types == ["go", "string"]
