"""SimpleAdjacencyMap perturbation model.

Ported from txpert's gspp/models/pert_models/simple_adjacency_map.py.
Each gene's embedding = MLP(adjacency_row). No learned message passing.
"""

import os

import torch
import torch.nn as nn

from torch_geometric.utils import to_undirected, add_self_loops


def _make_activation(name: str) -> nn.Module:
    """Create a fresh activation module instance by name."""
    ACTIVATIONS = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "silu": nn.SiLU,
        "mish": nn.Mish,
        "softplus": nn.Softplus,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "none": nn.Identity,
    }
    return ACTIVATIONS[name]()


class SimpleAdjacencyMapModel(nn.Module):
    """Adjacency matrix-based perturbation model.

    Uses rows of the (dense) adjacency matrix as node features,
    passed through a 2-layer MLP to produce gene embeddings.

    Forward signature matches PerturbationGNN: forward(edge_index, edge_weight).
    """

    def __init__(
        self,
        num_genes: int,
        edge_index: torch.Tensor,
        hidden_dim: int = 128,
        out_dim: int = 64,
        dropout: float = 0.2,
        use_ffn: bool = True,
        normalize: str = "none",
        adj_deg: int = 1,
        self_loops: bool = False,
        poly_sum: bool = False,
        only_reverse: bool = True,
        make_undirected: bool = True,
        add_column: bool = False,
        activation: str = "relu",
    ):
        super().__init__()
        self.num_genes = num_genes
        self.only_reverse = only_reverse
        self.adj_deg = adj_deg
        self.poly_sum = poly_sum
        self.make_undirected = make_undirected
        self.add_column = add_column
        self.normalize = normalize

        if self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_genes)

        if make_undirected:
            edge_index = to_undirected(edge_index)

        adj = self._build_adjacency_matrix(edge_index, num_genes)
        self.register_buffer("adj_matrix", adj)
        node_feat_dim = adj.size(1)

        if use_ffn:
            self.map_to_latent = nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                _make_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.map_to_latent = nn.Linear(node_feat_dim, out_dim)

        if poly_sum:
            self.alpha = nn.Parameter(torch.tensor(0.0))

    def _build_adjacency_matrix(
        self, edge_index: torch.Tensor, num_genes: int
    ) -> torch.Tensor:
        adj = torch.zeros(num_genes, num_genes)
        adj[edge_index[0], edge_index[1]] = 1.0

        if self.normalize == "sym":
            deg = adj.sum(dim=1)
            deg_inv_sqrt = torch.pow(deg, -0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
            adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        elif self.normalize == "row":
            deg = adj.sum(dim=1, keepdim=True)
            deg[deg == 0] = 1.0
            adj = adj / deg
        elif self.normalize == "col":
            deg = adj.sum(dim=0, keepdim=True)
            deg[deg == 0] = 1.0
            adj = adj / deg

        if self.only_reverse:
            return adj.t()

        mats = []
        cur = torch.eye(num_genes, dtype=adj.dtype)
        for _ in range(1, int(self.adj_deg) + 1):
            cur = cur.matmul(adj)
            if self.add_column:
                mats.append(torch.cat([cur, cur.t()], dim=1))
            else:
                mats.append(cur)

        if self.poly_sum:
            self._poly_mats = mats
            adj = mats[0]
        else:
            adj = torch.cat(mats, dim=1)

        return adj

    def forward(self, edge_index, edge_weight=None):
        if self.poly_sum:
            beta = torch.sigmoid(self.alpha)
            poly_adj = torch.zeros_like(self.adj_matrix)
            for i, mat in enumerate(self._poly_mats):
                poly_adj = poly_adj + (beta ** i) * mat.to(self.adj_matrix.device)
            return self.map_to_latent(poly_adj)

        return self.map_to_latent(self.adj_matrix)


class LearnableEmbeddingModel(nn.Module):
    """Learnable embedding table for no-graph-prior ablation.

    Each gene gets a freely learnable embedding vector.
    Forward signature matches SimpleAdjacencyMapModel/PerturbationGNN.
    """

    def __init__(self, num_genes: int, out_dim: int = 64):
        super().__init__()
        self.num_genes = num_genes
        self.embedding = nn.Embedding(num_genes, out_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, edge_index, edge_weight=None):
        return self.embedding.weight


# ---------------------------------------------------------------------------
# Frozen external per-gene embeddings (GenePT, ESM2, ...) -> the adjacency-map MLP
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS. The perturbation-representation ablation asks what the GO+STRING adjacency
# row contributes to Y06. SimpleAdjacencyMapModel is `MLP(adjacency_row[g])`; this module is
# `MLP(frozen_table[g])` with the SAME MLP (map_to_latent, built from the same adj_* knobs), so
# the only thing that changes between the arms is the source of the per-gene input vector.
#
# MISSING-GENE POLICY (documented, and logged at build time):
#   * A gene of the perturbation vocabulary that has no row in the table gets an all-zero input
#     row PLUS one shared, learned `missing_embedding` vector (initialised at zero, so at step 0
#     a missing gene's input is exactly the zero vector -- which, under the default `zscore`
#     normalisation, is the mean present gene).
#   * All missing genes therefore share one representation and are indistinguishable from each
#     other. That is the honest reading of "no information about this gene", and it is why
#     coverage is printed at build time and stored in the checkpoint.
#
# NORMALISATION (`norm`, applied once at build over PRESENT rows only; missing rows stay 0):
#   zscore : per-dimension standardisation (default). Puts GenePT (~0.02/dim) and ESM2
#            (~0.1-1/dim) on the same input scale for a fixed-lr MLP.
#   l2     : unit-norm rows.
#   none   : the raw table.
# The normalised table is a persistent buffer, so a checkpoint reloads self-contained and
# bit-exact without the table file; the path + sha256 are recorded for provenance only.

PERT_EMBEDDING_NORMS = ("zscore", "l2", "none")


def _file_sha256(path: str, chunk: int = 1 << 24) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_gene_embedding_table(path: str):
    """Read a per-gene embedding table into ``{symbol: np.ndarray[float32]}``.

    Accepted formats:
      * ``.pkl`` / ``.pickle``: a dict ``{gene_symbol: vector}`` (the GenePT asset format,
        ``data/assets/genept_occam_rhj_fewshot.pkl``: 2,376 keys x 3,072, lists of float64).
      * ``.npz``: arrays ``genes`` (str, [N]) and ``embeddings`` (float, [N, D]).
    """
    import pickle
    import numpy as np

    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=False)
        genes = [str(g) for g in z["genes"]]
        emb = np.asarray(z["embeddings"], dtype=np.float32)
        table = {g: emb[i] for i, g in enumerate(genes)}
    else:
        with open(path, "rb") as fh:
            raw = pickle.load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a dict {{symbol: vector}}, got {type(raw).__name__}")
        table = {str(k): np.asarray(v, dtype=np.float32).reshape(-1) for k, v in raw.items()}
    if not table:
        raise ValueError(f"{path}: empty embedding table")
    dims = {v.shape[0] for v in table.values()}
    if len(dims) != 1:
        raise ValueError(f"{path}: inconsistent embedding dims {sorted(dims)[:5]}")
    return table


def build_pert_embedding_matrix(path: str, pert2id: dict, norm: str = "zscore"):
    """Align a per-gene table to ``pert2id`` order.

    Returns ``(matrix [num_genes, D] float32 tensor, present bool tensor [num_genes], info)``
    where ``info`` holds the path, sha256, dim, norm and coverage counts.
    """
    import numpy as np

    if not os.path.isabs(path) and not os.path.exists(path):
        # relative paths resolve against the repo root too (src/occam/vcell -> repo)
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
        cand = os.path.join(repo, path)
        if os.path.exists(cand):
            path = cand
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"pert_embedding_path not found: {path}\n"
            f"The per-gene tables are on the Hub under assets/ (see data/assets/README.md):\n"
            f"  hf download bendidiihab/Occam --include 'assets/*' --local-dir /tmp/dl && "
            f"mkdir -p data/assets && mv /tmp/dl/assets/* data/assets/")
    if norm not in PERT_EMBEDDING_NORMS:
        raise ValueError(f"pert_embedding_norm={norm!r}; expected one of {PERT_EMBEDDING_NORMS}")
    table = read_gene_embedding_table(path)
    dim = next(iter(table.values())).shape[0]
    num_genes = max(pert2id.values()) + 1
    mat = np.zeros((num_genes, dim), dtype=np.float32)
    present = np.zeros(num_genes, dtype=bool)
    for sym, idx in pert2id.items():
        vec = table.get(sym)
        if vec is not None:
            mat[idx] = vec
            present[idx] = True
    n_present = int(present.sum())
    if n_present == 0:
        raise ValueError(f"{path}: 0/{num_genes} perturbation-vocabulary genes found in the table "
                         f"-- wrong key space (symbols expected)?")
    if norm == "zscore":
        mu = mat[present].mean(axis=0, keepdims=True)
        sd = mat[present].std(axis=0, keepdims=True)
        sd[sd < 1e-8] = 1.0
        mat[present] = (mat[present] - mu) / sd
    elif norm == "l2":
        nrm = np.linalg.norm(mat[present], axis=1, keepdims=True)
        nrm[nrm < 1e-12] = 1.0
        mat[present] = mat[present] / nrm
    sha = _file_sha256(path)
    # A downloaded / rebuilt table must be the one the sidecar describes: a silently different
    # table changes the arm. Sidecars without a top-level sha256 (older competitor tables) skip.
    _side = path + ".json"
    if os.path.exists(_side):
        import json
        with open(_side) as fh:
            _want = json.load(fh).get("sha256")
        if _want and _want != sha:
            raise ValueError(f"{path}: sha256 {sha[:12]}... does not match its sidecar "
                             f"{_side} ({_want[:12]}...). Re-download (data/assets/README.md) "
                             f"or rebuild the table.")
    info = {
        "path": os.path.abspath(path),
        "sha256": sha,
        "dim": int(dim),
        "norm": norm,
        "n_table_keys": len(table),
        "n_vocab": int(num_genes),
        "n_present": n_present,
        "n_missing": int(num_genes - n_present),
        "missing_policy": "zero_row_plus_shared_learned_missing_vector",
    }
    return torch.from_numpy(mat), torch.from_numpy(present), info


class FrozenEmbeddingModel(nn.Module):
    """Frozen per-gene embedding table -> the SimpleAdjacencyMapModel MLP.

    Output: ``[num_genes, out_dim]``, same as SimpleAdjacencyMapModel / LearnableEmbeddingModel.
    Forward signature matches them: ``forward(edge_index, edge_weight=None)`` (both ignored).

    Trainable: ``map_to_latent`` (identical construction to SimpleAdjacencyMapModel) and the
    single ``missing_embedding`` vector. The table itself is a buffer and receives no gradient.
    """

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        present_mask: torch.Tensor,
        hidden_dim: int = 128,
        out_dim: int = 64,
        dropout: float = 0.2,
        use_ffn: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        if embedding_matrix.dim() != 2:
            raise ValueError(f"embedding_matrix must be 2-D, got {tuple(embedding_matrix.shape)}")
        num_genes, in_dim = embedding_matrix.shape
        if present_mask.shape != (num_genes,):
            raise ValueError("present_mask must have shape [num_genes]")
        self.num_genes = int(num_genes)
        self.in_dim = int(in_dim)
        self.register_buffer("embedding_table", embedding_matrix.detach().float().clone())
        self.register_buffer("present_mask", present_mask.detach().bool().clone())
        self.missing_embedding = nn.Parameter(torch.zeros(in_dim))

        if use_ffn:
            self.map_to_latent = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                _make_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.map_to_latent = nn.Linear(in_dim, out_dim)

    @classmethod
    def empty_like_state(cls, state_dict: dict, **kwargs) -> "FrozenEmbeddingModel":
        """Build a shell with the buffer shapes of a saved state dict (reload path)."""
        tbl = state_dict["embedding_table"]
        return cls(torch.zeros_like(tbl, dtype=torch.float32),
                   torch.zeros(tbl.shape[0], dtype=torch.bool), **kwargs)

    def node_features(self) -> torch.Tensor:
        miss = (~self.present_mask).unsqueeze(1).to(self.embedding_table.dtype)
        return self.embedding_table + miss * self.missing_embedding.unsqueeze(0)

    def forward(self, edge_index=None, edge_weight=None):
        return self.map_to_latent(self.node_features())


# ---------------------------------------------------------------------------
# ALL knowledge sources concatenated: [adjacency row || table_1 || table_2 ...] -> the SAME MLP
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS. Ablation arm "all sources": does giving the adjacency-map MLP the GO+STRING
# row AND the frozen GenePT / ESM2 vectors beat any single source? Everything but the input
# width of the first Linear is identical to SimpleAdjacencyMapModel (same adj_* knobs, same
# adjacency construction, same MLP shape).
#
# PER-SOURCE SCALING (so no source dominates the first layer by scale alone):
#   * The adjacency block is left EXACTLY as SimpleAdjacencyMapModel builds it. Under the
#     rhj_fewshot settings (adj_normalize=none, only_reverse=True) that is a binary 0/1 row,
#     so its squared row norm is the node degree (mean ~ a few dozen), whatever the width.
#   * Each table is first normalised per dimension exactly as FrozenEmbeddingModel does
#     (`pert_embedding_norm`, default zscore -> squared row norm ~= D: 3,072 for GenePT, 1,280
#     for ESM2), then multiplied by ONE scalar so its mean squared row norm over present genes
#     EQUALS the adjacency block's mean squared row norm over nodes with degree > 0.
#   Rationale: the first-layer pre-activation is sum_b W_b x_b; with a fan-in init its variance
#   contribution from block b is proportional to ||x_b||^2. Energy matching therefore gives
#   each source an equal share at init. Without it a z-scored GenePT block would carry ~100x
#   the energy of the adjacency row and the arm would be "GenePT + a little graph".
#   The adjacency is the reference and is not rescaled, so the graph part of the input is
#   bit-identical to the default model's.
#
# MISSING GENES: per table, the FrozenEmbeddingModel policy -- zero row + one learned
#   zero-initialised missing vector PER TABLE (in the scaled space). The adjacency block keeps
#   its default behaviour (a gene with no edges has an all-zero row; no extra vector).
#
# The first Linear is applied block-wise (sum of F.linear over weight column slices), which is
# mathematically identical to Linear(cat(blocks)) and keeps the state-dict layout of a plain
# concat Linear, but never materialises the [N, N + sum D] concatenation (~1.7 GB at 18,216
# nodes) or saves it for backward.


def _row_sq_norms(mat: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """Squared L2 norm of every row, in float64, chunked (no full-size temporaries)."""
    out = torch.empty(mat.shape[0], dtype=torch.float64)
    for i in range(0, mat.shape[0], chunk):
        out[i:i + chunk] = mat[i:i + chunk].double().pow(2).sum(dim=1).cpu()
    return out


def _block_energy(mat: torch.Tensor, rows: torch.Tensor) -> float:
    """Mean squared L2 row norm over the selected rows."""
    if int(rows.sum()) == 0:
        return 0.0
    return float(_row_sq_norms(mat)[rows.cpu()].mean())


class ConcatKnowledgeModel(SimpleAdjacencyMapModel):
    """[adjacency row || frozen table_1 || ... ] -> the SimpleAdjacencyMapModel MLP.

    ``tables`` is a list of ``(matrix [num_genes, D_k], present [num_genes])`` already aligned to
    the perturbation vocabulary (``build_pert_embedding_matrix``). ``table_scales`` (reload path
    only) bypasses the energy matching and uses the given scalars; normally leave it None.

    Trainable: ``map_to_latent`` (first Linear has width ``adj_width + sum D_k``) and one
    ``missing_embeddings[k]`` vector per table. Adjacency and tables are buffers (no grad).
    """

    def __init__(self, num_genes: int, edge_index: torch.Tensor, tables, hidden_dim: int = 128,
                 out_dim: int = 64, dropout: float = 0.2, use_ffn: bool = True,
                 normalize: str = "none", adj_deg: int = 1, self_loops: bool = False,
                 only_reverse: bool = True, make_undirected: bool = True,
                 activation: str = "relu", table_scales=None):
        if not tables:
            raise ValueError("ConcatKnowledgeModel needs at least one table")
        super().__init__(num_genes=num_genes, edge_index=edge_index, hidden_dim=hidden_dim,
                         out_dim=out_dim, dropout=dropout, use_ffn=use_ffn, normalize=normalize,
                         adj_deg=adj_deg, self_loops=self_loops, poly_sum=False,
                         only_reverse=only_reverse, make_undirected=make_undirected,
                         add_column=False, activation=activation)
        self.use_ffn = bool(use_ffn)
        self.adj_width = int(self.adj_matrix.size(1))
        adj_sq = _row_sq_norms(self.adj_matrix)
        adj_present = adj_sq > 0
        self.adj_n_present = int(adj_present.sum())
        self.adj_energy = float(adj_sq[adj_present].mean()) if self.adj_n_present else 0.0
        self.n_tables = len(tables)
        self.table_dims = []
        self.table_scales = []
        self.missing_embeddings = nn.ParameterList()
        for k, (mat, present) in enumerate(tables):
            if mat.dim() != 2 or mat.shape[0] != self.adj_matrix.shape[0]:
                raise ValueError(f"table {k}: shape {tuple(mat.shape)} does not match "
                                 f"{self.adj_matrix.shape[0]} adjacency nodes")
            if present.shape != (mat.shape[0],):
                raise ValueError(f"table {k}: present mask must have shape [num_genes]")
            mat = mat.detach().float().clone()
            present = present.detach().bool().clone()
            if table_scales is not None:
                scale = float(table_scales[k])
            else:
                e_tab = _block_energy(mat, present)
                scale = (self.adj_energy / e_tab) ** 0.5 if (e_tab > 0 and self.adj_energy > 0) \
                    else 1.0
            mat = mat * scale
            self.register_buffer(f"emb_table_{k}", mat)
            self.register_buffer(f"emb_present_{k}", present)
            self.missing_embeddings.append(nn.Parameter(torch.zeros(mat.shape[1])))
            self.table_dims.append(int(mat.shape[1]))
            self.table_scales.append(scale)
        self.in_dim = self.adj_width + sum(self.table_dims)

        # Same construction as SimpleAdjacencyMapModel, wider first layer.
        if use_ffn:
            self.map_to_latent = nn.Sequential(
                nn.Linear(self.in_dim, hidden_dim),
                _make_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.map_to_latent = nn.Linear(self.in_dim, out_dim)

    @classmethod
    def empty_like_state(cls, state_dict: dict, table_scales, **kwargs) -> "ConcatKnowledgeModel":
        """Shell with the buffer shapes of a saved state dict (reload path; no files read)."""
        adj = state_dict["adj_matrix"]
        k, tables = 0, []
        while f"emb_table_{k}" in state_dict:
            t = state_dict[f"emb_table_{k}"]
            tables.append((torch.zeros_like(t, dtype=torch.float32),
                           torch.zeros(t.shape[0], dtype=torch.bool)))
            k += 1
        return cls(num_genes=adj.shape[0], edge_index=torch.zeros(2, 0, dtype=torch.long),
                   tables=tables, table_scales=table_scales, **kwargs)

    def node_feature_blocks(self):
        blocks = [self.adj_matrix]
        for k in range(self.n_tables):
            tbl = getattr(self, f"emb_table_{k}")
            miss = (~getattr(self, f"emb_present_{k}")).unsqueeze(1).to(tbl.dtype)
            blocks.append(tbl + miss * self.missing_embeddings[k].unsqueeze(0))
        return blocks

    def node_features(self) -> torch.Tensor:
        """The concatenated input (materialised; for tests / inspection only)."""
        return torch.cat(self.node_feature_blocks(), dim=1)

    def forward(self, edge_index=None, edge_weight=None):
        first = self.map_to_latent[0] if self.use_ffn else self.map_to_latent
        w = first.weight
        h, off = None, 0
        for b in self.node_feature_blocks():
            width = b.shape[1]
            # bias rides on the first block, so under autocast it is added in the same dtype
            # as nn.Linear would add it
            part = nn.functional.linear(b, w[:, off:off + width],
                                        first.bias if h is None else None)
            h = part if h is None else h + part
            off += width
        if self.use_ffn:
            for layer in list(self.map_to_latent)[1:]:
                h = layer(h)
        return h
