"""SimpleAdjacencyMap perturbation model.

Ported from txpert's gspp/models/pert_models/simple_adjacency_map.py.
Each gene's embedding = MLP(adjacency_row). No learned message passing.
"""

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
