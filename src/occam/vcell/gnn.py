"""Graph Neural Network for perturbation gene embeddings.

Ported from TxPM's basic_gnn.py, adapted for standalone use with Occam.
The GNN learns embeddings for all genes in the perturbation vocabulary via
message passing on a prior knowledge graph.
"""

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, GINConv, GATConv


GNN_LAYER_DICT = {
    "gat_v2": GATv2Conv,
    "gcn": GCNConv,
    "gin": GINConv,
    "gat": GATConv,
}

GNN_ACTIVATION_DICT = {
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(),
    "silu": nn.SiLU(),          # aka swish
    "mish": nn.Mish(),
    "softplus": nn.Softplus(),
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
    "selu": nn.SELU(),
    "none": nn.Identity(),
}


class PerturbationGNN(nn.Module):
    """GNN that produces embeddings for all perturbation genes.

    Architecture:
        nn.Embedding(num_genes, hidden_dim) as node features
        → N GNN layers (GATv2Conv by default) with optional skip connections
        → Linear projection to out_dim

    Args:
        num_genes: Total genes in pert2id vocabulary.
        hidden_dim: Hidden dimension for GNN layers.
        out_dim: Output embedding dimension per gene.
        layer_type: GNN layer type (gat_v2, gcn, gin, gat).
        num_layers: Number of GNN layers.
        skip: Skip connection type.
        dropout: Dropout rate.
        num_heads: Number of attention heads (for GAT variants).
        concat: Whether to concatenate multi-head outputs (vs sum).
        add_self_loops: Whether to add self-loops to the graph.
        use_edge_weight: Whether to pass edge weights to GNN layers.
    """

    def __init__(
        self,
        num_genes: int,
        hidden_dim: int = 64,
        out_dim: Optional[int] = None,
        layer_type: str = "gat_v2",
        num_layers: int = 4,
        skip: Literal[
            "none", "skip_cat", "skip_cat_learned", "skip_sum_learned"
        ] = "none",
        dropout: float = 0.0,
        num_heads: int = 2,
        concat: bool = False,
        add_self_loops: bool = False,
        use_edge_weight: bool = False,
        no_gnn: bool = False,
        activation: str = "gelu",
    ):
        super().__init__()

        if out_dim is None:
            out_dim = hidden_dim

        if "gat" not in layer_type:
            concat = False

        self.num_genes = num_genes
        self.num_layers = num_layers
        self.skip = skip
        self.concat = concat
        self.heads = num_heads
        self.use_edge_weight = use_edge_weight
        self.dropout = dropout
        self.no_gnn = no_gnn
        self.activation = GNN_ACTIVATION_DICT[activation]

        # Learned node features: one embedding per gene
        self.pert_embeddings = nn.Embedding(num_genes, hidden_dim)

        if no_gnn:
            self.out_nn = nn.Linear(hidden_dim, out_dim)
            self.gnn_layers = nn.ModuleList()
            self.skip_layers = nn.ModuleList()
            return

        gnn_layer_cls = GNN_LAYER_DICT[layer_type]

        self.gnn_layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()

        in_dim = hidden_dim
        for _ in range(num_layers):
            layer_kwargs = dict(
                in_channels=in_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                concat=concat,
                add_self_loops=add_self_loops,
            )
            if use_edge_weight:
                layer_kwargs["edge_dim"] = 1

            # GCN/GIN don't accept heads/concat/edge_dim
            if layer_type in ("gcn", "gin"):
                layer_kwargs = dict(in_channels=in_dim, out_channels=hidden_dim)

            self.gnn_layers.append(gnn_layer_cls(**layer_kwargs))

            # Skip connection bookkeeping
            res_dim = 0
            if "skip" in skip:
                res_dim = in_dim
                if "learned" in skip:
                    self.skip_layers.append(nn.Linear(in_dim, hidden_dim))
                    res_dim = hidden_dim
                if "cat" not in skip:
                    res_dim = 0

            out_channels = num_heads * hidden_dim if concat else hidden_dim
            in_dim = res_dim + out_channels

        self.out_nn = nn.Linear(in_dim, out_dim)

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute embeddings for ALL genes simultaneously.

        Args:
            edge_index: (2, num_edges) graph connectivity.
            edge_weight: (num_edges,) optional edge weights.

        Returns:
            Tensor of shape (num_genes, out_dim).
        """
        x = self.pert_embeddings(
            torch.arange(self.num_genes, device=self.pert_embeddings.weight.device)
        )

        if self.no_gnn:
            return self.out_nn(x)

        if not self.use_edge_weight:
            edge_weight = None

        for idx, layer in enumerate(self.gnn_layers):
            if "skip" in self.skip:
                res = x
                if "learned" in self.skip:
                    res = self.skip_layers[idx](res)

            x = layer(x, edge_index, edge_attr=edge_weight)

            if self.skip == "none":
                pass
            elif "sum" in self.skip:
                if self.concat:
                    x = x + torch.cat(self.heads * [res], dim=-1)
                else:
                    x = x + res
            elif "cat" in self.skip:
                x = torch.cat([x, res], dim=-1)

            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.out_nn(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x
