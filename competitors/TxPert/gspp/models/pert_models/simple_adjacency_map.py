from typing import Union, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import to_undirected, add_self_loops

from gspp.data.graphmodule import GSPGraph


class SimpleAdjacencyMapModel(nn.Module):
    """
    Simple adjacency matrix-based model for perturbation prediction with the same interface as GNN.
    
    This model uses adjacency matrix rows as node features, which are then linearly mapped
    to embeddings. Each node's embedding is computed from its row in the adjacency matrix,
    capturing the local neighborhood structure directly.

    Args:
        graph (GSPGraph): Graph object containing the edge index and edge weight.
        in_dim (int): Dimension of the input data (not used, kept for compatibility).
        hidden_dim (int): Dimension of the hidden layers (embedding dimension).
        out_dim (int): Dimension of the output data.
        dropout (float): Dropout rate.
        normalize (str): How to normalize adjacency matrix: 'none', 'sym', 'row', or 'col'.
        make_undirected (bool): Whether to make the graph undirected.
        device (str): Device to run the model on.
    """

    def __init__(
        self,
        graph: Union[GSPGraph, Tuple],
        in_dim: int = None,
        hidden_dim: int = 64,
        out_dim: int = 64,
        dropout: float = 0.0,
        use_ffn: bool = True,
        normalize: str = "none",
        adj_deg: int = 1,
        self_loops: bool = False,
        poly_sum: bool = False,
        undir_or_addcol: str = "undir",
        only_reverse: bool = False,
        kg_list: list = None,
        device: str = "cpu",
    ):
        super(SimpleAdjacencyMapModel, self).__init__()
        
        self.dropout = dropout
        self.device = device
        self.normalize = normalize

        self.use_cntr = False
        self.cat_cntr = False

        self.only_reverse = only_reverse

        self.adj_deg = adj_deg
        
        if undir_or_addcol == "undir":
            self.make_undirected = True
            self.add_column = False
        elif undir_or_addcol == "addcol":
            self.make_undirected = False
            self.add_column = True
        elif undir_or_addcol == "none":
            self.make_undirected = False
            self.add_column = False
        else:
            raise ValueError(f"Invalid undir_or_addcol value: {undir_or_addcol}")

        self.poly_sum = poly_sum

        out_dim = hidden_dim if out_dim is None else out_dim

        if isinstance(graph, Tuple):
            edge_index, edge_weight, num_perts = graph
            self.num_perts = num_perts
        else:
            _, _, self.num_perts = next(iter(graph.graph_dict.values()))
            all_edge_indices = []
            print("graph_dict keys:", graph.graph_dict.keys())
            print("kg_list:", kg_list)
            for name, G in graph.graph_dict.items():
                if kg_list is not None and name not in kg_list:
                    continue
                G_edge_index, G_edge_weight, _ = G
                G_edge_index = G_edge_index.to(device, dtype=torch.long)
                all_edge_indices.append(G_edge_index)
            
            edge_index = torch.cat(all_edge_indices, dim=1)
            edge_index = torch.unique(edge_index, dim=1)

        if self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_perts)

        if self.make_undirected:
            edge_index = to_undirected(edge_index)

        self.edge_index = edge_index.to(device)

        self.adj_matrix = self._build_adjacency_matrix()
        self.node_feat_dim = self.adj_matrix.size(1)

        if use_ffn:
            self.map_to_latent = nn.Sequential(
                nn.Linear(self.node_feat_dim, hidden_dim),
                nn.ReLU(), 
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            ).to(device)
        else:
            self.map_to_latent = nn.Linear(self.node_feat_dim, out_dim).to(device)

        if self.poly_sum:
            self.alpha = torch.nn.Parameter(torch.tensor(0.0))
            self.register_parameter("alpha", self.alpha)

    def _build_adjacency_matrix(self):
        """
        Build adjacency matrix from edge_index.
        
        Returns:
            torch.Tensor: Dense adjacency matrix of shape (num_perts, num_perts)
        """
        adj = torch.zeros(self.num_perts, self.num_perts, device=self.device)
        adj[self.edge_index[0], self.edge_index[1]] = 1.0

        # Normalize adjacency matrix if requested
        if self.normalize == "sym":
            # Symmetric normalization: D^(-1/2) A D^(-1/2)
            deg = adj.sum(dim=1)
            deg_inv_sqrt = torch.pow(deg, -0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
            adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        elif self.normalize == "row":
            # Row normalization: D^(-1) A
            deg = adj.sum(dim=1, keepdim=True)
            deg[deg == 0] = 1.0
            adj = adj / deg
        elif self.normalize == "col":
            # Column normalization: A D^(-1)
            deg = adj.sum(dim=0, keepdim=True)
            deg[deg == 0] = 1.0
            adj = adj / deg

        if self.only_reverse:
            adj = adj.t()
            return adj

        mats = []
        cur = torch.eye(self.num_perts, device=self.device, dtype=adj.dtype)
        for i in range(1, int(self.adj_deg) + 1):
            cur = cur.matmul(adj)
            if self.add_column:
                mats.append(torch.cat([cur, cur.t()], dim=1))
            else:
                mats.append(cur)

        if self.poly_sum:
            self.mats = mats
            adj = mats[0]
        else:
            adj = torch.cat(mats, dim=1)
        
        return adj

    def forward(self, edge_index, edge_weight, z_intrinsic=None):
        """
        Forward pass that computes node embeddings using adjacency matrix rows.
        
        Args:
            edge_index: Edge indices (used for compatibility, but uses stored adjacency matrix)
            edge_weight: Edge weights (used for compatibility, ignored in this model)
            z_intrinsic: Intrinsic embeddings (optional, not used currently)
        
        Returns:
            Node embeddings of shape (num_perts, out_dim)
        """
        if self.poly_sum:
            self.beta = torch.sigmoid(self.alpha)
            poly_adj_matrix = torch.zeros_like(self.adj_matrix, device=self.device)
            for i in range(len(self.mats)):
                poly_adj_matrix += (self.beta ** i) * self.mats[i]
                
            x = self.map_to_latent(poly_adj_matrix)
        else:
            x = self.map_to_latent(self.adj_matrix)
        
        return x

