from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GATConv


class QueryGraphGAT(nn.Module):
    """2-layer GAT graph reasoning module for query graph inference.

    Inputs:
        nodes: [N, in_dim]  — per-node feature vectors.
        edges: [2, E]       — COO-format edge index (source, target).

    Architecture:
        Layer 1: GATConv(in_dim, hidden_dim, heads) → ELU → LayerNorm
        Layer 2: GATConv(hidden_dim, out_dim,   heads) → ELU → LayerNorm
        Residual projection: Linear(hidden_dim, out_dim) aligns L1 output
                             with L2 output for the sum.

    Output:
        enhanced_nodes: [N, out_dim]
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 256,
        out_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.heads = heads

        # Layer 1
        self.gat1 = GATConv(
            in_dim, hidden_dim,
            heads=heads, concat=False, dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Layer 2
        self.gat2 = GATConv(
            hidden_dim, out_dim,
            heads=heads, concat=False, dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(out_dim)

        # Residual projection: align L1 output (hidden_dim) to L2 output (out_dim)
        self.projection = nn.Linear(hidden_dim, out_dim)

    def forward(self, nodes: Tensor, edges: Tensor) -> Tensor:
        """Forward pass.

        Args:
            nodes: [N, in_dim] node feature matrix.
            edges: [2, E] COO edge index (dtype long).

        Returns:
            enhanced_nodes: [N, out_dim].
        """
        edge_index = edges.long()

        # Layer 1
        x = self.gat1(nodes, edge_index)
        x = nn.functional.elu(x)
        x = self.norm1(x)

        # Residual branch: project L1 output
        residual = self.projection(x)

        # Layer 2
        x = self.gat2(x, edge_index)
        x = nn.functional.elu(x)
        x = self.norm2(x)

        # Residual connection
        x = x + residual

        return x
