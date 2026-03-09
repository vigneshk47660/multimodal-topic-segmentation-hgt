"""
Stage 5: Multimodal Temporal Fusion using Heterogeneous Graph Transformer (HGT)
Implements Eq. 7-8 from the paper.
h_i = HGT(s_i, N(u_i))  (Eq. 7)
H = {h1, h2, ..., hN}    (Eq. 8)
Models temporal progression and cross-modal interactions via a heterogeneous graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Optional, Tuple

from models.instructional_unit_builder import ModalityType


class HeterogeneousGraphBuilder:
    """
    Builds a heterogeneous temporal graph from instructional units.
    Node types: modality labels (text, equation, table, diagram)
    Edge types: temporal_next, temporal_prev, cross_modal
    """

    def __init__(
        self,
        temporal_window: int = 1,
        cross_modal_threshold: float = 0.5,
    ):
        self.temporal_window = temporal_window
        self.cross_modal_threshold = cross_modal_threshold
        self.edge_type_to_idx = {
            "temporal_next": 0,
            "temporal_prev": 1,
            "cross_modal": 2,
        }
        self.node_type_to_idx = {
            "text": 0,
            "equation": 1,
            "table": 2,
            "diagram": 3,
        }

    def build_graph(
        self,
        shared_embeddings: torch.Tensor,
        modalities: List[ModalityType],
    ) -> Dict[str, torch.Tensor]:
        """
        Build heterogeneous graph structure.
        Returns edge_index, edge_type, node_type tensors.
        """
        N = shared_embeddings.shape[0]
        device = shared_embeddings.device
        src_nodes, dst_nodes, edge_types = [], [], []

        # Temporal edges (forward and backward within window)
        for i in range(N):
            for w in range(1, self.temporal_window + 1):
                if i + w < N:
                    # Forward temporal edge
                    src_nodes.append(i)
                    dst_nodes.append(i + w)
                    edge_types.append(self.edge_type_to_idx["temporal_next"])
                    # Backward temporal edge
                    src_nodes.append(i + w)
                    dst_nodes.append(i)
                    edge_types.append(self.edge_type_to_idx["temporal_prev"])

        # Cross-modal semantic edges
        norms = F.normalize(shared_embeddings, p=2, dim=-1)
        sim_matrix = torch.mm(norms, norms.t())

        for i in range(N):
            for j in range(i + 1, N):
                if modalities[i] != modalities[j]:
                    if sim_matrix[i, j].item() > self.cross_modal_threshold:
                        src_nodes.extend([i, j])
                        dst_nodes.extend([j, i])
                        edge_types.extend([
                            self.edge_type_to_idx["cross_modal"],
                            self.edge_type_to_idx["cross_modal"]
                        ])

        # Node types
        node_types = torch.tensor(
            [self.node_type_to_idx.get(m.value, 0) for m in modalities],
            dtype=torch.long, device=device
        )

        if len(src_nodes) == 0:
            # Fallback: add self-loops
            src_nodes = list(range(N))
            dst_nodes = list(range(N))
            edge_types = [0] * N

        edge_index = torch.tensor(
            [src_nodes, dst_nodes], dtype=torch.long, device=device
        )
        edge_type = torch.tensor(edge_types, dtype=torch.long, device=device)

        return {
            "edge_index": edge_index,
            "edge_type": edge_type,
            "node_type": node_types,
            "num_nodes": N,
            "num_edge_types": len(self.edge_type_to_idx),
            "num_node_types": len(self.node_type_to_idx),
        }


class HGTAttentionHead(nn.Module):
    """Single attention head for HGT with type-specific transformations."""

    def __init__(self, hidden_dim: int, num_node_types: int, num_edge_types: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Type-specific key, query, value projections
        self.W_Q = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)
            for _ in range(num_node_types)
        ])
        self.W_K = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)
            for _ in range(num_node_types)
        ])
        self.W_V = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)
            for _ in range(num_node_types)
        ])

        # Edge-type specific attention
        self.W_edge = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)
            for _ in range(num_edge_types)
        ])

        self.mu = nn.Parameter(torch.ones(num_edge_types))
        self.scale = math.sqrt(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        node_type: torch.Tensor,
    ) -> torch.Tensor:
        N = x.shape[0]
        device = x.device
        src, dst = edge_index[0], edge_index[1]

        # Compute type-specific Q, K, V
        Q = torch.zeros_like(x)
        K = torch.zeros_like(x)
        V = torch.zeros_like(x)

        for ntype in range(len(self.W_Q)):
            mask = (node_type == ntype)
            if mask.any():
                Q[mask] = x[mask] @ self.W_Q[ntype]
                K[mask] = x[mask] @ self.W_K[ntype]
                V[mask] = x[mask] @ self.W_V[ntype]

        # Compute edge-type specific attention scores
        q_dst = Q[dst]
        k_src = K[src]

        # Apply edge-type transformation
        k_transformed = torch.zeros_like(k_src)
        for etype in range(len(self.W_edge)):
            emask = (edge_type == etype)
            if emask.any():
                k_transformed[emask] = k_src[emask] @ self.W_edge[etype]

        # Attention scores
        attn_scores = (q_dst * k_transformed).sum(dim=-1) / self.scale

        # Apply mu prior per edge type
        mu_per_edge = self.mu[edge_type]
        attn_scores = attn_scores * mu_per_edge

        # Softmax over incoming edges per destination node
        attn_weights = self._sparse_softmax(attn_scores, dst, N)

        # Aggregate values
        v_src = V[src]
        weighted_v = v_src * attn_weights.unsqueeze(-1)

        # Scatter add to destination nodes
        out = torch.zeros(N, self.hidden_dim, device=device)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted_v), weighted_v)

        return out

    def _sparse_softmax(
        self, scores: torch.Tensor, indices: torch.Tensor, N: int
    ) -> torch.Tensor:
        """Compute softmax over groups defined by indices."""
        scores_max = torch.zeros(N, device=scores.device).fill_(-1e9)
        scores_max.scatter_reduce_(0, indices, scores, reduce='amax')
        scores = scores - scores_max[indices]
        exp_scores = torch.exp(scores)
        sum_exp = torch.zeros(N, device=scores.device)
        sum_exp.scatter_add_(0, indices, exp_scores)
        return exp_scores / (sum_exp[indices] + 1e-10)


class HGTLayer(nn.Module):
    """Single Heterogeneous Graph Transformer layer."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_node_types: int,
        num_edge_types: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        self.attention_heads = nn.ModuleList([
            HGTAttentionHead(self.head_dim, num_node_types, num_edge_types)
            for _ in range(num_heads)
        ])

        self.W_out = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        node_type: torch.Tensor,
    ) -> torch.Tensor:
        # Split input into heads
        N, D = x.shape
        x_heads = x.view(N, self.num_heads, self.head_dim)

        # Multi-head attention
        head_outputs = []
        for i, head in enumerate(self.attention_heads):
            h_out = head(x_heads[:, i, :], edge_index, edge_type, node_type)
            head_outputs.append(h_out)

        # Concatenate heads
        multi_head_out = torch.cat(head_outputs, dim=-1)
        multi_head_out = self.W_out(multi_head_out)
        multi_head_out = self.dropout(multi_head_out)

        # Residual + LayerNorm
        x = self.layer_norm1(x + multi_head_out)

        # FFN + Residual + LayerNorm
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + ffn_out)

        return x


class HeterogeneousGraphTransformer(nn.Module):
    """
    Full HGT model for multimodal temporal fusion.
    h_i = HGT(s_i, N(u_i))  (Eq. 7)
    H = {h1, h2, ..., hN}    (Eq. 8)
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        num_node_types: int = 4,
        num_edge_types: int = 3,
        dropout: float = 0.1,
        temporal_window: int = 1,
        cross_modal_threshold: float = 0.5,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.graph_builder = HeterogeneousGraphBuilder(
            temporal_window=temporal_window,
            cross_modal_threshold=cross_modal_threshold,
        )
        self.layers = nn.ModuleList([
            HGTLayer(hidden_dim, num_heads, num_node_types, num_edge_types, dropout)
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        shared_embeddings: torch.Tensor,
        modalities: List[ModalityType],
        prebuilt_graph: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            shared_embeddings: (N, k) from cross-modal projection
            modalities: list of ModalityType
            prebuilt_graph: optional pre-built graph dict
        Returns:
            dict with 'fused_representations' H = {h1, ..., hN} and graph info
        """
        x = self.input_proj(shared_embeddings)

        if prebuilt_graph is None:
            graph = self.graph_builder.build_graph(shared_embeddings, modalities)
        else:
            graph = prebuilt_graph

        edge_index = graph["edge_index"]
        edge_type = graph["edge_type"]
        node_type = graph["node_type"]

        for layer in self.layers:
            x = layer(x, edge_index, edge_type, node_type)

        x = self.output_norm(x)

        return {
            "fused_representations": x,
            "graph": graph,
        }


def build_hgt(
    input_dim: int = 256,
    hidden_dim: int = 256,
    num_heads: int = 8,
    num_layers: int = 3,
    dropout: float = 0.1,
) -> HeterogeneousGraphTransformer:
    """Convenience function for building Stage 5 module."""
    return HeterogeneousGraphTransformer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )
