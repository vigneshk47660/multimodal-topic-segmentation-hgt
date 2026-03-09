"""
Stage 4: Shared Semantic Space Alignment using Cross-Modal Projection
Implements Eq. 6 from the paper.
s_i = W_{m_i} * e_i + b_{m_i}  (Eq. 6)
Projects modality-specific embeddings into a shared k-dimensional space.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional

from models.instructional_unit_builder import ModalityType


class CrossModalProjection(nn.Module):
    """
    Projects modality-specific embeddings into a shared semantic space.
    Each modality has its own learnable projection: W_{m_i} and b_{m_i}.
    s_i = W_{m_i} * e_i + b_{m_i}  (Eq. 6)
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        shared_dim: int = 256,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.shared_dim = shared_dim
        self.projections = nn.ModuleDict()

        for modality_name, input_dim in modality_dims.items():
            layers = [nn.Linear(input_dim, shared_dim)]
            if use_layer_norm:
                layers.append(nn.LayerNorm(shared_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(shared_dim, shared_dim))

            self.projections[modality_name] = nn.Sequential(*layers)

    def forward(
        self,
        embeddings: torch.Tensor,
        modalities: List[ModalityType],
    ) -> torch.Tensor:
        """
        Project all embeddings into shared space.
        Args:
            embeddings: (N, d_max) padded or per-unit embeddings
            modalities: list of ModalityType for each unit
        Returns:
            S: (N, k) shared semantic space representations
        """
        N = embeddings.shape[0]
        device = embeddings.device
        shared = torch.zeros(N, self.shared_dim, device=device)

        # Group by modality for efficient projection
        modality_groups: Dict[str, List[int]] = {}
        for idx, mod in enumerate(modalities):
            mod_name = mod.value
            if mod_name not in modality_groups:
                modality_groups[mod_name] = []
            modality_groups[mod_name].append(idx)

        for mod_name, indices in modality_groups.items():
            if mod_name in self.projections:
                idx_tensor = torch.tensor(indices, device=device)
                mod_embeddings = embeddings[idx_tensor]
                # Get the input dim expected by this projection
                expected_dim = self.projections[mod_name][0].in_features
                if mod_embeddings.shape[-1] != expected_dim:
                    # Pad or truncate
                    if mod_embeddings.shape[-1] < expected_dim:
                        pad = torch.zeros(
                            mod_embeddings.shape[0],
                            expected_dim - mod_embeddings.shape[-1],
                            device=device
                        )
                        mod_embeddings = torch.cat([mod_embeddings, pad], dim=-1)
                    else:
                        mod_embeddings = mod_embeddings[:, :expected_dim]

                projected = self.projections[mod_name](mod_embeddings)
                shared[idx_tensor] = projected

        return shared

    def project_single(
        self, embedding: torch.Tensor, modality: ModalityType
    ) -> torch.Tensor:
        """Project a single embedding into shared space."""
        mod_name = modality.value
        if mod_name in self.projections:
            expected_dim = self.projections[mod_name][0].in_features
            if embedding.shape[-1] != expected_dim:
                if embedding.shape[-1] < expected_dim:
                    pad = torch.zeros(
                        expected_dim - embedding.shape[-1],
                        device=embedding.device
                    )
                    embedding = torch.cat([embedding, pad], dim=-1)
                else:
                    embedding = embedding[:expected_dim]
            return self.projections[mod_name](embedding.unsqueeze(0)).squeeze(0)
        return embedding


def build_cross_modal_projection(
    modality_dims: Optional[Dict[str, int]] = None,
    shared_dim: int = 256,
) -> CrossModalProjection:
    """Convenience function for building Stage 4 module."""
    if modality_dims is None:
        modality_dims = {
            "text": 384,
            "equation": 768,
            "table": 768,
            "diagram": 768,
        }
    return CrossModalProjection(modality_dims, shared_dim)
