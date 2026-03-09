"""
Stage 3: Modality-Specific Representation Learning using Neural Encoders
Implements Eq. 4-5 from the paper.
e_i = f_{m_i}(c_tilde_i)  (Eq. 4)
E = {e1, e2, ..., eN}      (Eq. 5)
Encoders: SBERT (text), MathBERT (equations), TAPAS (tables), ViT (diagrams)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

from models.instructional_unit_builder import InstructionalUnit, ModalityType


@dataclass
class EncoderConfig:
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    equation_model: str = "tbs17/MathBERT"
    table_model: str = "google/tapas-base"
    diagram_model: str = "google/vit-base-patch16-224"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 512


class TextEncoder(nn.Module):
    """SBERT-based encoder for textual instructional units."""

    def __init__(self, model_name: str, device: str = "cpu"):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        self.output_dim = self.model.get_sentence_embedding_dimension()

    def forward(self, texts: List[str]) -> torch.Tensor:
        embeddings = self.model.encode(
            texts, convert_to_tensor=True,
            device=self.device, show_progress_bar=False
        )
        return embeddings

    def encode_single(self, text: str) -> torch.Tensor:
        return self.forward([text]).squeeze(0)


class EquationEncoder(nn.Module):
    """MathBERT-based encoder for mathematical equation units."""

    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 256):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.max_length = max_length
        self.output_dim = self.model.config.hidden_size

    @torch.no_grad()
    def forward(self, equations: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            equations, return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_length
        ).to(self.device)
        outputs = self.model(**inputs)
        # Use [CLS] token representation
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings

    def encode_single(self, equation: str) -> torch.Tensor:
        return self.forward([equation]).squeeze(0)


class TableEncoder(nn.Module):
    """TAPAS-based encoder for table instructional units."""

    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 512):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.max_length = max_length
        self.output_dim = self.model.config.hidden_size

    def _prepare_table_input(self, table_text: str) -> dict:
        """Convert linearized table text back to TAPAS-compatible format."""
        cells = table_text.split(' [SEP] ')
        # Fallback: treat as flat text if not parseable as table
        if len(cells) < 2:
            inputs = self.tokenizer(
                table_text, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_length
            )
            return inputs

        # Attempt to structure as table query
        import pandas as pd
        ncols = min(max(int(len(cells) ** 0.5), 2), 10)
        rows = [cells[i:i + ncols] for i in range(0, len(cells), ncols)]
        # Pad rows to same length
        max_cols = max(len(r) for r in rows)
        rows = [r + [''] * (max_cols - len(r)) for r in rows]

        if len(rows) > 1:
            headers = rows[0]
            data_rows = rows[1:]
        else:
            headers = [f"col_{j}" for j in range(max_cols)]
            data_rows = rows

        df = pd.DataFrame(data_rows, columns=headers[:max_cols])
        try:
            inputs = self.tokenizer(
                table=df, queries=["what is in this table?"],
                return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_length
            )
        except Exception:
            inputs = self.tokenizer(
                table_text, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_length
            )
        return inputs

    @torch.no_grad()
    def forward(self, tables: List[str]) -> torch.Tensor:
        all_embeddings = []
        for table_text in tables:
            inputs = self._prepare_table_input(table_text)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(embedding)
        return torch.cat(all_embeddings, dim=0)

    def encode_single(self, table_text: str) -> torch.Tensor:
        return self.forward([table_text]).squeeze(0)


class DiagramEncoder(nn.Module):
    """ViT-based encoder for diagram/figure instructional units.
    Since diagrams in transcripts are typically captions/references,
    we use a text-based fallback with ViT feature projection.
    """

    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 512):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel, ViTModel, ViTFeatureExtractor
        # Use ViT for actual image inputs
        try:
            self.vit_model = ViTModel.from_pretrained(model_name).to(device)
            self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
            self.has_vit = True
        except Exception:
            self.has_vit = False

        # Fallback text encoder for caption/reference-based diagram units
        fallback_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(fallback_name)
        self.text_model = AutoModel.from_pretrained(fallback_name).to(device)
        self.text_model.eval()
        self.device = device
        self.max_length = max_length
        self.output_dim = 768

    @torch.no_grad()
    def forward(self, diagram_inputs: List[Union[str, "PIL.Image.Image"]]) -> torch.Tensor:
        embeddings = []
        for inp in diagram_inputs:
            if isinstance(inp, str):
                # Text-based diagram reference/caption
                tokens = self.tokenizer(
                    inp, return_tensors="pt", padding=True,
                    truncation=True, max_length=self.max_length
                ).to(self.device)
                out = self.text_model(**tokens)
                emb = out.last_hidden_state[:, 0, :]
            else:
                # Actual image input
                if self.has_vit:
                    pixel_values = self.feature_extractor(
                        images=inp, return_tensors="pt"
                    ).pixel_values.to(self.device)
                    out = self.vit_model(pixel_values=pixel_values)
                    emb = out.last_hidden_state[:, 0, :]
                else:
                    emb = torch.randn(1, self.output_dim, device=self.device)
            embeddings.append(emb)
        return torch.cat(embeddings, dim=0)

    def encode_single(self, diagram_input: Union[str, "PIL.Image.Image"]) -> torch.Tensor:
        return self.forward([diagram_input]).squeeze(0)


class ModalitySpecificEncoders(nn.Module):
    """
    Unified interface for all modality-specific encoders.
    e_i = f_{m_i}(c_tilde_i)  (Eq. 4)
    """

    def __init__(self, config: Optional[EncoderConfig] = None):
        super().__init__()
        if config is None:
            config = EncoderConfig()
        self.config = config
        self.device = config.device
        self._encoders: Dict[ModalityType, nn.Module] = {}
        self._initialized = set()

    def _lazy_init_encoder(self, modality: ModalityType):
        """Lazy initialization to avoid loading all models at once."""
        if modality in self._initialized:
            return
        if modality == ModalityType.TEXT:
            self._encoders[modality] = TextEncoder(
                self.config.text_model, self.device
            )
        elif modality == ModalityType.EQUATION:
            self._encoders[modality] = EquationEncoder(
                self.config.equation_model, self.device, self.config.max_length
            )
        elif modality == ModalityType.TABLE:
            self._encoders[modality] = TableEncoder(
                self.config.table_model, self.device, self.config.max_length
            )
        elif modality == ModalityType.DIAGRAM:
            self._encoders[modality] = DiagramEncoder(
                self.config.diagram_model, self.device, self.config.max_length
            )
        self._initialized.add(modality)

    def get_output_dim(self, modality: ModalityType) -> int:
        dims = {
            ModalityType.TEXT: 384,
            ModalityType.EQUATION: 768,
            ModalityType.TABLE: 768,
            ModalityType.DIAGRAM: 768,
        }
        return dims.get(modality, 768)

    def encode_units(self, units: List[InstructionalUnit]) -> Dict[str, torch.Tensor]:
        """
        Encode all instructional units using modality-specific encoders.
        Returns E = {e1, e2, ..., eN}  (Eq. 5)
        """
        # Group units by modality for batched encoding
        modality_groups: Dict[ModalityType, List[int]] = {}
        for idx, unit in enumerate(units):
            mod = unit.modality
            if mod not in modality_groups:
                modality_groups[mod] = []
            modality_groups[mod].append(idx)

        # Encode each modality group
        embeddings = [None] * len(units)
        for modality, indices in modality_groups.items():
            self._lazy_init_encoder(modality)
            encoder = self._encoders[modality]

            contents = [
                units[i].normalized_content or units[i].content
                for i in indices
            ]

            # Batch encode
            batch_embeddings = encoder.forward(contents)

            for i, idx in enumerate(indices):
                embeddings[idx] = batch_embeddings[i]

        # Stack into tensor
        embedding_tensor = torch.stack(embeddings, dim=0)

        # Build metadata
        modalities = [unit.modality for unit in units]
        temporal_indices = torch.tensor(
            [unit.temporal_index for unit in units], dtype=torch.long
        )

        return {
            "embeddings": embedding_tensor,        # (N, d_m) variable dim per modality
            "modalities": modalities,               # List[ModalityType]
            "temporal_indices": temporal_indices,    # (N,)
        }


def encode_instructional_units(
    units: List[InstructionalUnit],
    config: Optional[EncoderConfig] = None
) -> Dict[str, torch.Tensor]:
    """Convenience function for Stage 3."""
    encoder = ModalitySpecificEncoders(config)
    return encoder.encode_units(units)
