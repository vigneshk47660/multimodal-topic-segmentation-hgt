"""
Unified HLC Multimodal Topic Segmentation Pipeline
Combines all stages from Algorithm 1 and Algorithm 2.
Full end-to-end: L -> U -> U_tilde -> E -> S -> H -> B -> Segments
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import yaml
import os

from models.instructional_unit_builder import (
    InstructionalUnitBuilder, InstructionalUnit, ModalityType
)
from models.hlc_normalizer import HLCNormalizer
from models.modality_encoders import ModalitySpecificEncoders, EncoderConfig
from models.cross_modal_projection import CrossModalProjection
from models.hgt_fusion import HeterogeneousGraphTransformer
from models.change_point_detection import (
    SimilarityProfiler, NeuralChangePointDetector
)
from models.segment_formation import SegmentFormation, TopicSegment


class HLCMultimodalSegmentationPipeline(nn.Module):
    """
    End-to-end pipeline for multimodal topic segmentation in lecture videos.

    Pipeline:
        1. Instructional Unit Builder    (Eq. 1-2)
        2. HLC Normalization             (Eq. 3)
        3. Modality-Specific Encoding    (Eq. 4-5)
        4. Cross-Modal Projection        (Eq. 6)
        5. HGT Temporal Fusion           (Eq. 7-8)
        6. Similarity + Change-Point     (Eq. 9-11)
        7. FAISS Segment Formation       (Eq. 12-14, Algorithm 2)
    """

    def __init__(self, config: Optional[Dict] = None, config_path: Optional[str] = None):
        super().__init__()

        if config is None and config_path is not None:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        elif config is None:
            config = self._default_config()

        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Stage 1: Instructional Unit Builder (rule-based, no parameters)
        self.unit_builder = InstructionalUnitBuilder()

        # Stage 2: HLC Normalizer (rule-based, no parameters)
        self.normalizer = HLCNormalizer()

        # Stage 3: Modality-Specific Encoders
        encoder_cfg = EncoderConfig(
            text_model=config["encoders"]["text"]["model_name"],
            equation_model=config["encoders"]["equation"]["model_name"],
            table_model=config["encoders"]["table"]["model_name"],
            diagram_model=config["encoders"]["diagram"]["model_name"],
            device=self.device,
        )
        self.modality_encoders = ModalitySpecificEncoders(encoder_cfg)

        # Stage 4: Cross-Modal Projection
        modality_dims = {
            "text": config["encoders"]["text"]["output_dim"],
            "equation": config["encoders"]["equation"]["output_dim"],
            "table": config["encoders"]["table"]["output_dim"],
            "diagram": config["encoders"]["diagram"]["output_dim"],
        }
        shared_dim = config["model"]["shared_dim"]
        self.cross_modal_projection = CrossModalProjection(
            modality_dims=modality_dims,
            shared_dim=shared_dim,
            dropout=config["model"]["hgt_dropout"],
        )

        # Stage 5: HGT Fusion
        self.hgt = HeterogeneousGraphTransformer(
            input_dim=shared_dim,
            hidden_dim=config["model"]["hgt_hidden_dim"],
            num_heads=config["model"]["hgt_num_heads"],
            num_layers=config["model"]["hgt_num_layers"],
            num_node_types=config["model"]["num_modalities"],
            num_edge_types=3,  # temporal_next, temporal_prev, cross_modal
            dropout=config["model"]["hgt_dropout"],
            temporal_window=config["graph"]["temporal_window"],
            cross_modal_threshold=config["graph"]["cross_modal_similarity_threshold"],
        )

        # Stage 6: Change-Point Detection
        self.change_point_detector = NeuralChangePointDetector(
            input_dim=config["model"]["hgt_hidden_dim"],
            hidden_dim=128,
            min_segment_length=config["change_point"]["min_segment_length"],
            penalty=config["change_point"]["penalty"],
        )

        # Stage 7: Segment Formation
        self.segment_formation = SegmentFormation(
            merge_threshold=config["faiss"]["merge_threshold"],
            nlist=config["faiss"]["nlist"],
            nprobe=config["faiss"]["nprobe"],
        )

    def _default_config(self):
        return {
            "model": {
                "shared_dim": 256,
                "hgt_hidden_dim": 256,
                "hgt_num_heads": 8,
                "hgt_num_layers": 3,
                "hgt_dropout": 0.1,
                "num_modalities": 4,
            },
            "encoders": {
                "text": {"model_name": "sentence-transformers/all-MiniLM-L6-v2", "output_dim": 384},
                "equation": {"model_name": "tbs17/MathBERT", "output_dim": 768},
                "table": {"model_name": "google/tapas-base", "output_dim": 768},
                "diagram": {"model_name": "google/vit-base-patch16-224", "output_dim": 768},
            },
            "graph": {
                "temporal_window": 1,
                "cross_modal_similarity_threshold": 0.5,
            },
            "change_point": {
                "method": "neural",
                "penalty": 1.0,
                "min_segment_length": 3,
                "kernel": "rbf",
            },
            "faiss": {
                "merge_threshold": 0.85,
                "nlist": 10,
                "nprobe": 5,
            },
        }

    def forward(
        self,
        transcript: Optional[str] = None,
        prebuilt_units: Optional[List[InstructionalUnit]] = None,
        return_intermediates: bool = False,
    ) -> Dict:
        """
        Full pipeline execution: L -> Segments

        Args:
            transcript: raw lecture transcript string
            prebuilt_units: alternatively, pre-built instructional units
            return_intermediates: if True, return all intermediate outputs

        Returns:
            dict with 'segments', 'boundaries', and optionally intermediate results
        """
        intermediates = {}

        # Stage 1: Build Instructional Units (Eq. 1-2)
        if prebuilt_units is not None:
            units = prebuilt_units
        else:
            assert transcript is not None, "Provide either transcript or prebuilt_units"
            units = self.unit_builder.build(transcript)
        if return_intermediates:
            intermediates["units"] = units

        # Stage 2: Normalize and Label (Eq. 3)
        normalized_units = self.normalizer.normalize(units)
        if return_intermediates:
            intermediates["normalized_units"] = normalized_units

        # Stage 3: Modality-Specific Encoding (Eq. 4-5)
        encoding_result = self.modality_encoders.encode_units(normalized_units)
        embeddings = encoding_result["embeddings"]
        modalities = encoding_result["modalities"]
        if return_intermediates:
            intermediates["embeddings"] = embeddings

        # Stage 4: Cross-Modal Projection (Eq. 6)
        shared_embeddings = self.cross_modal_projection(embeddings, modalities)
        if return_intermediates:
            intermediates["shared_embeddings"] = shared_embeddings

        # Stage 5: HGT Fusion (Eq. 7-8)
        hgt_result = self.hgt(shared_embeddings, modalities)
        fused_representations = hgt_result["fused_representations"]
        if return_intermediates:
            intermediates["fused_representations"] = fused_representations
            intermediates["graph"] = hgt_result["graph"]

        # Stage 6: Change-Point Detection (Eq. 9-11)
        cpd_result = self.change_point_detector(
            fused_representations,
            method=self.config["change_point"]["method"],
            penalty=self.config["change_point"]["penalty"],
        )
        boundaries = cpd_result["boundaries"]
        if return_intermediates:
            intermediates["similarity_sequence"] = cpd_result["similarity_sequence"]
            intermediates["boundary_scores"] = cpd_result["scores"]

        # Stage 7: Segment Formation (Eq. 12-14, Algorithm 2)
        segments = self.segment_formation.form_segments(
            boundaries, fused_representations, normalized_units
        )

        result = {
            "segments": segments,
            "boundaries": boundaries,
            "num_segments": len(segments),
            "num_units": len(normalized_units),
        }

        if return_intermediates:
            result["intermediates"] = intermediates

        return result

    def save_checkpoint(self, path: str):
        """Save trainable parameters."""
        state = {
            "cross_modal_projection": self.cross_modal_projection.state_dict(),
            "hgt": self.hgt.state_dict(),
            "change_point_detector": self.change_point_detector.state_dict(),
            "config": self.config,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(state, path)

    def load_checkpoint(self, path: str):
        """Load trainable parameters."""
        state = torch.load(path, map_location=self.device)
        self.cross_modal_projection.load_state_dict(state["cross_modal_projection"])
        self.hgt.load_state_dict(state["hgt"])
        self.change_point_detector.load_state_dict(state["change_point_detector"])
