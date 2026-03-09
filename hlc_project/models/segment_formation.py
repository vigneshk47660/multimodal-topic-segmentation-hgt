"""
Stage 7: Topic Segment Formation using FAISS-Assisted Semantic Grouping
Implements Eq. 12-14 and Algorithm 2 from the paper.
B = {b1, b2, ..., bK}  (Eq. 12)
S_j = {u_{b_{j-1}+1}, ..., u_{b_j}}  (Eq. 13)
z_j = sum_{u_i in S_j} h_i  (Eq. 14)
FAISS-based semantic validation and merging of similar segments.
"""

import torch
import torch.nn.functional as F
import numpy as np
import faiss
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from models.instructional_unit_builder import InstructionalUnit, ModalityType


@dataclass
class TopicSegment:
    """Represents a single topic segment with constituent units."""
    segment_id: int
    unit_indices: List[int]
    units: List[InstructionalUnit]
    aggregate_vector: Optional[np.ndarray] = None
    modality_distribution: Dict[str, int] = field(default_factory=dict)
    start_temporal: int = 0
    end_temporal: int = 0
    topic_label: Optional[str] = None

    def __post_init__(self):
        if self.units:
            self.start_temporal = min(u.temporal_index for u in self.units)
            self.end_temporal = max(u.temporal_index for u in self.units)
            self.modality_distribution = {}
            for u in self.units:
                mod = u.modality.value
                self.modality_distribution[mod] = self.modality_distribution.get(mod, 0) + 1


class SegmentFormation:
    """
    Converts boundary indices into meaningful multimodal topic segments.
    Implements Algorithm 2: Semantic Segment Formation and Validation.
    """

    def __init__(
        self,
        merge_threshold: float = 0.85,
        nlist: int = 10,
        nprobe: int = 5,
        use_gpu_faiss: bool = False,
    ):
        self.merge_threshold = merge_threshold
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu_faiss = use_gpu_faiss

    def construct_segments(
        self,
        boundaries: List[int],
        fused_representations: torch.Tensor,
        units: List[InstructionalUnit],
    ) -> List[TopicSegment]:
        """
        Partition lecture timeline using detected boundaries (Eq. 13).
        S1 = {u1,...,u_{b1}}, S2 = {u_{b1+1},...,u_{b2}}, ..., S_{K+1} = {u_{bK+1},...,u_N}
        """
        N = len(units)
        sorted_boundaries = sorted(set(boundaries))
        sorted_boundaries = [b for b in sorted_boundaries if 0 < b < N]

        # Build segment boundaries
        seg_starts = [0] + [b for b in sorted_boundaries]
        seg_ends = [b for b in sorted_boundaries] + [N]

        segments = []
        for seg_id, (start, end) in enumerate(zip(seg_starts, seg_ends)):
            if start >= end:
                continue
            indices = list(range(start, end))
            seg_units = [units[i] for i in indices]

            # Compute aggregate vector (Eq. 14): z_j = sum_{u_i in S_j} h_i
            seg_vectors = fused_representations[start:end]
            agg_vector = seg_vectors.sum(dim=0).detach().cpu().numpy()

            segment = TopicSegment(
                segment_id=seg_id,
                unit_indices=indices,
                units=seg_units,
                aggregate_vector=agg_vector,
            )
            segments.append(segment)

        return segments

    def faiss_semantic_validation(
        self,
        segments: List[TopicSegment],
    ) -> List[TopicSegment]:
        """
        FAISS-based approximate nearest neighbor search for semantic validation.
        Merges segments with high semantic similarity (Algorithm 2, lines 9-15).
        """
        if len(segments) < 2:
            return segments

        # Build FAISS index from segment aggregate vectors
        vectors = np.stack([seg.aggregate_vector for seg in segments]).astype(np.float32)
        # L2 normalize for cosine similarity
        faiss.normalize_L2(vectors)
        dim = vectors.shape[1]

        # Use IVF index for larger collections, flat for small
        if len(segments) > 50:
            nlist = min(self.nlist, len(segments) // 2)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(vectors)
            index.nprobe = min(self.nprobe, nlist)
        else:
            index = faiss.IndexFlatIP(dim)

        index.add(vectors)

        # Search for nearest neighbors
        k = min(3, len(segments))
        D, I = index.search(vectors, k)

        # Identify merge candidates (adjacent segments with high similarity)
        merge_pairs = []
        for i in range(len(segments)):
            for j_idx in range(1, k):  # Skip self (j_idx=0)
                j = I[i][j_idx]
                sim = D[i][j_idx]
                if sim > self.merge_threshold and abs(i - j) == 1:
                    pair = (min(i, j), max(i, j))
                    if pair not in merge_pairs:
                        merge_pairs.append(pair)

        # Merge segments (greedy, from right to left to preserve indices)
        merge_pairs.sort(key=lambda x: x[0], reverse=True)
        merged_indices = set()

        for i, j in merge_pairs:
            if i in merged_indices or j in merged_indices:
                continue
            # Merge j into i
            segments[i].unit_indices.extend(segments[j].unit_indices)
            segments[i].units.extend(segments[j].units)
            segments[i].aggregate_vector = (
                segments[i].aggregate_vector + segments[j].aggregate_vector
            )
            segments[i].end_temporal = max(
                segments[i].end_temporal, segments[j].end_temporal
            )
            # Update modality distribution
            for mod, count in segments[j].modality_distribution.items():
                segments[i].modality_distribution[mod] = (
                    segments[i].modality_distribution.get(mod, 0) + count
                )
            merged_indices.add(j)

        # Remove merged segments and re-index
        refined = [seg for idx, seg in enumerate(segments) if idx not in merged_indices]
        for new_id, seg in enumerate(refined):
            seg.segment_id = new_id
            seg.unit_indices.sort()

        return refined

    def form_segments(
        self,
        boundaries: List[int],
        fused_representations: torch.Tensor,
        units: List[InstructionalUnit],
    ) -> List[TopicSegment]:
        """
        Full segment formation pipeline (Algorithm 2).
        1. Timeline Partitioning
        2. Aggregate Vector Computation
        3. FAISS-Based Semantic Validation
        4. Refinement
        """
        # Step 1-2: Construct segments with aggregate vectors
        segments = self.construct_segments(
            boundaries, fused_representations, units
        )

        # Step 3: FAISS-based semantic validation and merging
        segments = self.faiss_semantic_validation(segments)

        return segments

    @staticmethod
    def segments_to_dict(segments: List[TopicSegment]) -> List[Dict]:
        """Convert segments to serializable format."""
        result = []
        for seg in segments:
            result.append({
                "segment_id": seg.segment_id,
                "unit_indices": seg.unit_indices,
                "start_temporal": seg.start_temporal,
                "end_temporal": seg.end_temporal,
                "num_units": len(seg.units),
                "modality_distribution": seg.modality_distribution,
                "topic_label": seg.topic_label,
            })
        return result


def form_topic_segments(
    boundaries: List[int],
    fused_representations: torch.Tensor,
    units: List[InstructionalUnit],
    merge_threshold: float = 0.85,
) -> List[TopicSegment]:
    """Convenience function for Stage 7."""
    formation = SegmentFormation(merge_threshold=merge_threshold)
    return formation.form_segments(boundaries, fused_representations, units)
