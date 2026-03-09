"""
Dataset loader for HLC multimodal topic segmentation.
Handles synthetic and real-world datasets.
"""

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple

from models.instructional_unit_builder import InstructionalUnit, ModalityType


class HLCDataset(Dataset):
    """PyTorch Dataset for heterogeneous lecture content."""

    def __init__(self, data_path: str, split: str = "train"):
        filepath = os.path.join(data_path, f"{split}.json")
        with open(filepath, "r", encoding="utf-8") as f:
            self.lectures = json.load(f)
        self.split = split

    def __len__(self) -> int:
        return len(self.lectures)

    def __getitem__(self, idx: int) -> Dict:
        lecture = self.lectures[idx]
        units = []
        for u in lecture["units"]:
            try:
                modality = ModalityType(u["modality"])
            except ValueError:
                modality = ModalityType.TEXT
            unit = InstructionalUnit(
                content=u["content"],
                temporal_index=u.get("temporal_index", 0),
                modality=modality,
                metadata=u.get("metadata", {}),
            )
            units.append(unit)

        return {
            "lecture_id": lecture["lecture_id"],
            "units": units,
            "ground_truth_boundaries": lecture["ground_truth_boundaries"],
            "topic_labels": lecture.get("topic_labels", []),
            "num_units": lecture["num_units"],
            "num_topics": lecture["num_topics"],
            "domain": lecture.get("domain", "unknown"),
        }


def hlc_collate_fn(batch: List[Dict]) -> List[Dict]:
    """Custom collate: return list of dicts (variable-length sequences)."""
    return batch


def get_dataloader(
    data_path: str,
    split: str = "train",
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create DataLoader for HLC dataset."""
    dataset = HLCDataset(data_path, split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if split == "train" else False,
        num_workers=num_workers,
        collate_fn=hlc_collate_fn,
    )
