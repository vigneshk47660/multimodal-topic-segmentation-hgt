"""
Evaluation Metrics for Topic Segmentation
Pk, WindowDiff, Boundary F1, Precision, Recall
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


def _boundaries_to_segments(boundaries: List[int], n: int) -> np.ndarray:
    """Convert boundary indices to segment labels."""
    labels = np.zeros(n, dtype=int)
    seg_id = 0
    prev = 0
    for b in sorted(boundaries):
        labels[prev:b] = seg_id
        prev = b
        seg_id += 1
    labels[prev:] = seg_id
    return labels


def pk_score(
    predicted_boundaries: List[int],
    reference_boundaries: List[int],
    n: int,
    window_size: Optional[int] = None,
) -> float:
    """
    Pk metric (Beeferman et al., 1999).
    Lower is better.
    """
    pred_labels = _boundaries_to_segments(predicted_boundaries, n)
    ref_labels = _boundaries_to_segments(reference_boundaries, n)

    if window_size is None:
        num_ref_segments = len(reference_boundaries) + 1
        window_size = max(2, n // (2 * num_ref_segments))

    errors = 0
    total = 0
    for i in range(n - window_size):
        j = i + window_size
        pred_same = (pred_labels[i] == pred_labels[j])
        ref_same = (ref_labels[i] == ref_labels[j])
        if pred_same != ref_same:
            errors += 1
        total += 1

    return errors / max(total, 1)


def windowdiff_score(
    predicted_boundaries: List[int],
    reference_boundaries: List[int],
    n: int,
    window_size: Optional[int] = None,
) -> float:
    """
    WindowDiff metric (Pevzner & Hearst, 2002).
    Lower is better.
    """
    pred_labels = _boundaries_to_segments(predicted_boundaries, n)
    ref_labels = _boundaries_to_segments(reference_boundaries, n)

    if window_size is None:
        num_ref_segments = len(reference_boundaries) + 1
        window_size = max(2, n // (2 * num_ref_segments))

    errors = 0
    total = 0
    for i in range(n - window_size):
        j = i + window_size
        pred_boundaries_in_window = sum(
            1 for k in range(i, j) if pred_labels[k] != pred_labels[k + 1]
        )
        ref_boundaries_in_window = sum(
            1 for k in range(i, j) if ref_labels[k] != ref_labels[k + 1]
        )
        if pred_boundaries_in_window != ref_boundaries_in_window:
            errors += 1
        total += 1

    return errors / max(total, 1)


def boundary_precision_recall_f1(
    predicted_boundaries: List[int],
    reference_boundaries: List[int],
    tolerance: int = 2,
) -> Dict[str, float]:
    """
    Boundary detection Precision, Recall, F1 with tolerance window.
    """
    pred_set = set(predicted_boundaries)
    ref_set = set(reference_boundaries)

    if len(pred_set) == 0 and len(ref_set) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if len(pred_set) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if len(ref_set) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # True positives for precision: predicted boundaries near a reference
    tp_precision = 0
    for p in pred_set:
        if any(abs(p - r) <= tolerance for r in ref_set):
            tp_precision += 1

    # True positives for recall: reference boundaries near a prediction
    tp_recall = 0
    for r in ref_set:
        if any(abs(r - p) <= tolerance for p in pred_set):
            tp_recall += 1

    precision = tp_precision / len(pred_set)
    recall = tp_recall / len(ref_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_segmentation(
    predicted_boundaries: List[int],
    reference_boundaries: List[int],
    n: int,
    window_size: Optional[int] = None,
    tolerance: int = 2,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    """
    pk = pk_score(predicted_boundaries, reference_boundaries, n, window_size)
    wd = windowdiff_score(predicted_boundaries, reference_boundaries, n, window_size)
    prf = boundary_precision_recall_f1(predicted_boundaries, reference_boundaries, tolerance)

    return {
        "pk": pk,
        "windowdiff": wd,
        "boundary_precision": prf["precision"],
        "boundary_recall": prf["recall"],
        "boundary_f1": prf["f1"],
        "num_predicted_boundaries": len(predicted_boundaries),
        "num_reference_boundaries": len(reference_boundaries),
    }
