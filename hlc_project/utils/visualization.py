"""
Visualization utilities for topic segmentation results.
Generates similarity heatmaps, boundary plots, and segment visualizations.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
from typing import List, Dict, Optional


def plot_similarity_profile(
    similarity_sequence: np.ndarray,
    predicted_boundaries: List[int],
    gt_boundaries: Optional[List[int]] = None,
    save_path: str = "similarity_profile.png",
    title: str = "Similarity Profile with Detected Boundaries",
):
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(similarity_sequence))
    ax.plot(x, similarity_sequence, color="steelblue", linewidth=1.5, label="Cosine Similarity")
    ax.fill_between(x, similarity_sequence, alpha=0.2, color="steelblue")

    for b in predicted_boundaries:
        if b < len(similarity_sequence):
            ax.axvline(x=b, color="red", linestyle="--", alpha=0.8, linewidth=1.5)
    if gt_boundaries:
        for b in gt_boundaries:
            if b < len(similarity_sequence):
                ax.axvline(x=b, color="green", linestyle=":", alpha=0.7, linewidth=1.5)

    handles = [
        mpatches.Patch(color="red", label="Predicted Boundaries"),
    ]
    if gt_boundaries:
        handles.append(mpatches.Patch(color="green", label="Ground Truth Boundaries"))

    ax.set_xlabel("Unit Index", fontsize=12)
    ax.set_ylabel("Cosine Similarity", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(handles=handles, loc="lower left")
    ax.set_ylim(-0.1, 1.1)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_similarity_matrix(
    similarity_matrix: np.ndarray,
    predicted_boundaries: Optional[List[int]] = None,
    save_path: str = "similarity_matrix.png",
    title: str = "Full Similarity Matrix",
):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap="YlOrRd", vmin=-1, vmax=1,
                square=True, ax=ax, cbar_kws={"label": "Cosine Similarity"})

    if predicted_boundaries:
        for b in predicted_boundaries:
            ax.axhline(y=b, color="blue", linewidth=1, linestyle="--")
            ax.axvline(x=b, color="blue", linewidth=1, linestyle="--")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Unit Index")
    ax.set_ylabel("Unit Index")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_segment_overview(
    segments: List[Dict],
    n_units: int,
    save_path: str = "segment_overview.png",
    title: str = "Topic Segments Overview",
):
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(segments), 1)))

    for seg in segments:
        start = seg["unit_indices"][0] if seg.get("unit_indices") else seg.get("start_temporal", 0)
        end = seg["unit_indices"][-1] + 1 if seg.get("unit_indices") else seg.get("end_temporal", 0) + 1
        seg_id = seg.get("segment_id", 0)
        color = colors[seg_id % len(colors)]
        ax.barh(0, end - start, left=start, height=0.6, color=color,
                edgecolor="black", linewidth=0.5)
        mid = (start + end) / 2
        ax.text(mid, 0, f"S{seg_id}", ha="center", va="center", fontsize=9, fontweight="bold")

    ax.set_xlim(0, n_units)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("Unit Index", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_yticks([])
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_modality_distribution(
    segments: List[Dict],
    save_path: str = "modality_distribution.png",
):
    fig, axes = plt.subplots(1, len(segments), figsize=(4 * len(segments), 4))
    if len(segments) == 1:
        axes = [axes]

    for ax, seg in zip(axes, segments):
        dist = seg.get("modality_distribution", {})
        if not dist:
            continue
        mods = list(dist.keys())
        counts = list(dist.values())
        colors = {"text": "#4C72B0", "equation": "#DD8452",
                  "table": "#55A868", "diagram": "#C44E52"}
        bar_colors = [colors.get(m, "gray") for m in mods]
        ax.bar(mods, counts, color=bar_colors)
        ax.set_title(f"Segment {seg.get('segment_id', '?')}", fontsize=11)
        ax.set_ylabel("Count")

    plt.suptitle("Modality Distribution per Segment", fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_curves(
    history: List[Dict],
    save_path: str = "training_curves.png",
):
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_losses"]["total_loss"] for h in history]
    val_f1 = [h["val_metrics"].get("boundary_f1", 0) for h in history if h.get("val_metrics")]
    val_pk = [h["val_metrics"].get("pk", 0) for h in history if h.get("val_metrics")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, "b-o", markersize=3, label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    val_epochs = epochs[:len(val_f1)]
    ax2.plot(val_epochs, val_f1, "g-o", markersize=3, label="Boundary F1")
    ax2.plot(val_epochs, val_pk, "r-s", markersize=3, label="Pk (lower=better)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Validation Metrics")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
