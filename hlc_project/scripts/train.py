"""
Training script for HLC Multimodal Topic Segmentation Pipeline.
Handles end-to-end training with boundary detection loss.
"""

import os
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm

from models.pipeline import HLCMultimodalSegmentationPipeline
from data.dataloader import get_dataloader
from utils.metrics import evaluate_segmentation


class BoundaryDetectionLoss(nn.Module):
    """
    Combined loss for topic boundary detection.
    L = L_boundary + alpha * L_contrastive + beta * L_similarity
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.3, margin: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def boundary_loss(
        self, boundary_scores: torch.Tensor, gt_boundaries: List[int], n: int
    ) -> torch.Tensor:
        """Binary cross-entropy loss for boundary prediction."""
        targets = torch.zeros(n - 1, device=boundary_scores.device)
        for b in gt_boundaries:
            if 0 <= b < n - 1:
                targets[b] = 1.0

        if boundary_scores.shape[0] != targets.shape[0]:
            min_len = min(boundary_scores.shape[0], targets.shape[0])
            boundary_scores = boundary_scores[:min_len]
            targets = targets[:min_len]

        return self.bce(boundary_scores, targets)

    def contrastive_loss(
        self,
        fused_representations: torch.Tensor,
        gt_boundaries: List[int],
        n: int,
    ) -> torch.Tensor:
        """
        Contrastive loss: same-segment units should be close,
        cross-segment units should be far apart.
        """
        if len(gt_boundaries) == 0:
            return torch.tensor(0.0, device=fused_representations.device)

        segment_labels = torch.zeros(n, dtype=torch.long, device=fused_representations.device)
        seg_id = 0
        prev = 0
        for b in sorted(gt_boundaries):
            segment_labels[prev:b] = seg_id
            prev = b
            seg_id += 1
        segment_labels[prev:] = seg_id

        # Sample pairs
        num_pairs = min(n * 2, 100)
        indices = torch.randint(0, n, (num_pairs, 2), device=fused_representations.device)

        emb_i = fused_representations[indices[:, 0]]
        emb_j = fused_representations[indices[:, 1]]
        same_segment = (segment_labels[indices[:, 0]] == segment_labels[indices[:, 1]]).float()

        dist = F.pairwise_distance(emb_i, emb_j)
        pos_loss = same_segment * dist.pow(2)
        neg_loss = (1 - same_segment) * F.relu(self.margin - dist).pow(2)

        return (pos_loss + neg_loss).mean()

    def similarity_consistency_loss(
        self, similarity_sequence: torch.Tensor, gt_boundaries: List[int], n: int
    ) -> torch.Tensor:
        """
        Encourage low similarity at boundaries and high similarity within segments.
        """
        targets = torch.ones_like(similarity_sequence) * 0.8
        for b in gt_boundaries:
            if 0 <= b < len(targets):
                targets[b] = 0.2
                if b > 0:
                    targets[b - 1] = min(targets[b - 1].item(), 0.5)
                if b + 1 < len(targets):
                    targets[b + 1] = min(targets[b + 1].item(), 0.5)

        return F.mse_loss(similarity_sequence, targets)

    def forward(
        self,
        boundary_scores: torch.Tensor,
        fused_representations: torch.Tensor,
        similarity_sequence: torch.Tensor,
        gt_boundaries: List[int],
        n: int,
    ) -> Dict[str, torch.Tensor]:
        l_boundary = self.boundary_loss(boundary_scores, gt_boundaries, n)
        l_contrastive = self.contrastive_loss(fused_representations, gt_boundaries, n)
        l_similarity = self.similarity_consistency_loss(
            similarity_sequence, gt_boundaries, n
        )

        total = l_boundary + self.alpha * l_contrastive + self.beta * l_similarity

        return {
            "total_loss": total,
            "boundary_loss": l_boundary,
            "contrastive_loss": l_contrastive,
            "similarity_loss": l_similarity,
        }


class Trainer:
    """Training loop for the HLC pipeline."""

    def __init__(self, config_path: str = "configs/default.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = HLCMultimodalSegmentationPipeline(
            config=self.config
        ).to(self.device)

        self.loss_fn = BoundaryDetectionLoss()

        # Only optimize trainable components (Stages 4-6)
        trainable_params = list(self.pipeline.cross_modal_projection.parameters()) + \
                          list(self.pipeline.hgt.parameters()) + \
                          list(self.pipeline.change_point_detector.parameters())

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        os.makedirs(self.config["paths"]["checkpoint_dir"], exist_ok=True)
        os.makedirs(self.config["paths"]["log_dir"], exist_ok=True)

    def train_epoch(self, dataloader) -> Dict[str, float]:
        self.pipeline.train()
        epoch_losses = {"total_loss": 0, "boundary_loss": 0,
                       "contrastive_loss": 0, "similarity_loss": 0}
        count = 0

        for batch in tqdm(dataloader, desc="Training"):
            for sample in batch:
                self.optimizer.zero_grad()
                try:
                    result = self.pipeline(
                        prebuilt_units=sample["units"],
                        return_intermediates=True,
                    )

                    intermediates = result["intermediates"]
                    fused = intermediates["fused_representations"]
                    sim_seq = intermediates.get("similarity_sequence")
                    boundary_scores = intermediates.get("boundary_scores")

                    if sim_seq is None or boundary_scores is None:
                        continue

                    if isinstance(boundary_scores, torch.Tensor):
                        scores_for_loss = boundary_scores
                    else:
                        scores_for_loss = sim_seq

                    losses = self.loss_fn(
                        boundary_scores=scores_for_loss,
                        fused_representations=fused,
                        similarity_sequence=sim_seq,
                        gt_boundaries=sample["ground_truth_boundaries"],
                        n=sample["num_units"],
                    )

                    losses["total_loss"].backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.pipeline.parameters(),
                        self.config["training"]["gradient_clip"],
                    )
                    self.optimizer.step()

                    for k in epoch_losses:
                        epoch_losses[k] += losses[k].item()
                    count += 1

                except Exception as e:
                    print(f"Error processing sample: {e}")
                    continue

        if count > 0:
            for k in epoch_losses:
                epoch_losses[k] /= count
        return epoch_losses

    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict[str, float]:
        self.pipeline.eval()
        all_metrics = []

        for batch in tqdm(dataloader, desc="Evaluating"):
            for sample in batch:
                try:
                    result = self.pipeline(prebuilt_units=sample["units"])
                    predicted_boundaries = result["boundaries"]
                    gt_boundaries = sample["ground_truth_boundaries"]
                    n = sample["num_units"]

                    metrics = evaluate_segmentation(
                        predicted_boundaries, gt_boundaries, n,
                        window_size=self.config["evaluation"]["window_size"],
                    )
                    all_metrics.append(metrics)
                except Exception as e:
                    print(f"Error evaluating sample: {e}")
                    continue

        if not all_metrics:
            return {}

        avg_metrics = {}
        for key in all_metrics[0]:
            vals = [m[key] for m in all_metrics]
            avg_metrics[key] = float(np.mean(vals))
            avg_metrics[f"{key}_std"] = float(np.std(vals))

        return avg_metrics

    def train(self, data_path: str):
        train_loader = get_dataloader(
            data_path, "train",
            batch_size=self.config["training"]["batch_size"],
        )
        val_loader = get_dataloader(
            data_path, "val",
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
        )

        best_val_f1 = 0
        patience_counter = 0
        history = []

        for epoch in range(self.config["training"]["epochs"]):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.config['training']['epochs']}")

            train_losses = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            self.scheduler.step()

            record = {
                "epoch": epoch + 1,
                "train_losses": train_losses,
                "val_metrics": val_metrics,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            history.append(record)

            print(f"  Train Loss: {train_losses['total_loss']:.4f}")
            if val_metrics:
                print(f"  Val Pk: {val_metrics.get('pk', 0):.4f} | "
                      f"Val WD: {val_metrics.get('windowdiff', 0):.4f} | "
                      f"Val F1: {val_metrics.get('boundary_f1', 0):.4f}")

                val_f1 = val_metrics.get("boundary_f1", 0)
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    self.pipeline.save_checkpoint(
                        os.path.join(self.config["paths"]["checkpoint_dir"], "best_model.pt")
                    )
                    print(f"  >> New best model (F1={best_val_f1:.4f})")
                else:
                    patience_counter += 1

                if patience_counter >= self.config["training"]["patience"]:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

        # Save training history
        with open(os.path.join(self.config["paths"]["log_dir"], "history.json"), "w") as f:
            json.dump(history, f, indent=2, default=str)

        return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_path", type=str, default="./data/synthetic_hlc")
    args = parser.parse_args()

    trainer = Trainer(config_path=args.config)
    trainer.train(data_path=args.data_path)
