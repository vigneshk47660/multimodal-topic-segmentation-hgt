"""
Evaluation script for HLC Multimodal Topic Segmentation Pipeline.
Runs inference on test set and reports all metrics.
"""

import os
import json
import yaml
import torch
import numpy as np
from typing import Dict, List
from tqdm import tqdm

from models.pipeline import HLCMultimodalSegmentationPipeline
from data.dataloader import get_dataloader
from utils.metrics import evaluate_segmentation


def evaluate_model(
    config_path: str = "configs/default.yaml",
    checkpoint_path: str = None,
    data_path: str = "./data/synthetic_hlc",
    split: str = "test",
    output_path: str = "./outputs/evaluation_results.json",
) -> Dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pipeline = HLCMultimodalSegmentationPipeline(config=config)

    if checkpoint_path and os.path.exists(checkpoint_path):
        pipeline.load_checkpoint(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}")

    pipeline.eval()
    dataloader = get_dataloader(data_path, split, batch_size=1, shuffle=False)

    all_results = []
    all_metrics = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating on {split}"):
            for sample in batch:
                try:
                    result = pipeline(
                        prebuilt_units=sample["units"],
                        return_intermediates=True,
                    )

                    predicted = result["boundaries"]
                    gt = sample["ground_truth_boundaries"]
                    n = sample["num_units"]

                    metrics = evaluate_segmentation(
                        predicted, gt, n,
                        window_size=config["evaluation"]["window_size"],
                    )
                    all_metrics.append(metrics)

                    all_results.append({
                        "lecture_id": sample["lecture_id"],
                        "domain": sample["domain"],
                        "num_units": n,
                        "num_topics": sample["num_topics"],
                        "predicted_boundaries": predicted,
                        "ground_truth_boundaries": gt,
                        "num_predicted_segments": result["num_segments"],
                        "metrics": metrics,
                    })
                except Exception as e:
                    print(f"Error: {e}")

    # Aggregate metrics
    summary = {}
    if all_metrics:
        for key in all_metrics[0]:
            vals = [m[key] for m in all_metrics]
            summary[f"mean_{key}"] = float(np.mean(vals))
            summary[f"std_{key}"] = float(np.std(vals))

    # Per-domain breakdown
    domain_metrics = {}
    for r in all_results:
        d = r["domain"]
        if d not in domain_metrics:
            domain_metrics[d] = []
        domain_metrics[d].append(r["metrics"])

    domain_summary = {}
    for domain, metrics_list in domain_metrics.items():
        domain_summary[domain] = {}
        for key in metrics_list[0]:
            vals = [m[key] for m in metrics_list]
            domain_summary[domain][f"mean_{key}"] = float(np.mean(vals))

    output = {
        "overall_metrics": summary,
        "domain_metrics": domain_summary,
        "num_samples": len(all_results),
        "per_sample_results": all_results,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Samples: {len(all_results)}")
    for k, v in summary.items():
        if k.startswith("mean_"):
            metric_name = k.replace("mean_", "")
            std = summary.get(f"std_{metric_name}", 0)
            print(f"  {metric_name:25s}: {v:.4f} +/- {std:.4f}")

    print(f"\nResults saved to {output_path}")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="./data/synthetic_hlc")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output", type=str, default="./outputs/evaluation_results.json")
    args = parser.parse_args()

    evaluate_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        split=args.split,
        output_path=args.output,
    )
