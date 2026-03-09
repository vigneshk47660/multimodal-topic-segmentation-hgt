"""
Main entry point for HLC Multimodal Topic Segmentation Pipeline.
Supports: generate_data, train, evaluate, infer (single transcript)
"""

import argparse
import json
import os
import sys
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def generate_data(args):
    from data.synthetic_dataset import generate_synthetic_dataset
    stats = generate_synthetic_dataset(
        output_dir=args.output_dir,
        num_lectures=args.num_lectures,
        seed=args.seed,
    )
    print(json.dumps(stats, indent=2))


def train(args):
    from scripts.train import Trainer
    trainer = Trainer(config_path=args.config)
    trainer.train(data_path=args.data_path)


def evaluate(args):
    from scripts.evaluate import evaluate_model
    evaluate_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        split=args.split,
        output_path=args.output,
    )


def infer(args):
    """Run inference on a single transcript."""
    from models.pipeline import HLCMultimodalSegmentationPipeline
    from models.segment_formation import SegmentFormation

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    pipeline = HLCMultimodalSegmentationPipeline(config=config)
    if args.checkpoint and os.path.exists(args.checkpoint):
        pipeline.load_checkpoint(args.checkpoint)
    pipeline.eval()

    # Load transcript
    if args.input.endswith(".json"):
        with open(args.input, "r") as f:
            data = json.load(f)
        if isinstance(data, dict) and "transcript" in data:
            transcript = data["transcript"]
        else:
            transcript = None
            from models.instructional_unit_builder import InstructionalUnitBuilder
            builder = InstructionalUnitBuilder()
            units = builder.build_from_file(args.input)
    else:
        with open(args.input, "r") as f:
            transcript = f.read()

    with torch.no_grad():
        if transcript:
            result = pipeline(transcript=transcript, return_intermediates=True)
        else:
            result = pipeline(prebuilt_units=units, return_intermediates=True)

    segments = result["segments"]
    boundaries = result["boundaries"]

    print(f"\n{'='*60}")
    print("TOPIC SEGMENTATION RESULTS")
    print(f"{'='*60}")
    print(f"Total Units: {result['num_units']}")
    print(f"Detected Boundaries: {boundaries}")
    print(f"Number of Segments: {result['num_segments']}")

    for seg in segments:
        print(f"\n--- Segment {seg.segment_id} ---")
        print(f"  Units: {seg.unit_indices[0]}..{seg.unit_indices[-1]}")
        print(f"  Modalities: {seg.modality_distribution}")
        print(f"  Preview: {seg.units[0].content[:100]}...")

    # Save results
    if args.output:
        output = {
            "boundaries": boundaries,
            "num_segments": result["num_segments"],
            "segments": SegmentFormation.segments_to_dict(segments),
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="HLC Multimodal Topic Segmentation Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate synthetic data
    gen = subparsers.add_parser("generate", help="Generate synthetic HLC dataset")
    gen.add_argument("--output_dir", type=str, default="./data/synthetic_hlc")
    gen.add_argument("--num_lectures", type=int, default=100)
    gen.add_argument("--seed", type=int, default=42)

    # Train
    tr = subparsers.add_parser("train", help="Train the pipeline")
    tr.add_argument("--config", type=str, default="configs/default.yaml")
    tr.add_argument("--data_path", type=str, default="./data/synthetic_hlc")

    # Evaluate
    ev = subparsers.add_parser("evaluate", help="Evaluate on test set")
    ev.add_argument("--config", type=str, default="configs/default.yaml")
    ev.add_argument("--checkpoint", type=str, default=None)
    ev.add_argument("--data_path", type=str, default="./data/synthetic_hlc")
    ev.add_argument("--split", type=str, default="test")
    ev.add_argument("--output", type=str, default="./outputs/evaluation_results.json")

    # Infer
    inf = subparsers.add_parser("infer", help="Run inference on a transcript")
    inf.add_argument("--config", type=str, default="configs/default.yaml")
    inf.add_argument("--checkpoint", type=str, default=None)
    inf.add_argument("--input", type=str, required=True, help="Path to transcript file")
    inf.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    if args.command == "generate":
        generate_data(args)
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "infer":
        infer(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
