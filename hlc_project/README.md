# Multimodal Topic Segmentation in Lecture Videos Using Heterogeneous Graph Transformers

> **Paper**: *Multimodal Topic Segmentation in Lecture Videos Using Heterogeneous Graph Transformers*  
> **Authors**: Vignesh. K, S.R. Balasundaram  
> **Affiliation**: Department of Computer Applications, National Institute of Technology, Tiruchirappalli  


This repository contains the official implementation of the paper. If you use this code, please cite our manuscript (see [Citation](#citation)).

---

## Architecture Overview

The pipeline consists of seven stages:

| Stage | Component | Paper Reference |
|-------|-----------|-----------------|
| 1 | Instructional Unit Builder | Eq. 1–2 |
| 2 | HLC Normalization & Modality ID | Eq. 3 |
| 3 | Modality-Specific Encoding (SBERT, MathBERT, TAPAS, ViT) | Eq. 4–5 |
| 4 | Cross-Modal Projection | Eq. 6 |
| 5 | Heterogeneous Graph Transformer (HGT) | Eq. 7–8 |
| 6 | Similarity Profiling & Neural Change-Point Detection | Eq. 9–11 |
| 7 | FAISS-Assisted Semantic Grouping | Eq. 12–14, Algorithm 2 |

## Project Structure

```
hlc_project/
├── configs/
│   └── default.yaml              # Full configuration
├── models/
│   ├── instructional_unit_builder.py  # Stage 1
│   ├── hlc_normalizer.py             # Stage 2
│   ├── modality_encoders.py          # Stage 3
│   ├── cross_modal_projection.py     # Stage 4
│   ├── hgt_fusion.py                 # Stage 5
│   ├── change_point_detection.py     # Stage 6
│   ├── segment_formation.py          # Stage 7
│   └── pipeline.py                   # Unified pipeline
├── data/
│   ├── synthetic_dataset.py      # Synthetic HLC generator
│   └── dataloader.py             # PyTorch data loading
├── scripts/
│   ├── train.py                  # Training with losses
│   └── evaluate.py               # Evaluation metrics
├── utils/
│   ├── metrics.py                # Pk, WindowDiff, F1
│   └── visualization.py         # Plots and charts
├── run.py                        # Main CLI entry point
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/<your-repo>/hlc-multimodal-segmentation.git
cd hlc-multimodal-segmentation
pip install -r requirements.txt
```

### Requirements
- Python >= 3.9
- PyTorch >= 2.0
- CUDA (recommended for training)

## Quick Start

### 1. Generate Synthetic Dataset
```bash
python run.py generate --num_lectures 100 --output_dir ./data/synthetic_hlc
```

### 2. Train
```bash
python run.py train --config configs/default.yaml --data_path ./data/synthetic_hlc
```

### 3. Evaluate
```bash
python run.py evaluate --config configs/default.yaml \
    --checkpoint ./checkpoints/best_model.pt \
    --data_path ./data/synthetic_hlc \
    --split test
```

### 4. Inference on a Single Transcript
```bash
python run.py infer --input transcript.txt --output results.json
```

## Pretrained Encoders

| Modality | Model | Source |
|----------|-------|--------|
| Text | SBERT (all-MiniLM-L6-v2) | sentence-transformers |
| Equations | MathBERT (tbs17/MathBERT) | HuggingFace |
| Tables | TAPAS (google/tapas-base) | HuggingFace |
| Diagrams | ViT (google/vit-base-patch16-224) | HuggingFace |

## Evaluation Metrics

- **Pk** (lower is better): Probabilistic boundary error
- **WindowDiff** (lower is better): Window-based segmentation error
- **Boundary F1** (higher is better): Precision/Recall of boundary detection

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{vignesh2025multimodal,
  title={Multimodal Topic Segmentation in Lecture Videos Using Heterogeneous Graph Transformers},
  author={Vignesh, K and Balasundaram, S.R.},
  journal={The Visual Computer},
  year={2025}
}
```

## License

This project is released for academic research purposes.
