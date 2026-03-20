# AAA-Segmentation

Hybrid Residual Attention U-Net + Swin Transformer for Abdominal Aortic Aneurysm (AAA) segmentation from 2D CT slices.

## Project Overview

This repository provides a modular and production-style deep learning pipeline for medical image segmentation:

- Dataset loading and train/validation split
- Hybrid segmentation model (Residual U-Net + Attention + Swin Transformer)
- Training loop with BCE + Dice loss
- Validation and model checkpointing
- Inference on unseen CT slices
- Metric utilities and visualization helpers

## Repository Structure

```text
AAA-Segmentation/
├── data/
│   ├── images/
│   └── masks/
├── dataset/
│   └── dataset_loader.py
├── models/
│   ├── attention.py
│   ├── hybrid_model.py
│   ├── residual_unet.py
│   └── swin_encoder.py
├── utils/
│   ├── metrics.py
│   ├── preprocessing.py
│   └── visualization.py
├── training/
│   └── train.py
├── inference/
│   └── predict.py
├── outputs/
│   ├── models/
│   ├── predictions/
│   └── plots/
├── config.py
├── requirements.txt
├── README.md
└── main.py
```

## Dataset Format

Place your paired CT slices and masks using matching file names:

```text
data/
  images/
    case_001.png
    case_002.jpg
  masks/
    case_001.png
    case_002.jpg
```

Supported image extensions: `.png`, `.jpg`, `.jpeg`

## Suggested Dataset Sources

Possible public sources for AAA and abdominal vessel segmentation studies:

- The Cancer Imaging Archive (TCIA)
- Medical Segmentation Decathlon style abdominal datasets
- Institution-specific anonymized datasets (if available and ethically approved)

Always follow patient privacy, licensing, and ethics requirements before use.

## Model Architecture

The `HybridAAASegmentation` model combines:

1. Residual U-Net encoder/decoder blocks
2. Attention gates for skip-connection filtering
3. Swin Transformer feature extraction using `timm`
4. Feature fusion in bottleneck and skip stages
5. Sigmoid output for binary mask prediction

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Training

```bash
python main.py --train
```

Outputs:

- Best model: `outputs/models/best_model.pth`
- Curves: `outputs/plots/loss_vs_epoch.png`, `outputs/plots/dice_vs_epoch.png`

## Inference

Predict for one image:

```bash
python main.py --predict --input data/images/case_001.png --model-path outputs/models/best_model.pth
```

Predict for a folder:

```bash
python main.py --predict --input data/images --model-path outputs/models/best_model.pth
```

Predictions are saved to:

- `outputs/predictions/`

## Configurable Hyperparameters

Defined in `config.py`:

- `IMAGE_SIZE = 256`
- `BATCH_SIZE = 4`
- `EPOCHS = 20`
- `LR = 0.0001`
- `DEVICE = cuda if available else cpu`

## Notes

- GPU training is enabled automatically when CUDA is available.
- Code follows a modular structure to support research experimentation.
- You can extend the dataset loader with augmentations for better generalization.
