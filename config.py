from pathlib import Path

import torch

torch.backends.cudnn.benchmark = True


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
MASKS_DIR = DATA_DIR / "masks"

OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
PLOTS_DIR = OUTPUTS_DIR / "plots"

IMAGE_SIZE = 512
BASE_CHANNELS = 24
BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-4
TRAIN_SPLIT = 0.8
RANDOM_STATE = 42
NUM_WORKERS = 2

# Inference/post-processing defaults. Tune threshold for precision-recall trade-off.
PRED_THRESHOLD = 0.35
POSTPROCESS_KERNEL_SIZE = (5, 5)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

