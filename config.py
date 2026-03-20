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
BATCH_SIZE = 2
EPOCHS = 60
LR = 1e-4
TRAIN_SPLIT = 0.8
RANDOM_STATE = 42
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

