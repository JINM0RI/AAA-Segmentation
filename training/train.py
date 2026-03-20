from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
	BATCH_SIZE,
	DEVICE,
	EPOCHS,
	IMAGE_SIZE,
	IMAGES_DIR,
	LR,
	MASKS_DIR,
	MODELS_DIR,
	NUM_WORKERS,
	PREDICTIONS_DIR,
	PLOTS_DIR,
	RANDOM_STATE,
	TRAIN_SPLIT,
)
from dataset.dataset_loader import create_datasets
from models.hybrid_model import HybridAAASegmentation
from utils.losses import DiceLoss
from utils.metrics import evaluate_segmentation, train_dice_score


def _run_one_epoch(
	model: nn.Module,
	loader: DataLoader,
	dice_loss_fn: nn.Module,
	bce_loss_fn: nn.Module,
	optimizer: optim.Optimizer | None,
	device: torch.device,
) -> tuple[float, float]:
	is_train = optimizer is not None
	model.train(is_train)

	total_loss = 0.0
	total_dice = 0.0

	loop = tqdm(loader, desc="Train" if is_train else "Val", leave=False)
	for images, masks in loop:
		images = images.to(device)
		masks = masks.to(device).float()

		if is_train:
			optimizer.zero_grad(set_to_none=True)

		with torch.set_grad_enabled(is_train):
			preds = model(images)

			# Model currently outputs probabilities; convert to logits for BCEWithLogits + DiceLoss.
			preds_logits = torch.logit(preds.clamp(1e-6, 1.0 - 1e-6))
			loss = dice_loss_fn(preds_logits, masks) + bce_loss_fn(preds_logits, masks)

			if is_train:
				loss.backward()
				optimizer.step()

		batch_dice = train_dice_score(preds.detach(), masks.detach()).item()
		total_loss += loss.item()
		total_dice += batch_dice

	avg_loss = total_loss / max(len(loader), 1)
	avg_dice = total_dice / max(len(loader), 1)
	return avg_loss, avg_dice


def _save_training_plots(history: dict[str, list[float]], output_dir: Path) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)

	epochs = range(1, len(history["train_loss"]) + 1)

	plt.figure(figsize=(8, 5))
	plt.plot(epochs, history["train_loss"], label="Train Loss")
	plt.plot(epochs, history["val_loss"], label="Val Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("Loss vs Epoch")
	plt.legend()
	plt.grid(alpha=0.2)
	plt.tight_layout()
	plt.savefig(output_dir / "loss_vs_epoch.png", dpi=200)
	plt.close()

	plt.figure(figsize=(8, 5))
	plt.plot(epochs, history["train_dice"], label="Train Dice")
	plt.plot(epochs, history["val_dice"], label="Val Dice")
	plt.xlabel("Epoch")
	plt.ylabel("Dice Score")
	plt.title("Dice Score vs Epoch")
	plt.legend()
	plt.grid(alpha=0.2)
	plt.tight_layout()
	plt.savefig(output_dir / "dice_vs_epoch.png", dpi=200)
	plt.close()


def train_model() -> dict[str, list[float]]:
	MODELS_DIR.mkdir(parents=True, exist_ok=True)
	PLOTS_DIR.mkdir(parents=True, exist_ok=True)

	train_dataset, val_dataset = create_datasets(
		images_dir=IMAGES_DIR,
		masks_dir=MASKS_DIR,
		image_size=IMAGE_SIZE,
		train_split=TRAIN_SPLIT,
		random_state=RANDOM_STATE,
		train_transform=A.Compose(
			[
				A.HorizontalFlip(p=0.5),
				A.VerticalFlip(p=0.5),
				A.RandomRotate90(p=0.5),
			]
		),
		val_transform=None,
	)

	train_loader = DataLoader(
		train_dataset,
		batch_size=BATCH_SIZE,
		shuffle=True,
		num_workers=NUM_WORKERS,
		pin_memory=torch.cuda.is_available(),
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=BATCH_SIZE,
		shuffle=False,
		num_workers=NUM_WORKERS,
		pin_memory=torch.cuda.is_available(),
	)

	model = HybridAAASegmentation(image_size=IMAGE_SIZE).to(DEVICE)
	dice_loss_fn = DiceLoss()
	bce_loss_fn = nn.BCEWithLogitsLoss()
	optimizer = optim.Adam(model.parameters(), lr=LR)

	best_val_dice = -1.0
	best_model_path = MODELS_DIR / "best_model.pth"

	history = {
		"train_loss": [],
		"val_loss": [],
		"train_dice": [],
		"val_dice": [],
	}

	print(f"Training on device: {DEVICE}")
	for epoch in range(EPOCHS):
		print(f"Epoch {epoch + 1}/{EPOCHS}")

		train_loss, train_dice = _run_one_epoch(
			model=model,
			loader=train_loader,
			dice_loss_fn=dice_loss_fn,
			bce_loss_fn=bce_loss_fn,
			optimizer=optimizer,
			device=DEVICE,
		)

		with torch.no_grad():
			val_loss, val_dice = _run_one_epoch(
				model=model,
				loader=val_loader,
				dice_loss_fn=dice_loss_fn,
				bce_loss_fn=bce_loss_fn,
				optimizer=None,
				device=DEVICE,
			)

		history["train_loss"].append(train_loss)
		history["val_loss"].append(val_loss)
		history["train_dice"].append(train_dice)
		history["val_dice"].append(val_dice)

		print(
			"  "
			f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
			f"train_dice={train_dice:.4f}, val_dice={val_dice:.4f}"
		)

		if val_dice > best_val_dice:
			best_val_dice = val_dice
			torch.save(model.state_dict(), best_model_path)
			print(f"  Saved best model to {best_model_path}")

	_save_training_plots(history, PLOTS_DIR)
	print(f"Training plots saved in: {PLOTS_DIR}")

	# Final training metric snapshot from the last completed epoch.
	final_train_loss = history["train_loss"][-1]
	final_val_loss = history["val_loss"][-1]
	final_train_dice = history["train_dice"][-1]
	final_val_dice = history["val_dice"][-1]

	print("\n## Training Summary\n")
	print(f"Train Loss: {final_train_loss:.4f}")
	print(f"Validation Loss: {final_val_loss:.4f}")
	print(f"Train Dice: {final_train_dice:.4f}")
	print(f"Validation Dice: {final_val_dice:.4f}")

	# Dataset-level evaluation based on saved prediction masks and ground truth masks.
	try:
		eval_metrics = evaluate_segmentation(pred_dir=PREDICTIONS_DIR, mask_dir=MASKS_DIR)
		print("\n## Evaluation Summary\n")
		print(f"Dice Score: {eval_metrics['dice']:.4f}")
		print(f"IoU Score: {eval_metrics['iou']:.4f}")
		print(f"Precision: {eval_metrics['precision']:.4f}")
		print(f"Recall: {eval_metrics['recall']:.4f}")
	except (FileNotFoundError, ValueError) as exc:
		print("\n## Evaluation Summary\n")
		print(f"Evaluation skipped: {exc}")

	return history

