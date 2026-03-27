from pathlib import Path
import os

import albumentations as A
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BASE_CHANNELS,
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
from utils.metrics import evaluate_segmentation, train_dice_score


def _dice_loss_from_probs(probs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-7) -> torch.Tensor:
    targets = targets.float()
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice_coeff = (2.0 * intersection + smooth) / (denominator + smooth)
    return (1.0 - dice_coeff).mean()


def _weighted_bce_from_probs(probs: torch.Tensor, targets: torch.Tensor, pos_weight: float) -> torch.Tensor:
    """Apply stronger penalties to foreground mistakes for highly imbalanced masks."""
    weights = targets * pos_weight + (1.0 - targets)
    return F.binary_cross_entropy(probs, targets, weight=weights)


def _estimate_foreground_ratio(loader: DataLoader, max_batches: int = 60) -> float:
    """Estimate dataset foreground ratio from training masks for class-imbalance weighting."""
    fg_pixels = 0.0
    total_pixels = 0.0

    for batch_idx, (_, masks) in enumerate(loader):
        masks = masks.float()
        fg_pixels += float(masks.sum().item())
        total_pixels += float(masks.numel())
        if batch_idx + 1 >= max_batches:
            break

    if total_pixels <= 0:
        return 0.01
    return max(fg_pixels / total_pixels, 1e-6)


def _run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    pos_weight: float,
    optimizer: optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_dice = 0.0

    loop = tqdm(loader, desc="Train" if is_train else "Val", leave=False)
    for images, masks in loop:
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        masks = masks.to(device, non_blocking=True).float()

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                preds = model(images)

            # BCELoss is not autocast-safe; compute losses in float32.
            preds = preds.float().clamp(1e-6, 1.0 - 1e-6)
            masks_f = masks.float()
            dice_loss = _dice_loss_from_probs(preds, masks_f)
            bce_loss = _weighted_bce_from_probs(preds, masks_f, pos_weight=pos_weight)
            loss = dice_loss + bce_loss

            if not torch.isfinite(loss):
                return float("nan"), float("nan")

            if is_train:
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(0.05, 0.05),
                    rotate=(-20, 20),
                    p=0.4,
                ),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
            ]
        ),
        val_transform=None,
    )

    loader_kwargs = {
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **loader_kwargs,
    )

    if DEVICE.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    model = HybridAAASegmentation(image_size=IMAGE_SIZE, base_channels=BASE_CHANNELS).to(DEVICE)
    if DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    if DEVICE.type == "cuda" and os.getenv("ENABLE_TORCH_COMPILE", "0") == "1" and hasattr(torch, "compile"):
        try:
            import triton  # type: ignore  # noqa: F401
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as exc:
            print(f"torch.compile disabled: {exc}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LR * 0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    foreground_ratio = _estimate_foreground_ratio(train_loader)
    # Keep weights in a stable range; very large values can destabilize training.
    pos_weight = float(min(max((1.0 - foreground_ratio) / foreground_ratio, 1.0), 80.0))
    print(f"Estimated foreground ratio: {foreground_ratio:.6f}, positive class weight: {pos_weight:.2f}")

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
            pos_weight=pos_weight,
            optimizer=optimizer,
            scaler=scaler,
            device=DEVICE,
        )

        with torch.no_grad():
            val_loss, val_dice = _run_one_epoch(
                model=model,
                loader=val_loader,
                pos_weight=pos_weight,
                optimizer=None,
                scaler=scaler,
                device=DEVICE,
            )

        if not (torch.isfinite(torch.tensor(train_loss)) and torch.isfinite(torch.tensor(val_loss))):
            print("  Stopping early: non-finite loss detected (NaN/Inf).")
            break

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)

        print(
            "  "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_dice={train_dice:.4f}, val_dice={val_dice:.4f}, "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            print(f"  Saved best model to {best_model_path}")

    if not history["train_loss"]:
        raise RuntimeError("Training stopped before completing a valid epoch due to non-finite loss.")

    _save_training_plots(history, PLOTS_DIR)
    print(f"Training plots saved in: {PLOTS_DIR}")

    final_train_loss = history["train_loss"][-1]
    final_val_loss = history["val_loss"][-1]
    final_train_dice = history["train_dice"][-1]
    final_val_dice = history["val_dice"][-1]

    print("\n## Training Summary\n")
    print(f"Train Loss: {final_train_loss:.4f}")
    print(f"Validation Loss: {final_val_loss:.4f}")
    print(f"Train Dice: {final_train_dice:.4f}")
    print(f"Validation Dice: {final_val_dice:.4f}")

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
