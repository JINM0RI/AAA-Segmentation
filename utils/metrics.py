from pathlib import Path

import cv2
import numpy as np
import torch


def _to_binary_flat(arr: np.ndarray | torch.Tensor) -> np.ndarray:
	"""Convert mask values to flat binary {0,1} numpy array."""
	if isinstance(arr, torch.Tensor):
		arr = arr.detach().cpu().numpy()
	arr = np.asarray(arr)

	if arr.ndim == 3 and arr.shape[0] == 1:
		arr = arr[0]
	elif arr.ndim == 3 and arr.shape[-1] == 1:
		arr = arr[..., 0]

	binary = (arr > 0).astype(np.uint8)
	return binary.reshape(-1)


def _confusion(pred: np.ndarray | torch.Tensor, mask: np.ndarray | torch.Tensor) -> tuple[float, float, float]:
	p = _to_binary_flat(pred)
	m = _to_binary_flat(mask)

	tp = float(np.logical_and(p == 1, m == 1).sum())
	fp = float(np.logical_and(p == 1, m == 0).sum())
	fn = float(np.logical_and(p == 0, m == 1).sum())
	return tp, fp, fn


def dice_score(pred: np.ndarray | torch.Tensor, mask: np.ndarray | torch.Tensor, eps: float = 1e-7) -> float:
	"""Dice = (2*TP)/(2*TP+FP+FN)."""
	tp, fp, fn = _confusion(pred, mask)
	return float((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps))


def iou_score(pred: np.ndarray | torch.Tensor, mask: np.ndarray | torch.Tensor, eps: float = 1e-7) -> float:
	"""IoU = TP/(TP+FP+FN)."""
	tp, fp, fn = _confusion(pred, mask)
	return float((tp + eps) / (tp + fp + fn + eps))


def precision_score(pred: np.ndarray | torch.Tensor, mask: np.ndarray | torch.Tensor, eps: float = 1e-7) -> float:
	"""Precision = TP/(TP+FP)."""
	tp, fp, _ = _confusion(pred, mask)
	return float((tp + eps) / (tp + fp + eps))


def recall_score(pred: np.ndarray | torch.Tensor, mask: np.ndarray | torch.Tensor, eps: float = 1e-7) -> float:
	"""Recall = TP/(TP+FN)."""
	tp, _, fn = _confusion(pred, mask)
	return float((tp + eps) / (tp + fn + eps))


def train_dice_score(
	preds: torch.Tensor,
	targets: torch.Tensor,
	threshold: float = 0.5,
	eps: float = 1e-7,
) -> torch.Tensor:
	"""Torch Dice helper for batch-wise training loop logging."""
	preds_bin = (preds > threshold).float()
	targets = targets.float()

	intersection = (preds_bin * targets).sum(dim=(1, 2, 3))
	denominator = preds_bin.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
	dice = (2.0 * intersection + eps) / (denominator + eps)
	return dice.mean()


def evaluate_segmentation(pred_dir: str | Path, mask_dir: str | Path) -> dict[str, float]:
	"""Evaluate dataset-level Dice/IoU/Precision/Recall using saved mask images."""
	pred_dir = Path(pred_dir)
	mask_dir = Path(mask_dir)

	if not pred_dir.exists():
		raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
	if not mask_dir.exists():
		raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

	pred_files = [p for p in pred_dir.iterdir() if p.is_file()]
	if not pred_files:
		raise ValueError(f"No prediction files found in: {pred_dir}")

	mask_lookup = {p.stem: p for p in mask_dir.iterdir() if p.is_file()}

	all_dice: list[float] = []
	all_iou: list[float] = []
	all_precision: list[float] = []
	all_recall: list[float] = []

	for pred_path in sorted(pred_files):
		pred_stem = pred_path.stem
		# Support predicted names like case001_pred.png -> case001.png.
		mask_stem = pred_stem[:-5] if pred_stem.endswith("_pred") else pred_stem
		mask_path = mask_lookup.get(mask_stem)
		if mask_path is None:
			continue

		pred_img = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
		mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
		if pred_img is None or mask_img is None:
			continue

		pred_bin = (pred_img > 127).astype(np.uint8)
		mask_bin = (mask_img > 127).astype(np.uint8)

		all_dice.append(dice_score(pred_bin, mask_bin))
		all_iou.append(iou_score(pred_bin, mask_bin))
		all_precision.append(precision_score(pred_bin, mask_bin))
		all_recall.append(recall_score(pred_bin, mask_bin))

	if not all_dice:
		raise ValueError("No matched prediction/mask pairs were found for evaluation.")

	return {
		"dice": float(np.mean(all_dice)),
		"iou": float(np.mean(all_iou)),
		"precision": float(np.mean(all_precision)),
		"recall": float(np.mean(all_recall)),
	}

