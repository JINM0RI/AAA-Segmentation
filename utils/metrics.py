from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def _align_mask_to_prediction_shape(mask_img: np.ndarray, pred_img: np.ndarray) -> np.ndarray:
	"""Resize mask to prediction size when dimensions differ."""
	if mask_img.shape[:2] == pred_img.shape[:2]:
		return mask_img

	height, width = pred_img.shape[:2]
	return cv2.resize(mask_img, (width, height), interpolation=cv2.INTER_NEAREST)


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

		mask_img = _align_mask_to_prediction_shape(mask_img, pred_img)

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


def plot_mask_comparison_accuracy(
	pred_dir: str | Path,
	mask_dir: str | Path,
	output_path: str | Path,
	max_items: int = 30,
) -> dict[str, float | str]:
	"""Create a comparison graph of Ground Truth vs Predicted mask foreground ratio."""
	pred_dir = Path(pred_dir)
	mask_dir = Path(mask_dir)
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	if not pred_dir.exists():
		raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
	if not mask_dir.exists():
		raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

	pred_files = [p for p in pred_dir.iterdir() if p.is_file()]
	if not pred_files:
		raise ValueError(f"No prediction files found in: {pred_dir}")

	mask_lookup = {p.stem: p for p in mask_dir.iterdir() if p.is_file()}

	names: list[str] = []
	gt_vals: list[float] = []
	pred_vals: list[float] = []

	for pred_path in sorted(pred_files):
		pred_stem = pred_path.stem
		mask_stem = pred_stem[:-5] if pred_stem.endswith("_pred") else pred_stem
		mask_path = mask_lookup.get(mask_stem)
		if mask_path is None:
			continue

		pred_img = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
		mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
		if pred_img is None or mask_img is None:
			continue

		mask_img = _align_mask_to_prediction_shape(mask_img, pred_img)

		pred_bin = (pred_img > 127).astype(np.uint8)
		mask_bin = (mask_img > 127).astype(np.uint8)

		names.append(mask_stem)
		gt_vals.append(float(mask_bin.mean()))
		pred_vals.append(float(pred_bin.mean()))

	if not gt_vals:
		raise ValueError("No matched prediction/mask pairs were found for plotting.")

	if len(names) > max_items:
		names = names[:max_items]
		gt_vals = gt_vals[:max_items]
		pred_vals = pred_vals[:max_items]

	x = np.arange(len(names))
	width = 0.38
	mean_gt = float(np.mean(gt_vals))
	mean_pred = float(np.mean(pred_vals))
	max_ratio = max(gt_vals + pred_vals)
	# Foreground ratios are often very small in medical masks; zoom y-axis for visibility.
	y_upper = min(1.0, max(0.01, max_ratio * 1.35))

	plt.figure(figsize=(max(10, len(names) * 0.45), 6))
	gt_bars = plt.bar(x - width / 2, gt_vals, width=width, label="Ground Truth Mask", color="#2a9d8f")
	pred_bars = plt.bar(x + width / 2, pred_vals, width=width, label="Predicted Mask", color="#e76f51")
	plt.ylim(0.0, y_upper)
	plt.xticks(x, names, rotation=60, ha="right")
	plt.xlabel("Image")
	plt.ylabel("Foreground Pixel Ratio")
	plt.title("Ground Truth Mask vs Predicted Mask")
	plt.grid(axis="y", alpha=0.25)
	plt.legend()

	# Label bars so tiny differences remain readable even when values are close.
	for bar in list(gt_bars) + list(pred_bars):
		height = bar.get_height()
		if height > 0:
			plt.text(
				bar.get_x() + bar.get_width() / 2,
				height + y_upper * 0.01,
				f"{height:.4f}",
				ha="center",
				va="bottom",
				fontsize=7,
				rotation=90,
			)

	plt.tight_layout()
	plt.savefig(output_path, dpi=220)
	plt.close()

	return {
		"mean_ground_truth": mean_gt,
		"mean_predicted": mean_pred,
		"num_images": float(len(names)),
		"plot_path": str(output_path),
	}

