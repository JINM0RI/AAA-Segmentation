from pathlib import Path

import cv2
import numpy as np
import torch


def load_grayscale_image(image_path: str | Path, image_size: int = 256) -> np.ndarray:
	image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
	if image is None:
		raise ValueError(f"Unable to load image: {image_path}")

	image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
	image = image.astype(np.float32) / 255.0
	return image


def load_binary_mask(mask_path: str | Path, image_size: int = 256) -> np.ndarray:
	mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
	if mask is None:
		raise ValueError(f"Unable to load mask: {mask_path}")

	mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
	mask = (mask > 127).astype(np.float32)
	return mask


def postprocess_mask(
	mask: np.ndarray | torch.Tensor,
	threshold: float = 0.5,
	kernel_size: int | tuple[int, int] = (3, 3),
) -> np.ndarray:
	"""Convert prediction to a clean binary mask using threshold + morphology.

	Supports mask shapes (H, W) and (1, H, W), with numpy arrays or torch tensors.
	Returns uint8 mask with values 0 or 255 for direct image saving.
	"""
	# Convert tensor input to numpy for OpenCV operations.
	if isinstance(mask, torch.Tensor):
		mask_np = mask.detach().cpu().numpy()
	else:
		mask_np = np.asarray(mask)

	# Normalize supported shapes to (H, W).
	if mask_np.ndim == 3 and mask_np.shape[0] == 1:
		mask_np = mask_np[0]
	elif mask_np.ndim == 3 and mask_np.shape[-1] == 1:
		mask_np = mask_np[..., 0]

	if mask_np.ndim != 2:
		raise ValueError(f"Unsupported mask shape for post-processing: {mask_np.shape}")

	# Keep this utility for backward compatibility with existing calls.
	return refine_segmentation(mask_np, threshold=threshold, kernel_size=kernel_size)


def _fill_internal_holes(binary_mask: np.ndarray) -> np.ndarray:
	"""Fill holes inside the foreground region using flood-fill from the border."""
	height, width = binary_mask.shape
	flood_input = (binary_mask * 255).astype(np.uint8)
	mask = np.zeros((height + 2, width + 2), dtype=np.uint8)

	# Flood fill background from top-left corner, then invert to identify enclosed holes.
	flood_filled = flood_input.copy()
	cv2.floodFill(flood_filled, mask, (0, 0), 255)
	holes = cv2.bitwise_not(flood_filled)
	filled = cv2.bitwise_or(flood_input, holes)
	return (filled > 0).astype(np.uint8)


def refine_segmentation(
	mask: np.ndarray | torch.Tensor,
	threshold: float = 0.35,
	kernel_size: tuple[int, int] | int = (5, 5),
) -> np.ndarray:
	"""Refine segmentation masks for anatomically plausible AAA predictions.

	Pipeline:
	1) probability thresholding,
	2) morphology (open then close),
	3) largest connected component,
	4) edge smoothing,
	5) hole filling.

	Input shapes supported: (H, W), (1, H, W), (H, W, 1)
	Returns uint8 mask with values {0, 255}.
	"""
	# Convert tensor predictions to CPU numpy arrays for OpenCV operations.
	if isinstance(mask, torch.Tensor):
		mask_np = mask.detach().cpu().numpy()
	else:
		mask_np = np.asarray(mask)

	# Normalize mask shape to 2D for connected component and morphology analysis.
	if mask_np.ndim == 3 and mask_np.shape[0] == 1:
		mask_np = mask_np[0]
	elif mask_np.ndim == 3 and mask_np.shape[-1] == 1:
		mask_np = mask_np[..., 0]

	if mask_np.ndim != 2:
		raise ValueError(f"Unsupported mask shape for refinement: {mask_np.shape}")

	if isinstance(kernel_size, int):
		kernel_size = (kernel_size, kernel_size)

	# Step A: Lower threshold can recover weak but clinically relevant edges.
	binary_mask = (mask_np > threshold).astype(np.uint8)

	# Step B: Opening removes tiny speckle artifacts; closing repairs discontinuities.
	kernel = np.ones(kernel_size, dtype=np.uint8)
	opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
	morphed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

	# Step C: Keep only the largest connected region to suppress fragmented blobs.
	num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(morphed, connectivity=8)
	if num_labels > 1:
		largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
		largest_component = (labels == largest_label).astype(np.uint8)
	else:
		largest_component = morphed

	# Step D: Slight blur + re-threshold smooths jagged contour boundaries.
	blurred = cv2.GaussianBlur(largest_component.astype(np.float32), (5, 5), 0)
	smoothed = (blurred > 0.5).astype(np.uint8)

	# Step E: Fill enclosed holes to enforce a contiguous aneurysm lumen region.
	filled = _fill_internal_holes(smoothed)

	return (filled * 255).astype(np.uint8)

