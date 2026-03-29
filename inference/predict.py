from pathlib import Path

import cv2
import numpy as np
import torch

from config import (
	BASE_CHANNELS,
	DEVICE,
	IMAGE_SIZE,
	MODELS_DIR,
	POSTPROCESS_KERNEL_SIZE,
	PREDICTIONS_DIR,
	PRED_THRESHOLD,
)
from models.hybrid_model import HybridAAASegmentation
from utils.preprocessing import refine_segmentation


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _load_input_image(image_path: Path, image_size: int) -> torch.Tensor:
	image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
	if image is None:
		raise ValueError(f"Could not load image: {image_path}")

	image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
	image = image.astype(np.float32) / 255.0
	tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
	return tensor


def _save_visualizations(
	image_tensor: torch.Tensor,
	binary_mask: np.ndarray,
	base_name: str,
	vis_dir: Path,
) -> None:
	"""Optionally save original CT slice, binary mask, and red overlay preview."""
	original_dir = vis_dir / "original"
	mask_dir = vis_dir / "mask"
	mask_binary_dir = vis_dir / "mask_binary"
	overlay_dir = vis_dir / "overlay"
	original_dir.mkdir(parents=True, exist_ok=True)
	mask_dir.mkdir(parents=True, exist_ok=True)
	mask_binary_dir.mkdir(parents=True, exist_ok=True)
	overlay_dir.mkdir(parents=True, exist_ok=True)

	# Recover the input slice from normalized tensor and convert to uint8 grayscale.
	image_uint8 = (image_tensor.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
	cv2.imwrite(str(original_dir / f"{base_name}.png"), image_uint8)

	# Save a binary mask copy for reproducibility/debugging.
	cv2.imwrite(str(mask_binary_dir / f"{base_name}_mask_binary.png"), binary_mask)

	# Save a red mask image so the segmented region is directly visible.
	mask_red = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
	mask_red[:, :, 2] = binary_mask
	cv2.imwrite(str(mask_dir / f"{base_name}_mask.png"), mask_red)

	# Visualization-only overlay for highlighted segmented region.
	# This does not feed back into prediction or saved binary mask generation.
	image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
	red_layer = np.zeros_like(image_bgr)
	red_layer[:, :, 2] = 255
	alpha = 0.4
	overlay = image_bgr.copy()
	blended = cv2.addWeighted(image_bgr, 1.0 - alpha, red_layer, alpha, 0)
	mask_bool = binary_mask > 0
	if mask_bool.any():
		overlay[mask_bool] = blended[mask_bool]
	cv2.imwrite(str(overlay_dir / f"{base_name}_overlay.png"), overlay)


def _collect_input_images(input_path: Path) -> list[Path]:
	if input_path.is_file():
		return [input_path]

	if input_path.is_dir():
		files = [
			p
			for p in input_path.iterdir()
			if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
		]
		files.sort()
		return files

	raise ValueError(f"Input path does not exist: {input_path}")


def run_inference(
	input_path: str | Path,
	model_path: str | Path,
	output_dir: str | Path = PREDICTIONS_DIR,
	threshold: float = PRED_THRESHOLD,
	kernel_size: tuple[int, int] = POSTPROCESS_KERNEL_SIZE,
	save_visualizations: bool = False,
	vis_dir: str | Path | None = None,
) -> list[Path]:
	input_path = Path(input_path)
	model_path = Path(model_path)
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	if vis_dir is None:
		vis_dir = output_dir / "visualizations"
	else:
		vis_dir = Path(vis_dir)

	if not model_path.is_absolute() and not model_path.exists():
		candidate = MODELS_DIR / model_path
		if candidate.exists():
			model_path = candidate

	if not model_path.exists():
		raise FileNotFoundError(f"Model file not found: {model_path}")

	model = HybridAAASegmentation(image_size=IMAGE_SIZE, base_channels=BASE_CHANNELS).to(DEVICE)
	state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
	model.load_state_dict(state_dict)
	model.eval()

	input_images = _collect_input_images(input_path)
	if not input_images:
		raise ValueError("No valid PNG/JPG images found for inference.")

	saved_files: list[Path] = []
	with torch.no_grad():
		for image_file in input_images:
			tensor = _load_input_image(image_file, IMAGE_SIZE).to(DEVICE)
			pred = model(tensor)

			# Ensure valid probability map if a model variant emits logits.
			if pred.min().item() < 0.0 or pred.max().item() > 1.0:
				pred = torch.sigmoid(pred)

			# Refine with morphology, largest component, boundary smoothing, and hole filling.
			pred_mask = refine_segmentation(
				pred.squeeze(0).squeeze(0),
				threshold=threshold,
				kernel_size=kernel_size,
			)

			output_file = output_dir / f"{image_file.stem}_pred.png"
			cv2.imwrite(str(output_file), pred_mask)
			saved_files.append(output_file)

			if save_visualizations:
				_save_visualizations(
					image_tensor=tensor,
					binary_mask=pred_mask,
					base_name=image_file.stem,
					vis_dir=vis_dir,
				)

	return saved_files

