from pathlib import Path
from typing import Any, Callable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _collect_image_mask_pairs(
	images_dir: Path,
	masks_dir: Path,
) -> List[Tuple[Path, Path]]:
	images_dir = Path(images_dir)
	masks_dir = Path(masks_dir)

	mask_lookup = {
		mask_path.stem: mask_path
		for mask_path in masks_dir.iterdir()
		if mask_path.suffix.lower() in VALID_EXTENSIONS
	}

	pairs: List[Tuple[Path, Path]] = []
	for image_path in images_dir.iterdir():
		if image_path.suffix.lower() not in VALID_EXTENSIONS:
			continue
		if image_path.stem in mask_lookup:
			pairs.append((image_path, mask_lookup[image_path.stem]))

	pairs.sort(key=lambda item: item[0].name)
	return pairs


class AAASegmentationDataset(Dataset):
	"""Dataset for loading 2D CT slice images and binary masks."""

	def __init__(
		self,
		pairs: Sequence[Tuple[Path, Path]],
		image_size: int = 256,
		transform: Callable[..., Any] | None = None,
	) -> None:
		self.pairs = list(pairs)
		self.image_size = image_size
		self.transform = transform

	def __len__(self) -> int:
		return len(self.pairs)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		image_path, mask_path = self.pairs[idx]

		image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
		mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

		if image is None:
			raise ValueError(f"Could not read image file: {image_path}")
		if mask is None:
			raise ValueError(f"Could not read mask file: {mask_path}")

		image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
		mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

		image = image.astype(np.float32) / 255.0
		mask = (mask > 127).astype(np.float32)

		if self.transform is not None:
			# Albumentations expects named inputs and applies identical spatial ops to mask.
			transformed = self.transform(image=image, mask=mask)
			image, mask = transformed["image"], transformed["mask"]

		# Keep tensor compatibility: float dtype and channel-first (1, H, W).
		image = image.astype(np.float32)
		mask = mask.astype(np.float32)

		image_tensor = torch.from_numpy(image).unsqueeze(0)
		mask_tensor = torch.from_numpy(mask).unsqueeze(0)
		return image_tensor, mask_tensor


def create_datasets(
	images_dir: str | Path,
	masks_dir: str | Path,
	image_size: int = 256,
	train_split: float = 0.8,
	random_state: int = 42,
	train_transform: Callable[..., Any] | None = None,
	val_transform: Callable[..., Any] | None = None,
) -> Tuple[AAASegmentationDataset, AAASegmentationDataset]:
	"""Create train and validation dataset splits from image/mask directories."""
	pairs = _collect_image_mask_pairs(Path(images_dir), Path(masks_dir))
	if not pairs:
		raise ValueError("No image-mask pairs found in the provided directories.")

	if len(pairs) == 1:
		train_pairs = pairs
		val_pairs = pairs
	else:
		train_pairs, val_pairs = train_test_split(
			pairs,
			train_size=train_split,
			random_state=random_state,
			shuffle=True,
		)

	train_dataset = AAASegmentationDataset(
		train_pairs,
		image_size=image_size,
		transform=train_transform,
	)
	val_dataset = AAASegmentationDataset(
		val_pairs,
		image_size=image_size,
		transform=val_transform,
	)
	return train_dataset, val_dataset

