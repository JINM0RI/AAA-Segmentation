from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _to_numpy_2d(arr: np.ndarray) -> np.ndarray:
	if arr.ndim == 3 and arr.shape[0] == 1:
		return arr[0]
	if arr.ndim == 3 and arr.shape[-1] == 1:
		return arr[..., 0]
	return arr


def plot_segmentation_results(
	image: np.ndarray,
	ground_truth: np.ndarray,
	prediction: np.ndarray,
	save_path: str | Path | None = None,
	show: bool = False,
) -> None:
	image = _to_numpy_2d(np.asarray(image))
	ground_truth = _to_numpy_2d(np.asarray(ground_truth))
	prediction = _to_numpy_2d(np.asarray(prediction))

	fig, axes = plt.subplots(1, 3, figsize=(12, 4))

	axes[0].imshow(image, cmap="gray")
	axes[0].set_title("Input CT Slice")
	axes[0].axis("off")

	axes[1].imshow(ground_truth, cmap="gray")
	axes[1].set_title("Ground Truth Mask")
	axes[1].axis("off")

	axes[2].imshow(prediction, cmap="gray")
	axes[2].set_title("Predicted Mask")
	axes[2].axis("off")

	fig.tight_layout()

	if save_path is not None:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(save_path, dpi=200, bbox_inches="tight")

	if show:
		plt.show()

	plt.close(fig)

