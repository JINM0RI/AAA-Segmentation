import torch
import torch.nn as nn


class DiceLoss(nn.Module):
	"""Dice loss for binary segmentation using logits as input."""

	def __init__(self, smooth: float = 1e-7) -> None:
		super().__init__()
		self.smooth = smooth

	def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		# Dice is computed on probabilities, so convert logits with sigmoid internally.
		probs = torch.sigmoid(logits)
		targets = targets.float()

		intersection = (probs * targets).sum(dim=(1, 2, 3))
		denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
		dice_coeff = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
		loss = 1.0 - dice_coeff
		return loss.mean()
