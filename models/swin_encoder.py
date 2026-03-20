from typing import List

import timm
import torch
import torch.nn as nn


class SwinFeatureExtractor(nn.Module):
	"""Swin Transformer encoder that returns multi-scale feature maps."""

	def __init__(
		self,
		model_name: str = "swin_tiny_patch4_window7_224",
		pretrained: bool = True,
		image_size: int = 256,
		out_indices: tuple[int, ...] = (0, 1, 2, 3),
	) -> None:
		super().__init__()
		self.backbone = timm.create_model(
			model_name,
			pretrained=pretrained,
			features_only=True,
			out_indices=out_indices,
			img_size=image_size,
		)
		self.feature_channels = list(self.backbone.feature_info.channels())

	def _to_nchw(self, feature: torch.Tensor, expected_channels: int) -> torch.Tensor:
		if feature.ndim != 4:
			raise ValueError("Expected 4D feature map from Swin backbone.")

		if feature.shape[1] == expected_channels:
			return feature
		if feature.shape[-1] == expected_channels:
			return feature.permute(0, 3, 1, 2).contiguous()
		return feature

	def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
		# Swin model expects 3-channel input; duplicate grayscale channel.
		if x.shape[1] == 1:
			x = x.repeat(1, 3, 1, 1)

		features = self.backbone(x)
		features = [
			self._to_nchw(feature, channels)
			for feature, channels in zip(features, self.feature_channels)
		]
		return features

