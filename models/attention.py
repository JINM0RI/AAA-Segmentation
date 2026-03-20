import torch
import torch.nn as nn


class AttentionGate(nn.Module):
	"""Attention gate that suppresses irrelevant skip-connection features."""

	def __init__(self, gating_channels: int, skip_channels: int, inter_channels: int) -> None:
		super().__init__()

		self.g_proj = nn.Sequential(
			nn.Conv2d(gating_channels, inter_channels, kernel_size=1, bias=False),
			nn.BatchNorm2d(inter_channels),
		)
		self.x_proj = nn.Sequential(
			nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
			nn.BatchNorm2d(inter_channels),
		)
		self.psi = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
			nn.Sigmoid(),
		)

	def forward(self, gating: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
		gate_features = self.g_proj(gating)
		skip_features = self.x_proj(skip)
		attn = self.psi(gate_features + skip_features)
		return skip * attn

