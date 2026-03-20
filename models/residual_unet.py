import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
	"""Two-layer residual block with BatchNorm and ReLU."""

	def __init__(self, in_channels: int, out_channels: int) -> None:
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels)

		self.shortcut = nn.Identity()
		if in_channels != out_channels:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
				nn.BatchNorm2d(out_channels),
			)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		identity = self.shortcut(x)

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = out + identity
		out = self.relu(out)
		return out


class EncoderBlock(nn.Module):
	"""Encoder block with a residual block and spatial downsampling."""

	def __init__(self, in_channels: int, out_channels: int) -> None:
		super().__init__()
		self.res = ResidualBlock(in_channels, out_channels)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

	def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		feat = self.res(x)
		pooled = self.pool(feat)
		return feat, pooled


class DecoderBlock(nn.Module):
	"""Decoder block with transposed convolution, skip concatenation, and residual refinement."""

	def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
		super().__init__()
		self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
		self.res = ResidualBlock(out_channels + skip_channels, out_channels)

	def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
		x = self.up(x)
		x = torch.cat([x, skip], dim=1)
		x = self.res(x)
		return x

