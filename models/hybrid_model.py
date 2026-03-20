import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import AttentionGate
from models.residual_unet import DecoderBlock, EncoderBlock, ResidualBlock
from models.swin_encoder import SwinFeatureExtractor


class HybridAAASegmentation(nn.Module):
	"""Hybrid Residual Attention U-Net with Swin Transformer feature fusion."""

	def __init__(
		self,
		in_channels: int = 1,
		out_channels: int = 1,
		base_channels: int = 32,
		image_size: int = 256,
		pretrained_swin: bool = True,
	) -> None:
		super().__init__()

		self.enc1 = EncoderBlock(in_channels, base_channels)
		self.enc2 = EncoderBlock(base_channels, base_channels * 2)
		self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4)
		self.enc4 = EncoderBlock(base_channels * 4, base_channels * 8)
		self.bottleneck = ResidualBlock(base_channels * 8, base_channels * 16)

		self.swin_encoder = SwinFeatureExtractor(
			pretrained=pretrained_swin,
			image_size=image_size,
		)
		swin_channels = self.swin_encoder.feature_channels

		self.swin_to_bottleneck_1 = nn.Conv2d(swin_channels[2], base_channels * 16, kernel_size=1)
		self.swin_to_bottleneck_2 = nn.Conv2d(swin_channels[3], base_channels * 16, kernel_size=1)
		self.bottleneck_fuse = ResidualBlock(base_channels * 32, base_channels * 16)

		self.swin_to_skip4 = nn.Conv2d(swin_channels[1], base_channels * 8, kernel_size=1)
		self.swin_to_skip3 = nn.Conv2d(swin_channels[0], base_channels * 4, kernel_size=1)
		self.skip4_fuse = ResidualBlock(base_channels * 16, base_channels * 8)
		self.skip3_fuse = ResidualBlock(base_channels * 8, base_channels * 4)

		self.att4 = AttentionGate(base_channels * 16, base_channels * 8, base_channels * 4)
		self.att3 = AttentionGate(base_channels * 8, base_channels * 4, base_channels * 2)
		self.att2 = AttentionGate(base_channels * 4, base_channels * 2, base_channels)
		self.att1 = AttentionGate(base_channels * 2, base_channels, base_channels // 2)

		self.dec1 = DecoderBlock(base_channels * 16, base_channels * 8, base_channels * 8)
		self.dec2 = DecoderBlock(base_channels * 8, base_channels * 4, base_channels * 4)
		self.dec3 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2)
		self.dec4 = DecoderBlock(base_channels * 2, base_channels, base_channels)

		self.segmentation_head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		e1, p1 = self.enc1(x)
		e2, p2 = self.enc2(p1)
		e3, p3 = self.enc3(p2)
		e4, p4 = self.enc4(p3)

		b = self.bottleneck(p4)

		swin_feats = self.swin_encoder(x)
		s1, s2, s3, s4 = swin_feats

		s3 = self.swin_to_bottleneck_1(s3)
		s4 = self.swin_to_bottleneck_2(s4)
		s3 = F.interpolate(s3, size=b.shape[2:], mode="bilinear", align_corners=False)
		s4 = F.interpolate(s4, size=b.shape[2:], mode="bilinear", align_corners=False)
		b = self.bottleneck_fuse(torch.cat([b, s3 + s4], dim=1))

		sw_skip4 = self.swin_to_skip4(s2)
		sw_skip4 = F.interpolate(sw_skip4, size=e4.shape[2:], mode="bilinear", align_corners=False)
		skip4 = self.skip4_fuse(torch.cat([e4, sw_skip4], dim=1))

		sw_skip3 = self.swin_to_skip3(s1)
		sw_skip3 = F.interpolate(sw_skip3, size=e3.shape[2:], mode="bilinear", align_corners=False)
		skip3 = self.skip3_fuse(torch.cat([e3, sw_skip3], dim=1))

		g4 = F.interpolate(b, size=skip4.shape[2:], mode="bilinear", align_corners=False)
		skip4 = self.att4(g4, skip4)
		d1 = self.dec1(b, skip4)

		g3 = F.interpolate(d1, size=skip3.shape[2:], mode="bilinear", align_corners=False)
		skip3 = self.att3(g3, skip3)
		d2 = self.dec2(d1, skip3)

		g2 = F.interpolate(d2, size=e2.shape[2:], mode="bilinear", align_corners=False)
		skip2 = self.att2(g2, e2)
		d3 = self.dec3(d2, skip2)

		g1 = F.interpolate(d3, size=e1.shape[2:], mode="bilinear", align_corners=False)
		skip1 = self.att1(g1, e1)
		d4 = self.dec4(d3, skip1)

		logits = self.segmentation_head(d4)
		return torch.sigmoid(logits)

