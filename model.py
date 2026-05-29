"""
CNN-based image denoising models with attention mechanisms.

Models:
  - DnCNN: Standard 17-layer baseline
  - UNetDenoise: U-Net with skip connections
  - DnCNNWithAttention: DnCNN + LMAB (Lightweight Mixed Attention Block)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    """Lightweight spatial attention using depthwise separable convolution."""

    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        # Depthwise conv reduces params vs standard conv
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=channels, bias=False
        )
        self.pointwise = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(x)
        y = self.pointwise(y)
        return x * self.sigmoid(y)


class LightweightMixedAttentionBlock(nn.Module):
    """
    LMAB: Lightweight Mixed Attention Block.

    Combines channel attention and spatial attention.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(self.avg_pool(x))
        x = x * y
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        x = x * self.spatial_sigmoid(y)
        return x


class CBAM(nn.Module):
    """
    Standard CBAM for ablation comparison.
    Slightly heavier than LMAB due to extra convolutions in spatial branch.
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        # Channel attention with both avg and max pool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.ch_sigmoid = nn.Sigmoid()

        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sp_sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        ch_att = self.ch_sigmoid(self.fc(avg_out) + self.fc(max_out))
        x = x * ch_att

        # Spatial
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sp_att = self.sp_sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * sp_att
        return x


# ---------------------------------------------------------------------------
# Residual blocks
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Residual block: Conv -> BN -> ReLU -> Conv -> [Attention] -> + residual.
    attention_type: 'lmab' | 'cbam' | 'channel_only' | 'spatial_only' | None
    """

    def __init__(self, channels: int, attention_type: str | None = 'lmab'):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

        self.attention_type = attention_type
        self.attention: nn.Module | None = None
        if attention_type == 'lmab':
            self.attention = LightweightMixedAttentionBlock(channels)
        elif attention_type == 'cbam':
            self.attention = CBAM(channels)
        elif attention_type == 'channel_only':
            self.attention = ChannelAttention(channels)
        elif attention_type == 'spatial_only':
            self.attention = SpatialAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn(self.conv1(x)))
        out = self.conv2(out)
        if self.attention is not None:
            out = self.attention(out)
        return out + residual


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class DnCNN(nn.Module):
    """
    Standard DnCNN for image denoising.
    17 layers of Conv+BN+ReLU, global residual learning.
    """

    def __init__(self, in_channels: int = 1, num_features: int = 64, num_layers: int = 17):
        super().__init__()
        layers = []
        # First conv
        layers.append(nn.Conv2d(in_channels, num_features, 3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        # Middle layers
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
        # Output conv
        layers.append(nn.Conv2d(num_features, in_channels, 3, padding=1, bias=False))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = self.model(x)
        return x - noise


class DnCNNWithAttention(nn.Module):
    """
    DnCNN variant with LMAB attention inserted in each residual block.
    This is the proposed 'Ours' model.

    The architecture:
      Input -> Conv+ReLU (shallow feature extraction)
           -> N x ResidualBlock (with LMAB)
           -> Conv (reconstruction)
           -> Output = Input - predicted_noise
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_features: int = 64,
        num_blocks: int = 8,
        attention_type: str | None = 'lmab',
    ):
        super().__init__()
        self.attention_type = attention_type
        self.shallow = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(num_features, attention_type) for _ in range(num_blocks)
        ])
        self.reconstruct = nn.Conv2d(num_features, in_channels, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.shallow(x)
        for block in self.blocks:
            feat = block(feat)
        noise = self.reconstruct(feat)
        return x - noise, feat


class DoubleConv(nn.Module):
    """Two sequential conv layers for U-Net."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetDenoise(nn.Module):
    """
    U-Net for image denoising with noise residual learning.
    Encoder-decoder with skip connections.
    """

    def __init__(self, in_channels: int = 1, features: int = 48):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, features)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(features, features * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = DoubleConv(features * 4, features * 8)
        # Decoder
        self.up3 = nn.ConvTranspose2d(features * 8, features * 4, 2, stride=2)
        self.dec3 = DoubleConv(features * 8, features * 4)
        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, 2, stride=2)
        self.dec2 = DoubleConv(features * 4, features * 2)
        self.up1 = nn.ConvTranspose2d(features * 2, features, 2, stride=2)
        self.dec1 = DoubleConv(features * 2, features)
        # Output
        self.out_conv = nn.Conv2d(features, in_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad input to be divisible by 8 (3 pooling layers → 2³ = 8)
        _, _, h, w = x.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect') if pad_h or pad_w else x

        # Encoder
        e1 = self.enc1(x_padded)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        # Bottleneck
        b = self.bottleneck(self.pool3(e3))
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        noise = self.out_conv(d1)
        # Crop back to original input size
        noise = noise[:, :, :h, :w]
        return x - noise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(name: str, **kwargs):
    """Factory for model creation."""
    models = {
        'dncnn': DnCNN,
        'unet': UNetDenoise,
        'ours': DnCNNWithAttention,
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Choose from {list(models.keys())}")
    return models[name](**kwargs)
