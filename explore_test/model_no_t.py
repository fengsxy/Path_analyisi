"""
架构 B：无 timestep + 32×32 noisy image（未知噪声水平）

验证假设：没有任何 t 信息，模型不知道当前噪声水平，性能会显著下降

- 输入：x_64 (noisy, t未知) + x_32 (noisy, t未知)
- 输出：noise_pred_64
- 无 time embedding
- 32×32 是从原图 downsample 后加随机噪声
- 参数量：~58M
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlockNoTime(nn.Module):
    """Residual block without time embedding."""

    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class Attention(nn.Module):
    """Self-attention layer."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = (channels // num_heads) ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)

        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)

        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class NoTimestepModel(nn.Module):
    """
    无 timestep 模型，32×32 输入为 noisy image（噪声水平未知）。

    模型不知道任何噪声水平信息，是一个"盲去噪器"。
    假设：性能会显著下降，证明 t 的重要性。
    """

    def __init__(self, hidden_dims=[64, 128, 256, 512]):
        super().__init__()
        self.hidden_dims = hidden_dims

        # Initial convolution (6 channels: 3 for 64×64 + 3 for 32×32 upsampled)
        self.init_conv = nn.Conv2d(6, hidden_dims[0], 3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()

        in_dim = hidden_dims[0]
        for out_dim in hidden_dims:
            self.encoder_blocks.append(nn.ModuleList([
                ResidualBlockNoTime(in_dim, out_dim),
                ResidualBlockNoTime(out_dim, out_dim),
                Attention(out_dim) if out_dim >= 128 else nn.Identity(),
            ]))
            self.downsample.append(nn.Conv2d(out_dim, out_dim, 4, stride=2, padding=1))
            in_dim = out_dim

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResidualBlockNoTime(hidden_dims[-1], hidden_dims[-1]),
            Attention(hidden_dims[-1]),
            ResidualBlockNoTime(hidden_dims[-1], hidden_dims[-1]),
        ])

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()

        reversed_dims = list(reversed(hidden_dims))
        for i, out_dim in enumerate(reversed_dims):
            in_dim = reversed_dims[i - 1] if i > 0 else hidden_dims[-1]
            self.upsample.append(nn.ConvTranspose2d(in_dim, in_dim, 4, stride=2, padding=1))
            self.decoder_blocks.append(nn.ModuleList([
                ResidualBlockNoTime(in_dim + out_dim, out_dim),
                ResidualBlockNoTime(out_dim, out_dim),
                Attention(out_dim) if out_dim >= 128 else nn.Identity(),
            ]))

        # Output (只输出 3 channels for 64×64)
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, hidden_dims[0]),
            nn.SiLU(),
            nn.Conv2d(hidden_dims[0], 3, 3, padding=1),
        )

        self.in_channels = 3
        self.sample_size = 64

    def forward(self, x_64, x_32_noisy):
        """
        Args:
            x_64: Noisy 64×64 image [B, 3, 64, 64]
            x_32_noisy: Noisy 32×32 image (噪声水平未知) [B, 3, 32, 32]

        Returns:
            noise_pred_64: Predicted noise for 64×64 [B, 3, 64, 64]
        """
        # Upsample 32×32 noisy image to 64×64
        x_32_up = F.interpolate(x_32_noisy, size=(64, 64), mode='bilinear', align_corners=False)

        # Concatenate
        x = torch.cat([x_64, x_32_up], dim=1)  # [B, 6, 64, 64]

        # Initial conv
        x = self.init_conv(x)

        # Encoder with skip connections
        skips = []
        for (res1, res2, attn), down in zip(self.encoder_blocks, self.downsample):
            x = res1(x)
            x = res2(x)
            x = attn(x) if not isinstance(attn, nn.Identity) else x
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.bottleneck[0](x)
        x = self.bottleneck[1](x)
        x = self.bottleneck[2](x)

        # Decoder with skip connections
        for (res1, res2, attn), up in zip(self.decoder_blocks, self.upsample):
            x = up(x)
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = res1(x)
            x = res2(x)
            x = attn(x) if not isinstance(attn, nn.Identity) else x

        # Final conv - only predict 64×64 noise
        noise_pred_64 = self.final_conv(x)

        return noise_pred_64


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = NoTimestepModel()
    print(f"Parameters: {count_parameters(model):,}")

    x_64 = torch.randn(2, 3, 64, 64)
    x_32 = torch.randn(2, 3, 32, 32)

    out = model(x_64, x_32)
    print(f"Output shape: {out.shape}")
