"""
架构 A：单 timestep + 32×32 noisy image（未知噪声水平）

验证假设：32×32 提供了有用的图像信息，即使模型不知道它的噪声水平

- 输入：x_64 (noisy, t已知) + x_32 (noisy, t未知) + t_64
- 输出：noise_pred_64
- 32×32 是从原图 downsample 后加随机噪声，但 t_32 不传给模型
- 参数量：~58M
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('..')
from model import SinusoidalPositionEmbeddings, ResidualBlock, Attention


class SingleTimestepModel(nn.Module):
    """
    单 timestep 模型，32×32 输入为 noisy image（噪声水平未知）。

    训练时：32×32 从原图 downsample 后加随机噪声，但 t_32 不传给模型
    假设：32×32 仍然提供有用的图像信息
    """

    def __init__(self, hidden_dims=[64, 128, 256, 512]):
        super().__init__()
        self.hidden_dims = hidden_dims
        time_emb_dim = hidden_dims[0] * 4

        # Time embedding (单个 t)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dims[0]),
            nn.Linear(hidden_dims[0], time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial convolution (6 channels: 3 for 64×64 + 3 for 32×32 upsampled)
        self.init_conv = nn.Conv2d(6, hidden_dims[0], 3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()

        in_dim = hidden_dims[0]
        for out_dim in hidden_dims:
            self.encoder_blocks.append(nn.ModuleList([
                ResidualBlock(in_dim, out_dim, time_emb_dim),
                ResidualBlock(out_dim, out_dim, time_emb_dim),
                Attention(out_dim) if out_dim >= 128 else nn.Identity(),
            ]))
            self.downsample.append(nn.Conv2d(out_dim, out_dim, 4, stride=2, padding=1))
            in_dim = out_dim

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResidualBlock(hidden_dims[-1], hidden_dims[-1], time_emb_dim),
            Attention(hidden_dims[-1]),
            ResidualBlock(hidden_dims[-1], hidden_dims[-1], time_emb_dim),
        ])

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()

        reversed_dims = list(reversed(hidden_dims))
        for i, out_dim in enumerate(reversed_dims):
            in_dim = reversed_dims[i - 1] if i > 0 else hidden_dims[-1]
            self.upsample.append(nn.ConvTranspose2d(in_dim, in_dim, 4, stride=2, padding=1))
            self.decoder_blocks.append(nn.ModuleList([
                ResidualBlock(in_dim + out_dim, out_dim, time_emb_dim),
                ResidualBlock(out_dim, out_dim, time_emb_dim),
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

    def forward(self, x_64, x_32_noisy, t):
        """
        Args:
            x_64: Noisy 64×64 image [B, 3, 64, 64]
            x_32_noisy: Noisy 32×32 image (噪声水平未知) [B, 3, 32, 32]
            t: Timestep for 64×64 [B]

        Returns:
            noise_pred_64: Predicted noise for 64×64 [B, 3, 64, 64]
        """
        # Time embedding
        t_emb = self.time_mlp(t.float())

        # Upsample 32×32 noisy image to 64×64
        x_32_up = F.interpolate(x_32_noisy, size=(64, 64), mode='bilinear', align_corners=False)

        # Concatenate
        x = torch.cat([x_64, x_32_up], dim=1)  # [B, 6, 64, 64]

        # Initial conv
        x = self.init_conv(x)

        # Encoder with skip connections
        skips = []
        for (res1, res2, attn), down in zip(self.encoder_blocks, self.downsample):
            x = res1(x, t_emb)
            x = res2(x, t_emb)
            x = attn(x) if not isinstance(attn, nn.Identity) else x
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.bottleneck[0](x, t_emb)
        x = self.bottleneck[1](x)
        x = self.bottleneck[2](x, t_emb)

        # Decoder with skip connections
        for (res1, res2, attn), up in zip(self.decoder_blocks, self.upsample):
            x = up(x)
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = res1(x, t_emb)
            x = res2(x, t_emb)
            x = attn(x) if not isinstance(attn, nn.Identity) else x

        # Final conv - only predict 64×64 noise
        noise_pred_64 = self.final_conv(x)

        return noise_pred_64


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = SingleTimestepModel()
    print(f"Parameters: {count_parameters(model):,}")

    x_64 = torch.randn(2, 3, 64, 64)
    x_32 = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 1000, (2,))

    out = model(x_64, x_32, t)
    print(f"Output shape: {out.shape}")
