"""
Baseline Model (Non-LIFT) for AFHQ64

Single-scale 64×64 diffusion model without auxiliary 32×32 scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time embedding"""

    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add time embedding
        h = h + self.time_mlp(F.silu(t))[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class Attention(nn.Module):
    """Self-attention layer"""

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

        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)

        # Attention
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)

        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class BaselineModel(nn.Module):
    """
    Baseline Model for AFHQ64 (Non-LIFT, single scale)

    - Input: 64×64 RGB image (3 channels)
    - Output: 64×64 RGB noise prediction (3 channels)
    """

    def __init__(self, hidden_dims=[64, 128, 256]):
        super().__init__()
        self.hidden_dims = hidden_dims
        time_emb_dim = hidden_dims[0] * 4

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dims[0]),
            nn.Linear(hidden_dims[0], time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(3, hidden_dims[0], 3, padding=1)

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
                ResidualBlock(in_dim + out_dim, out_dim, time_emb_dim),  # Skip connection
                ResidualBlock(out_dim, out_dim, time_emb_dim),
                Attention(out_dim) if out_dim >= 128 else nn.Identity(),
            ]))

        # Output
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, hidden_dims[0]),
            nn.SiLU(),
            nn.Conv2d(hidden_dims[0], 3, 3, padding=1),
        )

    def forward(self, x, timesteps):
        """
        Args:
            x: Noisy 64×64 image [B, 3, 64, 64]
            timesteps: Timesteps [B]

        Returns:
            noise_pred: Predicted noise for 64×64 [B, 3, 64, 64]
        """
        # Time embedding
        t = self.time_mlp(timesteps)

        # Initial conv
        x = self.init_conv(x)

        # Encoder with skip connections
        skips = []
        for (res1, res2, attn), down in zip(self.encoder_blocks, self.downsample):
            x = res1(x, t)
            x = res2(x, t)
            x = attn(x) if not isinstance(attn, nn.Identity) else x
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.bottleneck[0](x, t)
        x = self.bottleneck[1](x)
        x = self.bottleneck[2](x, t)

        # Decoder with skip connections
        for (res1, res2, attn), up in zip(self.decoder_blocks, self.upsample):
            x = up(x)
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = res1(x, t)
            x = res2(x, t)
            x = attn(x) if not isinstance(attn, nn.Identity) else x

        # Output
        x = self.final_conv(x)

        return x


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = BaselineModel(hidden_dims=[64, 128, 256])
    print(f"Baseline parameters: {count_parameters(model):,}")

    x = torch.randn(2, 3, 64, 64)
    t = torch.randint(0, 1000, (2,))
    noise_pred = model(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {noise_pred.shape}")
