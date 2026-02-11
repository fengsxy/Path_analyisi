"""Generate CIFAR images with 2-scale SLOT model and compute FID.

Supports sigma schedule modes:
  sync — same sigma for both scales
  orig, total — DP optimal paths from heatmap_2d.pth
"""
import argparse
import io
import math
import os
import pickle
import sys
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/ylong030/slot")
import dnnlib

# ── Torus helpers (2-scale) ───────────────────────────────────────────────
def convert_torus_2d(u):
    """Convert 2-scale RGB to torus: [B, 6, H, W] -> [B, 12, H, W]."""
    scales = u.chunk(2, dim=1)
    out = []
    for s in scales:
        s = (math.pi / 2) * torch.clamp(s, -1, 1)
        out.append(torch.cat([torch.cos(s), torch.sin(s)], dim=1))
    return torch.cat(out, dim=1)

def convert_from_torus_2d(t):
    """Convert torus back to 2-scale RGB: [B, 12, H, W] -> [B, 6, H, W]."""
    chunks = t.chunk(2, dim=1)
    out = []
    for c in chunks:
        cos_part, sin_part = c.chunk(2, dim=1)
        angle = torch.atan2(sin_part, cos_part)
        out.append(angle / (math.pi / 2))
    return torch.cat(out, dim=1)
# ── Sigma schedule builders ───────────────────────────────────────────────
def _rho_schedule(n, sigma_max, sigma_min, rho, device):
    if n <= 1:
        return torch.full((n,), sigma_min, device=device, dtype=torch.float32)
    idx = torch.arange(n, dtype=torch.float64, device=device)
    s = (sigma_max ** (1/rho) + idx/(n-1) * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
    return s.to(torch.float32)

def build_sigma_schedule_2d(num_steps, mode, sigma_min=0.002, sigma_max=80.0, rho=7.0, device='cpu'):
    """Build [num_steps, 2] sigma schedule for 2-scale model."""
    sigmas = torch.zeros(num_steps, 2, device=device, dtype=torch.float32)
    if mode == "sync":
        t = _rho_schedule(num_steps, sigma_max, sigma_min, rho, device)
        sigmas[:] = t.unsqueeze(1).expand(-1, 2)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return sigmas

def build_dp_schedule_2d(heatmap_path, path_type, device='cpu'):
    """Build sigma schedule from DP path in heatmap_2d.pth."""
    data = torch.load(heatmap_path, map_location='cpu', weights_only=False)
    sigma_grid = data['sigma_grid'].numpy()
    if path_type == 'orig':
        path = data['path_orig']
    elif path_type == 'total':
        path = data['path_total']
    else:
        raise ValueError(f"Unknown path_type: {path_type}")
    num_steps = len(path) - 1
    sigmas = torch.zeros(num_steps, 2, device=device, dtype=torch.float32)
    for step in range(num_steps):
        i_orig, i_2x = path[step + 1]
        sigmas[step, 0] = float(sigma_grid[i_orig])
        sigmas[step, 1] = float(sigma_grid[i_2x])
    return sigmas
# ── Heun sampler (reuse from generate_and_fid.py — already generic) ──────
from generate_and_fid import edm_sampler_torus_heun, load_cifar_real_features, compute_fid

# ── Model loading ─────────────────────────────────────────────────────────
def load_2scale_model(path, device):
    """Load 2-scale model from .pkl (EMA) or .pth (checkpoint)."""
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data['ema'].eval().to(device)
    else:
        from training.networks import EDMPrecondSlot
        net = EDMPrecondSlot(
            img_resolution=32, img_channels=12, label_dim=0, sigma_data=0.5,
            num_scales=2, model_type='SongUNet', model_channels=128,
            channel_mult=[2, 2, 2], augment_dim=9, dropout=0.13,
            embedding_type='positional', encoder_type='standard',
            decoder_type='standard', channel_mult_noise=1, resample_filter=[1, 1],
        ).to(device)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if 'ema' in ckpt:
            net.load_state_dict(ckpt['ema'])
        else:
            net.load_state_dict(ckpt['net'])
        return net.eval()

# ── Generation + FID ──────────────────────────────────────────────────────
@torch.no_grad()
def generate_batch(net, batch_size, sigmas_schedule, seed, device):
    rng = torch.Generator(device=device).manual_seed(seed)
    C = 12  # 2 scales × 6 torus channels
    H = W = 32
    latents = torch.randn(batch_size, C, H, W, generator=rng, device=device)
    cos_part, sin_part = latents.chunk(2, dim=1)
    norm = torch.sqrt(cos_part ** 2 + sin_part ** 2 + 1e-8)
    latents = torch.cat([cos_part / norm, sin_part / norm], dim=1)
    x_out = edm_sampler_torus_heun(net, latents, sigmas_schedule)
    images = convert_from_torus_2d(x_out)
    return images[:, :3]  # orig scale
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_images', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_steps', type=int, default=18)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--path_type', type=str, default='sync',
                        choices=['sync', 'orig', 'total'])
    parser.add_argument('--model_path', type=str, default=None,
                        help='.pkl or .pth checkpoint for 2-scale model')
    parser.add_argument('--heatmap_path', type=str, default='results/heatmap_2d.pth')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}')
    os.makedirs(args.output_dir, exist_ok=True)

    net = load_2scale_model(args.model_path, device)

    if args.path_type in ('orig', 'total'):
        sigmas = build_dp_schedule_2d(args.heatmap_path, args.path_type, device=device)
        num_steps = sigmas.shape[0]
    else:
        num_steps = args.num_steps
        sigmas = build_sigma_schedule_2d(num_steps, args.path_type, device=device)

    print(f"Path type: {args.path_type}, steps: {num_steps}")
    print(f"sigma_orig: {sigmas[:3, 0].tolist()}...{sigmas[-2:, 0].tolist()}")
    print(f"sigma_2x:   {sigmas[:3, 1].tolist()}...{sigmas[-2:, 1].tolist()}")

    cache_path = os.path.join(args.output_dir, 'cifar10_real_features.npz')
    real_mu, real_sigma = load_cifar_real_features(cache_path, device)

    print(f"\nGenerating {args.num_images} images...")
    num_batches = (args.num_images + args.batch_size - 1) // args.batch_size
    all_images = []
    for i in tqdm(range(num_batches), desc="Generating"):
        bs = min(args.batch_size, args.num_images - i * args.batch_size)
        imgs = generate_batch(net, bs, sigmas, args.seed + i, device)
        all_images.append(imgs.cpu())

    all_images_cat = torch.cat(all_images, dim=0)[:args.num_images]
    print(f"Generated {len(all_images_cat)} images, range [{all_images_cat.min():.2f}, {all_images_cat.max():.2f}]")

    print("\nComputing FID...")
    fid_batches = [all_images_cat[i:i+args.batch_size] for i in range(0, len(all_images_cat), args.batch_size)]
    fid = compute_fid(fid_batches, real_mu, real_sigma, device)
    print(f"\nFID ({args.path_type}, {args.num_images} images): {fid:.4f}")

    csv_path = os.path.join(args.output_dir, 'fid_2scale_results.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a') as f:
        if write_header:
            f.write('path_type,num_images,num_steps,seed,fid\n')
        f.write(f'{args.path_type},{args.num_images},{num_steps},{args.seed},{fid:.4f}\n')
    print(f"Appended to {csv_path}")

    from torchvision.utils import save_image
    grid_path = os.path.join(args.output_dir, f'samples_2scale_{args.path_type}.png')
    save_image((all_images_cat[:64].clamp(-1,1)+1)/2, grid_path, nrow=8)
    print(f"Saved sample grid: {grid_path}")

if __name__ == '__main__':
    main()
