#!/usr/bin/env python3
"""
Generate images using DP path from 100×100 timestep-space heatmap.

This script uses the pre-computed DP paths from compute_heatmap_timestep.py
to generate images with the optimal timestep schedule.

Usage:
    python generate_with_dp_path.py --checkpoint checkpoints/lift_dual_timestep_1000ep.pth \
        --heatmap results/heatmap_timestep_1000ep.pth \
        --output_dir results/fid_lift_dp64_100_1000ep \
        --path_type dp_64 --num_images 1000
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from model import LIFTDualTimestepModel
from scheduler import DDIMScheduler


def ddim_step(scheduler, model_output, timestep, prev_timestep, sample, eta=0.0):
    """Single DDIM step with explicit prev_timestep."""
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else scheduler.final_alpha_cumprod
    )
    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5

    if scheduler.clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

    variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    std_dev_t = eta * variance**0.5

    pred_epsilon = (sample - alpha_prod_t**0.5 * pred_original_sample) / beta_prod_t**0.5
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * pred_epsilon
    prev_sample = alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction

    if eta > 0:
        noise = torch.randn_like(model_output)
        prev_sample += std_dev_t * noise

    return prev_sample


def path_to_timesteps(path, t_grid):
    """
    Convert path indices to actual timesteps.

    Path goes from (0,0) to (99,99) in index space.
    t_grid maps index to timestep: t_grid[0]=999 (high noise), t_grid[99]=0 (low noise)

    For generation, we need timesteps in decreasing order (high to low noise).
    So we reverse the path.
    """
    t_np = t_grid.numpy() if torch.is_tensor(t_grid) else t_grid

    # Path is from (0,0) to (99,99), which is t=999 to t=0
    # For generation we go from high noise to low noise, so path order is correct
    # But t_grid[0] = 0 and t_grid[99] = 999? Let's check...
    # Actually t_grid = linspace(0, 999, 100), so t_grid[0]=0, t_grid[99]=999
    # Path (0,0) means index 0, which is t=0 (low noise)
    # Path (99,99) means index 99, which is t=999 (high noise)

    # Wait, the DP starts from (0,0) which should be high noise (t=999)
    # So t_grid should be reversed: t_grid[0]=999, t_grid[99]=0

    # Let's just compute: path index i -> timestep = 999 - i * (999/99)
    # Or use: timestep = 999 * (1 - i / 99)

    timesteps_64 = [int(999 * (1 - p[0] / 99)) for p in path]
    timesteps_32 = [int(999 * (1 - p[1] / 99)) for p in path]

    return timesteps_64, timesteps_32


@torch.no_grad()
def generate_batch_with_path(model, scheduler, batch_size, timesteps_64, timesteps_32, device, eta=0.0):
    """Generate a batch of images using pre-computed timestep paths."""
    num_steps = len(timesteps_64)

    x_64 = torch.randn(batch_size, 3, 64, 64, device=device)
    x_32 = torch.randn(batch_size, 3, 32, 32, device=device)

    for i in range(num_steps):
        t_64 = timesteps_64[i]
        t_32 = timesteps_32[i]

        t_64_tensor = torch.tensor([t_64], device=device).expand(batch_size)
        t_32_tensor = torch.tensor([t_32], device=device).expand(batch_size)

        noise_pred_64, noise_pred_32 = model(x_64, x_32, t_64_tensor, t_32_tensor)

        # Get prev timesteps
        if i < num_steps - 1:
            prev_t_64 = timesteps_64[i + 1]
            prev_t_32 = timesteps_32[i + 1]
        else:
            prev_t_64 = 0
            prev_t_32 = 0

        x_64 = ddim_step(scheduler, noise_pred_64, t_64, prev_t_64, x_64, eta)
        x_32 = ddim_step(scheduler, noise_pred_32, t_32, prev_t_32, x_32, eta)

    x_64 = (x_64 + 1) * 0.5
    return x_64


def save_images(images, output_dir, start_idx):
    """Save batch of images as PNG files."""
    for i, img in enumerate(images):
        img_np = img.cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        img_pil.save(os.path.join(output_dir, f'{start_idx + i:05d}.png'))


def parse_args():
    parser = argparse.ArgumentParser(description='Generate images with DP path from 100×100 heatmap')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--heatmap', type=str, required=True, help='Path to heatmap_timestep_XXXep.pth')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--path_type', type=str, choices=['dp_64', 'dp_total'], required=True,
                        help='Which DP path to use')
    parser.add_argument('--num_images', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_steps', type=int, default=18, help='Number of generation steps (default 18)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hidden_dims = checkpoint.get('hidden_dims', [64, 128, 256, 512])

    model = LIFTDualTimestepModel(hidden_dims=hidden_dims)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load heatmap and path
    print(f"Loading heatmap: {args.heatmap}")
    heatmap_data = torch.load(args.heatmap, weights_only=False)

    t_grid = heatmap_data['t_grid']

    if args.path_type == 'dp_64':
        path = heatmap_data['path_64']
        print(f"Using DP-64 path ({len(path)} points)")
    else:
        path = heatmap_data['path_total']
        print(f"Using DP-Total path ({len(path)} points)")

    # Convert path to timesteps
    timesteps_64, timesteps_32 = path_to_timesteps(path, t_grid)

    print(f"Timestep schedule:")
    print(f"  t_64: {timesteps_64[0]} -> {timesteps_64[-1]}")
    print(f"  t_32: {timesteps_32[0]} -> {timesteps_32[-1]}")

    # Create scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="cosine",
        clip_sample=True
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate images
    print(f"\nGenerating {args.num_images} images with {args.path_type} path...")
    print(f"Output directory: {args.output_dir}")

    num_batches = (args.num_images + args.batch_size - 1) // args.batch_size
    generated = 0

    for _ in tqdm(range(num_batches), desc="Generating"):
        current_batch_size = min(args.batch_size, args.num_images - generated)

        images = generate_batch_with_path(
            model, scheduler, current_batch_size,
            timesteps_64, timesteps_32, device, args.eta
        )

        save_images(images, args.output_dir, generated)
        generated += current_batch_size

    print(f"\nGenerated {generated} images to {args.output_dir}")


if __name__ == "__main__":
    main()
