#!/usr/bin/env python3
"""
Generate images using optimal path from 30×30 heatmap.

Uses the pre-computed path and step allocation from compute_heatmap_30.py.

Usage:
    python generate_with_path_30.py --checkpoint checkpoints/lift_dual_timestep_100ep.pth \
        --heatmap results/heatmap_30_100ep.pth --output_dir results/fid_dp64_100ep \
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
    """Single DDIM step."""
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


@torch.no_grad()
def generate_batch(model, scheduler, batch_size, timesteps_64, timesteps_32, device, eta=0.0):
    """Generate a batch using pre-computed timestep schedule."""
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
    """Save batch of images."""
    for i, img in enumerate(images):
        img_np = img.cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        Image.fromarray(img_np).save(os.path.join(output_dir, f'{start_idx + i:05d}.png'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--heatmap', type=str, required=True, help='Path to heatmap_30_XXXep.pth')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--path_type', type=str, choices=['dp_64', 'dp_total'], required=True)
    parser.add_argument('--num_images', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hidden_dims = ckpt.get('hidden_dims', [64, 128, 256, 512])

    model = LIFTDualTimestepModel(hidden_dims=hidden_dims)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device).eval()

    # Load heatmap and get timesteps
    print(f"Loading heatmap: {args.heatmap}")
    heatmap_data = torch.load(args.heatmap, weights_only=False)

    if args.path_type == 'dp_64':
        timesteps = heatmap_data['timesteps_64']
    else:
        timesteps = heatmap_data['timesteps_total']

    timesteps_64 = timesteps['t_64']
    timesteps_32 = timesteps['t_32']

    print(f"Path type: {args.path_type}")
    print(f"Steps: {len(timesteps_64)}")
    print(f"t_64: {timesteps_64[:3]}...{timesteps_64[-3:]}")
    print(f"t_32: {timesteps_32[:3]}...{timesteps_32[-3:]}")

    # Scheduler
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="cosine", clip_sample=True)

    # Generate
    os.makedirs(args.output_dir, exist_ok=True)

    num_batches = (args.num_images + args.batch_size - 1) // args.batch_size
    generated = 0

    for _ in tqdm(range(num_batches), desc="Generating"):
        batch_size = min(args.batch_size, args.num_images - generated)
        images = generate_batch(model, scheduler, batch_size, timesteps_64, timesteps_32, device, args.eta)
        save_images(images, args.output_dir, generated)
        generated += batch_size

    print(f"Generated {generated} images to {args.output_dir}")


if __name__ == "__main__":
    main()
