#!/usr/bin/env python3
"""
Generate images using Baseline model for FID evaluation.

Usage:
    python generate_baseline_for_fid.py --checkpoint checkpoints/baseline_final.pth \
        --output_dir results/fid_baseline --num_images 1000
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from baseline_model import BaselineModel
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
def generate_batch(model, scheduler, batch_size, num_steps, device, eta=0.0):
    """Generate a batch of images using DDIM."""
    scheduler._set_timesteps(num_steps)
    timesteps = scheduler.timesteps

    x = torch.randn(batch_size, 3, 64, 64, device=device)

    for i, t in enumerate(timesteps):
        t_tensor = torch.tensor([t], device=device).expand(batch_size)
        noise_pred = model(x, t_tensor)

        prev_t = timesteps[i + 1] if i < len(timesteps) - 1 else 0
        x = ddim_step(scheduler, noise_pred, t, prev_t, x, eta)

    x = (x + 1) * 0.5
    return x


def save_images(images, output_dir, start_idx):
    """Save batch of images as PNG files."""
    for i, img in enumerate(images):
        img_np = img.cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        img_pil.save(os.path.join(output_dir, f'{start_idx + i:05d}.png'))


def parse_args():
    parser = argparse.ArgumentParser(description='Generate images with Baseline model for FID')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_images', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_steps', type=int, default=50)
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

    model = BaselineModel(hidden_dims=hidden_dims)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="cosine",
        clip_sample=True
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate images
    print(f"\nGenerating {args.num_images} images...")
    print(f"Output directory: {args.output_dir}")

    num_batches = (args.num_images + args.batch_size - 1) // args.batch_size
    generated = 0

    for batch_idx in tqdm(range(num_batches), desc="Generating"):
        current_batch_size = min(args.batch_size, args.num_images - generated)

        images = generate_batch(
            model, scheduler, current_batch_size, args.num_steps, device, args.eta
        )

        save_images(images, args.output_dir, generated)
        generated += current_batch_size

    print(f"\nGenerated {generated} images to {args.output_dir}")


if __name__ == "__main__":
    main()
