#!/usr/bin/env python3
"""
Evaluate FID in memory without saving individual images.

Only saves a 9x9 grid for visualization.

Usage:
    python eval_fid_memory.py --checkpoint checkpoints/baseline_100ep.pth --model_type baseline
    python eval_fid_memory.py --checkpoint checkpoints/lift_dual_timestep_100ep.pth --model_type lift
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3

from baseline_model import BaselineModel
from model import LIFTDualTimestepModel
from scheduler import DDIMScheduler
from data import AFHQ64Dataset


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
def generate_batch_baseline(model, scheduler, batch_size, num_steps, device, eta=0.0):
    """Generate a batch using baseline model."""
    scheduler._set_timesteps(num_steps)
    timesteps = scheduler.timesteps

    x = torch.randn(batch_size, 3, 64, 64, device=device)

    for i, t in enumerate(timesteps):
        t_tensor = torch.tensor([t], device=device).expand(batch_size).float()
        noise_pred = model(x, t_tensor)

        prev_t = timesteps[i + 1] if i < len(timesteps) - 1 else 0
        x = ddim_step(scheduler, noise_pred, t, prev_t, x, eta)

    return (x + 1) * 0.5


@torch.no_grad()
def generate_batch_lift(model, scheduler, batch_size, num_steps, device, eta=0.0):
    """Generate a batch using LIFT model (diagonal path)."""
    scheduler._set_timesteps(num_steps)
    timesteps = scheduler.timesteps

    x_64 = torch.randn(batch_size, 3, 64, 64, device=device)
    x_32 = torch.randn(batch_size, 3, 32, 32, device=device)

    for i, t in enumerate(timesteps):
        t_tensor = torch.tensor([t], device=device).expand(batch_size).float()
        noise_pred_64, noise_pred_32 = model(x_64, x_32, t_tensor, t_tensor)

        prev_t = timesteps[i + 1] if i < len(timesteps) - 1 else 0
        x_64 = ddim_step(scheduler, noise_pred_64, t, prev_t, x_64, eta)
        x_32 = ddim_step(scheduler, noise_pred_32, t, prev_t, x_32, eta)

    return (x_64 + 1) * 0.5


def get_inception_features(images, inception_model, device, batch_size=64):
    """Extract Inception features from images in memory."""
    # images: [N, 3, 64, 64] in [0, 1]
    # Resize to 299x299 for Inception
    features_list = []

    num_batches = (len(images) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Extracting features"):
        start = i * batch_size
        end = min(start + batch_size, len(images))
        batch = images[start:end].to(device)

        # Resize to 299x299
        batch_resized = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)

        # Get features
        with torch.no_grad():
            feat = inception_model(batch_resized)[0]

        # Pool if needed
        if feat.dim() == 4:
            feat = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1)

        features_list.append(feat.cpu())

    return torch.cat(features_list, dim=0).numpy()


def calculate_fid_from_features(feat_real, feat_gen):
    """Calculate FID from pre-computed features."""
    mu_real = np.mean(feat_real, axis=0)
    sigma_real = np.cov(feat_real, rowvar=False)

    mu_gen = np.mean(feat_gen, axis=0)
    sigma_gen = np.cov(feat_gen, rowvar=False)

    return fid_score.calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)


def save_grid(images, output_path, nrow=9):
    """Save a grid of images."""
    n = min(nrow * nrow, len(images))
    images = images[:n]

    # Make grid
    nrow = int(np.ceil(np.sqrt(n)))
    ncol = int(np.ceil(n / nrow))

    h, w = images.shape[2], images.shape[3]
    grid = np.zeros((nrow * h, ncol * w, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        row = i // ncol
        col = i % ncol
        img_np = img.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = img_np

    Image.fromarray(grid).save(output_path)
    print(f"Saved grid: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate FID in memory')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model_type', type=str, choices=['baseline', 'lift'], required=True)
    parser.add_argument('--num_images', type=int, default=15803, help='Number of images to generate')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_steps', type=int, default=18)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--cache_dir', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    print("Starting evaluation...", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hidden_dims = checkpoint.get('hidden_dims', [64, 128, 256, 512])

    if args.model_type == 'baseline':
        model = BaselineModel(hidden_dims=hidden_dims)
        generate_fn = generate_batch_baseline
    else:
        model = LIFTDualTimestepModel(hidden_dims=hidden_dims)
        generate_fn = generate_batch_lift

    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device).eval()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Scheduler
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="cosine", clip_sample=True)

    # Load Inception model
    print("Loading Inception model...")
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()

    # Load real images and extract features
    print(f"Loading real images...", flush=True)
    dataset = AFHQ64Dataset(split='train', cache_dir=args.cache_dir)
    num_real = min(args.num_images, len(dataset))
    print(f"Dataset loaded, {num_real} images to process", flush=True)

    real_images_list = []
    for i in range(num_real):
        real_images_list.append(dataset[i])
        if (i + 1) % 1000 == 0:
            print(f"  Loaded {i+1}/{num_real} real images", flush=True)

    real_images = torch.stack(real_images_list)
    real_images = (real_images + 1) * 0.5  # [-1,1] -> [0,1]
    del real_images_list
    print(f"Real images loaded: {real_images.shape}", flush=True)

    print(f"Extracting real features ({num_real} images)...")
    feat_real = get_inception_features(real_images, inception, device, batch_size=64)
    del real_images  # Free memory

    # Generate images and extract features
    print(f"\nGenerating {args.num_images} images...")
    gen_images_list = []
    num_batches = (args.num_images + args.batch_size - 1) // args.batch_size
    generated = 0

    for _ in tqdm(range(num_batches), desc="Generating"):
        batch_size = min(args.batch_size, args.num_images - generated)
        images = generate_fn(model, scheduler, batch_size, args.num_steps, device, args.eta)
        gen_images_list.append(images.cpu())
        generated += batch_size

    gen_images = torch.cat(gen_images_list, dim=0)
    del gen_images_list

    # Save 9x9 grid
    os.makedirs(args.output_dir, exist_ok=True)
    epoch = args.checkpoint.split('_')[-1].replace('ep.pth', '')
    grid_path = os.path.join(args.output_dir, f'grid_{args.model_type}_{epoch}ep.png')
    save_grid(gen_images, grid_path, nrow=9)

    # Extract generated features
    print(f"Extracting generated features ({len(gen_images)} images)...")
    feat_gen = get_inception_features(gen_images, inception, device, batch_size=64)
    del gen_images

    # Calculate FID
    print("\nCalculating FID...")
    fid = calculate_fid_from_features(feat_real, feat_gen)
    print(f"\n{'='*40}")
    print(f"FID: {fid:.2f}")
    print(f"{'='*40}")

    # Save result
    result_file = os.path.join(args.output_dir, f'fid_{args.model_type}_{epoch}ep.txt')
    with open(result_file, 'w') as f:
        f.write(f"FID: {fid:.4f}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Num images: {args.num_images}\n")
        f.write(f"Num steps: {args.num_steps}\n")
    print(f"Result saved: {result_file}")

    return fid


if __name__ == "__main__":
    main()
