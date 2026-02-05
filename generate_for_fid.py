#!/usr/bin/env python3
"""
Generate images for FID evaluation.

Supports both diagonal path and DP optimal path generation.

Usage:
    # Diagonal path
    python generate_for_fid.py --checkpoint checkpoints/lift_full_random_final.pth \
        --output_dir results/fid_diagonal --num_images 1000 --mode diagonal

    # DP optimal path
    python generate_for_fid.py --checkpoint checkpoints/lift_full_random_final.pth \
        --output_dir results/fid_dp --num_images 1000 --mode dp \
        --heatmap results/error_heatmap_chainrule.pth
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image

from model import LIFTDualTimestepModel
from scheduler import DDIMScheduler


def snr_to_timestep(snr, scheduler):
    """Convert SNR to timestep."""
    alpha_bar_target = snr / (snr + 1.0)
    alphas_cumprod = scheduler.alphas_cumprod.cpu().numpy()
    idx = np.argmin(np.abs(alphas_cumprod - alpha_bar_target))
    return int(idx)


def compute_dp_optimal_path(gamma_grid, error_matrix):
    """
    Compute DP optimal path on given error matrix.

    Args:
        gamma_grid: Array of gamma values
        error_matrix: Error matrix to optimize (can be error_total, error_scale0, etc.)

    Returns:
        optimal_gamma1: Optimal γ₁ for each γ₀
        total_cost: Total accumulated error along the path
    """
    num_points = len(gamma_grid)
    gamma_np = gamma_grid if isinstance(gamma_grid, np.ndarray) else gamma_grid.numpy()
    error_np = error_matrix if isinstance(error_matrix, np.ndarray) else error_matrix.numpy()

    dp = np.full((num_points, num_points), np.inf)
    parent = np.zeros((num_points, num_points), dtype=int)
    dp[0, 0] = error_np[0, 0]

    for i in range(1, num_points):
        min_dp_so_far = np.inf
        best_k = 0
        for j in range(num_points):
            if dp[i-1, j] < min_dp_so_far:
                min_dp_so_far = dp[i-1, j]
                best_k = j
            dp[i, j] = min_dp_so_far + error_np[i, j]
            parent[i, j] = best_k

    best_end_j = np.argmin(dp[num_points - 1, :])
    total_cost = dp[num_points - 1, best_end_j]

    path_j = [best_end_j]
    current_j = best_end_j
    for i in range(num_points - 1, 0, -1):
        current_j = parent[i, current_j]
        path_j.append(current_j)
    path_j = path_j[::-1]

    optimal_gamma1 = np.array([gamma_np[j] for j in path_j])
    return optimal_gamma1, total_cost


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
def generate_batch_diagonal(model, scheduler, batch_size, num_steps, device, eta=0.0):
    """Generate with diagonal path (γ₁ = γ₀)."""
    # Standard DDIM timesteps
    scheduler._set_timesteps(num_steps)
    timesteps = scheduler.timesteps

    x_64 = torch.randn(batch_size, 3, 64, 64, device=device)
    x_32 = torch.randn(batch_size, 3, 32, 32, device=device)

    for i, t in enumerate(timesteps):
        t_tensor = torch.tensor([t], device=device).expand(batch_size)

        # Same timestep for both scales (diagonal path)
        noise_pred_64, noise_pred_32 = model(x_64, x_32, t_tensor, t_tensor)

        prev_t = timesteps[i + 1] if i < len(timesteps) - 1 else 0
        x_64 = ddim_step(scheduler, noise_pred_64, t, prev_t, x_64, eta)
        x_32 = ddim_step(scheduler, noise_pred_32, t, prev_t, x_32, eta)

    x_64 = (x_64 + 1) * 0.5
    return x_64


@torch.no_grad()
def generate_batch_dp(model, scheduler, batch_size, num_steps, device,
                      gamma_grid, optimal_gamma1, eta=0.0):
    """Generate with DP optimal path."""
    gamma_np = gamma_grid if isinstance(gamma_grid, np.ndarray) else gamma_grid.numpy()

    # Log-uniform γ₀ schedule
    gamma0_schedule = np.logspace(np.log10(gamma_np[0]), np.log10(gamma_np[-1]), num_steps)

    # Interpolate γ₁ from DP path
    log_gamma = np.log10(gamma_np)
    log_gamma1_orig = np.log10(optimal_gamma1)
    gamma1_schedule = np.array([
        10 ** np.interp(np.log10(g0), log_gamma, log_gamma1_orig)
        for g0 in gamma0_schedule
    ])

    # Convert to timesteps
    timesteps_64 = [snr_to_timestep(g, scheduler) for g in gamma0_schedule]
    timesteps_32 = [snr_to_timestep(g, scheduler) for g in gamma1_schedule]

    x_64 = torch.randn(batch_size, 3, 64, 64, device=device)
    x_32 = torch.randn(batch_size, 3, 32, 32, device=device)

    for i in range(num_steps):
        t_64 = timesteps_64[i]
        t_32 = timesteps_32[i]

        t_64_tensor = torch.tensor([t_64], device=device).expand(batch_size)
        t_32_tensor = torch.tensor([t_32], device=device).expand(batch_size)

        noise_pred_64, noise_pred_32 = model(x_64, x_32, t_64_tensor, t_32_tensor)

        prev_t_64 = timesteps_64[i + 1] if i < num_steps - 1 else 0
        prev_t_32 = timesteps_32[i + 1] if i < num_steps - 1 else 0

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
    parser = argparse.ArgumentParser(description='Generate images for FID evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_images', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--mode', type=str, choices=['diagonal', 'dp', 'dp_64'], required=True,
                        help='diagonal: γ₁=γ₀, dp: optimize total error, dp_64: optimize 64×64 error only')
    parser.add_argument('--heatmap', type=str, default=None, help='Required for dp mode')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == 'dp' and args.heatmap is None:
        raise ValueError("--heatmap is required for dp mode")
    if args.mode == 'dp_64' and args.heatmap is None:
        raise ValueError("--heatmap is required for dp_64 mode")

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

    # Create scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="cosine",
        clip_sample=True
    )

    # Load DP path if needed
    gamma_grid = None
    optimal_gamma1 = None
    if args.mode == 'dp':
        print(f"Loading heatmap: {args.heatmap}")
        heatmap_data = torch.load(args.heatmap, weights_only=False)
        gamma_grid = heatmap_data['gamma_grid']
        error_total = heatmap_data['error_total']
        optimal_gamma1, total_cost = compute_dp_optimal_path(gamma_grid, error_total)
        print(f"DP path (total error): γ₁ range {optimal_gamma1[0]:.4f} -> {optimal_gamma1[-1]:.4f}")
        print(f"Total cost: {total_cost:.6e}")
    elif args.mode == 'dp_64':
        print(f"Loading heatmap: {args.heatmap}")
        heatmap_data = torch.load(args.heatmap, weights_only=False)
        gamma_grid = heatmap_data['gamma_grid']
        error_scale0 = heatmap_data['error_scale0']  # 64×64 error only
        optimal_gamma1, total_cost = compute_dp_optimal_path(gamma_grid, error_scale0)
        print(f"DP path (64×64 error only): γ₁ range {optimal_gamma1[0]:.4f} -> {optimal_gamma1[-1]:.4f}")
        print(f"Total cost: {total_cost:.6e}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate images
    print(f"\nGenerating {args.num_images} images with {args.mode} path...")
    print(f"Output directory: {args.output_dir}")

    num_batches = (args.num_images + args.batch_size - 1) // args.batch_size
    generated = 0

    for batch_idx in tqdm(range(num_batches), desc="Generating"):
        current_batch_size = min(args.batch_size, args.num_images - generated)

        if args.mode == 'diagonal':
            images = generate_batch_diagonal(
                model, scheduler, current_batch_size, args.num_steps, device, args.eta
            )
        else:  # dp or dp_64
            images = generate_batch_dp(
                model, scheduler, current_batch_size, args.num_steps, device,
                gamma_grid, optimal_gamma1, args.eta
            )

        save_images(images, args.output_dir, generated)
        generated += current_batch_size

    print(f"\nGenerated {generated} images to {args.output_dir}")


if __name__ == "__main__":
    main()
