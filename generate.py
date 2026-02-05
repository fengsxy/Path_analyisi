#!/usr/bin/env python3
"""
Image Generation Script for LIFT Baseline Model.

Generates images using the dual-scale LIFT Baseline model.
Supports both standard generation and DP-optimal path generation.

Usage:
    python generate.py --checkpoint /path/to/checkpoint.pth --num_images 8
    python generate.py --checkpoint /path/to/checkpoint.pth --num_images 8 --dp_path results/error_heatmap_chainrule.pth
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import LIFTDualTimestepModel
from scheduler import DDIMScheduler
from data import to_image


@torch.no_grad()
def generate_lift_baseline(model, scheduler, num_images, num_steps, device, eta=0.0):
    """
    Generate images using LIFT Baseline model.

    The model processes both 64×64 and 32×32 scales simultaneously.
    We denoise both scales in parallel using the same timesteps.
    """
    scheduler._set_timesteps(num_steps)

    # Initialize with random noise
    x_64 = torch.randn(num_images, 3, 64, 64, device=device)
    x_32 = torch.randn(num_images, 3, 32, 32, device=device)

    for t in tqdm(scheduler.timesteps, desc="Generating"):
        t_tensor = torch.tensor([t], device=device).expand(num_images)

        # Get noise predictions for both scales
        noise_pred_64, noise_pred_32 = model(x_64, x_32, t_tensor)

        # Denoise both scales
        x_64 = scheduler._step(noise_pred_64, t, x_64, eta=eta)
        x_32 = scheduler._step(noise_pred_32, t, x_32, eta=eta)

    # Unnormalize to [0, 1]
    x_64 = (x_64 + 1) * 0.5
    x_32 = (x_32 + 1) * 0.5

    return x_64, x_32


def snr_to_timestep(snr, scheduler):
    """
    Convert SNR to approximate timestep.

    In DDPM: SNR = alpha_bar / (1 - alpha_bar)
    So: alpha_bar = SNR / (SNR + 1)
    """
    alpha_bar_target = snr / (snr + 1.0)
    alphas_cumprod = scheduler.alphas_cumprod.cpu().numpy()
    idx = np.argmin(np.abs(alphas_cumprod - alpha_bar_target))
    return int(idx)


def compute_dp_optimal_path(gamma_grid, error_total):
    """
    Compute optimal path using Dynamic Programming on total error.

    Returns:
        path_j: Array of γ₁ indices along optimal path
        optimal_gamma1: Optimal γ₁ for each γ₀
    """
    num_points = len(gamma_grid)
    error_np = error_total if isinstance(error_total, np.ndarray) else error_total.numpy()

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

    path_j = [best_end_j]
    current_j = best_end_j

    for i in range(num_points - 1, 0, -1):
        current_j = parent[i, current_j]
        path_j.append(current_j)

    path_j = path_j[::-1]

    gamma_np = gamma_grid if isinstance(gamma_grid, np.ndarray) else gamma_grid.numpy()
    optimal_gamma1 = np.array([gamma_np[j] for j in path_j])

    return np.array(path_j), optimal_gamma1


def interpolate_gamma1_schedule(gamma_grid, optimal_gamma1, num_steps):
    """
    Interpolate the DP optimal γ₁ schedule to num_steps points.

    Args:
        gamma_grid: Original γ₀ grid from heatmap
        optimal_gamma1: Optimal γ₁ for each γ₀ from DP
        num_steps: Number of generation steps

    Returns:
        gamma0_schedule: γ₀ values for each step (high SNR to low SNR for generation)
        gamma1_schedule: γ₁ values for each step (interpolated from DP path)
    """
    gamma_np = gamma_grid if isinstance(gamma_grid, np.ndarray) else gamma_grid.numpy()

    # For generation: go from high SNR (clean) to low SNR (noisy) in reverse
    # Actually generation goes from noisy to clean, so:
    # Start at low SNR (high noise), end at high SNR (low noise)
    # But DDIM steps go from high timestep to low timestep
    # High timestep = low SNR = high noise
    # Low timestep = high SNR = low noise

    # Create γ₀ schedule: from high SNR to low SNR (will be reversed for generation)
    gamma0_schedule = np.logspace(
        np.log10(gamma_np[-1]),  # High SNR (low noise)
        np.log10(gamma_np[0]),   # Low SNR (high noise)
        num_steps
    )

    # Interpolate γ₁ from the DP optimal path
    log_gamma0_orig = np.log10(gamma_np)
    log_gamma1_orig = np.log10(optimal_gamma1)
    log_gamma0_new = np.log10(gamma0_schedule)

    log_gamma1_new = np.interp(log_gamma0_new, log_gamma0_orig, log_gamma1_orig)
    gamma1_schedule = 10 ** log_gamma1_new

    return gamma0_schedule, gamma1_schedule


def compute_adaptive_schedule(gamma_grid, error_total, optimal_gamma1, num_steps):
    """
    Compute schedule based on DP optimal path.

    Uses log-uniform distribution for γ₀, and interpolates γ₁ from DP path.

    Generation order: high timestep (low SNR, high noise) -> low timestep (high SNR, low noise)
    So γ₀ should go from low to high.

    Args:
        gamma_grid: γ grid from heatmap
        error_total: Total error matrix
        optimal_gamma1: DP optimal γ₁ for each γ₀
        num_steps: Total number of generation steps

    Returns:
        gamma0_schedule: γ₀ values for each step (low SNR to high SNR)
        gamma1_schedule: γ₁ values for each step
    """
    gamma_np = gamma_grid if isinstance(gamma_grid, np.ndarray) else gamma_grid.numpy()

    # Use log-uniform distribution for γ₀
    gamma0_schedule = np.logspace(
        np.log10(gamma_np[0]),   # Low SNR (high noise)
        np.log10(gamma_np[-1]),  # High SNR (low noise)
        num_steps
    )

    # Interpolate γ₁ from DP optimal path
    log_gamma = np.log10(gamma_np)
    log_gamma1_orig = np.log10(optimal_gamma1)

    gamma1_schedule = []
    for g0 in gamma0_schedule:
        log_g1 = np.interp(np.log10(g0), log_gamma, log_gamma1_orig)
        gamma1_schedule.append(10 ** log_g1)

    gamma0_schedule = np.array(gamma0_schedule)
    gamma1_schedule = np.array(gamma1_schedule)

    return gamma0_schedule, gamma1_schedule


@torch.no_grad()
def generate_with_dp_path(model, scheduler, num_images, num_steps, device,
                          gamma_grid, error_total, optimal_gamma1, eta=0.0,
                          adaptive=True):
    """
    Generate images using DP optimal path for dual-scale scheduling.

    The key insight: 64×64 and 32×32 scales use different timesteps based on
    the DP optimal path that minimizes total error.

    Args:
        model: LIFTDualTimestepModel
        scheduler: DDIMScheduler
        num_images: Number of images to generate
        num_steps: Number of generation steps
        device: torch device
        gamma_grid: γ grid from heatmap
        error_total: Total error matrix (for adaptive scheduling)
        optimal_gamma1: DP optimal γ₁ for each γ₀
        eta: DDIM eta parameter
        adaptive: If True, use error-adaptive step distribution

    Returns:
        x_64: Generated 64×64 images
        x_32: Generated 32×32 images
    """
    # Get schedules
    if adaptive:
        gamma0_schedule, gamma1_schedule = compute_adaptive_schedule(
            gamma_grid, error_total, optimal_gamma1, num_steps
        )
        print("Using adaptive schedule (more steps where error is high)")
    else:
        gamma0_schedule, gamma1_schedule = interpolate_gamma1_schedule(
            gamma_grid, optimal_gamma1, num_steps
        )
        print("Using uniform schedule")

    # Convert γ (SNR) to timesteps
    # gamma0_schedule is already in generation order (high SNR -> low SNR)
    # which corresponds to low timestep -> high timestep
    # But we need high timestep -> low timestep for generation
    timesteps_64 = [snr_to_timestep(g, scheduler) for g in gamma0_schedule]
    timesteps_32 = [snr_to_timestep(g, scheduler) for g in gamma1_schedule]

    print(f"Timestep ranges:")
    print(f"  64×64: {timesteps_64[0]} -> {timesteps_64[-1]}")
    print(f"  32×32: {timesteps_32[0]} -> {timesteps_32[-1]}")
    print(f"  γ₀: {gamma0_schedule[0]:.4f} -> {gamma0_schedule[-1]:.4f}")
    print(f"  γ₁: {gamma1_schedule[0]:.4f} -> {gamma1_schedule[-1]:.4f}")

    # Initialize with random noise
    x_64 = torch.randn(num_images, 3, 64, 64, device=device)
    x_32 = torch.randn(num_images, 3, 32, 32, device=device)

    for i in tqdm(range(num_steps), desc="Generating (DP path)"):
        t_64 = timesteps_64[i]
        t_32 = timesteps_32[i]

        t_64_tensor = torch.tensor([t_64], device=device).expand(num_images)
        t_32_tensor = torch.tensor([t_32], device=device).expand(num_images)

        # Get noise predictions with different timesteps for each scale
        noise_pred_64, noise_pred_32 = model(x_64, x_32, t_64_tensor, t_32_tensor)

        # Compute prev timesteps
        if i < num_steps - 1:
            prev_t_64 = timesteps_64[i + 1]
            prev_t_32 = timesteps_32[i + 1]
        else:
            prev_t_64 = 0
            prev_t_32 = 0

        # Custom DDIM step for each scale
        x_64 = ddim_step(scheduler, noise_pred_64, t_64, prev_t_64, x_64, eta)
        x_32 = ddim_step(scheduler, noise_pred_32, t_32, prev_t_32, x_32, eta)

    # Unnormalize to [0, 1]
    x_64 = (x_64 + 1) * 0.5
    x_32 = (x_32 + 1) * 0.5

    return x_64, x_32


def ddim_step(scheduler, model_output, timestep, prev_timestep, sample, eta=0.0):
    """
    Single DDIM step with explicit prev_timestep.
    """
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else scheduler.final_alpha_cumprod
    )
    beta_prod_t = 1 - alpha_prod_t

    # Predict x_0
    pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5

    if scheduler.clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

    # Compute variance
    variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    std_dev_t = eta * variance**0.5

    # Predict noise direction
    pred_epsilon = (sample - alpha_prod_t**0.5 * pred_original_sample) / beta_prod_t**0.5

    # Compute prev sample
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * pred_epsilon
    prev_sample = alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction

    if eta > 0:
        noise = torch.randn_like(model_output)
        prev_sample += std_dev_t * noise

    return prev_sample


def plot_grid(images_64, images_32, output_path, title="Generated Images"):
    """Plot a grid of generated images showing both scales."""
    num_images = len(images_64)
    fig, axes = plt.subplots(num_images, 2, figsize=(6, 3 * num_images))

    if num_images == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_images):
        # 64x64 image
        img_64 = images_64[i].cpu().permute(1, 2, 0).numpy()
        img_64 = np.clip(img_64, 0, 1)
        axes[i, 0].imshow(img_64)
        axes[i, 0].set_title(f'64×64 #{i+1}')
        axes[i, 0].axis('off')

        # 32x32 image
        img_32 = images_32[i].cpu().permute(1, 2, 0).numpy()
        img_32 = np.clip(img_32, 0, 1)
        axes[i, 1].imshow(img_32)
        axes[i, 1].set_title(f'32×32 #{i+1}')
        axes[i, 1].axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.close()


def plot_single_scale(images, output_path, title="Generated Images", nrow=4):
    """Plot a grid of single-scale images."""
    num_images = len(images)
    ncol = min(nrow, num_images)
    nrow_actual = (num_images + ncol - 1) // ncol

    fig, axes = plt.subplots(nrow_actual, ncol, figsize=(3 * ncol, 3 * nrow_actual))

    if nrow_actual == 1 and ncol == 1:
        axes = np.array([[axes]])
    elif nrow_actual == 1:
        axes = axes.reshape(1, -1)
    elif ncol == 1:
        axes = axes.reshape(-1, 1)

    for i in range(nrow_actual):
        for j in range(ncol):
            idx = i * ncol + j
            if idx < num_images:
                img = images[idx].cpu().permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                axes[i, j].imshow(img)
                axes[i, j].set_title(f'#{idx+1}')
            axes[i, j].axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Generate images with LIFT model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--device', type=int, default=0, help='CUDA device')
    parser.add_argument('--num_images', type=int, default=8, help='Number of images to generate')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of sampling steps')
    parser.add_argument('--eta', type=float, default=0.0, help='DDIM eta (0=deterministic, 1=DDPM)')
    parser.add_argument('--output', type=str, default='results/generated.png', help='Output path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dp_path', type=str, default=None,
                        help='Path to error heatmap .pth file for DP optimal path generation')
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Get model configuration from checkpoint
    hidden_dims = checkpoint.get('hidden_dims', [64, 128, 256, 512])
    print(f"Hidden dims: {hidden_dims}")

    # Create model
    model = LIFTDualTimestepModel(hidden_dims=hidden_dims)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="cosine",
        clip_sample=True
    )

    # Generate images
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    if args.dp_path:
        # Load DP optimal path from heatmap
        print(f"\nLoading DP path from: {args.dp_path}")
        heatmap_data = torch.load(args.dp_path, weights_only=False)

        gamma_grid = heatmap_data['gamma_grid']
        error_total = heatmap_data['error_total']

        # Compute DP optimal path
        path_j, optimal_gamma1 = compute_dp_optimal_path(gamma_grid, error_total)

        print(f"DP optimal path computed")
        print(f"γ₁ range: {optimal_gamma1[0]:.4f} -> {optimal_gamma1[-1]:.4f}")

        print(f"\nGenerating {args.num_images} images with DP optimal path ({args.num_steps} steps)...")

        images_64, images_32 = generate_with_dp_path(
            model, scheduler, args.num_images, args.num_steps, device,
            gamma_grid, error_total, optimal_gamma1, eta=args.eta, adaptive=True
        )

        title = f"DP Optimal Path Generation ({args.num_steps} steps)"
    else:
        # Standard generation with same timesteps for both scales
        print(f"\nGenerating {args.num_images} images with standard method ({args.num_steps} steps)...")

        images_64, images_32 = generate_lift_baseline(
            model, scheduler, args.num_images, args.num_steps, device, eta=args.eta
        )

        title = f"Standard Generation ({args.num_steps} steps)"

    # Plot both scales side by side
    plot_grid(images_64, images_32, args.output, title)

    # Also save 64x64 images separately
    output_64 = args.output.replace('.png', '_64x64.png')
    plot_single_scale(images_64, output_64, "64×64 Generated Images")

    print("Done!")


if __name__ == "__main__":
    main()
