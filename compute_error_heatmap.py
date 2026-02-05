#!/usr/bin/env python3
"""
Compute 2-Scale Error Heatmap using Hutchinson Estimator for LIFT Dual Timestep Model.

For each (γ₀, γ₁) combination, computes the discretization error (vHv)
per scale. This helps identify the optimal noise schedule for generation.

Following Greg Ver Steeg's observation, we apply the chain-rule factor:
    γ(SNR) = 1 / (SNR * (1 + SNR))

This ensures the error decays as SNR^{-2} at high SNR, matching the
theoretical expectation from stochastic localization.

Usage:
    python compute_error_heatmap.py --checkpoint checkpoints/lift_full_random_final.pth --device 0
"""

import os
import argparse
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.func import jvp, vmap

from model import LIFTDualTimestepModel
from scheduler import DDIMScheduler
from data import AFHQ64Dataset


def chain_rule_factor(snr):
    """
    Compute the chain-rule correction factor.

    The model operates on x_t = α*x_0 + σ*ε (DDIM parameterization).
    To convert Jacobian from ∂x̂/∂x_t to ∂x̂/∂z where z = SNR*x_0 + √SNR*ε,
    we need to multiply by:

        γ(SNR) = 1 / (SNR * (1 + SNR))

    This is because:
        x_t = z / √(SNR * (1 + SNR))
        ∂x_t/∂z = 1 / √(SNR * (1 + SNR))

    And for element-wise squared Jacobian:
        (J_z)² = (J_{x_t})² / (SNR * (1 + SNR))
    """
    return 1.0 / (snr * (1.0 + snr))


def snr_to_timestep(snr, scheduler):
    """
    Convert SNR to approximate timestep.

    In DDPM: SNR = alpha_bar / (1 - alpha_bar)
    So: alpha_bar = SNR / (SNR + 1)
    """
    alpha_bar_target = snr / (snr + 1.0)
    alphas_cumprod = scheduler.alphas_cumprod.cpu().numpy()
    idx = np.argmin(np.abs(alphas_cumprod - alpha_bar_target))
    return idx


def add_noise_at_timestep(x, t, scheduler):
    """Add noise at given timestep using scheduler."""
    noise = torch.randn_like(x)
    t_tensor = torch.tensor([t], device=x.device).expand(x.shape[0])
    noisy = scheduler.add_noise(x, noise, t_tensor)
    return noisy


def get_Hv_vmap_4d(f, z, v, K=8, chunk_size=None):
    """
    Compute v^T (J ⊙ J) v using Hutchinson estimator.

    This computes the Jacobian with respect to the model's input (x_t).
    The chain-rule factor must be applied separately to convert to z-space.
    """
    z, v = z.detach(), v.detach()
    scale = torch.sqrt(v).unsqueeze(0)

    eps = torch.empty((K, *z.shape), device=z.device, dtype=z.dtype).bernoulli_(0.5).mul_(2).add_(-1)
    u_batch = eps * scale

    def single_sample_jvp(u):
        return jvp(f, (z,), (u,))[1]

    batched_out = vmap(single_sample_jvp, chunk_size=chunk_size)(u_batch)
    return (batched_out ** 2).mean(dim=0)


def get_vHv(f, z, v, K=8, chunk_size=None):
    """Compute vHv (scalar per batch element)."""
    hv = get_Hv_vmap_4d(f, z, v, K=K, chunk_size=chunk_size)
    hv = hv * v.expand_as(hv)
    return hv.sum(dim=(1, 2, 3))


def compute_error_heatmap(model, scheduler, x_batch, gamma_range, num_points=20, K=4, device='cpu', apply_chain_rule=True):
    """
    Compute error heatmap for all (γ₀, γ₁) combinations.

    Args:
        model: Trained LIFTDualTimestepModel
        scheduler: DDIM scheduler
        x_batch: Clean image batch [B, 3, 64, 64]
        gamma_range: [min_gamma, max_gamma]
        num_points: Number of points per axis
        K: Hutchinson samples
        apply_chain_rule: Whether to apply the chain-rule factor γ(SNR)

    Returns:
        gamma_grid, error_scale0, error_scale1, error_total
    """
    gamma_grid = torch.logspace(
        math.log10(gamma_range[0]),
        math.log10(gamma_range[1]),
        steps=num_points,
        device=device,
    )

    error_scale0 = torch.zeros(num_points, num_points, device=device)
    error_scale1 = torch.zeros(num_points, num_points, device=device)

    # Create 32x32 version of clean images
    x_batch_32 = F.interpolate(x_batch, size=(32, 32), mode='bilinear', align_corners=False)

    model.eval()

    total_iters = num_points * num_points
    pbar = tqdm(total=total_iters, desc="Computing error heatmap")

    for i, g0 in enumerate(gamma_grid):
        # Convert gamma to timestep for scale 0 (64x64)
        t_64 = snr_to_timestep(g0.item(), scheduler)

        # Chain-rule factor for scale 0
        gamma_factor_0 = chain_rule_factor(g0.item()) if apply_chain_rule else 1.0

        for j, g1 in enumerate(gamma_grid):
            # Convert gamma to timestep for scale 1 (32x32)
            t_32 = snr_to_timestep(g1.item(), scheduler)

            # Chain-rule factor for scale 1
            gamma_factor_1 = chain_rule_factor(g1.item()) if apply_chain_rule else 1.0

            # Add noise at respective timesteps
            z_64 = add_noise_at_timestep(x_batch, t_64, scheduler)
            z_32 = add_noise_at_timestep(x_batch_32, t_32, scheduler)

            t_64_tensor = torch.tensor([t_64], device=device).expand(x_batch.shape[0])
            t_32_tensor = torch.tensor([t_32], device=device).expand(x_batch.shape[0])

            # Define function for 64x64 scale (varying z_64, fixed z_32)
            def f_64(z_in):
                noise_pred_64, _ = model(z_in, z_32, t_64_tensor, t_32_tensor)
                return noise_pred_64

            # Define function for 32x32 scale (fixed z_64, varying z_32)
            def f_32(z_in):
                _, noise_pred_32 = model(z_64, z_in, t_64_tensor, t_32_tensor)
                return noise_pred_32

            v_64 = torch.ones_like(z_64[:, :1]) / (z_64.shape[-2] * z_64.shape[-1])
            v_32 = torch.ones_like(z_32[:, :1]) / (z_32.shape[-2] * z_32.shape[-1])

            with torch.no_grad():
                # Compute vHv in x_t space
                vhv_64_xt = get_vHv(f_64, z_64, v_64, K=K, chunk_size=None)
                vhv_32_xt = get_vHv(f_32, z_32, v_32, K=K, chunk_size=None)

                # Apply chain-rule factor to convert to z-space (SNR parameterization)
                error_scale0[i, j] = vhv_64_xt.mean() * gamma_factor_0
                error_scale1[i, j] = vhv_32_xt.mean() * gamma_factor_1

            pbar.update(1)

    pbar.close()

    error_total = error_scale0 + error_scale1

    return gamma_grid.cpu(), error_scale0.cpu(), error_scale1.cpu(), error_total.cpu()


def plot_heatmaps(gamma_grid, error_scale0, error_scale1, error_total, output_path, title_suffix=""):
    """Plot error heatmaps."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    gamma_np = gamma_grid.numpy()

    extent = [np.log10(gamma_np[0]), np.log10(gamma_np[-1]),
              np.log10(gamma_np[0]), np.log10(gamma_np[-1])]

    titles = ['Scale 0 (64×64) Error', 'Scale 1 (32×32) Error', 'Total Error']
    data = [error_scale0.numpy(), error_scale1.numpy(), error_total.numpy()]

    for ax, title, d in zip(axes, titles, data):
        d_log = np.log10(d + 1e-10)

        im = ax.imshow(d_log.T, origin='lower', extent=extent, aspect='auto', cmap='viridis')
        ax.set_xlabel('log₁₀(γ₀) - Scale 0 SNR')
        ax.set_ylabel('log₁₀(γ₁) - Scale 1 SNR')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='log₁₀(vHv)')

    plt.suptitle(f'2-Scale Discretization Error Heatmap{title_suffix}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Heatmap saved to: {output_path}")
    plt.close()


def plot_1d_error_curve(gamma_grid, error_scale0, error_scale1, output_path):
    """
    Plot 1D error curves along the diagonal (γ₀ = γ₁).

    This helps verify that error decays as SNR^{-2} at high SNR.
    """
    gamma_np = gamma_grid.numpy()
    n = len(gamma_np)

    # Extract diagonal (γ₀ = γ₁)
    error_diag_0 = np.array([error_scale0[i, i].item() for i in range(n)])
    error_diag_1 = np.array([error_scale1[i, i].item() for i in range(n)])
    error_diag_total = error_diag_0 + error_diag_1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Error vs SNR
    axes[0].loglog(gamma_np, error_diag_0, 'b-o', label='Scale 0 (64×64)', markersize=4)
    axes[0].loglog(gamma_np, error_diag_1, 'r-s', label='Scale 1 (32×32)', markersize=4)
    axes[0].loglog(gamma_np, error_diag_total, 'k--', label='Total', linewidth=2)

    # Add reference line for SNR^{-2} decay
    snr_ref = gamma_np
    ref_line = error_diag_total[0] * (snr_ref[0] / snr_ref) ** 2
    axes[0].loglog(snr_ref, ref_line, 'g:', label='∝ SNR⁻²', linewidth=2)

    axes[0].set_xlabel('SNR (γ)')
    axes[0].set_ylabel('Error (vHv)')
    axes[0].set_title('Error vs SNR (Diagonal: γ₀ = γ₁)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: Error * SNR^2 (should be roughly constant at high SNR)
    scaled_error_0 = error_diag_0 * gamma_np ** 2
    scaled_error_1 = error_diag_1 * gamma_np ** 2
    scaled_error_total = error_diag_total * gamma_np ** 2

    axes[1].semilogx(gamma_np, scaled_error_0, 'b-o', label='Scale 0 (64×64)', markersize=4)
    axes[1].semilogx(gamma_np, scaled_error_1, 'r-s', label='Scale 1 (32×32)', markersize=4)
    axes[1].semilogx(gamma_np, scaled_error_total, 'k--', label='Total', linewidth=2)

    axes[1].set_xlabel('SNR (γ)')
    axes[1].set_ylabel('Error × SNR²')
    axes[1].set_title('Scaled Error (should plateau at high SNR)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"1D error curve saved to: {output_path}")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Compute error heatmap for LIFT Dual Timestep')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--cache_dir', type=str, default=None, help='HuggingFace cache directory')
    parser.add_argument('--device', type=int, default=0, help='CUDA device')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of images for estimation')
    parser.add_argument('--gamma_min', type=float, default=1e-2, help='Minimum gamma (SNR)')
    parser.add_argument('--gamma_max', type=float, default=1e2, help='Maximum gamma (SNR)')
    parser.add_argument('--num_points', type=int, default=15, help='Number of points per axis')
    parser.add_argument('--K', type=int, default=4, help='Hutchinson samples')
    parser.add_argument('--output', type=str, default='results/error_heatmap.png', help='Output path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_chain_rule', action='store_true', help='Disable chain-rule factor (for comparison)')
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
    print(f"Hidden dims: {hidden_dims}")
    print(f"Training mode: {checkpoint.get('training_mode', 'N/A')}")

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

    # Load dataset
    print("Loading dataset...")
    dataset = AFHQ64Dataset(split='train', cache_dir=args.cache_dir)

    indices = np.random.choice(len(dataset), args.batch_size, replace=False)
    x_batch = torch.stack([dataset[int(i)] for i in indices], dim=0).to(device)
    print(f"Batch shape: {x_batch.shape}")

    apply_chain_rule = not args.no_chain_rule
    print(f"\nChain-rule factor: {'ENABLED' if apply_chain_rule else 'DISABLED'}")
    print(f"  γ(SNR) = 1 / (SNR × (1 + SNR))")

    # Compute error heatmap
    print(f"\nComputing error heatmap...")
    print(f"Gamma range: [{args.gamma_min}, {args.gamma_max}]")
    print(f"Grid size: {args.num_points} × {args.num_points}")
    print(f"Hutchinson samples: {args.K}")

    gamma_grid, error_scale0, error_scale1, error_total = compute_error_heatmap(
        model, scheduler, x_batch,
        gamma_range=[args.gamma_min, args.gamma_max],
        num_points=args.num_points,
        K=args.K,
        device=device,
        apply_chain_rule=apply_chain_rule
    )

    # Save results
    results = {
        'gamma_grid': gamma_grid,
        'error_scale0': error_scale0,
        'error_scale1': error_scale1,
        'error_total': error_total,
        'apply_chain_rule': apply_chain_rule,
        'args': vars(args)
    }
    results_path = args.output.replace('.png', '.pth')
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(results, results_path)
    print(f"Results saved to: {results_path}")

    # Plot heatmaps
    title_suffix = " (with chain-rule)" if apply_chain_rule else " (without chain-rule)"
    plot_heatmaps(gamma_grid, error_scale0, error_scale1, error_total, args.output, title_suffix)

    # Plot 1D error curve along diagonal
    curve_path = args.output.replace('.png', '_1d_curve.png')
    plot_1d_error_curve(gamma_grid, error_scale0, error_scale1, curve_path)

    # Print summary
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    print(f"Scale 0 (64×64) error range: [{error_scale0.min():.6e}, {error_scale0.max():.6e}]")
    print(f"Scale 1 (32×32) error range: [{error_scale1.min():.6e}, {error_scale1.max():.6e}]")
    print(f"Total error range: [{error_total.min():.6e}, {error_total.max():.6e}]")


if __name__ == "__main__":
    main()
