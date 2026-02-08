#!/usr/bin/env python3
"""
Cross-Scale Jacobian Analysis for LIFT Dual-Scale Diffusion Model.

Computes 4 Jacobian heatmaps on a 30×30 (t_64, t_32) grid:
  J_HH: ∂f_H/∂x_H  (self: 64→64)
  J_HL: ∂f_H/∂x_L  (cross: 32→64)
  J_LH: ∂f_L/∂x_H  (cross: 64→32)
  J_LL: ∂f_L/∂x_L  (self: 32→32)

Usage:
    python compute_cross_jacobian.py \
        --checkpoint checkpoints/lift_dual_timestep_400ep.pth \
        --output results/cross_jacobian_400ep.png \
        --device 0 --batch_size 16 --K 4
"""

import os
import argparse
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
    """Chain-rule correction: x_t space → z (SNR) space."""
    return 1.0 / (snr * (1.0 + snr))


def timestep_to_snr(t, scheduler):
    """Convert timestep to SNR."""
    if t >= len(scheduler.alphas_cumprod):
        t = len(scheduler.alphas_cumprod) - 1
    if t < 0:
        t = 0
    alpha_bar = scheduler.alphas_cumprod[int(t)]
    if alpha_bar >= 1.0:
        return 1e6
    if alpha_bar <= 0.0:
        return 1e-6
    return (alpha_bar / (1 - alpha_bar)).item()


def add_noise_at_timestep(x, t, scheduler):
    """Add noise at given timestep."""
    noise = torch.randn_like(x)
    t_tensor = torch.tensor([t], device=x.device).expand(x.shape[0])
    return scheduler.add_noise(x, noise, t_tensor)


def get_vHv(f, z, v_in, K=8, v_out=None):
    """
    Compute vHv using Hutchinson estimator with JVP.

    Args:
        f: function mapping input → output
        z: input tensor
        v_in: input variance (for scaling Rademacher vectors), shape matches z
        K: number of Hutchinson samples
        v_out: output variance (for weighting output). If None, uses v_in
               (same-scale case). For cross-scale, output shape differs from input.
    """
    z, v_in = z.detach(), v_in.detach()
    scale = torch.sqrt(v_in).unsqueeze(0)

    eps = torch.empty((K, *z.shape), device=z.device, dtype=z.dtype).bernoulli_(0.5).mul_(2).add_(-1)
    u_batch = eps * scale

    def single_sample_jvp(u):
        return jvp(f, (z,), (u,))[1]

    batched_out = vmap(single_sample_jvp, chunk_size=None)(u_batch)
    hv = (batched_out ** 2).mean(dim=0)

    if v_out is not None:
        v_out = v_out.detach()
        hv = hv * v_out.expand_as(hv)
    else:
        hv = hv * v_in.expand_as(hv)
    return hv.sum(dim=(1, 2, 3))


def compute_cross_jacobian_heatmap(model, scheduler, x_batch, device='cpu', K=4):
    """
    Compute 30×30 cross-scale Jacobian heatmaps.

    Returns 4 error matrices:
        error_HH: ∂f_H/∂x_H (self, 64→64)
        error_HL: ∂f_H/∂x_L (cross, 32→64)
        error_LH: ∂f_L/∂x_H (cross, 64→32)
        error_LL: ∂f_L/∂x_L (self, 32→32)
    """
    num_points = 30

    t_grid = torch.linspace(999, 0, steps=num_points, device=device).long()
    snr_grid = torch.tensor([timestep_to_snr(t.item(), scheduler) for t in t_grid], device=device)

    error_HH = torch.zeros(num_points, num_points, device=device)
    error_HL = torch.zeros(num_points, num_points, device=device)
    error_LH = torch.zeros(num_points, num_points, device=device)
    error_LL = torch.zeros(num_points, num_points, device=device)

    x_batch_32 = F.interpolate(x_batch, size=(32, 32), mode='bilinear', align_corners=False)

    model.eval()

    pbar = tqdm(total=num_points * num_points, desc="Computing cross-Jacobian heatmap")

    for i in range(num_points):
        t_64_int = t_grid[i].item()
        snr_64 = snr_grid[i].item()
        gamma_64 = chain_rule_factor(snr_64)

        for j in range(num_points):
            t_32_int = t_grid[j].item()
            snr_32 = snr_grid[j].item()
            gamma_32 = chain_rule_factor(snr_32)

            z_64 = add_noise_at_timestep(x_batch, t_64_int, scheduler)
            z_32 = add_noise_at_timestep(x_batch_32, t_32_int, scheduler)

            t_64_tensor = torch.tensor([t_64_int], device=device).expand(x_batch.shape[0])
            t_32_tensor = torch.tensor([t_32_int], device=device).expand(x_batch.shape[0])

            # J_HH: ∂f_H/∂x_H — vary x_64, read 64 output
            def f_HH(z_in):
                out, _ = model(z_in, z_32, t_64_tensor, t_32_tensor)
                return out

            # J_HL: ∂f_H/∂x_L — vary x_32, read 64 output
            def f_HL(z_in):
                out, _ = model(z_64, z_in, t_64_tensor, t_32_tensor)
                return out

            # J_LH: ∂f_L/∂x_H — vary x_64, read 32 output
            def f_LH(z_in):
                _, out = model(z_in, z_32, t_64_tensor, t_32_tensor)
                return out

            # J_LL: ∂f_L/∂x_L — vary x_32, read 32 output
            def f_LL(z_in):
                _, out = model(z_64, z_in, t_64_tensor, t_32_tensor)
                return out

            v_64 = torch.ones_like(z_64[:, :1]) / (64 * 64)
            v_32 = torch.ones_like(z_32[:, :1]) / (32 * 32)

            with torch.no_grad():
                # Self Jacobians (same as existing — v_in == v_out)
                error_HH[i, j] = get_vHv(f_HH, z_64, v_64, K=K).mean() * gamma_64
                error_LL[i, j] = get_vHv(f_LL, z_32, v_32, K=K).mean() * gamma_32

                # Cross Jacobians (v_in and v_out differ in spatial size)
                # J_HL: input x_L (32), output f_H (64) → v_in=v_32, v_out=v_64
                error_HL[i, j] = get_vHv(f_HL, z_32, v_32, K=K, v_out=v_64).mean() * gamma_32
                # J_LH: input x_H (64), output f_L (32) → v_in=v_64, v_out=v_32
                error_LH[i, j] = get_vHv(f_LH, z_64, v_64, K=K, v_out=v_32).mean() * gamma_64

            pbar.update(1)

    pbar.close()

    return (t_grid.cpu(), snr_grid.cpu(),
            error_HH.cpu(), error_HL.cpu(), error_LH.cpu(), error_LL.cpu())


def plot_cross_jacobian(t_grid, error_HH, error_HL, error_LH, error_LL, output_path):
    """Plot 2×2 panel of cross-scale Jacobian heatmaps."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    t_np = t_grid.numpy() if torch.is_tensor(t_grid) else t_grid
    extent = [t_np[0], t_np[-1], t_np[0], t_np[-1]]

    panels = [
        (axes[0, 0], error_HH, r'$J_{HH}$: $\partial f_H / \partial x_H$ (self)'),
        (axes[0, 1], error_HL, r'$J_{HL}$: $\partial f_H / \partial x_L$ (cross)'),
        (axes[1, 0], error_LH, r'$J_{LH}$: $\partial f_L / \partial x_H$ (cross)'),
        (axes[1, 1], error_LL, r'$J_{LL}$: $\partial f_L / \partial x_L$ (self)'),
    ]

    # Use shared color scale across all panels
    all_errors = [e.numpy() if torch.is_tensor(e) else e for _, e, _ in panels]
    vmin = min(np.log10(e[e > 0].min() + 1e-10) for e in all_errors if (e > 0).any())
    vmax = max(np.log10(e.max() + 1e-10) for e in all_errors)

    for ax, error, title in panels:
        error_np = error.numpy() if torch.is_tensor(error) else error
        log_error = np.log10(error_np + 1e-10)

        im = ax.imshow(log_error.T, origin='lower', extent=extent,
                       aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)

        # Diagonal reference
        ax.plot([t_np[0], t_np[-1]], [t_np[0], t_np[-1]],
                'w--', linewidth=1.5, alpha=0.5)

        ax.set_xlabel('$t_{64}$', fontsize=11)
        ax.set_ylabel('$t_{32}$', fontsize=11)
        ax.set_title(title, fontsize=13)
        plt.colorbar(im, ax=ax, label=r'$\log_{10}$(Error)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Cross-scale Jacobian analysis for LIFT')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='results/cross_jacobian.png')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--K', type=int, default=4, help='Hutchinson samples')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ema', action='store_true', help='Use EMA weights')
    parser.add_argument('--cache_dir', type=str, default=None)
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
    if args.ema:
        print("Using EMA weights")
        model.load_state_dict(ckpt['ema_state']['shadow'])
    else:
        model.load_state_dict(ckpt['model_state'])
    model = model.to(device).eval()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Scheduler
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="cosine", clip_sample=True)

    # Load data
    print("Loading dataset...")
    dataset = AFHQ64Dataset(split='train', cache_dir=args.cache_dir)
    indices = np.random.choice(len(dataset), args.batch_size, replace=False)
    x_batch = torch.stack([dataset[int(i)] for i in indices], dim=0).to(device)

    # Compute cross-Jacobian heatmaps
    print(f"\nComputing 30×30 cross-Jacobian heatmap (K={args.K}, batch={args.batch_size})...")
    t_grid, snr_grid, error_HH, error_HL, error_LH, error_LL = compute_cross_jacobian_heatmap(
        model, scheduler, x_batch, device=device, K=args.K
    )

    # Save results
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    results = {
        't_grid': t_grid,
        'snr_grid': snr_grid,
        'error_HH': error_HH,
        'error_HL': error_HL,
        'error_LH': error_LH,
        'error_LL': error_LL,
        'args': vars(args),
    }

    results_path = args.output.replace('.png', '.pth')
    torch.save(results, results_path)
    print(f"Results saved: {results_path}")

    # Plot 2×2 panel
    plot_cross_jacobian(t_grid, error_HH, error_HL, error_LH, error_LL, args.output)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Cross-Scale Jacobian Summary")
    print("=" * 60)
    for name, err in [('J_HH (∂f_H/∂x_H)', error_HH),
                      ('J_HL (∂f_H/∂x_L)', error_HL),
                      ('J_LH (∂f_L/∂x_H)', error_LH),
                      ('J_LL (∂f_L/∂x_L)', error_LL)]:
        print(f"  {name}: mean={err.mean():.4e}, max={err.max():.4e}, "
              f"min={err[err > 0].min():.4e}")

    # Cross-scale coupling ratio
    ratio_HL = error_HL.mean() / error_HH.mean()
    ratio_LH = error_LH.mean() / error_LL.mean()
    print(f"\n  Cross/Self ratio (J_HL/J_HH): {ratio_HL:.4f}")
    print(f"  Cross/Self ratio (J_LH/J_LL): {ratio_LH:.4f}")
    print(f"  → {'Strong' if ratio_HL > 0.1 else 'Weak'} 32→64 coupling")
    print(f"  → {'Strong' if ratio_LH > 0.1 else 'Weak'} 64→32 coupling")


if __name__ == "__main__":
    main()
