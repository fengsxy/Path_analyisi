#!/usr/bin/env python3
"""
Plot separate heatmaps for DP-64 and DP-Total paths.

Usage:
    python plot_heatmap_separate.py --heatmap results/error_heatmap_80ep.pth --output figures/
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt


def compute_dp_path(gamma_grid, error_matrix):
    """Compute DP optimal path on given error matrix."""
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


def plot_heatmap_with_path(gamma_grid, error_matrix, optimal_gamma1,
                           title, output_path, cost):
    """Plot single heatmap with its optimal path."""
    gamma_np = gamma_grid if isinstance(gamma_grid, np.ndarray) else gamma_grid.numpy()
    error_np = error_matrix if isinstance(error_matrix, np.ndarray) else error_matrix.numpy()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Log scale for better visualization
    log_gamma = np.log10(gamma_np)
    extent = [log_gamma[0], log_gamma[-1], log_gamma[0], log_gamma[-1]]

    # Plot heatmap (log scale)
    error_log = np.log10(error_np + 1e-10)
    im = ax.imshow(error_log.T, origin='lower', extent=extent,
                   aspect='auto', cmap='viridis')

    # Plot optimal path
    log_gamma0 = log_gamma
    log_gamma1_opt = np.log10(optimal_gamma1)
    ax.plot(log_gamma0, log_gamma1_opt, 'r-', linewidth=3, label='DP Optimal Path')
    ax.plot(log_gamma0, log_gamma1_opt, 'wo', markersize=6, markeredgewidth=2)

    # Plot diagonal for reference
    ax.plot(log_gamma, log_gamma, 'w--', linewidth=2, alpha=0.5, label='Diagonal (γ₁=γ₀)')

    ax.set_xlabel('log₁₀(γ₀) - 64×64 SNR', fontsize=14, fontweight='bold')
    ax.set_ylabel('log₁₀(γ₁) - 32×32 SNR', fontsize=14, fontweight='bold')
    ax.set_title(f'{title}\nTotal Cost: {cost:.6e}', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='log₁₀(Error)')
    cbar.ax.tick_params(labelsize=11)

    # Add path info
    info_text = f'Path: γ₁ ∈ [{optimal_gamma1[0]:.4f}, {optimal_gamma1[-1]:.4f}]'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_comparison(gamma_grid, error_scale0, error_total,
                   optimal_gamma1_64, optimal_gamma1_total,
                   cost_64, cost_total, output_path):
    """Plot side-by-side comparison."""
    gamma_np = gamma_grid if isinstance(gamma_grid, np.ndarray) else gamma_grid.numpy()
    log_gamma = np.log10(gamma_np)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: DP-64
    error_log = np.log10(error_scale0.numpy() + 1e-10)
    extent = [log_gamma[0], log_gamma[-1], log_gamma[0], log_gamma[-1]]

    im1 = axes[0].imshow(error_log.T, origin='lower', extent=extent,
                         aspect='auto', cmap='viridis')
    log_gamma1_opt = np.log10(optimal_gamma1_64)
    axes[0].plot(log_gamma, log_gamma1_opt, 'r-', linewidth=3, label='DP-64 Path')
    axes[0].plot(log_gamma, log_gamma1_opt, 'wo', markersize=6, markeredgewidth=2)
    axes[0].plot(log_gamma, log_gamma, 'w--', linewidth=2, alpha=0.5, label='Diagonal')

    axes[0].set_xlabel('log₁₀(γ₀) - 64×64 SNR', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('log₁₀(γ₁) - 32×32 SNR', fontsize=13, fontweight='bold')
    axes[0].set_title(f'DP-64: Optimize 64×64 Error Only\nCost: {cost_64:.6e}',
                     fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    plt.colorbar(im1, ax=axes[0], label='log₁₀(Error 64×64)')

    # Right: DP-Total
    error_log = np.log10(error_total.numpy() + 1e-10)

    im2 = axes[1].imshow(error_log.T, origin='lower', extent=extent,
                         aspect='auto', cmap='viridis')
    log_gamma1_opt = np.log10(optimal_gamma1_total)
    axes[1].plot(log_gamma, log_gamma1_opt, 'r-', linewidth=3, label='DP-Total Path')
    axes[1].plot(log_gamma, log_gamma1_opt, 'wo', markersize=6, markeredgewidth=2)
    axes[1].plot(log_gamma, log_gamma, 'w--', linewidth=2, alpha=0.5, label='Diagonal')

    axes[1].set_xlabel('log₁₀(γ₀) - 64×64 SNR', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('log₁₀(γ₁) - 32×32 SNR', fontsize=13, fontweight='bold')
    axes[1].set_title(f'DP-Total: Optimize Total Error\nCost: {cost_total:.6e}',
                     fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    plt.colorbar(im2, ax=axes[1], label='log₁₀(Error Total)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot separate heatmaps for DP paths')
    parser.add_argument('--heatmap', type=str, required=True,
                       help='Path to error heatmap .pth file')
    parser.add_argument('--output', type=str, default='figures/',
                       help='Output directory for plots')
    args = parser.parse_args()

    # Load heatmap
    print(f"Loading heatmap: {args.heatmap}")
    data = torch.load(args.heatmap, weights_only=False)

    gamma_grid = data['gamma_grid']
    error_scale0 = data['error_scale0']  # 64×64 error
    error_total = data['error_total']    # Total error

    # Compute DP paths
    print("Computing DP-64 path...")
    optimal_gamma1_64, cost_64 = compute_dp_path(gamma_grid, error_scale0)
    print(f"  γ₁ range: {optimal_gamma1_64[0]:.4f} -> {optimal_gamma1_64[-1]:.4f}")
    print(f"  Total cost: {cost_64:.6e}")

    print("Computing DP-Total path...")
    optimal_gamma1_total, cost_total = compute_dp_path(gamma_grid, error_total)
    print(f"  γ₁ range: {optimal_gamma1_total[0]:.4f} -> {optimal_gamma1_total[-1]:.4f}")
    print(f"  Total cost: {cost_total:.6e}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Extract epoch from filename
    basename = os.path.basename(args.heatmap)
    epoch_str = basename.replace('error_heatmap_', '').replace('.pth', '')

    # Plot individual heatmaps
    print("\nGenerating plots...")

    output_64 = os.path.join(args.output, f'heatmap_dp64_{epoch_str}.png')
    plot_heatmap_with_path(gamma_grid, error_scale0, optimal_gamma1_64,
                          f'DP-64 Path: Optimize 64×64 Error ({epoch_str})',
                          output_64, cost_64)

    output_total = os.path.join(args.output, f'heatmap_dp_total_{epoch_str}.png')
    plot_heatmap_with_path(gamma_grid, error_total, optimal_gamma1_total,
                          f'DP-Total Path: Optimize Total Error ({epoch_str})',
                          output_total, cost_total)

    # Plot comparison
    output_comparison = os.path.join(args.output, f'heatmap_comparison_{epoch_str}.png')
    plot_comparison(gamma_grid, error_scale0, error_total,
                   optimal_gamma1_64, optimal_gamma1_total,
                   cost_64, cost_total, output_comparison)

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"DP-64 Path:   γ₁ ∈ [{optimal_gamma1_64[0]:.4f}, {optimal_gamma1_64[-1]:.4f}]")
    print(f"DP-Total Path: γ₁ ∈ [{optimal_gamma1_total[0]:.4f}, {optimal_gamma1_total[-1]:.4f}]")
    print("="*60)


if __name__ == "__main__":
    main()
