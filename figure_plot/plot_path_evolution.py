#!/usr/bin/env python3
"""
Plot evolution of optimal paths across training epochs.

Top row: Paths on 64×64 Error heatmaps
Bottom row: Paths on Total Error heatmaps
"""

import os
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


def main():
    epochs = [20, 40, 60, 80, 100]

    # Load all heatmaps
    print("Loading heatmaps...")
    data_list = []
    for epoch in epochs:
        heatmap_path = f'results/error_heatmap_{epoch}ep.pth'
        if not os.path.exists(heatmap_path):
            print(f"⚠️  Missing: {heatmap_path}")
            return
        data = torch.load(heatmap_path, weights_only=False)
        data_list.append(data)
        print(f"  ✓ Loaded {epoch}ep")

    # Create figure with 2 rows × 5 columns
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))

    for col, (epoch, data) in enumerate(zip(epochs, data_list)):
        gamma_grid = data['gamma_grid']
        error_scale0 = data['error_scale0']  # 64×64 error
        error_total = data['error_total']    # Total error

        gamma_np = gamma_grid if isinstance(gamma_grid, np.ndarray) else gamma_grid.numpy()
        log_gamma = np.log10(gamma_np)
        extent = [log_gamma[0], log_gamma[-1], log_gamma[0], log_gamma[-1]]

        # Compute paths
        optimal_gamma1_64, cost_64 = compute_dp_path(gamma_grid, error_scale0)
        optimal_gamma1_total, cost_total = compute_dp_path(gamma_grid, error_total)

        # Top row: 64×64 Error with its optimal path
        ax_top = axes[0, col]
        error_log = np.log10(error_scale0.numpy() + 1e-10)
        im_top = ax_top.imshow(error_log.T, origin='lower', extent=extent,
                               aspect='auto', cmap='viridis')

        log_gamma1_opt = np.log10(optimal_gamma1_64)
        ax_top.plot(log_gamma, log_gamma1_opt, 'r-', linewidth=2.5, label='Optimal Path')
        ax_top.plot(log_gamma, log_gamma1_opt, 'wo', markersize=5, markeredgewidth=1.5)
        ax_top.plot(log_gamma, log_gamma, 'w--', linewidth=1.5, alpha=0.5, label='Diagonal')

        ax_top.set_title(f'{epoch} Epochs\nγ₁: {optimal_gamma1_64[0]:.2f}→{optimal_gamma1_64[-1]:.2f}',
                        fontsize=12, fontweight='bold')

        if col == 0:
            ax_top.set_ylabel('64×64 Error\n\nlog₁₀(γ₁)', fontsize=12, fontweight='bold')
            ax_top.legend(fontsize=9, loc='upper left')
        else:
            ax_top.set_ylabel('')
            ax_top.set_yticklabels([])

        ax_top.set_xlabel('')
        ax_top.set_xticklabels([])

        # Bottom row: Total Error with its optimal path
        ax_bottom = axes[1, col]
        error_log = np.log10(error_total.numpy() + 1e-10)
        im_bottom = ax_bottom.imshow(error_log.T, origin='lower', extent=extent,
                                     aspect='auto', cmap='viridis')

        log_gamma1_opt = np.log10(optimal_gamma1_total)
        ax_bottom.plot(log_gamma, log_gamma1_opt, 'r-', linewidth=2.5, label='Optimal Path')
        ax_bottom.plot(log_gamma, log_gamma1_opt, 'wo', markersize=5, markeredgewidth=1.5)
        ax_bottom.plot(log_gamma, log_gamma, 'w--', linewidth=1.5, alpha=0.5, label='Diagonal')

        ax_bottom.set_title(f'γ₁: {optimal_gamma1_total[0]:.2f}→{optimal_gamma1_total[-1]:.2f}',
                           fontsize=11)
        ax_bottom.set_xlabel('log₁₀(γ₀)', fontsize=11)

        if col == 0:
            ax_bottom.set_ylabel('Total Error\n\nlog₁₀(γ₁)', fontsize=12, fontweight='bold')
            ax_bottom.legend(fontsize=9, loc='upper left')
        else:
            ax_bottom.set_ylabel('')
            ax_bottom.set_yticklabels([])

    # Add overall title
    fig.suptitle('Evolution of Optimal Generation Paths Across Training Epochs',
                fontsize=16, fontweight='bold', y=0.98)

    # Add colorbars
    cbar_top = fig.colorbar(im_top, ax=axes[0, :], location='right',
                           label='log₁₀(64×64 Error)', pad=0.02, aspect=30)
    cbar_bottom = fig.colorbar(im_bottom, ax=axes[1, :], location='right',
                               label='log₁₀(Total Error)', pad=0.02, aspect=30)

    plt.tight_layout(rect=[0, 0, 0.98, 0.96])

    output_path = 'figures/path_evolution_across_epochs.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()

    # Print summary
    print("\n" + "="*70)
    print("Path Evolution Summary")
    print("="*70)
    print(f"{'Epoch':<8} {'64×64 Error Path':<30} {'Total Error Path':<30}")
    print("-"*70)
    for epoch, data in zip(epochs, data_list):
        gamma_grid = data['gamma_grid']
        error_scale0 = data['error_scale0']
        error_total = data['error_total']

        opt_64, _ = compute_dp_path(gamma_grid, error_scale0)
        opt_total, _ = compute_dp_path(gamma_grid, error_total)

        path_64 = f"{opt_64[0]:.2f} → {opt_64[-1]:.2f}"
        path_total = f"{opt_total[0]:.2f} → {opt_total[-1]:.2f}"

        print(f"{epoch:<8} {path_64:<30} {path_total:<30}")
    print("="*70)


if __name__ == "__main__":
    main()
