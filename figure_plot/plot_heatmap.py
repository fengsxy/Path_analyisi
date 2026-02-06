#!/usr/bin/env python3
"""
Plot Error Heatmap with Optimal Path

Loads the computed error heatmap and plots the optimal path that minimizes
64×64 error with monotonic increasing γ₁ constraint.

Usage:
    python plot_heatmap.py --heatmap results/error_heatmap.pth
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt


def compute_optimal_path(gamma_grid, error_scale0, tolerance=0.05):
    """
    Compute optimal γ₁ schedule that minimizes 64×64 error.

    Constraint: γ₁ must be monotonically increasing.
    Starting from (γ_min, γ_min).

    If the error is nearly constant across γ₁ (within tolerance),
    we default to γ₁ = γ₀ (diagonal path) since there's no clear optimum.

    Args:
        gamma_grid: Array of gamma values
        error_scale0: Error matrix [num_gamma0, num_gamma1]
        tolerance: If (max-min)/min < tolerance, consider error constant

    Returns:
        optimal_gamma1: Optimal γ₁ for each γ₀
    """
    num_points = len(gamma_grid)
    optimal_gamma1 = [gamma_grid[0]]  # Start at γ_min
    current_min_j = 0

    for i in range(1, num_points):
        # Check if error is nearly constant across γ₁ for this γ₀
        row = error_scale0[i, current_min_j:]
        row_min = row.min()
        row_max = row.max()

        if row_min > 0 and (row_max - row_min) / row_min < tolerance:
            # Error is nearly constant - default to diagonal (γ₁ = γ₀)
            j_opt = i
        else:
            # Find minimum with monotonic constraint
            j_opt_relative = np.argmin(row)
            j_opt = current_min_j + j_opt_relative

        # Ensure monotonic constraint
        j_opt = max(j_opt, current_min_j)

        optimal_gamma1.append(gamma_grid[j_opt])
        current_min_j = j_opt

    return np.array(optimal_gamma1)


def compute_optimal_path_dp(gamma_grid, error_total):
    """
    Compute optimal path using Dynamic Programming on total error.

    DP formulation:
        dp[i][j] = minimum cumulative total error to reach (γ₀=i, γ₁=j)

    Constraint: γ₁ must be monotonically increasing (can only move right or stay)

    Transition: dp[i][j] = min(dp[i-1][k] for k <= j) + error_total[i][j]

    Args:
        gamma_grid: Array of gamma values
        error_total: Total error matrix [num_gamma0, num_gamma1]

    Returns:
        path_i: Array of γ₀ indices along optimal path
        path_j: Array of γ₁ indices along optimal path
        optimal_gamma1: Optimal γ₁ for each γ₀
        total_cost: Total accumulated error along the path
    """
    num_points = len(gamma_grid)
    error_np = error_total if isinstance(error_total, np.ndarray) else error_total.numpy()

    # dp[i][j] = minimum cumulative error to reach (i, j)
    dp = np.full((num_points, num_points), np.inf)
    # parent[i][j] = the j index we came from at row i-1
    parent = np.zeros((num_points, num_points), dtype=int)

    # Initialize first row: can start at any j (but typically start at j=0)
    # We start from (0, 0) as the initial point
    dp[0, 0] = error_np[0, 0]

    # Fill DP table
    for i in range(1, num_points):
        # For each j at row i, find the best k <= j from row i-1
        min_dp_so_far = np.inf
        best_k = 0

        for j in range(num_points):
            # Update running minimum from row i-1
            if dp[i-1, j] < min_dp_so_far:
                min_dp_so_far = dp[i-1, j]
                best_k = j

            # dp[i][j] = best way to reach (i, j) with monotonic constraint
            dp[i, j] = min_dp_so_far + error_np[i, j]
            parent[i, j] = best_k

    # Find the best ending point at the last row
    best_end_j = np.argmin(dp[num_points - 1, :])
    total_cost = dp[num_points - 1, best_end_j]

    # Backtrack to find the path
    path_j = [best_end_j]
    current_j = best_end_j

    for i in range(num_points - 1, 0, -1):
        current_j = parent[i, current_j]
        path_j.append(current_j)

    path_j = path_j[::-1]  # Reverse to get forward order
    path_i = list(range(num_points))

    # Convert to gamma values
    gamma_np = gamma_grid if isinstance(gamma_grid, np.ndarray) else gamma_grid.numpy()
    optimal_gamma1 = np.array([gamma_np[j] for j in path_j])

    return np.array(path_i), np.array(path_j), optimal_gamma1, total_cost


def plot_heatmap_with_path(gamma_grid, error_scale0, error_scale1, error_total, output_path):
    """Plot error heatmaps with optimal path overlay."""
    gamma_np = gamma_grid if isinstance(gamma_grid, np.ndarray) else gamma_grid.numpy()
    error_scale0_np = error_scale0 if isinstance(error_scale0, np.ndarray) else error_scale0.numpy()
    error_scale1_np = error_scale1 if isinstance(error_scale1, np.ndarray) else error_scale1.numpy()
    error_total_np = error_total if isinstance(error_total, np.ndarray) else error_total.numpy()

    # Compute optimal path
    optimal_gamma1 = compute_optimal_path(gamma_np, error_scale0_np)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    extent = [np.log10(gamma_np[0]), np.log10(gamma_np[-1]),
              np.log10(gamma_np[0]), np.log10(gamma_np[-1])]

    titles = ['Scale 0 (64×64) Error', 'Scale 1 (32×32) Error', 'Total Error']
    data_list = [error_scale0_np, error_scale1_np, error_total_np]

    for ax, title, d in zip(axes, titles, data_list):
        d_log = np.log10(d + 1e-10)
        im = ax.imshow(d_log.T, origin='lower', extent=extent, aspect='auto', cmap='viridis')

        # Plot optimal path
        ax.plot(np.log10(gamma_np), np.log10(optimal_gamma1),
                'r-', linewidth=2.5, label='Optimal γ₁ (monotonic)')

        ax.set_xlabel('log₁₀(γ₀) - 64×64 SNR')
        ax.set_ylabel('log₁₀(γ₁) - 32×32 SNR')
        ax.set_title(title)
        ax.legend(loc='upper left', fontsize=8)
        plt.colorbar(im, ax=ax, label='log₁₀(vHv)')

    plt.suptitle('Optimal Path for Minimizing 64×64 Error (Monotonic Increasing γ₁)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.close()

    return optimal_gamma1


def plot_heatmap_with_dp_path(gamma_grid, error_scale0, error_scale1, error_total, output_path):
    """
    Plot error heatmaps with DP optimal path overlay on total error.

    The DP path minimizes cumulative total error with monotonic γ₁ constraint.
    """
    gamma_np = gamma_grid if isinstance(gamma_grid, np.ndarray) else gamma_grid.numpy()
    error_scale0_np = error_scale0 if isinstance(error_scale0, np.ndarray) else error_scale0.numpy()
    error_scale1_np = error_scale1 if isinstance(error_scale1, np.ndarray) else error_scale1.numpy()
    error_total_np = error_total if isinstance(error_total, np.ndarray) else error_total.numpy()

    # Compute DP optimal path on total error
    path_i, path_j, optimal_gamma1, total_cost = compute_optimal_path_dp(gamma_np, error_total_np)

    print(f"DP optimal path total cost: {total_cost:.6e}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    extent = [np.log10(gamma_np[0]), np.log10(gamma_np[-1]),
              np.log10(gamma_np[0]), np.log10(gamma_np[-1])]

    titles = ['Scale 0 (64×64) Error', 'Scale 1 (32×32) Error', 'Total Error']
    data_list = [error_scale0_np, error_scale1_np, error_total_np]

    for ax, title, d in zip(axes, titles, data_list):
        d_log = np.log10(d + 1e-10)
        im = ax.imshow(d_log.T, origin='lower', extent=extent, aspect='auto', cmap='viridis')

        # Plot DP optimal path
        ax.plot(np.log10(gamma_np), np.log10(optimal_gamma1),
                'r-', linewidth=2.5, label='DP Optimal (total error)')

        # Plot diagonal for reference
        ax.plot(np.log10(gamma_np), np.log10(gamma_np),
                'w--', linewidth=1.5, alpha=0.7, label='Diagonal (γ₁=γ₀)')

        ax.set_xlabel('log₁₀(γ₀) - 64×64 SNR')
        ax.set_ylabel('log₁₀(γ₁) - 32×32 SNR')
        ax.set_title(title)
        ax.legend(loc='upper left', fontsize=8)
        plt.colorbar(im, ax=ax, label='log₁₀(vHv)')

    plt.suptitle(f'DP Optimal Path for Minimizing Total Error (cost={total_cost:.4e})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.close()

    return path_i, path_j, optimal_gamma1, total_cost


def plot_error_along_path(gamma_grid, error_scale0, optimal_gamma1, output_path):
    """Plot error along the optimal path."""
    gamma_np = gamma_grid if isinstance(gamma_grid, np.ndarray) else gamma_grid.numpy()
    error_scale0_np = error_scale0 if isinstance(error_scale0, np.ndarray) else error_scale0.numpy()

    num_points = len(gamma_np)

    # Compute error along optimal path
    optimal_error = []
    current_min_j = 0

    optimal_error.append(error_scale0_np[0, 0])

    for i in range(1, num_points):
        j_opt_relative = np.argmin(error_scale0_np[i, current_min_j:])
        j_opt = current_min_j + j_opt_relative
        optimal_error.append(error_scale0_np[i, j_opt])
        current_min_j = j_opt

    optimal_error = np.array(optimal_error)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(gamma_np, optimal_error, 'r-', linewidth=2, label='Optimal path - 64×64 error')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('γ₀ (64×64 SNR)')
    ax.set_ylabel('64×64 Error (vHv)')
    ax.set_title('64×64 Error Along Optimal Monotonic Path')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.close()


def print_optimal_schedule(gamma_grid, error_scale0, optimal_gamma1):
    """Print the optimal schedule."""
    gamma_np = gamma_grid if isinstance(gamma_grid, np.ndarray) else gamma_grid.numpy()
    error_scale0_np = error_scale0 if isinstance(error_scale0, np.ndarray) else error_scale0.numpy()

    num_points = len(gamma_np)

    print("\n" + "="*60)
    print("Optimal γ₁ schedule (monotonic increasing) to minimize 64×64 error")
    print("="*60)
    print(f"{'γ₀ (64×64)':<15} {'Optimal γ₁ (32×32)':<20} {'64×64 Error':<15}")
    print("-"*60)

    for i in range(num_points):
        # Find j index for optimal_gamma1[i]
        j_opt = np.argmin(np.abs(gamma_np - optimal_gamma1[i]))
        error = error_scale0_np[i, j_opt]
        print(f"{gamma_np[i]:<15.4f} {optimal_gamma1[i]:<20.4f} {error:<15.6f}")


def parse_args():
    parser = argparse.ArgumentParser(description='Plot error heatmap with optimal path')
    parser.add_argument('--heatmap', type=str, required=True, help='Path to error heatmap .pth file')
    parser.add_argument('--output', type=str, default=None, help='Output path (default: same dir as heatmap)')
    parser.add_argument('--dp', action='store_true', help='Use DP to find optimal path on total error')
    parser.add_argument('--dp_64', action='store_true', help='Use DP to find optimal path on 64×64 error only')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load heatmap data
    print(f"Loading heatmap: {args.heatmap}")
    data = torch.load(args.heatmap, weights_only=False)

    gamma_grid = data['gamma_grid']
    error_scale0 = data['error_scale0']
    error_scale1 = data['error_scale1']
    error_total = data['error_total']

    # Determine output paths
    if args.output:
        output_dir = os.path.dirname(args.output) or '.'
        base_name = os.path.splitext(os.path.basename(args.output))[0]
    else:
        output_dir = os.path.dirname(args.heatmap) or '.'
        base_name = 'error_heatmap_with_path'

    os.makedirs(output_dir, exist_ok=True)

    if args.dp_64:
        # Plot heatmap with DP optimal path on 64×64 error only
        dp64_output = os.path.join(output_dir, f'{base_name}_dp_64.png')
        path_i, path_j, optimal_gamma1, total_cost = plot_heatmap_with_dp_path(
            gamma_grid, error_scale0, error_scale1, error_scale0, dp64_output  # Use error_scale0 for DP
        )

        # Print DP schedule
        gamma_np = gamma_grid if isinstance(gamma_grid, np.ndarray) else gamma_grid.numpy()
        error_scale0_np = error_scale0 if isinstance(error_scale0, np.ndarray) else error_scale0.numpy()

        print("\n" + "="*70)
        print("DP Optimal γ₁ schedule (monotonic) to minimize cumulative 64×64 error")
        print("="*70)
        print(f"{'γ₀ (64×64)':<15} {'Optimal γ₁ (32×32)':<20} {'64×64 Error':<15}")
        print("-"*70)

        for i in range(len(gamma_np)):
            j = path_j[i]
            print(f"{gamma_np[i]:<15.4f} {optimal_gamma1[i]:<20.4f} {error_scale0_np[i, j]:<15.6e}")

        print("-"*70)
        print(f"Total cumulative cost: {total_cost:.6e}")

    elif args.dp:
        # Plot heatmap with DP optimal path on total error
        dp_output = os.path.join(output_dir, f'{base_name}_dp.png')
        path_i, path_j, optimal_gamma1, total_cost = plot_heatmap_with_dp_path(
            gamma_grid, error_scale0, error_scale1, error_total, dp_output
        )

        # Print DP schedule
        gamma_np = gamma_grid if isinstance(gamma_grid, np.ndarray) else gamma_grid.numpy()
        error_total_np = error_total if isinstance(error_total, np.ndarray) else error_total.numpy()

        print("\n" + "="*70)
        print("DP Optimal γ₁ schedule (monotonic) to minimize cumulative total error")
        print("="*70)
        print(f"{'γ₀ (64×64)':<15} {'Optimal γ₁ (32×32)':<20} {'Total Error':<15}")
        print("-"*70)

        for i in range(len(gamma_np)):
            j = path_j[i]
            print(f"{gamma_np[i]:<15.4f} {optimal_gamma1[i]:<20.4f} {error_total_np[i, j]:<15.6e}")

        print("-"*70)
        print(f"Total cumulative cost: {total_cost:.6e}")
    else:
        # Plot heatmap with greedy path
        heatmap_output = os.path.join(output_dir, f'{base_name}.png')
        optimal_gamma1 = plot_heatmap_with_path(
            gamma_grid, error_scale0, error_scale1, error_total, heatmap_output
        )

        # Print schedule
        print_optimal_schedule(gamma_grid, error_scale0, optimal_gamma1)


if __name__ == "__main__":
    main()
