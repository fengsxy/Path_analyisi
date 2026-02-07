#!/usr/bin/env python3
"""
Compute 30×30 Error Heatmap and Find Optimal Path with Step Allocation.

Two-step approach based on "Align Your Steps" (AYS):
1. Find optimal path on 30×30 grid (58 steps, minimize total error)
2. Allocate N sampling steps based on error distribution (equal error per step)

Usage:
    python compute_heatmap_30.py --checkpoint checkpoints/lift_dual_timestep_100ep.pth \
        --output results/heatmap_30_100ep.png --num_steps 18 --device 0
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
    """
    Compute the chain-rule correction factor to convert from x_t space to z space.

    In stochastic localization: z = SNR * x + sqrt(SNR) * ε
    In DDPM: x_t = sqrt(α̅) * x + sqrt(1-α̅) * ε = z / sqrt(SNR * (1 + SNR))

    The Jacobian transforms as:
        H_z = H_{x_t} / (SNR * (1 + SNR))

    At high SNR, H_z ~ 1/SNR² (goes to zero), which is the expected behavior
    since x̂ ≈ z/SNR and ∂x̂/∂z ≈ I/SNR.
    """
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
    noisy = scheduler.add_noise(x, noise, t_tensor)
    return noisy


def get_vHv(f, z, v, K=8):
    """Compute vHv using Hutchinson estimator with JVP."""
    z, v = z.detach(), v.detach()
    scale = torch.sqrt(v).unsqueeze(0)

    eps = torch.empty((K, *z.shape), device=z.device, dtype=z.dtype).bernoulli_(0.5).mul_(2).add_(-1)
    u_batch = eps * scale

    def single_sample_jvp(u):
        return jvp(f, (z,), (u,))[1]

    batched_out = vmap(single_sample_jvp, chunk_size=None)(u_batch)
    hv = (batched_out ** 2).mean(dim=0)
    hv = hv * v.expand_as(hv)
    return hv.sum(dim=(1, 2, 3))


def compute_error_heatmap_30(model, scheduler, x_batch, device='cpu', K=4):
    """
    Compute 30×30 error heatmap in timestep space.

    Grid convention (matching original SNR-space heatmap):
    - Index 0 → t=999 (high noise, low SNR) → high error
    - Index 29 → t=0 (low noise, high SNR) → low error

    This way:
    - Bottom-left (0,0) = high noise = start of generation
    - Top-right (29,29) = low noise = end of generation
    - DP path goes from (0,0) to (29,29)

    Returns:
        t_grid: [30] timestep values (999 to 0, decreasing)
        error_64: [30, 30] error for 64×64 scale
        error_32: [30, 30] error for 32×32 scale
        error_total: [30, 30] total error
    """
    num_points = 30

    # Timestep grid: 999 to 0 (high noise to low noise)
    # This matches the original SNR-space convention where:
    # - index 0 = low SNR (high noise) = high error
    # - index 29 = high SNR (low noise) = low error
    t_grid = torch.linspace(999, 0, steps=num_points, device=device).long()
    snr_grid = torch.tensor([timestep_to_snr(t.item(), scheduler) for t in t_grid], device=device)

    error_64 = torch.zeros(num_points, num_points, device=device)
    error_32 = torch.zeros(num_points, num_points, device=device)

    x_batch_32 = F.interpolate(x_batch, size=(32, 32), mode='bilinear', align_corners=False)

    model.eval()

    pbar = tqdm(total=num_points * num_points, desc="Computing 30×30 heatmap")

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

            def f_64(z_in):
                out, _ = model(z_in, z_32, t_64_tensor, t_32_tensor)
                return out

            def f_32(z_in):
                _, out = model(z_64, z_in, t_64_tensor, t_32_tensor)
                return out

            v_64 = torch.ones_like(z_64[:, :1]) / (64 * 64)
            v_32 = torch.ones_like(z_32[:, :1]) / (32 * 32)

            with torch.no_grad():
                vhv_64 = get_vHv(f_64, z_64, v_64, K=K).mean() * gamma_64
                vhv_32 = get_vHv(f_32, z_32, v_32, K=K).mean() * gamma_32

                error_64[i, j] = vhv_64
                error_32[i, j] = vhv_32

            pbar.update(1)

    pbar.close()

    error_total = error_64 + error_32

    return t_grid.cpu(), snr_grid.cpu(), error_64.cpu(), error_32.cpu(), error_total.cpu()


def find_optimal_path(error_matrix):
    """
    Find optimal path from (0,0) to (29,29) using DP.

    Each step moves exactly one cell: right OR up.
    Total: 58 steps, 59 points.

    Returns:
        path: List of (i, j) tuples, length 59
        total_cost: Total error along path
    """
    error_np = error_matrix.numpy() if torch.is_tensor(error_matrix) else error_matrix
    n = error_np.shape[0]  # 30

    # dp[i][j] = minimum cost to reach (i, j)
    dp = np.full((n, n), np.inf)
    parent = np.full((n, n, 2), -1, dtype=int)

    dp[0, 0] = error_np[0, 0]

    # Fill DP table
    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0:
                continue

            candidates = []
            if i > 0 and dp[i-1, j] < np.inf:
                candidates.append((dp[i-1, j], i-1, j))
            if j > 0 and dp[i, j-1] < np.inf:
                candidates.append((dp[i, j-1], i, j-1))

            if candidates:
                best = min(candidates, key=lambda x: x[0])
                dp[i, j] = best[0] + error_np[i, j]
                parent[i, j] = [best[1], best[2]]

    # Backtrack to find path
    path = []
    i, j = n - 1, n - 1
    while i >= 0 and j >= 0:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        pi, pj = parent[i, j]
        i, j = pi, pj

    path = path[::-1]
    total_cost = dp[n-1, n-1]

    return path, total_cost


def find_optimal_path_n_steps(error_matrix, num_steps, max_jump=5):
    """
    Find optimal path from (0,0) to (n-1,n-1) with exactly num_steps steps.

    Cost = integral of error along the path segment, approximated as:
        (error_start + error_end) / 2 × step_size

    Constraint: each step can move at most max_jump cells in each direction.
    This prevents unrealistic large jumps that DDIM can't handle well.

    This guarantees:
    - Start at (0,0) = high noise (t=999)
    - End at (n-1,n-1) = low noise (t=0)
    - Exactly num_steps steps (num_steps+1 points)
    - Each step moves at most max_jump in i and max_jump in j
    - Minimum total integrated error

    Args:
        error_matrix: [n, n] error values
        num_steps: Number of steps (e.g., 18)
        max_jump: Maximum jump per step in each direction (default 5 ≈ 170 timesteps)

    Returns:
        path: List of (i, j) tuples, length num_steps+1
        total_cost: Total integrated error along path
    """
    error_np = error_matrix.numpy() if torch.is_tensor(error_matrix) else error_matrix
    n = error_np.shape[0]  # 30

    INF = float('inf')
    dp = np.full((n, n, num_steps + 1), INF)
    parent = np.full((n, n, num_steps + 1, 3), -1, dtype=int)  # (pi, pj, pk)

    dp[0, 0, 0] = 0  # Start with 0 cost (we'll add segment costs)

    # For each step k, compute transitions
    for k in range(num_steps):
        for i in range(n):
            for j in range(n):
                if dp[i, j, k] == INF:
                    continue

                # Remaining steps after this one
                remaining = num_steps - k - 1

                # From (i,j), try reachable (ni, nj) within max_jump
                for ni in range(i, min(i + max_jump + 1, n)):
                    for nj in range(j, min(j + max_jump + 1, n)):
                        if ni == i and nj == j:
                            continue  # must move

                        # Check if we can reach end from (ni, nj) in remaining steps
                        dist_to_end = (n - 1 - ni) + (n - 1 - nj)
                        # With max_jump constraint, we can cover at most 2*max_jump per step
                        max_reachable = remaining * 2 * max_jump
                        if dist_to_end > max_reachable:
                            continue  # can't reach end in time
                        if remaining == 0 and (ni != n-1 or nj != n-1):
                            continue  # last step must reach end

                        # Cost = integral of error along segment (trapezoidal)
                        step_size = (ni - i) + (nj - j)
                        error_start = error_np[i, j]
                        error_end = error_np[ni, nj]
                        step_cost = (error_start + error_end) / 2 * step_size

                        new_cost = dp[i, j, k] + step_cost
                        if new_cost < dp[ni, nj, k + 1]:
                            dp[ni, nj, k + 1] = new_cost
                            parent[ni, nj, k + 1] = [i, j, k]

    # Check if we reached the end
    if dp[n-1, n-1, num_steps] == INF:
        print(f"Warning: Cannot reach (n-1,n-1) in exactly {num_steps} steps with max_jump={max_jump}")
        # Try to find any reachable solution
        best_k = -1
        for k in range(num_steps + 1):
            if dp[n-1, n-1, k] < INF:
                best_k = k
        if best_k == -1:
            # Increase max_jump and retry
            print(f"Retrying with larger max_jump...")
            return find_optimal_path_n_steps(error_matrix, num_steps, max_jump=max_jump + 2)
        print(f"Using {best_k} steps instead")
        final_k = best_k
    else:
        final_k = num_steps

    # Backtrack
    path = []
    i, j, k = n - 1, n - 1, final_k
    while k >= 0:
        path.append((i, j))
        if k == 0:
            break
        pi, pj, pk = parent[i, j, k]
        i, j, k = pi, pj, pk

    path = path[::-1]
    total_cost = dp[n-1, n-1, final_k]

    return path, total_cost


def allocate_steps(path, error_matrix, num_steps):
    """
    Allocate N sampling points along path based on error distribution.

    Goal: Each segment has approximately equal cumulative error.

    Args:
        path: List of (i, j) tuples (59 points)
        error_matrix: Error values
        num_steps: Number of sampling steps (e.g., 18)

    Returns:
        sample_indices: Indices into path for sampling points
        sample_points: List of (i, j) tuples for sampling
    """
    error_np = error_matrix.numpy() if torch.is_tensor(error_matrix) else error_matrix

    # Get error at each path point
    path_errors = [error_np[i, j] for i, j in path]
    total_error = sum(path_errors)
    target_per_step = total_error / num_steps

    sample_indices = [0]  # Always include start
    cumulative = path_errors[0]

    for k in range(1, len(path)):
        cumulative += path_errors[k]

        # Check if we've accumulated enough error for next sample
        if cumulative >= target_per_step * len(sample_indices) and len(sample_indices) < num_steps:
            sample_indices.append(k)

    # Always include end
    if sample_indices[-1] != len(path) - 1:
        if len(sample_indices) < num_steps + 1:
            sample_indices.append(len(path) - 1)
        else:
            sample_indices[-1] = len(path) - 1

    # Ensure we have exactly num_steps + 1 points
    while len(sample_indices) < num_steps + 1:
        # Add points in largest gaps
        gaps = []
        for i in range(len(sample_indices) - 1):
            gap = sample_indices[i+1] - sample_indices[i]
            gaps.append((gap, i))
        gaps.sort(reverse=True)

        if gaps and gaps[0][0] > 1:
            idx = gaps[0][1]
            new_point = (sample_indices[idx] + sample_indices[idx+1]) // 2
            sample_indices.insert(idx + 1, new_point)
        else:
            break

    sample_indices = sample_indices[:num_steps + 1]
    sample_points = [path[k] for k in sample_indices]

    return sample_indices, sample_points


def path_to_timesteps(sample_points, t_grid):
    """Convert sample points to actual timesteps.

    With the new convention:
    - t_grid[0] = 999 (high noise, low SNR)
    - t_grid[29] = 0 (low noise, high SNR)
    - Path goes from (0,0) to (29,29), i.e., t=999 to t=0

    For generation, we already have high noise first, so no reversal needed.
    """
    t_np = t_grid.numpy() if torch.is_tensor(t_grid) else t_grid

    timesteps_64 = [int(t_np[p[0]]) for p in sample_points]
    timesteps_32 = [int(t_np[p[1]]) for p in sample_points]

    # No reversal needed - path already goes from high noise to low noise

    return timesteps_64, timesteps_32


def plot_heatmap_with_path(t_grid, error_matrix, path, sample_points, output_path, title):
    """Plot heatmap with sampling path (connected yellow points)."""
    fig, ax = plt.subplots(figsize=(8, 7))

    t_np = t_grid.numpy() if torch.is_tensor(t_grid) else t_grid
    error_np = error_matrix.numpy() if torch.is_tensor(error_matrix) else error_matrix

    extent = [t_np[0], t_np[-1], t_np[0], t_np[-1]]

    im = ax.imshow(np.log10(error_np + 1e-10).T, origin='lower', extent=extent,
                   aspect='auto', cmap='viridis')

    # Diagonal reference
    ax.plot([t_np[0], t_np[-1]], [t_np[0], t_np[-1]], 'w--', linewidth=1.5, alpha=0.5, label='Diagonal')

    # Sampling path (red line connecting yellow points)
    sample_t64 = [t_np[p[0]] for p in sample_points]
    sample_t32 = [t_np[p[1]] for p in sample_points]
    ax.plot(sample_t64, sample_t32, 'r-', linewidth=2, alpha=0.8, label=f'Path ({len(sample_points)} pts)')
    ax.scatter(sample_t64, sample_t32, c='yellow', s=50, zorder=5,
               edgecolors='black', linewidths=1)

    ax.set_xlabel('t_64 (timestep)', fontsize=12)
    ax.set_ylabel('t_32 (timestep)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log₁₀(Error)', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_comparison(t_grid, error_64, error_total, path_64, path_total,
                    samples_64, samples_total, output_path, num_steps=18):
    """Plot comparison of Diagonal, DP-64 and DP-Total paths."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    t_np = t_grid.numpy() if torch.is_tensor(t_grid) else t_grid
    extent = [t_np[0], t_np[-1], t_np[0], t_np[-1]]
    n = len(t_np)

    # Compute diagonal path
    diagonal_indices = np.linspace(0, n-1, num_steps + 1).astype(int)
    diagonal_path = [(i, i) for i in diagonal_indices]

    data = [
        (error_64, samples_64, '64×64 Error', 'DP-64'),
        (error_total, samples_total, 'Total Error', 'DP-Total'),
    ]

    for ax, (error, samples, title, label) in zip(axes, data):
        error_np = error.numpy() if torch.is_tensor(error) else error

        im = ax.imshow(np.log10(error_np + 1e-10).T, origin='lower', extent=extent,
                       aspect='auto', cmap='viridis')

        # Diagonal reference (white dashed)
        ax.plot([t_np[0], t_np[-1]], [t_np[0], t_np[-1]], 'w--', linewidth=1.5, alpha=0.5)

        # Diagonal sampling path (cyan)
        diag_t64 = [t_np[p[0]] for p in diagonal_path]
        diag_t32 = [t_np[p[1]] for p in diagonal_path]
        ax.plot(diag_t64, diag_t32, 'c-', linewidth=2, alpha=0.7, label=f'Diagonal ({len(diagonal_path)} pts)')
        ax.scatter(diag_t64, diag_t32, c='cyan', s=30, zorder=4, edgecolors='black', linewidths=0.5)

        # DP sampling path (red line + yellow points)
        sample_t64 = [t_np[p[0]] for p in samples]
        sample_t32 = [t_np[p[1]] for p in samples]
        ax.plot(sample_t64, sample_t32, 'r-', linewidth=2, alpha=0.8, label=f'{label} ({len(samples)} pts)')
        ax.scatter(sample_t64, sample_t32, c='yellow', s=50, zorder=5,
                   edgecolors='black', linewidths=1)

        ax.set_xlabel('t_64', fontsize=11)
        ax.set_ylabel('t_32', fontsize=11)
        ax.set_title(f'{title}', fontsize=12)
        ax.legend(loc='upper left', fontsize=9)
        plt.colorbar(im, ax=ax, label='log₁₀(Error)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Compute 30×30 heatmap with optimal path')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='results/heatmap_30.png')
    parser.add_argument('--num_steps', type=int, default=18, help='Number of sampling steps')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--K', type=int, default=4, help='Hutchinson samples')
    parser.add_argument('--seed', type=int, default=42)
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

    # Step 1: Compute 30×30 heatmap
    print("\n[Step 1] Computing 30×30 heatmap...")
    t_grid, snr_grid, error_64, error_32, error_total = compute_error_heatmap_30(
        model, scheduler, x_batch, device=device, K=args.K
    )

    # Step 2: Find optimal paths
    print("\n[Step 2] Finding optimal paths...")
    path_64, cost_64 = find_optimal_path(error_64)
    path_total, cost_total = find_optimal_path(error_total)
    print(f"  DP-64 path cost: {cost_64:.6e}")
    print(f"  DP-Total path cost: {cost_total:.6e}")

    # Step 3: Find optimal N-step paths (guarantees start and end points)
    print(f"\n[Step 3] Finding optimal {args.num_steps}-step paths...")
    samples_64, cost_64_n = find_optimal_path_n_steps(error_64, args.num_steps)
    samples_total, cost_total_n = find_optimal_path_n_steps(error_total, args.num_steps)
    print(f"  DP-64 ({args.num_steps} steps): cost={cost_64_n:.6e}, points={len(samples_64)}")
    print(f"  DP-Total ({args.num_steps} steps): cost={cost_total_n:.6e}, points={len(samples_total)}")

    # Verify endpoints
    n = 30
    print(f"  DP-64 path: {samples_64[0]} -> {samples_64[-1]} (should be (0,0) -> ({n-1},{n-1}))")
    print(f"  DP-Total path: {samples_total[0]} -> {samples_total[-1]}")

    # Convert to timesteps
    ts_64_64, ts_64_32 = path_to_timesteps(samples_64, t_grid)
    ts_total_64, ts_total_32 = path_to_timesteps(samples_total, t_grid)

    # Verify timesteps end at 0
    print(f"\n  DP-64 timesteps: t_64={ts_64_64[0]}->{ts_64_64[-1]}, t_32={ts_64_32[0]}->{ts_64_32[-1]}")
    print(f"  DP-Total timesteps: t_64={ts_total_64[0]}->{ts_total_64[-1]}, t_32={ts_total_32[0]}->{ts_total_32[-1]}")

    # Save results
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    results = {
        't_grid': t_grid,
        'snr_grid': snr_grid,
        'error_64': error_64,
        'error_32': error_32,
        'error_total': error_total,
        'path_64': path_64,  # Full 58-step path
        'path_total': path_total,
        'cost_64': cost_64,
        'cost_total': cost_total,
        'samples_64': samples_64,  # N-step optimal path
        'samples_total': samples_total,
        'cost_64_n': cost_64_n,
        'cost_total_n': cost_total_n,
        'timesteps_64': {'t_64': ts_64_64, 't_32': ts_64_32},
        'timesteps_total': {'t_64': ts_total_64, 't_32': ts_total_32},
        'num_steps': args.num_steps,
    }

    results_path = args.output.replace('.png', '.pth')
    torch.save(results, results_path)
    print(f"\nResults saved: {results_path}")

    # Plot
    base = args.output.replace('.png', '')

    plot_heatmap_with_path(t_grid, error_64, path_64, samples_64,
                           f"{base}_64.png", "64×64 Error with Optimal Path")

    plot_heatmap_with_path(t_grid, error_total, path_total, samples_total,
                           f"{base}_total.png", "Total Error with Optimal Path")

    plot_comparison(t_grid, error_64, error_total, path_64, path_total,
                    samples_64, samples_total, args.output)

    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Heatmap: 30×30")
    print(f"Path length: {len(path_64)} points (58 steps)")
    print(f"Sampling: {args.num_steps} steps ({len(samples_64)} points)")
    print(f"\nDP-64 timesteps (t_64): {ts_64_64[:5]}...{ts_64_64[-3:]}")
    print(f"DP-64 timesteps (t_32): {ts_64_32[:5]}...{ts_64_32[-3:]}")


if __name__ == "__main__":
    main()
