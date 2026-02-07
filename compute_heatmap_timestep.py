#!/usr/bin/env python3
"""
Compute 100×100 Error Heatmap in Timestep Space with DDIM-aware DP Path.

Key features:
1. Uses timestep (0-999) as coordinates instead of SNR (gamma)
2. 100×100 grid for finer resolution
3. DP path considers DDIM step constraints:
   - Both t_64 and t_32 must be monotonically decreasing
   - Given num_steps, find optimal path from (t_max, t_max) to (0, 0)

Usage:
    python compute_heatmap_timestep.py --checkpoint checkpoints/lift_dual_timestep_1000ep.pth \
        --output results/heatmap_timestep_1000ep.png --device 0
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
    """Compute the chain-rule correction factor."""
    return 1.0 / (snr * (1.0 + snr))


def timestep_to_snr(t, scheduler):
    """Convert timestep to SNR."""
    if t >= len(scheduler.alphas_cumprod):
        t = len(scheduler.alphas_cumprod) - 1
    alpha_bar = scheduler.alphas_cumprod[int(t)]
    if alpha_bar >= 1.0:
        return 1e6  # Very high SNR for t=0
    return (alpha_bar / (1 - alpha_bar)).item()


def add_noise_at_timestep(x, t, scheduler):
    """Add noise at given timestep using scheduler."""
    noise = torch.randn_like(x)
    t_tensor = torch.tensor([t], device=x.device).expand(x.shape[0])
    noisy = scheduler.add_noise(x, noise, t_tensor)
    return noisy


def get_Hv_vmap_4d(f, z, v, K=8, chunk_size=None):
    """Compute v^T (J ⊙ J) v using Hutchinson estimator."""
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


def compute_error_heatmap_timestep(model, scheduler, x_batch, num_points=100, K=4, device='cpu', apply_chain_rule=True):
    """
    Compute error heatmap in timestep space.

    Args:
        model: Trained LIFTDualTimestepModel
        scheduler: DDIM scheduler
        x_batch: Clean image batch [B, 3, 64, 64]
        num_points: Number of points per axis (default 100)
        K: Hutchinson samples
        apply_chain_rule: Whether to apply the chain-rule factor

    Returns:
        t_grid: Timestep grid (0 to 999)
        snr_grid: Corresponding SNR values
        error_scale0: Error for 64×64 scale
        error_scale1: Error for 32×32 scale
        error_total: Total error
    """
    # Create timestep grid (0 to 999, but we sample num_points)
    t_grid = torch.linspace(0, 999, steps=num_points, device=device).long()

    # Compute corresponding SNR values
    snr_grid = torch.tensor([timestep_to_snr(t.item(), scheduler) for t in t_grid], device=device)

    error_scale0 = torch.zeros(num_points, num_points, device=device)
    error_scale1 = torch.zeros(num_points, num_points, device=device)

    # Create 32x32 version of clean images
    x_batch_32 = F.interpolate(x_batch, size=(32, 32), mode='bilinear', align_corners=False)

    model.eval()

    total_iters = num_points * num_points
    pbar = tqdm(total=total_iters, desc="Computing error heatmap (timestep space)")

    for i, t_64 in enumerate(t_grid):
        t_64_int = t_64.item()
        snr_64 = snr_grid[i].item()
        gamma_factor_0 = chain_rule_factor(snr_64) if apply_chain_rule else 1.0

        for j, t_32 in enumerate(t_grid):
            t_32_int = t_32.item()
            snr_32 = snr_grid[j].item()
            gamma_factor_1 = chain_rule_factor(snr_32) if apply_chain_rule else 1.0

            # Add noise at respective timesteps
            z_64 = add_noise_at_timestep(x_batch, t_64_int, scheduler)
            z_32 = add_noise_at_timestep(x_batch_32, t_32_int, scheduler)

            t_64_tensor = torch.tensor([t_64_int], device=device).expand(x_batch.shape[0])
            t_32_tensor = torch.tensor([t_32_int], device=device).expand(x_batch.shape[0])

            # Define function for 64x64 scale
            def f_64(z_in):
                noise_pred_64, _ = model(z_in, z_32, t_64_tensor, t_32_tensor)
                return noise_pred_64

            # Define function for 32x32 scale
            def f_32(z_in):
                _, noise_pred_32 = model(z_64, z_in, t_64_tensor, t_32_tensor)
                return noise_pred_32

            v_64 = torch.ones_like(z_64[:, :1]) / (z_64.shape[-2] * z_64.shape[-1])
            v_32 = torch.ones_like(z_32[:, :1]) / (z_32.shape[-2] * z_32.shape[-1])

            with torch.no_grad():
                vhv_64_xt = get_vHv(f_64, z_64, v_64, K=K, chunk_size=None)
                vhv_32_xt = get_vHv(f_32, z_32, v_32, K=K, chunk_size=None)

                error_scale0[i, j] = vhv_64_xt.mean() * gamma_factor_0
                error_scale1[i, j] = vhv_32_xt.mean() * gamma_factor_1

            pbar.update(1)

    pbar.close()

    error_total = error_scale0 + error_scale1

    return t_grid.cpu(), snr_grid.cpu(), error_scale0.cpu(), error_scale1.cpu(), error_total.cpu()


def compute_dp_optimal_path(error_matrix, num_steps, num_points):
    """
    Compute optimal path using 2D DP.

    Problem:
    - Start: (0, 0) - bottom-left corner (high noise: t_64=999, t_32=999)
    - End: (num_points-1, num_points-1) - top-right corner (low noise: t_64=0, t_32=0)
    - Steps: exactly num_steps (num_steps+1 points)
    - Constraint: can only move right/up, i.e., i' >= i and j' >= j, at least one strictly greater
    - Objective: minimize sum of errors along path

    Args:
        error_matrix: [num_points, num_points] error values, E[i][j]
        num_steps: Number of steps (e.g., 18)
        num_points: Grid size (e.g., 100)

    Returns:
        path: List of (i, j) indices from (0,0) to (num_points-1, num_points-1)
        total_cost: Total accumulated error
    """
    error_np = error_matrix.numpy() if torch.is_tensor(error_matrix) else error_matrix

    INF = float('inf')

    # dp[k][i][j] = minimum cost to reach (i, j) at step k
    # We use dictionary for sparse storage since not all (i,j) are reachable at step k
    dp = [{} for _ in range(num_steps + 1)]

    # Initialize: step 0 at (0, 0)
    dp[0][(0, 0)] = (error_np[0, 0], None)  # (cost, parent)

    # Fill DP table
    print(f"Computing DP path with {num_steps} steps...")
    for k in range(num_steps):
        # For each position at step k, try all possible next positions
        for (i, j), (cost, _) in dp[k].items():
            # Can move to any (i', j') where i' >= i, j' >= j, and at least one strictly greater
            # To limit search space, we set a max jump distance
            max_jump = (num_points - 1) // (num_steps - k) + 2  # adaptive max jump

            for di in range(0, min(max_jump, num_points - i)):
                for dj in range(0, min(max_jump, num_points - j)):
                    if di == 0 and dj == 0:
                        continue  # must move at least one step

                    ni, nj = i + di, j + dj

                    if ni >= num_points or nj >= num_points:
                        continue

                    new_cost = cost + error_np[ni, nj]

                    if (ni, nj) not in dp[k + 1] or new_cost < dp[k + 1][(ni, nj)][0]:
                        dp[k + 1][(ni, nj)] = (new_cost, (i, j))

    # Find best path that ends at or near (num_points-1, num_points-1)
    target = (num_points - 1, num_points - 1)

    if target in dp[num_steps]:
        best_cost = dp[num_steps][target][0]
        best_end = target
    else:
        # Find closest ending point
        best_cost = INF
        best_end = None
        for (i, j), (cost, _) in dp[num_steps].items():
            # Prefer points closer to target
            dist = abs(i - (num_points - 1)) + abs(j - (num_points - 1))
            adjusted_cost = cost + dist * 1e-6  # small penalty for distance
            if adjusted_cost < best_cost:
                best_cost = cost
                best_end = (i, j)

    if best_end is None:
        print("Warning: DP failed to find valid path, using diagonal")
        path = []
        for k in range(num_steps + 1):
            idx = int(round(k * (num_points - 1) / num_steps))
            path.append((idx, idx))
        return path, sum(error_np[i, i] for i, _ in path)

    # Backtrack to find path
    path = []
    pos = best_end
    for k in range(num_steps, -1, -1):
        path.append(pos)
        if k > 0 and pos in dp[k]:
            _, parent = dp[k][pos]
            pos = parent

    path = path[::-1]

    return path, best_cost


def compute_dp_path_2d(error_matrix, num_steps, num_points):
    """
    Compute true 2D DP optimal path where both t_64 and t_32 can vary.

    Uses band constraint to make it tractable:
    - At step k, both i and j should be roughly at (num_steps-k)/num_steps * (num_points-1)
    - Allow deviation within a band

    Complexity: O(num_steps × band² × band²) ≈ O(num_steps × band⁴)
    With band=20, this is about 50 × 160000 = 8M operations

    Args:
        error_matrix: [num_points, num_points] error values
        num_steps: Number of generation steps
        num_points: Grid size

    Returns:
        path: List of (i, j) indices
        total_cost: Total accumulated error
    """
    error_np = error_matrix.numpy() if torch.is_tensor(error_matrix) else error_matrix

    INF = float('inf')
    band = min(20, num_points // 5)  # Band width around target

    # Target positions at each step
    def target_pos(k):
        t = (num_steps - k) * (num_points - 1) / num_steps
        return int(round(t)), int(round(t))

    # Valid range at step k
    def valid_range(k):
        ti, tj = target_pos(k)
        i_min = max(0, ti - band)
        i_max = min(num_points - 1, ti + band)
        j_min = max(0, tj - band)
        j_max = min(num_points - 1, tj + band)
        return i_min, i_max, j_min, j_max

    # DP with dictionary for sparse storage
    # dp[k][(i,j)] = (min_cost, parent)
    dp = [{} for _ in range(num_steps + 1)]

    # Initialize
    i0, j0 = num_points - 1, num_points - 1
    dp[0][(i0, j0)] = (error_np[i0, j0], None)

    # Fill DP
    for k in range(1, num_steps + 1):
        i_min, i_max, j_min, j_max = valid_range(k)

        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                best_cost = INF
                best_parent = None

                # Check all valid predecessors
                for (pi, pj), (pcost, _) in dp[k-1].items():
                    # Must have i <= pi and j <= pj (monotonic decrease)
                    # And at least one must strictly decrease
                    if i <= pi and j <= pj and (i < pi or j < pj):
                        cost = pcost + error_np[i, j]
                        if cost < best_cost:
                            best_cost = cost
                            best_parent = (pi, pj)

                if best_cost < INF:
                    dp[k][(i, j)] = (best_cost, best_parent)

    # Find best ending position
    best_cost = INF
    best_end = None
    for (i, j), (cost, _) in dp[num_steps].items():
        if cost < best_cost:
            best_cost = cost
            best_end = (i, j)

    if best_end is None:
        # Fallback to diagonal path
        print("Warning: 2D DP failed, using diagonal path")
        path = []
        for k in range(num_steps + 1):
            i = int(round((num_steps - k) * (num_points - 1) / num_steps))
            path.append((i, i))
        return path, sum(error_np[i, i] for i, _ in path)

    # Backtrack
    path = []
    pos = best_end
    for k in range(num_steps, -1, -1):
        path.append(pos)
        if k > 0:
            _, parent = dp[k][pos]
            pos = parent

    path = path[::-1]

    return path, best_cost


def plot_heatmaps_with_paths(t_grid, snr_grid, error_scale0, error_scale1, error_total,
                              path_64, path_total, output_path, title_suffix=""):
    """Plot all three heatmaps with their respective DP paths."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    t_np = t_grid.numpy() if torch.is_tensor(t_grid) else t_grid
    snr_np = snr_grid.numpy() if torch.is_tensor(snr_grid) else snr_grid

    # Extent for imshow (in timestep space)
    extent = [t_np[0], t_np[-1], t_np[0], t_np[-1]]

    data = [
        (error_scale0, path_64, 'Scale 0 (64×64) Error', 'r', 'DP-64'),
        (error_scale1, None, 'Scale 1 (32×32) Error', None, None),
        (error_total, path_total, 'Total Error', 'cyan', 'DP-Total'),
    ]

    for ax, (error, path, title, color, path_label) in zip(axes, data):
        error_np = error.numpy() if torch.is_tensor(error) else error

        # Plot heatmap (transpose for correct orientation: x=t_64, y=t_32)
        im = ax.imshow(np.log10(error_np + 1e-10).T, origin='lower', extent=extent,
                       aspect='auto', cmap='viridis')

        # Plot diagonal for reference
        ax.plot([t_np[0], t_np[-1]], [t_np[0], t_np[-1]], 'w--', linewidth=1, alpha=0.5, label='Diagonal')

        # Plot DP path
        if path and color:
            path_t64 = [t_np[p[0]] for p in path]
            path_t32 = [t_np[p[1]] for p in path]
            ax.plot(path_t64, path_t32, color=color, linewidth=2, label=path_label)
            ax.scatter(path_t64[::5], path_t32[::5], c=color, s=10, zorder=5)  # Mark every 5th point

        ax.set_xlabel('t_64 (timestep)')
        ax.set_ylabel('t_32 (timestep)')
        ax.set_title(title)
        ax.legend(loc='upper left', fontsize=8)
        plt.colorbar(im, ax=ax, label='log₁₀(vHv)')

    plt.suptitle(f'Error Heatmap (Timestep Space){title_suffix}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_single_heatmap_with_path(t_grid, error_matrix, path, output_path, title, path_label, color='r'):
    """Plot a single heatmap with DP path for detailed view."""
    fig, ax = plt.subplots(figsize=(8, 7))

    t_np = t_grid.numpy() if torch.is_tensor(t_grid) else t_grid
    error_np = error_matrix.numpy() if torch.is_tensor(error_matrix) else error_matrix

    extent = [t_np[0], t_np[-1], t_np[0], t_np[-1]]

    im = ax.imshow(np.log10(error_np + 1e-10).T, origin='lower', extent=extent,
                   aspect='auto', cmap='viridis')

    # Diagonal
    ax.plot([t_np[0], t_np[-1]], [t_np[0], t_np[-1]], 'w--', linewidth=1.5, alpha=0.7, label='Diagonal')

    # DP path
    if path:
        path_t64 = [t_np[p[0]] for p in path]
        path_t32 = [t_np[p[1]] for p in path]
        ax.plot(path_t64, path_t32, color=color, linewidth=2.5, label=f'{path_label} ({len(path)} steps)')
        ax.scatter(path_t64[::5], path_t32[::5], c=color, s=20, zorder=5)

    ax.set_xlabel('t_64 (timestep for 64×64)', fontsize=12)
    ax.set_ylabel('t_32 (timestep for 32×32)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log₁₀(Error)', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Compute error heatmap in timestep space')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--cache_dir', type=str, default=None, help='HuggingFace cache directory')
    parser.add_argument('--device', type=int, default=0, help='CUDA device')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of images for estimation')
    parser.add_argument('--num_points', type=int, default=100, help='Grid size (default 100)')
    parser.add_argument('--num_steps', type=int, default=18, help='Number of generation steps for DP path (default 18)')
    parser.add_argument('--K', type=int, default=4, help='Hutchinson samples')
    parser.add_argument('--output', type=str, default='results/heatmap_timestep.png', help='Output path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_chain_rule', action='store_true', help='Disable chain-rule factor')
    parser.add_argument('--dp_method', type=str, choices=['optimal', '2d_band'], default='optimal',
                        help='DP method: optimal (full 2D DP) or 2d_band (with band constraint)')
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

    # Compute error heatmap
    print(f"\nComputing {args.num_points}×{args.num_points} error heatmap in timestep space...")
    print(f"Hutchinson samples: {args.K}")

    t_grid, snr_grid, error_scale0, error_scale1, error_total = compute_error_heatmap_timestep(
        model, scheduler, x_batch,
        num_points=args.num_points,
        K=args.K,
        device=device,
        apply_chain_rule=apply_chain_rule
    )

    # Compute DP paths
    print(f"\nComputing DP paths with {args.num_steps} steps (method: {args.dp_method})...")

    if args.dp_method == 'optimal':
        path_64, cost_64 = compute_dp_optimal_path(error_scale0, args.num_steps, args.num_points)
        path_total, cost_total = compute_dp_optimal_path(error_total, args.num_steps, args.num_points)
    else:
        path_64, cost_64 = compute_dp_path_2d(error_scale0, args.num_steps, args.num_points)
        path_total, cost_total = compute_dp_path_2d(error_total, args.num_steps, args.num_points)

    print(f"DP-64 path cost: {cost_64:.6e}")
    print(f"DP-Total path cost: {cost_total:.6e}")

    # Save results
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    results = {
        't_grid': t_grid,
        'snr_grid': snr_grid,
        'error_scale0': error_scale0,
        'error_scale1': error_scale1,
        'error_total': error_total,
        'path_64': path_64,
        'path_total': path_total,
        'cost_64': cost_64,
        'cost_total': cost_total,
        'num_steps': args.num_steps,
        'num_points': args.num_points,
        'dp_method': args.dp_method,
        'apply_chain_rule': apply_chain_rule,
        'args': vars(args)
    }

    results_path = args.output.replace('.png', '.pth')
    torch.save(results, results_path)
    print(f"Results saved to: {results_path}")

    # Plot combined heatmaps
    title_suffix = f" ({args.num_steps} steps, {args.dp_method})"
    plot_heatmaps_with_paths(t_grid, snr_grid, error_scale0, error_scale1, error_total,
                              path_64, path_total, args.output, title_suffix)

    # Plot individual heatmaps with paths (for README)
    base_path = args.output.replace('.png', '')
    plot_single_heatmap_with_path(t_grid, error_scale0, path_64,
                                   f"{base_path}_64_with_path.png",
                                   "64×64 Error Heatmap with DP Path", "DP-64", 'r')
    plot_single_heatmap_with_path(t_grid, error_total, path_total,
                                   f"{base_path}_total_with_path.png",
                                   "Total Error Heatmap with DP Path", "DP-Total", 'cyan')

    # Print path summary
    print("\n" + "="*60)
    print("Path Summary")
    print("="*60)
    t_np = t_grid.numpy()
    print(f"DP-64 path: {len(path_64)} points")
    print(f"  Start: t_64={t_np[path_64[0][0]]:.0f}, t_32={t_np[path_64[0][1]]:.0f}")
    print(f"  End:   t_64={t_np[path_64[-1][0]]:.0f}, t_32={t_np[path_64[-1][1]]:.0f}")
    print(f"DP-Total path: {len(path_total)} points")
    print(f"  Start: t_64={t_np[path_total[0][0]]:.0f}, t_32={t_np[path_total[0][1]]:.0f}")
    print(f"  End:   t_64={t_np[path_total[-1][0]]:.0f}, t_32={t_np[path_total[-1][1]]:.0f}")

    # Print statistics
    print("\n" + "="*60)
    print("Error Statistics")
    print("="*60)
    print(f"Scale 0 (64×64) error range: [{error_scale0.min():.6e}, {error_scale0.max():.6e}]")
    print(f"Scale 1 (32×32) error range: [{error_scale1.min():.6e}, {error_scale1.max():.6e}]")
    print(f"Total error range: [{error_total.min():.6e}, {error_total.max():.6e}]")


if __name__ == "__main__":
    main()
