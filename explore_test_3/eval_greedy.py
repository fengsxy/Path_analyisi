#!/usr/bin/env python3
"""
Evaluate LIFT with Greedy Adaptive Path Sampling.

Compares greedy sample-adaptive paths against DP-Total (fixed path).

Usage:
    python eval_greedy.py --ema --epochs 400 --device 1 --num_images 1000 --force --output_dir /tmp/greedy_test
    python eval_greedy.py --ema --epochs 200 400 600 800 1000 --device 1 --num_images 15803
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import LIFTDualTimestepModel
from scheduler import DDIMScheduler
from data import AFHQ64Dataset
from eval_lift_dp import (
    load_lift_model, load_real_features, get_inception_features,
    calculate_fid_from_features, save_grid, generate_batch_with_path,
)
from compute_heatmap_30 import (
    compute_error_heatmap_30, find_optimal_path_n_steps_lambda,
    path_to_timesteps, plot_comparison,
)
from greedy_sampler import generate_batch_greedy

from pytorch_fid.inception import InceptionV3


def plot_greedy_paths(paths_64, paths_32, output_path, dp_timesteps_64=None, dp_timesteps_32=None):
    """Plot average greedy path with std bands, optionally overlaying DP-Total path.

    Args:
        paths_64: [B, num_steps+1] integer timesteps
        paths_32: [B, num_steps+1] integer timesteps
        output_path: where to save the plot
        dp_timesteps_64: list of DP-Total timesteps for 64-scale (optional)
        dp_timesteps_32: list of DP-Total timesteps for 32-scale (optional)
    """
    paths_64_np = paths_64.float().numpy()
    paths_32_np = paths_32.float().numpy()

    mean_64 = paths_64_np.mean(axis=0)
    std_64 = paths_64_np.std(axis=0)
    mean_32 = paths_32_np.mean(axis=0)
    std_32 = paths_32_np.std(axis=0)
    steps = np.arange(len(mean_64))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: t_64 and t_32 vs step
    ax = axes[0]
    ax.plot(steps, mean_64, 'b-', label='t_64 (greedy mean)', linewidth=2)
    ax.fill_between(steps, mean_64 - std_64, mean_64 + std_64, alpha=0.2, color='blue')
    ax.plot(steps, mean_32, 'r-', label='t_32 (greedy mean)', linewidth=2)
    ax.fill_between(steps, mean_32 - std_32, mean_32 + std_32, alpha=0.2, color='red')
    if dp_timesteps_64 is not None:
        ax.plot(range(len(dp_timesteps_64)), dp_timesteps_64, 'b--', label='t_64 (DP-Total)', linewidth=1.5, alpha=0.7)
    if dp_timesteps_32 is not None:
        ax.plot(range(len(dp_timesteps_32)), dp_timesteps_32, 'r--', label='t_32 (DP-Total)', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Timestep')
    ax.set_title('Greedy vs DP-Total: Timestep Schedule')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: 2D path (t_64 vs t_32)
    ax = axes[1]
    ax.plot(mean_64, mean_32, 'g-', label='Greedy (mean)', linewidth=2, zorder=3)
    ax.scatter(mean_64, mean_32, c=steps, cmap='viridis', s=30, zorder=4)
    # Plot a few individual sample paths (thin lines)
    n_show = min(5, paths_64_np.shape[0])
    for i in range(n_show):
        ax.plot(paths_64_np[i], paths_32_np[i], '-', alpha=0.15, color='gray', linewidth=0.5)
    if dp_timesteps_64 is not None and dp_timesteps_32 is not None:
        ax.plot(dp_timesteps_64, dp_timesteps_32, 'r--', label='DP-Total', linewidth=2, zorder=2)
    # Diagonal reference
    ax.plot([999, 0], [999, 0], 'k:', alpha=0.3, label='Diagonal')
    ax.set_xlabel('t_64')
    ax.set_ylabel('t_32')
    ax.set_title('2D Path: t_64 vs t_32')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved path plot: {output_path}", flush=True)


def compute_or_load_dp_total(checkpoint_path, epoch, device, output_dir,
                              num_steps=18, use_ema=False):
    """Compute or load DP-Total path for comparison."""
    ema_suffix = '_ema' if use_ema else ''
    heatmap_path = os.path.join(output_dir, f'heatmap_30{ema_suffix}_{epoch}ep.pth')

    # Check parent results dir for cached heatmap
    parent_results = os.path.join(os.path.dirname(__file__), '..', 'results')
    parent_heatmap = os.path.join(parent_results, f'heatmap_30{ema_suffix}_{epoch}ep.pth')

    cache_path = None
    if os.path.exists(heatmap_path):
        cache_path = heatmap_path
    elif os.path.exists(parent_heatmap):
        cache_path = parent_heatmap

    if cache_path is not None:
        print(f"  Loading cached heatmap: {cache_path}", flush=True)
        data = torch.load(cache_path, weights_only=False)
        t_grid = data['t_grid']
        error_64 = data['error_64']
        error_32 = data['error_32']
        log_snr = torch.log(data['snr_grid'])

        samples_total, cost_total = find_optimal_path_n_steps_lambda(
            error_64, error_32, log_snr, num_steps
        )
        ts_total_64, ts_total_32 = path_to_timesteps(samples_total, t_grid)
        return ts_total_64, ts_total_32

    # Need to compute from scratch
    print(f"  Computing 30x30 heatmap for DP-Total...", flush=True)
    model = load_lift_model(checkpoint_path, device, use_ema)
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="cosine", clip_sample=True)

    dataset = AFHQ64Dataset(split='train')
    indices = np.random.choice(len(dataset), 16, replace=False)
    x_batch = torch.stack([dataset[int(i)] for i in indices], dim=0).to(device)

    t_grid, snr_grid, error_64, error_32, error_total = compute_error_heatmap_30(
        model, scheduler, x_batch, device=device, K=4
    )

    log_snr = torch.log(snr_grid)
    samples_total, cost_total = find_optimal_path_n_steps_lambda(
        error_64, error_32, log_snr, num_steps
    )
    ts_total_64, ts_total_32 = path_to_timesteps(samples_total, t_grid)

    # Cache
    os.makedirs(output_dir, exist_ok=True)
    torch.save({
        't_grid': t_grid, 'snr_grid': snr_grid,
        'error_64': error_64, 'error_32': error_32, 'error_total': error_total,
    }, heatmap_path)

    del model
    torch.cuda.empty_cache()

    return ts_total_64, ts_total_32


def evaluate_greedy(checkpoint_path, feat_real, num_images, batch_size, device,
                    output_dir, epoch, num_steps=18, use_ema=False, bps=None,
                    seed=42):
    """Generate images with greedy sampler and compute FID."""
    model = load_lift_model(checkpoint_path, device, use_ema)
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="cosine", clip_sample=True)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()

    gen_images_list = []
    all_paths_64 = []
    all_paths_32 = []
    num_batches = (num_images + batch_size - 1) // batch_size
    generated = 0

    for b in tqdm(range(num_batches), desc="Generating (greedy)"):
        bs = min(batch_size, num_images - generated)
        images, p64, p32 = generate_batch_greedy(
            model, scheduler, bs, num_steps, device,
            bps=bps, verbose=(b == 0)
        )
        gen_images_list.append(images.cpu())
        all_paths_64.append(p64)
        all_paths_32.append(p32)
        generated += bs

    gen_images = torch.cat(gen_images_list, dim=0)
    paths_64 = torch.cat(all_paths_64, dim=0)  # [N, num_steps+1]
    paths_32 = torch.cat(all_paths_32, dim=0)
    del gen_images_list, all_paths_64, all_paths_32

    # Save grid
    ema_suffix = '_ema' if use_ema else ''
    grid_path = os.path.join(output_dir, f'grid_greedy{ema_suffix}_{epoch}ep.png')
    save_grid(gen_images, grid_path, nrow=9)
    print(f"  Saved: {grid_path}", flush=True)

    # FID
    feat_gen = get_inception_features(gen_images, inception, device, batch_size=64)
    del gen_images, model, inception
    torch.cuda.empty_cache()

    fid = calculate_fid_from_features(feat_real, feat_gen)
    return fid, paths_64, paths_32


def evaluate_dp_total(checkpoint_path, dp_ts_64, dp_ts_32, feat_real, num_images,
                      batch_size, device, output_dir, epoch, use_ema=False, seed=42):
    """Generate images with DP-Total path and compute FID."""
    model = load_lift_model(checkpoint_path, device, use_ema)
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="cosine", clip_sample=True)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()

    gen_images_list = []
    num_batches = (num_images + batch_size - 1) // batch_size
    generated = 0

    for _ in tqdm(range(num_batches), desc="Generating (DP-Total)"):
        bs = min(batch_size, num_images - generated)
        images = generate_batch_with_path(model, scheduler, bs, dp_ts_64, dp_ts_32, device)
        gen_images_list.append(images.cpu())
        generated += bs

    gen_images = torch.cat(gen_images_list, dim=0)
    del gen_images_list

    ema_suffix = '_ema' if use_ema else ''
    grid_path = os.path.join(output_dir, f'grid_dp_total{ema_suffix}_{epoch}ep.png')
    save_grid(gen_images, grid_path, nrow=9)
    print(f"  Saved: {grid_path}", flush=True)

    feat_gen = get_inception_features(gen_images, inception, device, batch_size=64)
    del gen_images, model, inception
    torch.cuda.empty_cache()

    fid = calculate_fid_from_features(feat_real, feat_gen)
    return fid


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate LIFT with Greedy Adaptive Path')
    parser.add_argument('--epochs', type=int, nargs='+', required=True)
    parser.add_argument('--num_images', type=int, default=15803)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_steps', type=int, default=18)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--force', action='store_true', help='Force re-evaluation')
    parser.add_argument('--ema', action='store_true', help='Use EMA weights')
    parser.add_argument('--bps', type=float, default=None, help='BPS budget (None=auto)')
    parser.add_argument('--skip_dp', action='store_true', help='Skip DP-Total comparison')
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 50, flush=True)
    print("Greedy Adaptive Path Evaluation", flush=True)
    print("=" * 50, flush=True)
    print(f"Epochs: {args.epochs}", flush=True)
    print(f"Images: {args.num_images}", flush=True)
    print(f"Steps: {args.num_steps}", flush=True)
    print(f"EMA: {args.ema}", flush=True)
    print(f"BPS: {args.bps if args.bps else 'auto'}", flush=True)
    print("", flush=True)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # Real features (use parent cache)
    parent_cache = os.path.join(os.path.dirname(__file__), '..', 'results', 'real_features.npz')
    print("\n[Step 1] Loading real features...", flush=True)
    feat_real = load_real_features(args.num_images, device, cache_path=parent_cache)
    print(f"Real features shape: {feat_real.shape}", flush=True)

    results = []
    ema_suffix = '_ema' if args.ema else ''

    for epoch in args.epochs:
        print(f"\n{'='*50}", flush=True)
        print(f"Epoch {epoch}", flush=True)
        print(f"{'='*50}", flush=True)

        if args.ema:
            ckpt_path = os.path.join(os.path.dirname(__file__), '..', f'checkpoints/lift_ema_{epoch}ep.pth')
        else:
            ckpt_path = os.path.join(os.path.dirname(__file__), '..', f'checkpoints/lift_dual_timestep_{epoch}ep.pth')

        if not os.path.exists(ckpt_path):
            print(f"[Skip] Checkpoint not found: {ckpt_path}", flush=True)
            continue

        grid_greedy = os.path.join(args.output_dir, f'grid_greedy{ema_suffix}_{epoch}ep.png')
        if os.path.exists(grid_greedy) and not args.force:
            print(f"[Skip] Already evaluated: {epoch}ep", flush=True)
            continue

        # --- Greedy ---
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        print(f"\n[Greedy] Generating...", flush=True)
        fid_greedy, paths_64, paths_32 = evaluate_greedy(
            ckpt_path, feat_real, args.num_images, args.batch_size, device,
            args.output_dir, epoch, args.num_steps, args.ema, args.bps, args.seed
        )
        print(f"  FID (Greedy): {fid_greedy:.2f}", flush=True)

        # --- DP-Total comparison ---
        fid_dp_total = float('nan')
        dp_ts_64, dp_ts_32 = None, None

        if not args.skip_dp:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

            print(f"\n[DP-Total] Computing path...", flush=True)
            dp_ts_64, dp_ts_32 = compute_or_load_dp_total(
                ckpt_path, epoch, device, args.output_dir,
                args.num_steps, args.ema
            )
            print(f"  DP-Total path: {len(dp_ts_64)} points", flush=True)

            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

            print(f"[DP-Total] Generating...", flush=True)
            fid_dp_total = evaluate_dp_total(
                ckpt_path, dp_ts_64, dp_ts_32, feat_real, args.num_images,
                args.batch_size, device, args.output_dir, epoch, args.ema, args.seed
            )
            print(f"  FID (DP-Total): {fid_dp_total:.2f}", flush=True)

        # --- Path visualization ---
        path_plot = os.path.join(args.output_dir, f'paths_greedy{ema_suffix}_{epoch}ep.png')
        plot_greedy_paths(paths_64, paths_32, path_plot, dp_ts_64, dp_ts_32)

        results.append((epoch, fid_greedy, fid_dp_total))

    # Save CSV
    csv_path = os.path.join(args.output_dir, f'fid_greedy{ema_suffix}_results.csv')
    with open(csv_path, 'w') as f:
        f.write("Epoch,FID_Greedy,FID_DP_Total\n")
        for epoch, fg, fd in results:
            f.write(f"{epoch},{fg:.4f},{fd:.4f}\n")

    print(f"\n{'='*50}", flush=True)
    print("Results Summary", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"{'Epoch':>6} {'Greedy':>10} {'DP-Total':>10}", flush=True)
    for epoch, fg, fd in results:
        print(f"{epoch:>6} {fg:>10.2f} {fd:>10.2f}", flush=True)
    print(f"\nSaved to: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
