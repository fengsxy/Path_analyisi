#!/usr/bin/env python3
"""
Evaluate LIFT with cross-Jacobian augmented DP paths.

Compares three cost functions for DP path optimization:
  1. DP-Total (existing): cost = J_HH + J_LL
  2. DP-Full:             cost = J_HH + J_LL + J_HL + J_LH
  3. DP-Self-Only:        cost = J_HH + J_LL  (same as DP-Total, for sanity check)

Usage:
    python eval_cross_dp.py --epochs 400 --device 0
    python eval_cross_dp.py --epochs 400 --device 0 --num_images 1000  # quick test
    python eval_cross_dp.py --ema --epochs 400 --device 0
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3

from model import LIFTDualTimestepModel
from scheduler import DDIMScheduler
from data import AFHQ64Dataset

from compute_heatmap_30 import (
    find_optimal_path_n_steps_lambda,
    path_to_timesteps,
)
from compute_cross_jacobian import compute_cross_jacobian_heatmap
from eval_lift_dp import (
    ddim_step,
    generate_batch_with_path,
    get_inception_features,
    calculate_fid_from_features,
    save_grid,
    load_real_features,
    load_lift_model,
)


def compute_or_load_cross_jacobian(checkpoint_path, epoch, device, output_dir,
                                   cache_dir=None, use_ema=False, suffix='',
                                   batch_size=16, K=4):
    """Compute or load cached cross-Jacobian heatmap."""
    # Cache in results/ so it's reusable across different output_dirs
    cache_path = os.path.join('results', f'cross_jacobian{suffix}_{epoch}ep.pth')

    if os.path.exists(cache_path):
        print(f"Loading cached cross-Jacobian: {cache_path}", flush=True)
        return torch.load(cache_path, weights_only=False)

    print(f"Computing cross-Jacobian heatmap (batch={batch_size}, K={K})...", flush=True)

    model = load_lift_model(checkpoint_path, device, use_ema)
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="cosine", clip_sample=True)

    dataset = AFHQ64Dataset(split='train', cache_dir=cache_dir)
    indices = np.random.choice(len(dataset), batch_size, replace=False)
    x_batch = torch.stack([dataset[int(i)] for i in indices], dim=0).to(device)

    t_grid, snr_grid, error_HH, error_HL, error_LH, error_LL = compute_cross_jacobian_heatmap(
        model, scheduler, x_batch, device=device, K=K
    )

    data = {
        't_grid': t_grid, 'snr_grid': snr_grid,
        'error_HH': error_HH, 'error_HL': error_HL,
        'error_LH': error_LH, 'error_LL': error_LL,
    }

    os.makedirs(output_dir, exist_ok=True)
    torch.save(data, cache_path)
    print(f"Saved: {cache_path}", flush=True)

    del model
    torch.cuda.empty_cache()
    return data


def build_cost_functions(cross_data):
    """Build different cost function pairs (error_64, error_32) from cross-Jacobian data."""
    eHH = cross_data['error_HH']
    eHL = cross_data['error_HL']
    eLH = cross_data['error_LH']
    eLL = cross_data['error_LL']

    # Each cost is a (error_64, error_32) tuple for λ-space DP
    # error_64 component uses eHH (+ cross terms scaled by α)
    # error_32 component uses eLL (+ cross terms scaled by α)
    costs = {
        'dp_total':   (eHH, eLL),                                                    # baseline
        'dp_w1':      (eHH + 1 * eHL, eLL + 1 * eLH),                                # α=1
        'dp_w10':     (eHH + 10 * eHL, eLL + 10 * eLH),                               # α=10
        'dp_w100':    (eHH + 100 * eHL, eLL + 100 * eLH),                             # α=100
        'dp_w1000':   (eHH + 1000 * eHL, eLL + 1000 * eLH),                           # α=1000
        'dp_mult':    (eHH * (1 + eHL / (eHH + 1e-12)), eLL * (1 + eLH / (eLL + 1e-12))),  # multiplicative
    }
    return costs


def evaluate_single_path(checkpoint_path, t_grid, cost_name, cost_pair, log_snr, num_steps,
                         feat_real, num_images, batch_size, device, output_dir, epoch,
                         use_ema=False, suffix='', seed=42):
    """Find DP path on given cost pair, generate images, compute FID."""
    # Find optimal path using λ-space DP
    error_64, error_32 = cost_pair
    samples, cost = find_optimal_path_n_steps_lambda(error_64, error_32, log_snr, num_steps)
    ts_64, ts_32 = path_to_timesteps(samples, t_grid)

    print(f"  {cost_name}: cost={cost:.4e}, path {samples[0]}->{samples[-1]}", flush=True)
    print(f"    t_64: {ts_64[0]}->{ts_64[-1]}, t_32: {ts_32[0]}->{ts_32[-1]}", flush=True)

    # Load model
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = load_lift_model(checkpoint_path, device, use_ema)
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="cosine", clip_sample=True)

    # Load Inception
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()

    # Generate
    gen_images_list = []
    num_batches = (num_images + batch_size - 1) // batch_size
    generated = 0

    for _ in tqdm(range(num_batches), desc=f"Generating ({cost_name})"):
        bs = min(batch_size, num_images - generated)
        images = generate_batch_with_path(model, scheduler, bs, ts_64, ts_32, device)
        gen_images_list.append(images.cpu())
        generated += bs

    gen_images = torch.cat(gen_images_list, dim=0)
    del gen_images_list

    # Save grid
    grid_path = os.path.join(output_dir, f'grid_lift{suffix}_{cost_name}_{epoch}ep.png')
    save_grid(gen_images, grid_path, nrow=9)
    print(f"    Grid: {grid_path}", flush=True)

    # FID
    feat_gen = get_inception_features(gen_images, inception, device, batch_size=64)
    del gen_images, model, inception
    torch.cuda.empty_cache()

    fid = calculate_fid_from_features(feat_real, feat_gen)
    return fid, cost, samples, ts_64, ts_32


def plot_path_comparison(t_grid, cost_matrices, all_samples, output_path):
    """Plot paths from different cost functions on the same heatmap."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t_np = t_grid.numpy() if torch.is_tensor(t_grid) else t_grid
    extent = [t_np[0], t_np[-1], t_np[0], t_np[-1]]

    # Use dp_total cost as background (sum of the tuple)
    e64, e32 = cost_matrices['dp_total']
    bg = e64 + e32 if torch.is_tensor(e64) else e64 + e32
    bg_np = bg.numpy() if torch.is_tensor(bg) else bg

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(np.log10(bg_np + 1e-10).T, origin='lower', extent=extent,
                   aspect='auto', cmap='viridis')

    # Diagonal reference
    ax.plot([t_np[0], t_np[-1]], [t_np[0], t_np[-1]], 'w--', lw=1.5, alpha=0.4, label='Diagonal')

    colors = {'dp_total': ('cyan', 's'), 'dp_w10': ('yellow', 'o'), 'dp_w100': ('red', 'D'),
               'dp_w1000': ('magenta', '^'), 'dp_mult': ('lime', 'v')}
    for name, samples in all_samples.items():
        color, marker = colors.get(name, ('white', 'o'))
        pts_64 = [t_np[p[0]] for p in samples]
        pts_32 = [t_np[p[1]] for p in samples]
        ax.plot(pts_64, pts_32, '-', color=color, lw=2, alpha=0.8, label=name)
        ax.scatter(pts_64, pts_32, c=color, s=40, zorder=5, edgecolors='black', linewidths=0.5, marker=marker)

    ax.set_xlabel('t_64', fontsize=12)
    ax.set_ylabel('t_32', fontsize=12)
    ax.set_title('DP Path Comparison (background: full cross-scale cost)', fontsize=13)
    ax.legend(loc='upper left', fontsize=10)
    plt.colorbar(im, ax=ax, label='log10(cost)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate LIFT with cross-Jacobian DP paths')
    parser.add_argument('--epochs', type=int, nargs='+', required=True)
    parser.add_argument('--num_images', type=int, default=15803)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_steps', type=int, default=18)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--ema', action='store_true', help='Use EMA weights')
    parser.add_argument('--costs', type=str, nargs='+', default=None,
                        help='Cost functions to evaluate (default: all)')
    parser.add_argument('--heatmap_batch_size', type=int, default=16)
    parser.add_argument('--heatmap_K', type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60, flush=True)
    print("Cross-Jacobian DP Path Evaluation", flush=True)
    print("=" * 60, flush=True)
    print(f"Epochs: {args.epochs}", flush=True)
    print(f"Images: {args.num_images}", flush=True)
    print(f"Steps: {args.num_steps}", flush=True)
    print(f"EMA: {args.ema}", flush=True)
    print("", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    ema_suffix = '_ema' if args.ema else ''

    # Load real features
    print("[Step 1] Loading real features...", flush=True)
    feat_real = load_real_features(args.num_images, device, args.cache_dir)
    print(f"Real features: {feat_real.shape}", flush=True)

    all_results = []

    for epoch in args.epochs:
        print(f"\n{'='*60}", flush=True)
        print(f"Epoch {epoch}", flush=True)
        print(f"{'='*60}", flush=True)

        if args.ema:
            ckpt_path = f'checkpoints/lift_ema_{epoch}ep.pth'
        else:
            ckpt_path = f'checkpoints/lift_dual_timestep_{epoch}ep.pth'

        if not os.path.exists(ckpt_path):
            print(f"[Skip] Checkpoint not found: {ckpt_path}", flush=True)
            continue

        # Compute or load cross-Jacobian
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        cross_data = compute_or_load_cross_jacobian(
            ckpt_path, epoch, device, args.output_dir,
            cache_dir=args.cache_dir, use_ema=args.ema, suffix=ema_suffix,
            batch_size=args.heatmap_batch_size, K=args.heatmap_K
        )

        # Build cost functions
        costs = build_cost_functions(cross_data)
        t_grid = cross_data['t_grid']
        log_snr = torch.log(cross_data['snr_grid'])

        # Print cross-scale stats
        eHH = cross_data['error_HH']
        eHL = cross_data['error_HL']
        eLH = cross_data['error_LH']
        eLL = cross_data['error_LL']
        print(f"\n  Cross-scale stats:", flush=True)
        print(f"    J_HL/J_HH mean ratio: {eHL.mean()/eHH.mean():.4f}", flush=True)
        print(f"    J_LH/J_LL mean ratio: {eLH.mean()/eLL.mean():.4f}", flush=True)
        print(f"    Cross contribution: {(eHL+eLH).mean()/(eHH+eLL+eHL+eLH).mean():.4f}", flush=True)

        # Evaluate each cost function
        epoch_results = {'epoch': epoch}
        all_samples = {}

        cost_names_to_eval = args.costs if args.costs else list(costs.keys())
        for cost_name in cost_names_to_eval:
            fid, cost, samples, ts_64, ts_32 = evaluate_single_path(
                ckpt_path, t_grid, cost_name, costs[cost_name], log_snr, args.num_steps,
                feat_real, args.num_images, args.batch_size, device, args.output_dir,
                epoch, use_ema=args.ema, suffix=ema_suffix, seed=args.seed
            )
            print(f"  FID ({cost_name}): {fid:.2f}", flush=True)
            epoch_results[f'fid_{cost_name}'] = fid
            epoch_results[f'cost_{cost_name}'] = cost
            all_samples[cost_name] = samples

        all_results.append(epoch_results)

        # Plot path comparison
        plot_path = os.path.join(args.output_dir, f'cross_dp_paths{ema_suffix}_{epoch}ep.png')
        plot_path_comparison(t_grid, costs, all_samples, plot_path)

    # Save CSV
    csv_path = os.path.join(args.output_dir, f'fid_cross_dp{ema_suffix}_results.csv')
    cost_names = args.costs if args.costs else ['dp_total', 'dp_w10', 'dp_w100', 'dp_w1000', 'dp_mult']
    with open(csv_path, 'w') as f:
        header = "Epoch," + ",".join(f"FID_{n}" for n in cost_names) + "\n"
        f.write(header)
        for r in all_results:
            vals = ",".join(f"{r[f'fid_{n}']:.4f}" for n in cost_names)
            f.write(f"{r['epoch']},{vals}\n")

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("Results Summary", flush=True)
    print(f"{'='*60}", flush=True)
    header = f"{'Epoch':>6}" + "".join(f" {n:>12}" for n in cost_names)
    print(header, flush=True)
    for r in all_results:
        vals = "".join(f" {r[f'fid_{n}']:>12.2f}" for n in cost_names)
        print(f"{r['epoch']:>6}{vals}", flush=True)
    print(f"\nSaved: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
