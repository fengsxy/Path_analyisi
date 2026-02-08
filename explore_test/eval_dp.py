#!/usr/bin/env python3
"""
Evaluate explore_test models with DP paths (heatmap + optimal path + FID).

For single_t and no_t models:
- Compute 30x30 error heatmap (error_64 only, no error_32)
- Find optimal N-step path via DP
- Generate images using DP path (x_32 constructed from pred_x0_64)
- Calculate FID

Usage:
    python eval_dp.py --model_type single_t --epochs 200 400 --device 0
    python eval_dp.py --model_type no_t --epochs 200 400 --device 1
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
from torch.func import jvp, vmap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scheduler import DDIMScheduler
from data import AFHQ64Dataset
from compute_heatmap_30 import (
    chain_rule_factor,
    timestep_to_snr,
    add_noise_at_timestep,
    get_vHv,
    find_optimal_path_n_steps_lambda,
    path_to_timesteps,
)

from model_single_t import SingleTimestepModel
from model_no_t import NoTimestepModel


def ddim_step(scheduler, model_output, timestep, prev_timestep, sample, eta=0.0):
    """Single DDIM step. Returns (prev_sample, pred_x0)."""
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else scheduler.final_alpha_cumprod
    )
    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5

    if scheduler.clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

    variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    std_dev_t = eta * variance**0.5

    pred_epsilon = (sample - alpha_prod_t**0.5 * pred_original_sample) / beta_prod_t**0.5
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * pred_epsilon
    prev_sample = alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction

    if eta > 0:
        noise = torch.randn_like(model_output)
        prev_sample += std_dev_t * noise

    return prev_sample, pred_original_sample


def compute_error_heatmap(model, model_type, scheduler, x_batch, device='cpu', K=4):
    """
    Compute 30x30 error heatmap for explore_test models.

    Only computes error_64 (models don't predict 32x noise).
    Grid: index 0 = t=999 (high noise), index 29 = t=0 (low noise).
    """
    num_points = 30
    t_grid = torch.linspace(999, 0, steps=num_points, device=device).long()
    snr_grid = torch.tensor([timestep_to_snr(t.item(), scheduler) for t in t_grid], device=device)

    error_64 = torch.zeros(num_points, num_points, device=device)
    x_batch_32 = F.interpolate(x_batch, size=(32, 32), mode='bilinear', align_corners=False)

    model.eval()
    pbar = tqdm(total=num_points * num_points, desc="Computing 30x30 heatmap")

    for i in range(num_points):
        t_64_int = t_grid[i].item()
        snr_64 = snr_grid[i].item()
        gamma_64 = chain_rule_factor(snr_64)

        for j in range(num_points):
            t_32_int = t_grid[j].item()

            z_64 = add_noise_at_timestep(x_batch, t_64_int, scheduler)
            z_32 = add_noise_at_timestep(x_batch_32, t_32_int, scheduler)

            t_64_tensor = torch.tensor([t_64_int], device=device).expand(x_batch.shape[0])

            if model_type == 'single_t':
                def f_64(z_in):
                    return model(z_in, z_32, t_64_tensor)
            else:
                def f_64(z_in):
                    return model(z_in, z_32)

            v_64 = torch.ones_like(z_64[:, :1]) / (64 * 64)

            with torch.no_grad():
                vhv_64 = get_vHv(f_64, z_64, v_64, K=K).mean() * gamma_64
                error_64[i, j] = vhv_64

            pbar.update(1)

    pbar.close()
    return t_grid.cpu(), snr_grid.cpu(), error_64.cpu()


@torch.no_grad()
def generate_batch_with_path(model, model_type, scheduler, batch_size,
                             timesteps_64, timesteps_32, device, eta=0.0):
    """
    Generate using DP path. x_32 constructed from pred_x0_64.

    At each step:
    1. Get noise_pred_64 from model
    2. DDIM step for x_64 (also get pred_x0_64)
    3. Construct x_32 for next step: downsample(pred_x0_64) + noise at t_32_next
    """
    num_steps = len(timesteps_64)
    x_64 = torch.randn(batch_size, 3, 64, 64, device=device)
    # Initial x_32: just downsample x_64 (both are pure noise at start)
    x_32 = F.interpolate(x_64, size=32, mode='bilinear', align_corners=False)

    for i in range(num_steps):
        t_64 = timesteps_64[i]
        t_64_tensor = torch.tensor([t_64], device=device).expand(batch_size).float()

        if model_type == 'single_t':
            noise_pred_64 = model(x_64, x_32, t_64_tensor)
        else:
            noise_pred_64 = model(x_64, x_32)

        # Get prev timesteps
        if i < num_steps - 1:
            prev_t_64 = timesteps_64[i + 1]
            prev_t_32 = timesteps_32[i + 1]
        else:
            prev_t_64 = 0
            prev_t_32 = 0

        # DDIM step for x_64 (returns prev_sample and pred_x0)
        x_64, pred_x0_64 = ddim_step(scheduler, noise_pred_64, t_64, prev_t_64, x_64, eta)

        # Construct x_32 for next step
        if i < num_steps - 1:
            pred_x0_32 = F.interpolate(pred_x0_64, size=32, mode='bilinear', align_corners=False)
            if prev_t_32 > 0:
                # Re-noise at t_32_next level
                alpha_bar_32 = scheduler.alphas_cumprod[prev_t_32]
                noise = torch.randn(batch_size, 3, 32, 32, device=device)
                x_32 = alpha_bar_32**0.5 * pred_x0_32 + (1 - alpha_bar_32)**0.5 * noise
            else:
                x_32 = pred_x0_32

    return (x_64 + 1) * 0.5


def get_inception_features(images, inception_model, device, batch_size=64):
    """Extract Inception features from images in memory."""
    features_list = []
    num_batches = (len(images) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Extracting features"):
        start = i * batch_size
        end = min(start + batch_size, len(images))
        batch = images[start:end].to(device)

        batch_resized = F.interpolate(batch, size=(299, 299), mode='bicubic', align_corners=False, antialias=True)
        batch_resized = torch.clamp(batch_resized, 0, 1)

        with torch.no_grad():
            feat = inception_model(batch_resized)[0]

        if feat.dim() == 4:
            feat = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1)

        features_list.append(feat.cpu())

    return torch.cat(features_list, dim=0).numpy()


def calculate_fid_from_features(feat_real, feat_gen):
    mu_real = np.mean(feat_real, axis=0)
    sigma_real = np.cov(feat_real, rowvar=False)
    mu_gen = np.mean(feat_gen, axis=0)
    sigma_gen = np.cov(feat_gen, rowvar=False)
    return fid_score.calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)


def save_grid(images, output_path, nrow=9):
    n = min(nrow * nrow, len(images))
    images = images[:n]
    h, w = images.shape[2], images.shape[3]
    grid = np.zeros((nrow * h, nrow * w, 3), dtype=np.uint8)
    for i, img in enumerate(images[:nrow*nrow]):
        row = i // nrow
        col = i % nrow
        img_np = img.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = img_np
    Image.fromarray(grid).save(output_path)


def load_real_features(num_images, device, cache_dir=None):
    """Load or compute real image features."""
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'real_features.npz')

    if os.path.exists(cache_path):
        print(f"Loading cached real features from {cache_path}", flush=True)
        data = np.load(cache_path)
        if data['num_images'] >= num_images:
            print(f"Using cached features ({data['num_images']} images)", flush=True)
            return data['features'][:num_images]

    print("Loading Inception model...", flush=True)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()

    print(f"Loading real images...", flush=True)
    dataset = AFHQ64Dataset(split='train', cache_dir=cache_dir)
    num_real = min(num_images, len(dataset))

    real_images_list = []
    for i in range(num_real):
        real_images_list.append(dataset[i])
        if (i + 1) % 2000 == 0:
            print(f"  Loaded {i+1}/{num_real} images", flush=True)

    real_images = torch.stack(real_images_list)
    real_images = (real_images + 1) * 0.5
    del real_images_list

    print(f"Extracting real features ({num_real} images)...", flush=True)
    features = get_inception_features(real_images, inception, device, batch_size=64)
    del real_images

    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    np.savez(cache_path, features=features, num_images=num_real)
    print(f"Cached features to {cache_path}", flush=True)
    return features


def load_model(checkpoint_path, model_type, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden_dims = checkpoint.get('hidden_dims', [64, 128, 256, 512])

    if model_type == 'single_t':
        model = SingleTimestepModel(hidden_dims=hidden_dims)
    else:
        model = NoTimestepModel(hidden_dims=hidden_dims)

    model.load_state_dict(checkpoint['model_state'])
    return model.to(device).eval()


def plot_heatmap_with_path(t_grid, error_matrix, samples, output_path, title, num_steps=18):
    """Plot heatmap with DP path and diagonal reference."""
    fig, ax = plt.subplots(figsize=(8, 7))

    t_np = t_grid.numpy() if torch.is_tensor(t_grid) else t_grid
    error_np = error_matrix.numpy() if torch.is_tensor(error_matrix) else error_matrix
    extent = [t_np[0], t_np[-1], t_np[0], t_np[-1]]
    n = len(t_np)

    im = ax.imshow(np.log10(error_np + 1e-10).T, origin='lower', extent=extent,
                   aspect='auto', cmap='viridis')

    # Diagonal reference
    ax.plot([t_np[0], t_np[-1]], [t_np[0], t_np[-1]], 'w--', linewidth=1.5, alpha=0.5)

    # Diagonal sampling path (cyan)
    diagonal_indices = np.linspace(0, n-1, num_steps + 1).astype(int)
    diag_t64 = [t_np[i] for i in diagonal_indices]
    diag_t32 = [t_np[i] for i in diagonal_indices]
    ax.plot(diag_t64, diag_t32, 'c-', linewidth=2, alpha=0.7, label=f'Diagonal ({len(diagonal_indices)} pts)')
    ax.scatter(diag_t64, diag_t32, c='cyan', s=30, zorder=4, edgecolors='black', linewidths=0.5)

    # DP path (red + yellow)
    sample_t64 = [t_np[p[0]] for p in samples]
    sample_t32 = [t_np[p[1]] for p in samples]
    ax.plot(sample_t64, sample_t32, 'r-', linewidth=2, alpha=0.8, label=f'DP ({len(samples)} pts)')
    ax.scatter(sample_t64, sample_t32, c='yellow', s=50, zorder=5, edgecolors='black', linewidths=1)

    ax.set_xlabel('t_64', fontsize=11)
    ax.set_ylabel('t_32', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    plt.colorbar(im, ax=ax, label='log10(Error)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}", flush=True)
    plt.close()


def compute_or_load_heatmap(checkpoint_path, model_type, epoch, device, output_dir,
                            num_steps=18, cache_dir=None):
    """Compute heatmap or load from cache, then compute optimal N-step path."""
    heatmap_path = os.path.join(output_dir, f'heatmap_30_{model_type}_{epoch}ep.pth')

    if os.path.exists(heatmap_path):
        print(f"Loading cached heatmap: {heatmap_path}", flush=True)
        data = torch.load(heatmap_path, weights_only=False)

        print(f"Computing optimal {num_steps}-step path...", flush=True)
        t_grid = data['t_grid']
        error_64 = data['error_64']
        snr_grid = data['snr_grid']
        log_snr = torch.log(snr_grid)

        samples, cost_n = find_optimal_path_n_steps_lambda(error_64, torch.zeros_like(error_64), log_snr, num_steps)
        ts_64, ts_32 = path_to_timesteps(samples, t_grid)

        print(f"  DP: {len(samples)} points, {samples[0]} -> {samples[-1]}", flush=True)

        data['samples'] = samples
        data['cost_n'] = cost_n
        data['timesteps'] = {'t_64': ts_64, 't_32': ts_32}
        data['num_steps'] = num_steps
        return data

    print(f"Computing 30x30 heatmap...", flush=True)
    model = load_model(checkpoint_path, model_type, device)
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="cosine", clip_sample=True)

    dataset = AFHQ64Dataset(split='train', cache_dir=cache_dir)
    indices = np.random.choice(len(dataset), 16, replace=False)
    x_batch = torch.stack([dataset[int(i)] for i in indices], dim=0).to(device)

    t_grid, snr_grid, error_64 = compute_error_heatmap(
        model, model_type, scheduler, x_batch, device=device, K=4
    )

    # Find optimal N-step path
    print(f"Finding optimal {num_steps}-step path...", flush=True)
    log_snr = torch.log(snr_grid)
    samples, cost_n = find_optimal_path_n_steps_lambda(error_64, torch.zeros_like(error_64), log_snr, num_steps)
    ts_64, ts_32 = path_to_timesteps(samples, t_grid)

    print(f"  DP: {len(samples)} points, {samples[0]} -> {samples[-1]}", flush=True)
    print(f"  Timesteps end at: t_64={ts_64[-1]}, t_32={ts_32[-1]}", flush=True)

    results = {
        't_grid': t_grid,
        'snr_grid': snr_grid,
        'error_64': error_64,
        'samples': samples,
        'cost_n': cost_n,
        'timesteps': {'t_64': ts_64, 't_32': ts_32},
        'num_steps': num_steps,
    }

    torch.save(results, heatmap_path)
    print(f"Saved: {heatmap_path}", flush=True)

    # Plot
    plot_path = os.path.join(output_dir, f'heatmap_30_{model_type}_{epoch}ep.png')
    plot_heatmap_with_path(t_grid, error_64, samples, plot_path,
                           f'{model_type} - 64x64 Error ({epoch}ep)', num_steps)

    del model
    torch.cuda.empty_cache()
    return results


def evaluate_dp_path(checkpoint_path, model_type, heatmap_data, feat_real, num_images,
                     batch_size, device, output_dir, epoch, eta=0.0):
    """Evaluate a DP path."""
    model = load_model(checkpoint_path, model_type, device)
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="cosine", clip_sample=True)

    timesteps = heatmap_data['timesteps']
    timesteps_64 = timesteps['t_64']
    timesteps_32 = timesteps['t_32']

    print(f"  Steps: {len(timesteps_64)}", flush=True)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()

    gen_images_list = []
    num_batches = (num_images + batch_size - 1) // batch_size
    generated = 0

    for _ in tqdm(range(num_batches), desc="Generating (DP)"):
        bs = min(batch_size, num_images - generated)
        images = generate_batch_with_path(model, model_type, scheduler, bs,
                                          timesteps_64, timesteps_32, device, eta)
        gen_images_list.append(images.cpu())
        generated += bs

    gen_images = torch.cat(gen_images_list, dim=0)
    del gen_images_list

    grid_path = os.path.join(output_dir, f'grid_{model_type}_dp_{epoch}ep.png')
    save_grid(gen_images, grid_path, nrow=9)
    print(f"  Saved: {grid_path}", flush=True)

    feat_gen = get_inception_features(gen_images, inception, device, batch_size=64)
    del gen_images, model, inception
    torch.cuda.empty_cache()

    fid = calculate_fid_from_features(feat_real, feat_gen)
    return fid


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate explore_test models with DP paths')
    parser.add_argument('--model_type', type=str, choices=['single_t', 'no_t'], required=True)
    parser.add_argument('--epochs', type=int, nargs='+', required=True)
    parser.add_argument('--num_images', type=int, default=15803)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_steps', type=int, default=18)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 50, flush=True)
    print("Explore Test - DP Path Evaluation", flush=True)
    print("=" * 50, flush=True)
    print(f"Model: {args.model_type}", flush=True)
    print(f"Epochs: {args.epochs}", flush=True)
    print(f"Images: {args.num_images}", flush=True)
    print(f"Steps: {args.num_steps}", flush=True)
    print("", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n[Step 1] Loading real features...", flush=True)
    feat_real = load_real_features(args.num_images, device, args.cache_dir)
    print(f"Real features shape: {feat_real.shape}", flush=True)

    print("\n[Step 2] Evaluating checkpoints...", flush=True)
    results = []

    for epoch in args.epochs:
        print(f"\n{'='*50}", flush=True)
        print(f"Epoch {epoch}", flush=True)
        print(f"{'='*50}", flush=True)

        ckpt_path = f'checkpoints/{args.model_type}_{epoch}ep.pth'
        grid_dp_path = os.path.join(args.output_dir, f'grid_{args.model_type}_dp_{epoch}ep.png')

        if not os.path.exists(ckpt_path):
            print(f"[Skip] Checkpoint not found: {ckpt_path}", flush=True)
            continue

        if os.path.exists(grid_dp_path) and not args.force:
            print(f"[Skip] Already evaluated: {epoch}ep", flush=True)
            continue

        # Compute or load heatmap
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        heatmap_data = compute_or_load_heatmap(
            ckpt_path, args.model_type, epoch, device, args.output_dir,
            num_steps=args.num_steps, cache_dir=args.cache_dir
        )

        # Evaluate DP path
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        fid_dp = evaluate_dp_path(
            ckpt_path, args.model_type, heatmap_data, feat_real,
            args.num_images, args.batch_size, device, args.output_dir, epoch, args.eta
        )
        print(f"  FID (DP): {fid_dp:.2f}", flush=True)

        results.append((epoch, fid_dp, heatmap_data['cost_n']))

    csv_path = os.path.join(args.output_dir, f'fid_{args.model_type}_dp_results.csv')
    with open(csv_path, 'w') as f:
        f.write("Epoch,FID_DP,Cost\n")
        for epoch, fid_dp, cost in results:
            f.write(f"{epoch},{fid_dp:.4f},{cost:.6e}\n")

    print(f"\n{'='*50}", flush=True)
    print("Results Summary", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"{'Epoch':>6} {'FID (DP)':>10}", flush=True)
    for epoch, fid_dp, _ in results:
        print(f"{epoch:>6} {fid_dp:>10.2f}", flush=True)
    print(f"\nSaved to: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
