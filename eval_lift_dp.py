#!/usr/bin/env python3
"""
Evaluate LIFT with DP paths (heatmap + DP-64 + DP-Total).

Features:
- Compute 30x30 heatmap and find optimal paths
- Generate images using DP-64 and DP-Total paths
- Calculate FID for both paths
- Reuse cached real features

Usage:
    python eval_lift_dp.py --epochs 1000 2000 --device 0
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3

from model import LIFTDualTimestepModel
from scheduler import DDIMScheduler
from data import AFHQ64Dataset

# Import heatmap functions
from compute_heatmap_30 import (
    compute_error_heatmap_30,
    find_optimal_path,
    find_optimal_path_n_steps_lambda,
    path_to_timesteps,
    plot_comparison
)


def ddim_step(scheduler, model_output, timestep, prev_timestep, sample, eta=0.0):
    """Single DDIM step."""
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

    return prev_sample


@torch.no_grad()
def generate_batch_with_path(model, scheduler, batch_size, timesteps_64, timesteps_32, device, eta=0.0):
    """Generate a batch using pre-computed timestep paths."""
    num_steps = len(timesteps_64)

    x_64 = torch.randn(batch_size, 3, 64, 64, device=device)
    x_32 = torch.randn(batch_size, 3, 32, 32, device=device)

    for i in range(num_steps):
        t_64 = timesteps_64[i]
        t_32 = timesteps_32[i]

        t_64_tensor = torch.tensor([t_64], device=device).expand(batch_size)
        t_32_tensor = torch.tensor([t_32], device=device).expand(batch_size)

        noise_pred_64, noise_pred_32 = model(x_64, x_32, t_64_tensor, t_32_tensor)

        # Get prev timesteps
        if i < num_steps - 1:
            prev_t_64 = timesteps_64[i + 1]
            prev_t_32 = timesteps_32[i + 1]
        else:
            prev_t_64 = 0
            prev_t_32 = 0

        x_64 = ddim_step(scheduler, noise_pred_64, t_64, prev_t_64, x_64, eta)
        x_32 = ddim_step(scheduler, noise_pred_32, t_32, prev_t_32, x_32, eta)

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
    """Calculate FID from pre-computed features."""
    mu_real = np.mean(feat_real, axis=0)
    sigma_real = np.cov(feat_real, rowvar=False)

    mu_gen = np.mean(feat_gen, axis=0)
    sigma_gen = np.cov(feat_gen, rowvar=False)

    return fid_score.calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)


def save_grid(images, output_path, nrow=9):
    """Save a grid of images."""
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


def load_real_features(num_images, device, cache_dir=None, cache_path='results/real_features.npz'):
    """Load or compute real image features."""
    if os.path.exists(cache_path):
        print(f"Loading cached real features from {cache_path}", flush=True)
        data = np.load(cache_path)
        if data['num_images'] >= num_images:
            print(f"Using cached features ({data['num_images']} images)", flush=True)
            return data['features'][:num_images]

    # Need to compute - load Inception
    print("Loading Inception model...", flush=True)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()

    print(f"Loading real images...", flush=True)
    dataset = AFHQ64Dataset(split='train', cache_dir=cache_dir)
    num_real = min(num_images, len(dataset))

    real_images_list = []
    for i in range(num_real):
        img = dataset[i]
        real_images_list.append(img)
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


def load_lift_model(checkpoint_path, device, use_ema=False):
    """Load LIFT model from checkpoint, optionally using EMA weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden_dims = checkpoint.get('hidden_dims', [64, 128, 256, 512])

    model = LIFTDualTimestepModel(hidden_dims=hidden_dims)

    if use_ema and 'ema_state' in checkpoint:
        ema_shadow = checkpoint['ema_state']['shadow']
        model_state = model.state_dict()
        for name in ema_shadow:
            if name in model_state:
                model_state[name] = ema_shadow[name]
        model.load_state_dict(model_state)
        print(f"  Loaded EMA weights (decay={checkpoint['ema_state'].get('decay', '?')})", flush=True)
    elif use_ema:
        print(f"  WARNING: --ema specified but no ema_state in checkpoint, using model_state", flush=True)
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint['model_state'])

    return model.to(device).eval()


def compute_or_load_heatmap(checkpoint_path, epoch, device, output_dir, num_steps=18,
                            cache_dir=None, use_ema=False, heatmap_suffix=''):
    """Compute heatmap or load from cache, then compute optimal N-step paths."""
    heatmap_path = os.path.join(output_dir, f'heatmap_30{heatmap_suffix}_{epoch}ep.pth')

    if os.path.exists(heatmap_path):
        print(f"Loading cached heatmap: {heatmap_path}", flush=True)
        data = torch.load(heatmap_path, weights_only=False)

        # Re-compute optimal N-step paths using new algorithm
        print(f"Computing optimal {num_steps}-step paths...", flush=True)
        t_grid = data['t_grid']
        error_64 = data['error_64']
        error_32 = data['error_32']
        error_total = data['error_total']
        log_snr = torch.log(data['snr_grid'])

        zeros = torch.zeros_like(error_64)
        samples_64, cost_64_n = find_optimal_path_n_steps_lambda(error_64, zeros, log_snr, num_steps)
        samples_total, cost_total_n = find_optimal_path_n_steps_lambda(error_64, error_32, log_snr, num_steps)

        ts_64_64, ts_64_32 = path_to_timesteps(samples_64, t_grid)
        ts_total_64, ts_total_32 = path_to_timesteps(samples_total, t_grid)

        print(f"  DP-64: {len(samples_64)} points, {samples_64[0]} -> {samples_64[-1]}", flush=True)
        print(f"  DP-Total: {len(samples_total)} points, {samples_total[0]} -> {samples_total[-1]}", flush=True)
        print(f"  Timesteps end at: t_64={ts_64_64[-1]}, t_32={ts_64_32[-1]}", flush=True)

        # Update data with new paths
        data['samples_64'] = samples_64
        data['samples_total'] = samples_total
        data['cost_64_n'] = cost_64_n
        data['cost_total_n'] = cost_total_n
        data['timesteps_64'] = {'t_64': ts_64_64, 't_32': ts_64_32}
        data['timesteps_total'] = {'t_64': ts_total_64, 't_32': ts_total_32}
        data['num_steps'] = num_steps

        return data

    print(f"Computing 30x30 heatmap...", flush=True)

    # Load model
    model = load_lift_model(checkpoint_path, device, use_ema)

    # Scheduler
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="cosine", clip_sample=True)

    # Load sample images for heatmap computation
    dataset = AFHQ64Dataset(split='train', cache_dir=cache_dir)
    indices = np.random.choice(len(dataset), 16, replace=False)
    x_batch = torch.stack([dataset[int(i)] for i in indices], dim=0).to(device)

    # Compute heatmap
    t_grid, snr_grid, error_64, error_32, error_total = compute_error_heatmap_30(
        model, scheduler, x_batch, device=device, K=4
    )

    # Find optimal full paths (58 steps)
    print("Finding optimal full paths...", flush=True)
    path_64, cost_64 = find_optimal_path(error_64)
    path_total, cost_total = find_optimal_path(error_total)

    # Find optimal N-step paths
    print(f"Finding optimal {num_steps}-step paths...", flush=True)
    log_snr = torch.log(snr_grid)
    samples_64, cost_64_n = find_optimal_path_n_steps_lambda(error_64, torch.zeros_like(error_64), log_snr, num_steps)
    samples_total, cost_total_n = find_optimal_path_n_steps_lambda(error_64, error_32, log_snr, num_steps)

    # Convert to timesteps
    ts_64_64, ts_64_32 = path_to_timesteps(samples_64, t_grid)
    ts_total_64, ts_total_32 = path_to_timesteps(samples_total, t_grid)

    print(f"  DP-64: {len(samples_64)} points, {samples_64[0]} -> {samples_64[-1]}", flush=True)
    print(f"  DP-Total: {len(samples_total)} points, {samples_total[0]} -> {samples_total[-1]}", flush=True)
    print(f"  Timesteps end at: t_64={ts_64_64[-1]}, t_32={ts_64_32[-1]}", flush=True)

    # Save results
    results = {
        't_grid': t_grid,
        'snr_grid': snr_grid,
        'error_64': error_64,
        'error_32': error_32,
        'error_total': error_total,
        'path_64': path_64,
        'path_total': path_total,
        'cost_64': cost_64,
        'cost_total': cost_total,
        'samples_64': samples_64,
        'samples_total': samples_total,
        'cost_64_n': cost_64_n,
        'cost_total_n': cost_total_n,
        'timesteps_64': {'t_64': ts_64_64, 't_32': ts_64_32},
        'timesteps_total': {'t_64': ts_total_64, 't_32': ts_total_32},
        'num_steps': num_steps,
    }

    torch.save(results, heatmap_path)
    print(f"Saved: {heatmap_path}", flush=True)

    # Plot
    plot_path = os.path.join(output_dir, f'heatmap_30{heatmap_suffix}_{epoch}ep.png')
    plot_comparison(t_grid, error_64, error_total, path_64, path_total,
                    samples_64, samples_total, plot_path)

    del model
    torch.cuda.empty_cache()

    return results


def evaluate_dp_path(checkpoint_path, heatmap_data, path_type, feat_real, num_images,
                     batch_size, device, output_dir, epoch, eta=0.0, use_ema=False, grid_suffix=''):
    """Evaluate a DP path."""
    # Load model
    model = load_lift_model(checkpoint_path, device, use_ema)

    # Scheduler
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="cosine", clip_sample=True)

    # Get timesteps
    if path_type == 'dp_64':
        timesteps = heatmap_data['timesteps_64']
    else:
        timesteps = heatmap_data['timesteps_total']

    timesteps_64 = timesteps['t_64']
    timesteps_32 = timesteps['t_32']

    print(f"  Path: {path_type}", flush=True)
    print(f"  Steps: {len(timesteps_64)}", flush=True)

    # Load Inception
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()

    # Generate images
    gen_images_list = []
    num_batches = (num_images + batch_size - 1) // batch_size
    generated = 0

    for _ in tqdm(range(num_batches), desc=f"Generating ({path_type})"):
        bs = min(batch_size, num_images - generated)
        images = generate_batch_with_path(model, scheduler, bs, timesteps_64, timesteps_32, device, eta)
        gen_images_list.append(images.cpu())
        generated += bs

    gen_images = torch.cat(gen_images_list, dim=0)
    del gen_images_list

    # Save grid
    grid_path = os.path.join(output_dir, f'grid_lift{grid_suffix}_{path_type}_{epoch}ep.png')
    save_grid(gen_images, grid_path, nrow=9)
    print(f"  Saved: {grid_path}", flush=True)

    # Extract features and compute FID
    feat_gen = get_inception_features(gen_images, inception, device, batch_size=64)
    del gen_images, model, inception
    torch.cuda.empty_cache()

    fid = calculate_fid_from_features(feat_real, feat_gen)
    return fid


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate LIFT with DP paths')
    parser.add_argument('--epochs', type=int, nargs='+', required=True)
    parser.add_argument('--num_images', type=int, default=15803)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_steps', type=int, default=18)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--force', action='store_true', help='Force re-evaluation even if results exist')
    parser.add_argument('--ema', action='store_true', help='Use EMA weights from checkpoint')
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 50, flush=True)
    print("LIFT DP Path Evaluation", flush=True)
    print("=" * 50, flush=True)
    print(f"Epochs: {args.epochs}", flush=True)
    print(f"Images: {args.num_images}", flush=True)
    print(f"Steps: {args.num_steps}", flush=True)
    print(f"EMA: {args.ema}", flush=True)
    print("", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load real features (only once!)
    print("\n[Step 1] Loading real features...", flush=True)
    feat_real = load_real_features(args.num_images, device, args.cache_dir)
    print(f"Real features shape: {feat_real.shape}", flush=True)

    # Evaluate each epoch
    print("\n[Step 2] Evaluating checkpoints...", flush=True)
    results = []

    ema_suffix = '_ema' if args.ema else ''

    for epoch in args.epochs:
        print(f"\n{'='*50}", flush=True)
        print(f"Epoch {epoch}", flush=True)
        print(f"{'='*50}", flush=True)

        if args.ema:
            ckpt_path = f'checkpoints/lift_ema_{epoch}ep.pth'
        else:
            ckpt_path = f'checkpoints/lift_dual_timestep_{epoch}ep.pth'
        grid_dp64_path = os.path.join(args.output_dir, f'grid_lift{ema_suffix}_dp_64_{epoch}ep.png')
        grid_dp_total_path = os.path.join(args.output_dir, f'grid_lift{ema_suffix}_dp_total_{epoch}ep.png')

        if not os.path.exists(ckpt_path):
            print(f"[Skip] Checkpoint not found: {ckpt_path}", flush=True)
            continue

        # Check if already evaluated (both grid files exist)
        if os.path.exists(grid_dp64_path) and os.path.exists(grid_dp_total_path) and not args.force:
            print(f"[Skip] Already evaluated: {epoch}ep", flush=True)
            continue

        # Compute or load heatmap
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        heatmap_data = compute_or_load_heatmap(
            ckpt_path, epoch, device, args.output_dir,
            num_steps=args.num_steps, cache_dir=args.cache_dir,
            use_ema=args.ema, heatmap_suffix=ema_suffix
        )

        # Evaluate DP-64
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        fid_dp64 = evaluate_dp_path(
            ckpt_path, heatmap_data, 'dp_64', feat_real,
            args.num_images, args.batch_size, device, args.output_dir, epoch, args.eta,
            use_ema=args.ema, grid_suffix=ema_suffix
        )
        print(f"  FID (DP-64): {fid_dp64:.2f}", flush=True)

        # Evaluate DP-Total
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        fid_dp_total = evaluate_dp_path(
            ckpt_path, heatmap_data, 'dp_total', feat_real,
            args.num_images, args.batch_size, device, args.output_dir, epoch, args.eta,
            use_ema=args.ema, grid_suffix=ema_suffix
        )
        print(f"  FID (DP-Total): {fid_dp_total:.2f}", flush=True)

        results.append((epoch, fid_dp64, fid_dp_total, heatmap_data['cost_64'], heatmap_data['cost_total']))

    # Save results
    csv_path = os.path.join(args.output_dir, f'fid_lift{ema_suffix}_dp_results.csv')
    with open(csv_path, 'w') as f:
        f.write("Epoch,FID_DP64,FID_DP_Total,Cost_64,Cost_Total\n")
        for epoch, fid64, fid_total, cost64, cost_total in results:
            f.write(f"{epoch},{fid64:.4f},{fid_total:.4f},{cost64:.6e},{cost_total:.6e}\n")

    print(f"\n{'='*50}", flush=True)
    print("Results Summary", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"{'Epoch':>6} {'DP-64':>10} {'DP-Total':>10}", flush=True)
    for epoch, fid64, fid_total, _, _ in results:
        print(f"{epoch:>6} {fid64:>10.2f} {fid_total:>10.2f}", flush=True)
    print(f"\nSaved to: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
