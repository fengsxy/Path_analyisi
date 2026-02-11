"""Explore alternative sigma scheduling paths for SLOT model.

The DP paths from the heatmap are L-shaped (degenerate) because the two scales'
errors are nearly independent. This script tries:
1. Parameterized diagonal-offset paths (2x leads orig by k steps)
2. DP with diagonal-proximity constraint
3. Log-SNR-weighted diagonal paths
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, "/home/ylong030/slot")
import dnnlib

# ── Reuse from generate_and_fid.py ──
import pickle
from generate_and_fid import (
    convert_torus, convert_from_torus,
    _rho_schedule, _rho_interp,
    edm_sampler_torus_heun,
    load_cifar_real_features, compute_fid,
)

def load_model(path, device):
    with dnnlib.util.open_url(path) as f:
        data = pickle.load(f)
    return data['ema'].eval().to(device)

# ── New path builders ──

def build_lead_follow_schedule(num_steps, lead_steps, sigma_min=0.002, sigma_max=80.0, rho=7.0, device='cpu'):
    """2x scale leads orig by `lead_steps` steps.

    Both scales follow the same rho schedule, but 2x is shifted ahead.
    lead_steps=0 is sync, lead_steps>0 means 2x denoises faster.
    """
    # Build a longer schedule and offset
    total = num_steps + lead_steps
    full_sched = _rho_schedule(total, sigma_max, sigma_min, rho, device)

    sigmas = torch.zeros(num_steps, 3, device=device, dtype=torch.float32)
    for k in range(num_steps):
        # orig follows the first num_steps of the schedule
        sigmas[k, 0] = full_sched[k]
        # 2x is shifted ahead by lead_steps
        idx_2x = min(k + lead_steps, total - 1)
        sigmas[k, 1] = full_sched[idx_2x]
        sigmas[k, 2] = full_sched[idx_2x]
    return sigmas


def build_speed_ratio_schedule(num_steps, ratio_2x, sigma_min=0.002, sigma_max=80.0, rho=7.0, device='cpu'):
    """2x scale denoises `ratio_2x` times faster than orig.

    ratio_2x=1.0 is sync. ratio_2x=1.5 means 2x reaches sigma_min 1.5x faster.
    """
    sigmas = torch.zeros(num_steps, 3, device=device, dtype=torch.float32)
    for k in range(num_steps):
        u_orig = k / max(1, num_steps - 1)
        u_2x = min(u_orig * ratio_2x, 1.0)
        sigmas[k, 0] = _rho_interp(u_orig, sigma_max, sigma_min, rho)
        s_2x = _rho_interp(u_2x, sigma_max, sigma_min, rho)
        sigmas[k, 1] = s_2x
        sigmas[k, 2] = s_2x
    return sigmas


def build_dp_constrained_schedule(heatmap_path, max_diag_dist=3, num_steps=18,
                                   sigma_min=0.002, sigma_max=80.0, device='cpu'):
    """DP path constrained to stay within max_diag_dist of the diagonal.

    This prevents the L-shaped degenerate paths.
    """
    data = torch.load(heatmap_path, map_location='cpu', weights_only=False)
    sigma_grid = data['sigma_grid'].numpy()
    e_orig = data['error_orig'].numpy()
    e_2x = data['error_2x'].numpy()
    n = e_orig.shape[0]

    # Compute log_snr
    sigma_data = 0.5
    snr = sigma_data**2 / sigma_grid**2
    log_snr = np.log(snr)

    INF = float('inf')
    max_jump = 5
    dp = np.full((n, n, num_steps + 1), INF)
    parent = np.full((n, n, num_steps + 1, 3), -1, dtype=int)
    dp[0, 0, 0] = 0

    for k in range(num_steps):
        for i in range(n):
            for j in range(n):
                if dp[i, j, k] == INF:
                    continue
                remaining = num_steps - k - 1
                for ni in range(i, min(i + max_jump + 1, n)):
                    for nj in range(j, min(j + max_jump + 1, n)):
                        if ni == i and nj == j:
                            continue
                        # Diagonal proximity constraint
                        if abs(ni - nj) > max_diag_dist:
                            continue
                        dist_to_end = (n - 1 - ni) + (n - 1 - nj)
                        if dist_to_end > remaining * 2 * max_jump:
                            continue
                        if remaining == 0 and (ni != n - 1 or nj != n - 1):
                            continue

                        dl64 = abs(log_snr[ni] - log_snr[i])
                        dl32 = abs(log_snr[nj] - log_snr[j])
                        c64 = (e_orig[i, j] + e_orig[ni, nj]) / 2 * dl64
                        c32 = (e_2x[i, j] + e_2x[ni, nj]) / 2 * dl32
                        new_cost = dp[i, j, k] + c64 + c32

                        if new_cost < dp[ni, nj, k + 1]:
                            dp[ni, nj, k + 1] = new_cost
                            parent[ni, nj, k + 1] = [i, j, k]

    if dp[n-1, n-1, num_steps] == INF:
        print(f"Warning: Cannot reach ({n-1},{n-1}) with max_diag_dist={max_diag_dist}, trying {max_diag_dist+2}")
        return build_dp_constrained_schedule(heatmap_path, max_diag_dist+2, num_steps, sigma_min, sigma_max, device)

    # Backtrack
    path = []
    i, j, k = n-1, n-1, num_steps
    while k >= 0:
        path.append((i, j))
        if k == 0:
            break
        pi, pj, pk = parent[i, j, k]
        i, j, k = pi, pj, pk
    path = path[::-1]

    # Convert to sigma schedule
    sigmas = torch.zeros(num_steps, 3, device=device, dtype=torch.float32)
    for step in range(num_steps):
        i_orig, i_2x = path[step + 1]
        sigmas[step, 0] = float(sigma_grid[i_orig])
        sigmas[step, 1] = float(sigma_grid[i_2x])
        sigmas[step, 2] = float(sigma_grid[i_2x])
    return sigmas, path


def generate_batch(net, batch_size, sigmas, seed, device):
    """Generate a batch of images."""
    rng = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(batch_size, 18, 32, 32, generator=rng, device=device)
    cos_part, sin_part = latents.chunk(2, dim=1)
    norm = torch.sqrt(cos_part ** 2 + sin_part ** 2 + 1e-8)
    latents = torch.cat([cos_part / norm, sin_part / norm], dim=1)
    x = edm_sampler_torus_heun(net, latents, sigmas)
    rgb = convert_from_torus(x)
    return rgb[:, :3]  # orig scale only


def evaluate_path(net, sigmas, num_images, batch_size, seed, real_mu, real_sigma, device):
    """Generate images with given sigma schedule and compute FID."""
    num_batches = (num_images + batch_size - 1) // batch_size
    all_images = []
    for i in tqdm(range(num_batches), desc="Generating", leave=False):
        bs = min(batch_size, num_images - i * batch_size)
        imgs = generate_batch(net, bs, sigmas, seed + i, device)
        all_images.append(imgs.cpu())
    all_images_cat = torch.cat(all_images, dim=0)[:num_images]
    fid_batches = [all_images_cat[i:i+batch_size] for i in range(0, len(all_images_cat), batch_size)]
    fid = compute_fid(fid_batches, real_mu, real_sigma, device)
    return fid


def build_path_sigmas(name, device, heatmap_path=None):
    """Build sigma schedule for a named path."""
    if name == 'sync':
        return build_lead_follow_schedule(18, 0, device=device)
    elif name.startswith('lead_'):
        lead = int(name.split('_')[1])
        return build_lead_follow_schedule(18, lead, device=device)
    elif name.startswith('ratio_'):
        ratio = float(name.split('_')[1])
        return build_speed_ratio_schedule(18, ratio, device=device)
    elif name.startswith('dp_diag_'):
        dist = int(name.split('_')[2])
        sigmas, path = build_dp_constrained_schedule(heatmap_path, max_diag_dist=dist, device=device)
        return sigmas
    else:
        raise ValueError(f"Unknown path: {name}")


ALL_PATHS = ['sync', 'lead_1', 'lead_2', 'lead_3', 'lead_4', 'lead_5',
             'lead_6', 'lead_7', 'lead_8',
             'ratio_1.2', 'ratio_1.5', 'ratio_1.8', 'ratio_2.0',
             'dp_diag_2', 'dp_diag_3', 'dp_diag_5']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/home/ylong030/slot/network-snapshot-052685.pkl')
    parser.add_argument('--heatmap_path', default='results/heatmap.pth')
    parser.add_argument('--num_images', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--paths', nargs='+', default=None,
                        help='Specific paths to evaluate (default: all)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}')
    net = load_model(args.model_path, device)

    cache_path = os.path.join(args.output_dir, 'cifar10_real_features.npz')
    real_mu, real_sigma = load_cifar_real_features(cache_path, device)

    paths_to_run = args.paths if args.paths else ALL_PATHS
    results = []

    for name in paths_to_run:
        print(f"\n=== {name} ===")
        sigmas = build_path_sigmas(name, device, args.heatmap_path)
        fid = evaluate_path(net, sigmas, args.num_images, args.batch_size, args.seed, real_mu, real_sigma, device)
        print(f"  FID: {fid:.4f}")
        results.append((name, fid))

    # Print summary
    print("\n" + "="*50)
    print("Summary (sorted by FID)")
    print("="*50)
    results.sort(key=lambda x: x[1])
    for name, fid in results:
        print(f"  {name:20s}: {fid:.4f}")

    # Save to CSV (append mode for 10k runs)
    csv_path = os.path.join(args.output_dir, 'path_exploration.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a') as f:
        if write_header:
            f.write('path_type,num_images,fid\n')
        for name, fid in results:
            f.write(f'{name},{args.num_images},{fid:.4f}\n')
    print(f"\nAppended to {csv_path}")


if __name__ == '__main__':
    main()
