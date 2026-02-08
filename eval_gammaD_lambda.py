#!/usr/bin/env python3
"""Evaluate γ_D + Δλ DP path: compute path, generate images, compute FID."""
import sys
sys.path.insert(0, '/home/ylong030/slot/simple_diffusion_clean')

import os
import torch
import numpy as np
from tqdm import tqdm

from compute_heatmap_30 import path_to_timesteps
from eval_lift_dp import (
    load_lift_model, load_real_features, generate_batch_with_path,
    get_inception_features, calculate_fid_from_features, save_grid
)
from scheduler import DDIMScheduler
from pytorch_fid.inception import InceptionV3


def find_optimal_path_lambda(error_64, error_32, log_snr, num_steps=18, max_jump=5):
    """DP with λ-space step size: each scale integrated separately."""
    e64 = error_64.numpy() if torch.is_tensor(error_64) else error_64
    e32 = error_32.numpy() if torch.is_tensor(error_32) else error_32
    lsnr = log_snr.numpy() if torch.is_tensor(log_snr) else log_snr
    n = e64.shape[0]
    INF = float('inf')
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
                        dist_to_end = (n - 1 - ni) + (n - 1 - nj)
                        if dist_to_end > remaining * 2 * max_jump:
                            continue
                        if remaining == 0 and (ni != n - 1 or nj != n - 1):
                            continue
                        dl64 = abs(lsnr[ni] - lsnr[i])
                        dl32 = abs(lsnr[nj] - lsnr[j])
                        c64 = (e64[i, j] + e64[ni, nj]) / 2 * dl64
                        c32 = (e32[i, j] + e32[ni, nj]) / 2 * dl32
                        new_cost = dp[i, j, k] + c64 + c32
                        if new_cost < dp[ni, nj, k + 1]:
                            dp[ni, nj, k + 1] = new_cost
                            parent[ni, nj, k + 1] = [i, j, k]
    if dp[n - 1, n - 1, num_steps] == INF:
        print("Warning: no path found!")
        return [], INF
    path = []
    i, j, k = n - 1, n - 1, num_steps
    while k >= 0:
        path.append((i, j))
        if k == 0:
            break
        pi, pj, pk = parent[i, j, k]
        i, j, k = pi, pj, pk
    return path[::-1], dp[n - 1, n - 1, num_steps]


# === Config ===
EPOCH = 400
DEVICE_ID = 0
NUM_IMAGES = 15803
BATCH_SIZE = 64
SEED = 42
OUTPUT_DIR = 'results/gammaD_lambda'

device = torch.device(f'cuda:{DEVICE_ID}')
torch.manual_seed(SEED)
np.random.seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60, flush=True)
print("γ_D + Δλ DP Path Evaluation", flush=True)
print(f"Epoch: {EPOCH}, Images: {NUM_IMAGES}, Device: {DEVICE_ID}", flush=True)
print("=" * 60, flush=True)

# 1) Load cached heatmap (with γ_B)
cache_path = f'results/heatmap_30_ema_{EPOCH}ep.pth'
print(f"\nLoading heatmap: {cache_path}", flush=True)
data = torch.load(cache_path, map_location='cpu', weights_only=False)

snr_grid = data['snr_grid']
error_64_B = data['error_64']
error_32_B = data['error_32']
t_grid = data['t_grid']
log_snr = torch.log(snr_grid)

# 2) Convert γ_B → γ_D
# γ_D/γ_B = SNR / (4*(1+SNR))
ratio_64 = snr_grid / (4.0 * (1.0 + snr_grid))
ratio_32 = snr_grid / (4.0 * (1.0 + snr_grid))
error_64_D = error_64_B * ratio_64.unsqueeze(1)
error_32_D = error_32_B * ratio_32.unsqueeze(0)

# 3) Compute DP path with Δλ step size
print("\nComputing γ_D + Δλ DP path (18 steps)...", flush=True)
path, cost = find_optimal_path_lambda(error_64_D, error_32_D, log_snr, num_steps=18)
print(f"  Cost: {cost:.6e}", flush=True)
print(f"  Path: {path[0]} -> {path[-1]}", flush=True)

# Convert to timesteps
timesteps_64, timesteps_32 = path_to_timesteps(path, t_grid)
print(f"  t_64: {timesteps_64}", flush=True)
print(f"  t_32: {timesteps_32}", flush=True)

# 4) Load real features
print(f"\nLoading real features ({NUM_IMAGES} images)...", flush=True)
feat_real = load_real_features(NUM_IMAGES, device)

# 5) Load model (EMA)
ckpt_path = f'checkpoints/lift_ema_{EPOCH}ep.pth'
print(f"\nLoading model: {ckpt_path}", flush=True)
model = load_lift_model(ckpt_path, device, use_ema=True)

# 6) Scheduler
scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="cosine", clip_sample=True)

# 7) Generate images with γ_D + Δλ path
print(f"\nGenerating {NUM_IMAGES} images with γ_D + Δλ path...", flush=True)
gen_images_list = []
num_batches = (NUM_IMAGES + BATCH_SIZE - 1) // BATCH_SIZE
generated = 0

for _ in tqdm(range(num_batches), desc="Generating (γ_D+Δλ)"):
    bs = min(BATCH_SIZE, NUM_IMAGES - generated)
    images = generate_batch_with_path(model, scheduler, bs, timesteps_64, timesteps_32, device)
    gen_images_list.append(images.cpu())
    generated += bs

gen_images = torch.cat(gen_images_list, dim=0)
del gen_images_list

# Save grid
grid_path = os.path.join(OUTPUT_DIR, f'grid_gammaD_lambda_{EPOCH}ep.png')
save_grid(gen_images, grid_path, nrow=9)
print(f"  Grid: {grid_path}", flush=True)

# 8) Compute FID
print("Computing FID...", flush=True)
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
inception = InceptionV3([block_idx]).to(device).eval()
feat_gen = get_inception_features(gen_images, inception, device, batch_size=64)
del gen_images, model, inception
torch.cuda.empty_cache()

fid_gammaD = calculate_fid_from_features(feat_real, feat_gen)
print(f"\n{'=' * 60}", flush=True)
print(f"FID (γ_D + Δλ, EMA {EPOCH}ep): {fid_gammaD:.4f}", flush=True)
print(f"[Reference] FID (γ_B + grid, EMA {EPOCH}ep): 29.45", flush=True)
print(f"{'=' * 60}", flush=True)

# Save result
with open(os.path.join(OUTPUT_DIR, 'results.txt'), 'w') as f:
    f.write(f"γ_D + Δλ DP Path Evaluation\n")
    f.write(f"Epoch: {EPOCH}, Images: {NUM_IMAGES}, Seed: {SEED}\n")
    f.write(f"FID (γ_D + Δλ): {fid_gammaD:.4f}\n")
    f.write(f"FID (γ_B + grid, reference): 29.45\n")
    f.write(f"t_64: {timesteps_64}\n")
    f.write(f"t_32: {timesteps_32}\n")
