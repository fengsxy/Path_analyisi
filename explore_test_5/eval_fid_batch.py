#!/usr/bin/env python3
"""
Batch evaluate FID for explore_test_5 models (diagonal path) with EMA.

All 3 regimes (same_t, dp_path, heuristic) use SingleTimestepModel.
Loads EMA weights from checkpoint['ema_state']['shadow'].

Usage:
    python eval_fid_batch.py --model_type same_t --epochs 200 400 --device 0
    python eval_fid_batch.py --model_type dp_path --epochs 200 400 --device 1
    python eval_fid_batch.py --model_type heuristic --epochs 200 400 --device 2
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scheduler import DDIMScheduler
from data import AFHQ64Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'explore_test'))
from model_single_t import SingleTimestepModel


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
def generate_batch(model, scheduler, batch_size, num_steps, device, eta=0.0):
    """Generate with SingleTimestepModel. x_32 = downsample(x_64) each step."""
    scheduler._set_timesteps(num_steps)
    timesteps = scheduler.timesteps

    x_64 = torch.randn(batch_size, 3, 64, 64, device=device)

    for i, t in enumerate(timesteps):
        t_tensor = torch.tensor([t], device=device).expand(batch_size).float()
        x_32 = F.interpolate(x_64, size=32, mode='bilinear', align_corners=False)
        noise_pred_64 = model(x_64, x_32, t_tensor)

        prev_t = timesteps[i + 1] if i < len(timesteps) - 1 else 0
        x_64 = ddim_step(scheduler, noise_pred_64, t, prev_t, x_64, eta)

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
    """Load or compute real image features (cache in parent results/)."""
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'real_features.npz')

    if os.path.exists(cache_path):
        print(f"Loading cached real features from {cache_path}", flush=True)
        data = np.load(cache_path)
        if data['num_images'] >= num_images:
            print(f"Using cached features ({data['num_images']} images)", flush=True)
            return data['features'][:num_images]
        print(f"Cache has {data['num_images']} images, need {num_images}. Recomputing...", flush=True)

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


def load_model(checkpoint_path, device):
    """Load SingleTimestepModel with EMA weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden_dims = checkpoint.get('hidden_dims', [64, 128, 256, 512])
    model = SingleTimestepModel(hidden_dims=hidden_dims)

    ema_shadow = checkpoint['ema_state']['shadow']
    for name, param in model.named_parameters():
        if name in ema_shadow:
            param.data = ema_shadow[name].to(device)

    return model.to(device).eval()


def evaluate_checkpoint(checkpoint_path, feat_real, num_images, num_steps,
                        batch_size, device, output_dir, eta=0.0):
    """Evaluate a single checkpoint."""
    model = load_model(checkpoint_path, device)
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="cosine", clip_sample=True)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()

    gen_images_list = []
    num_batches = (num_images + batch_size - 1) // batch_size
    generated = 0

    for _ in tqdm(range(num_batches), desc="Generating"):
        bs = min(batch_size, num_images - generated)
        images = generate_batch(model, scheduler, bs, num_steps, device, eta)
        gen_images_list.append(images.cpu())
        generated += bs

    gen_images = torch.cat(gen_images_list, dim=0)
    del gen_images_list

    epoch = checkpoint_path.split('_ema_')[-1].replace('ep.pth', '')
    grid_path = os.path.join(output_dir, f'grid_{os.path.basename(checkpoint_path).replace(".pth", "")}.png')
    save_grid(gen_images, grid_path, nrow=9)
    print(f"Saved: {grid_path}", flush=True)

    feat_gen = get_inception_features(gen_images, inception, device, batch_size=64)
    del gen_images, model, inception
    torch.cuda.empty_cache()

    return calculate_fid_from_features(feat_real, feat_gen)


def parse_args():
    parser = argparse.ArgumentParser(description='Batch evaluate FID for explore_test_5 models')
    parser.add_argument('--model_type', type=str, choices=['same_t', 'dp_path', 'heuristic'], required=True)
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
    print("Explore Test 5 - Batch FID Evaluation (Diagonal)", flush=True)
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
        print(f"\n{'='*40}", flush=True)
        print(f"Epoch {epoch}", flush=True)
        print(f"{'='*40}", flush=True)

        ckpt_path = f'checkpoints/{args.model_type}_ema_{epoch}ep.pth'
        grid_path = os.path.join(args.output_dir, f'grid_{args.model_type}_ema_{epoch}ep.png')

        if not os.path.exists(ckpt_path):
            print(f"[Skip] Checkpoint not found: {ckpt_path}", flush=True)
            continue

        if os.path.exists(grid_path) and not args.force:
            print(f"[Skip] Already evaluated: {grid_path}", flush=True)
            continue

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        fid = evaluate_checkpoint(
            ckpt_path, feat_real,
            args.num_images, args.num_steps, args.batch_size,
            device, args.output_dir, args.eta
        )

        print(f"FID: {fid:.2f}", flush=True)
        results.append((epoch, fid))

    csv_path = os.path.join(args.output_dir, f'fid_{args.model_type}_diag_results.csv')
    with open(csv_path, 'w') as f:
        f.write("Epoch,FID\n")
        for epoch, fid in results:
            f.write(f"{epoch},{fid:.4f}\n")

    print(f"\n{'='*50}", flush=True)
    print("Results Summary", flush=True)
    print(f"{'='*50}", flush=True)
    for epoch, fid in results:
        print(f"Epoch {epoch:4d}: FID = {fid:.2f}", flush=True)
    print(f"\nSaved to: {csv_path}", flush=True)


if __name__ == "__main__":
    main()