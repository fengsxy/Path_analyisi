#!/usr/bin/env python3
"""
Evaluate LIFT as a 32→64 super-resolution model.

Given a clean 32×32 image (t_32=0), denoise x_64 from pure noise (t_64: 999→0).
The LIFT model was trained with all (t_64, t_32) pairs including t_32=0.

Compares against bicubic upsampling baseline using PSNR and SSIM.

Usage:
    python eval_sr.py --ema --epochs 400 --device 1 --num_images 100
    python eval_sr.py --ema --epochs 200 400 600 800 1000 --device 1
"""

import os
import sys
import argparse
import csv

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import lpips

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import LIFTDualTimestepModel
from scheduler import DDIMScheduler
from data import AFHQ64Dataset, CelebAHQ64Dataset


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_psnr(img1, img2):
    """PSNR between two [0,1] tensors. Returns per-image [B] values."""
    mse = (img1 - img2).pow(2).mean(dim=[1, 2, 3])
    return 10 * torch.log10(1.0 / (mse + 1e-10))


def _gaussian_window(window_size, sigma):
    """Create 1D Gaussian kernel."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-coords.pow(2) / (2 * sigma ** 2))
    return g / g.sum()

def compute_ssim(img1, img2, window_size=11, sigma=1.5):
    """SSIM between two [0,1] tensors. Returns per-image [B] values."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    C = img1.shape[1]  # channels

    # Build 2D Gaussian window
    g1d = _gaussian_window(window_size, sigma).to(img1.device)
    window = g1d.unsqueeze(1) * g1d.unsqueeze(0)  # outer product
    window = window.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)  # [C,1,H,W]

    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=C)
    mu2 = F.conv2d(img2, window, padding=pad, groups=C)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # Mean over spatial and channel dims → per-image
    return ssim_map.mean(dim=[1, 2, 3])


# ---------------------------------------------------------------------------
# Model loading (reuse from parent eval_lift_dp.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# DDIM step (reuse from parent eval_lift_dp.py)
# ---------------------------------------------------------------------------

def ddim_step(scheduler, model_output, timestep, prev_timestep, sample, eta=0.0):
    """Single DDIM step."""
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else scheduler.final_alpha_cumprod
    )
    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

    if scheduler.clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

    variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    std_dev_t = eta * variance ** 0.5

    pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * pred_epsilon
    prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

    if eta > 0:
        noise = torch.randn_like(model_output)
        prev_sample += std_dev_t * noise

    return prev_sample


# ---------------------------------------------------------------------------
# Super-resolution generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_sr_batch(model, scheduler, x_32_clean, num_steps=18, device='cuda', eta=0.0):
    """Super-resolve: clean x_32 → generated x_64.

    Args:
        model: LIFTDualTimestepModel
        scheduler: DDIMScheduler
        x_32_clean: [B, 3, 32, 32] clean images in [-1, 1]
        num_steps: DDIM steps for x_64 denoising

    Returns: x_64 in [0, 1]
    """
    B = x_32_clean.shape[0]
    scheduler._set_timesteps(num_steps)
    timesteps = scheduler.timesteps  # descending: [999, 944, ...]

    x_64 = torch.randn(B, 3, 64, 64, device=device)
    t_32_tensor = torch.zeros(B, device=device, dtype=torch.long)

    for i, t in enumerate(timesteps):
        t_64_tensor = torch.full((B,), int(t), device=device, dtype=torch.long)
        eps_64, _ = model(x_64, x_32_clean, t_64_tensor, t_32_tensor)
        prev_t = int(timesteps[i + 1]) if i < len(timesteps) - 1 else 0
        x_64 = ddim_step(scheduler, eps_64, int(t), prev_t, x_64, eta=eta)

    return (x_64 + 1) * 0.5


# ---------------------------------------------------------------------------
# Bicubic baseline
# ---------------------------------------------------------------------------

def bicubic_upsample(x_32_01):
    """Bicubic 32→64 upsampling. Input/output in [0,1]."""
    return F.interpolate(x_32_01, size=64, mode='bicubic', align_corners=False).clamp(0, 1)


# ---------------------------------------------------------------------------
# Comparison grid
# ---------------------------------------------------------------------------

def save_comparison_grid(gt_images, bicubic_images, sr_images, output_path, num_show=4):
    """Save comparison grid: columns = [GT, Bicubic, LIFT SR] with headers.

    Args:
        gt_images, bicubic_images, sr_images: [N, 3, 64, 64] tensors in [0,1]
        output_path: path to save PNG
        num_show: number of rows (images) to show
    """
    n = min(num_show, len(gt_images))
    h, w = 64, 64
    header_h = 20
    labels = ["GT", "Bicubic", "LIFT SR"]

    # Grid: header + n rows × 3 columns
    grid_h = header_h + n * h
    grid_w = 3 * w
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255  # white background

    # Draw images
    for i in range(n):
        for col_idx, imgs in enumerate([gt_images, bicubic_images, sr_images]):
            img_np = imgs[i].permute(1, 2, 0).cpu().numpy()
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            y0 = header_h + i * h
            x0 = col_idx * w
            grid[y0:y0 + h, x0:x0 + w] = img_np

    # Add column headers
    pil_img = Image.fromarray(grid)
    draw = ImageDraw.Draw(pil_img)
    for col_idx, label in enumerate(labels):
        # Center text in column
        bbox = draw.textbbox((0, 0), label)
        text_w = bbox[2] - bbox[0]
        x = col_idx * w + (w - text_w) // 2
        draw.text((x, 2), label, fill=(0, 0, 0))

    pil_img.save(output_path)
    print(f"  Saved comparison grid: {output_path}", flush=True)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_epoch(model, scheduler, dataset, epoch, args, device, lpips_model):
    """Evaluate SR for one checkpoint. Returns dict of metrics."""
    num_images = min(args.num_images, len(dataset))

    # Load images
    gt_images_list = []
    for i in range(num_images):
        gt_images_list.append(dataset[i])
    gt_neg1_to_1 = torch.stack(gt_images_list)  # [-1, 1]
    gt_01 = (gt_neg1_to_1 + 1) * 0.5  # [0, 1]

    # Downsample to 32×32
    x_32_01 = F.interpolate(gt_01, size=32, mode='bicubic', align_corners=False).clamp(0, 1)
    x_32_neg1 = x_32_01 * 2 - 1  # [-1, 1] for model input

    # Bicubic baseline
    bicubic_01 = bicubic_upsample(x_32_01)

    # LIFT SR — process in batches
    sr_parts = []
    batch_size = args.batch_size
    for start in tqdm(range(0, num_images, batch_size), desc=f"SR generation ({epoch}ep)"):
        end = min(start + batch_size, num_images)
        batch_32 = x_32_neg1[start:end].to(device)
        sr_batch = generate_sr_batch(model, scheduler, batch_32,
                                     num_steps=args.num_steps, device=device, eta=0.0)
        sr_parts.append(sr_batch.cpu())
    sr_01 = torch.cat(sr_parts, dim=0)

    # Compute metrics
    psnr_bicubic = compute_psnr(bicubic_01, gt_01)
    psnr_sr = compute_psnr(sr_01, gt_01)
    ssim_bicubic = compute_ssim(bicubic_01, gt_01)
    ssim_sr = compute_ssim(sr_01, gt_01)

    # Compute LPIPS (input must be [-1, 1])
    gt_lpips = gt_01 * 2 - 1
    bicubic_lpips = bicubic_01 * 2 - 1
    sr_lpips = sr_01 * 2 - 1
    lpips_bicubic_vals = []
    lpips_sr_vals = []
    lpips_batch = 64
    for start in range(0, num_images, lpips_batch):
        end = min(start + lpips_batch, num_images)
        with torch.no_grad():
            lb = lpips_model(bicubic_lpips[start:end].to(device), gt_lpips[start:end].to(device))
            ls = lpips_model(sr_lpips[start:end].to(device), gt_lpips[start:end].to(device))
        lpips_bicubic_vals.append(lb.cpu().squeeze())
        lpips_sr_vals.append(ls.cpu().squeeze())
    lpips_bicubic = torch.cat(lpips_bicubic_vals)
    lpips_sr = torch.cat(lpips_sr_vals)

    metrics = {
        'epoch': epoch,
        'psnr_bicubic_mean': psnr_bicubic.mean().item(),
        'psnr_bicubic_std': psnr_bicubic.std().item(),
        'psnr_sr_mean': psnr_sr.mean().item(),
        'psnr_sr_std': psnr_sr.std().item(),
        'ssim_bicubic_mean': ssim_bicubic.mean().item(),
        'ssim_bicubic_std': ssim_bicubic.std().item(),
        'ssim_sr_mean': ssim_sr.mean().item(),
        'ssim_sr_std': ssim_sr.std().item(),
        'lpips_bicubic_mean': lpips_bicubic.mean().item(),
        'lpips_bicubic_std': lpips_bicubic.std().item(),
        'lpips_sr_mean': lpips_sr.mean().item(),
        'lpips_sr_std': lpips_sr.std().item(),
    }

    # Print
    print(f"  Bicubic — PSNR: {metrics['psnr_bicubic_mean']:.2f}±{metrics['psnr_bicubic_std']:.2f} dB, "
          f"SSIM: {metrics['ssim_bicubic_mean']:.4f}±{metrics['ssim_bicubic_std']:.4f}, "
          f"LPIPS: {metrics['lpips_bicubic_mean']:.4f}±{metrics['lpips_bicubic_std']:.4f}", flush=True)
    print(f"  LIFT SR — PSNR: {metrics['psnr_sr_mean']:.2f}±{metrics['psnr_sr_std']:.2f} dB, "
          f"SSIM: {metrics['ssim_sr_mean']:.4f}±{metrics['ssim_sr_std']:.4f}, "
          f"LPIPS: {metrics['lpips_sr_mean']:.4f}±{metrics['lpips_sr_std']:.4f}", flush=True)

    # Save comparison grid
    ema_tag = '_ema' if args.ema else ''
    dataset_tag = f'_{args.dataset}' if args.dataset != 'afhq' else ''
    grid_path = os.path.join(args.output_dir, f'sr_comparison{dataset_tag}{ema_tag}_{epoch}ep.png')
    save_comparison_grid(gt_01, bicubic_01, sr_01, grid_path)

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate LIFT as 32→64 super-resolution')
    parser.add_argument('--epochs', type=int, nargs='+', required=True)
    parser.add_argument('--ema', action='store_true', help='Use EMA weights')
    parser.add_argument('--dataset', type=str, default='afhq', choices=['afhq', 'celeba'],
                        help='Dataset to evaluate on (default: afhq)')
    parser.add_argument('--num_images', type=int, default=100)
    parser.add_argument('--num_steps', type=int, default=18)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--force', action='store_true', help='Force re-evaluation')
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 50, flush=True)
    print("LIFT Super-Resolution Evaluation (32→64)", flush=True)
    print("=" * 50, flush=True)
    print(f"Dataset: {args.dataset}", flush=True)
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

    # Load dataset once
    if args.dataset == 'celeba':
        print("\nLoading CelebA-HQ dataset...", flush=True)
        dataset = CelebAHQ64Dataset(split='train', cache_dir=args.cache_dir)
    else:
        print("\nLoading AFHQ dataset...", flush=True)
        dataset = AFHQ64Dataset(split='train', cache_dir=args.cache_dir)
    print(f"Dataset size: {len(dataset)}", flush=True)

    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="cosine", clip_sample=True)

    # Create LPIPS model once
    lpips_model = lpips.LPIPS(net='alex').to(device)

    all_results = []
    ema_tag = '_ema' if args.ema else ''
    dataset_tag = f'_{args.dataset}' if args.dataset != 'afhq' else ''

    for epoch in args.epochs:
        print(f"\n{'=' * 50}", flush=True)
        print(f"Epoch {epoch}", flush=True)
        print(f"{'=' * 50}", flush=True)

        if args.ema:
            ckpt_path = os.path.join('..', 'checkpoints', f'lift_ema_{epoch}ep.pth')
        else:
            ckpt_path = os.path.join('..', 'checkpoints', f'lift_dual_timestep_{epoch}ep.pth')

        if not os.path.exists(ckpt_path):
            print(f"[Skip] Checkpoint not found: {ckpt_path}", flush=True)
            continue

        # Check if already evaluated
        grid_path = os.path.join(args.output_dir, f'sr_comparison{dataset_tag}{ema_tag}_{epoch}ep.png')
        if os.path.exists(grid_path) and not args.force:
            print(f"[Skip] Already evaluated: {epoch}ep", flush=True)
            continue

        # Reset seed for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        model = load_lift_model(ckpt_path, device, use_ema=args.ema)
        metrics = evaluate_epoch(model, scheduler, dataset, epoch, args, device, lpips_model)
        all_results.append(metrics)

        del model
        torch.cuda.empty_cache()

    # Save CSV
    if all_results:
        csv_path = os.path.join(args.output_dir, f'sr_metrics{dataset_tag}{ema_tag}_results.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch',
                             'PSNR_Bicubic_Mean', 'PSNR_Bicubic_Std',
                             'PSNR_SR_Mean', 'PSNR_SR_Std',
                             'SSIM_Bicubic_Mean', 'SSIM_Bicubic_Std',
                             'SSIM_SR_Mean', 'SSIM_SR_Std',
                             'LPIPS_Bicubic_Mean', 'LPIPS_Bicubic_Std',
                             'LPIPS_SR_Mean', 'LPIPS_SR_Std'])
            for m in all_results:
                writer.writerow([
                    m['epoch'],
                    f"{m['psnr_bicubic_mean']:.4f}", f"{m['psnr_bicubic_std']:.4f}",
                    f"{m['psnr_sr_mean']:.4f}", f"{m['psnr_sr_std']:.4f}",
                    f"{m['ssim_bicubic_mean']:.6f}", f"{m['ssim_bicubic_std']:.6f}",
                    f"{m['ssim_sr_mean']:.6f}", f"{m['ssim_sr_std']:.6f}",
                    f"{m['lpips_bicubic_mean']:.6f}", f"{m['lpips_bicubic_std']:.6f}",
                    f"{m['lpips_sr_mean']:.6f}", f"{m['lpips_sr_std']:.6f}",
                ])

        print(f"\n{'=' * 50}", flush=True)
        print("Results Summary", flush=True)
        print(f"{'=' * 50}", flush=True)
        print(f"{'Epoch':>6} | {'Bicubic PSNR':>14} | {'SR PSNR':>14} | {'Bicubic SSIM':>14} | {'SR SSIM':>14} | {'Bicubic LPIPS':>14} | {'SR LPIPS':>14}", flush=True)
        print("-" * 105, flush=True)
        for m in all_results:
            print(f"{m['epoch']:>6} | "
                  f"{m['psnr_bicubic_mean']:>6.2f}±{m['psnr_bicubic_std']:<5.2f} | "
                  f"{m['psnr_sr_mean']:>6.2f}±{m['psnr_sr_std']:<5.2f} | "
                  f"{m['ssim_bicubic_mean']:>6.4f}±{m['ssim_bicubic_std']:<6.4f} | "
                  f"{m['ssim_sr_mean']:>6.4f}±{m['ssim_sr_std']:<6.4f} | "
                  f"{m['lpips_bicubic_mean']:>6.4f}±{m['lpips_bicubic_std']:<6.4f} | "
                  f"{m['lpips_sr_mean']:>6.4f}±{m['lpips_sr_std']:<6.4f}", flush=True)
        print(f"\nSaved to: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
