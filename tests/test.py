#!/usr/bin/env python3
"""
Test Script: Evaluate LIFT Dual Timestep Model

Tests three scenarios:
1. Both scales noisy - test denoising ability
2a. Clean 32×32 + Noisy 64×64 - test conditioning ability
2b. Clean 64×64 + Noisy 32×32 - test conditioning ability

Usage:
    python test.py --checkpoint checkpoints/lift_full_random_final.pth --device 0
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from model import LIFTDualTimestepModel
from scheduler import DDIMScheduler
from data import AFHQ64Dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Test LIFT Dual Timestep Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of test samples')
    parser.add_argument('--cache_dir', type=str, default=None, help='HuggingFace cache directory')
    parser.add_argument('--device', type=int, default=0, help='CUDA device')
    parser.add_argument('--timestep', type=int, default=500, help='Timestep for testing (0-999)')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--num_vis', type=int, default=4, help='Number of images to visualize')
    return parser.parse_args()


def add_noise_at_timestep(x, t, scheduler):
    """Add noise at given timestep."""
    noise = torch.randn_like(x)
    t_tensor = torch.tensor([t], device=x.device).expand(x.shape[0])
    noisy = scheduler.add_noise(x, noise, t_tensor)
    return noisy, noise


def to_image(tensor):
    """Convert tensor to displayable image."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    img = tensor.detach().cpu()
    img = (img + 1) * 0.5  # [-1, 1] -> [0, 1]
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()
    return img


def test_both_noisy(model, scheduler, dataloader, device, timestep, num_samples):
    """Test 1: Both scales noisy - test denoising ability."""
    model.eval()
    total_mse = 0
    total_mse_64 = 0
    total_mse_32 = 0
    count = 0

    t_tensor = torch.tensor([timestep], device=device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Test 1: Both Noisy"):
            if count >= num_samples:
                break

            images = batch.to(device)
            batch_size = images.shape[0]

            # Create 32x32 version
            images_32 = F.interpolate(images, size=(32, 32), mode='bilinear', align_corners=False)

            # Add noise to both scales
            noisy_64, noise_64 = add_noise_at_timestep(images, timestep, scheduler)
            noisy_32, noise_32 = add_noise_at_timestep(images_32, timestep, scheduler)

            # Predict noise
            t_batch = t_tensor.expand(batch_size)
            noise_pred_64, noise_pred_32 = model(noisy_64, noisy_32, t_batch, t_batch)

            # Compute MSE between predicted and actual noise
            mse_64 = F.mse_loss(noise_pred_64, noise_64)
            mse_32 = F.mse_loss(noise_pred_32, noise_32)

            total_mse_64 += mse_64.item() * batch_size
            total_mse_32 += mse_32.item() * batch_size
            total_mse += (mse_64.item() + mse_32.item()) / 2 * batch_size
            count += batch_size

    return {
        'total_mse': total_mse / count,
        'mse_64': total_mse_64 / count,
        'mse_32': total_mse_32 / count,
    }


def test_condition_clean32_noisy64(model, scheduler, dataloader, device, timestep, num_samples):
    """Test 2a: Clean 32×32 + Noisy 64×64 - test conditioning ability."""
    model.eval()
    total_mse = 0
    total_mse_64 = 0
    total_mse_32 = 0
    count = 0

    t_noisy = torch.tensor([timestep], device=device)
    t_clean = torch.tensor([0], device=device)  # t=0 means clean (no noise)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Test 2a: Clean32 + Noisy64"):
            if count >= num_samples:
                break

            images = batch.to(device)
            batch_size = images.shape[0]

            # Create 32x32 version
            images_32 = F.interpolate(images, size=(32, 32), mode='bilinear', align_corners=False)

            # Add noise only to 64x64
            noisy_64, noise_64 = add_noise_at_timestep(images, timestep, scheduler)
            clean_32 = images_32  # No noise

            # Predict noise
            t_64_batch = t_noisy.expand(batch_size)
            t_32_batch = t_clean.expand(batch_size)
            noise_pred_64, noise_pred_32 = model(noisy_64, clean_32, t_64_batch, t_32_batch)

            # MSE for 64x64 (noisy)
            mse_64 = F.mse_loss(noise_pred_64, noise_64)
            # MSE for 32x32 (should predict zero noise)
            mse_32 = F.mse_loss(noise_pred_32, torch.zeros_like(noise_pred_32))

            total_mse_64 += mse_64.item() * batch_size
            total_mse_32 += mse_32.item() * batch_size
            total_mse += (mse_64.item() + mse_32.item()) / 2 * batch_size
            count += batch_size

    return {
        'total_mse': total_mse / count,
        'mse_64': total_mse_64 / count,
        'mse_32': total_mse_32 / count,
    }


def test_condition_clean64_noisy32(model, scheduler, dataloader, device, timestep, num_samples):
    """Test 2b: Clean 64×64 + Noisy 32×32 - test conditioning ability."""
    model.eval()
    total_mse = 0
    total_mse_64 = 0
    total_mse_32 = 0
    count = 0

    t_noisy = torch.tensor([timestep], device=device)
    t_clean = torch.tensor([0], device=device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Test 2b: Clean64 + Noisy32"):
            if count >= num_samples:
                break

            images = batch.to(device)
            batch_size = images.shape[0]

            # Create 32x32 version
            images_32 = F.interpolate(images, size=(32, 32), mode='bilinear', align_corners=False)

            # Add noise only to 32x32
            clean_64 = images  # No noise
            noisy_32, noise_32 = add_noise_at_timestep(images_32, timestep, scheduler)

            # Predict noise
            t_64_batch = t_clean.expand(batch_size)
            t_32_batch = t_noisy.expand(batch_size)
            noise_pred_64, noise_pred_32 = model(clean_64, noisy_32, t_64_batch, t_32_batch)

            # MSE for 64x64 (should predict zero noise)
            mse_64 = F.mse_loss(noise_pred_64, torch.zeros_like(noise_pred_64))
            # MSE for 32x32 (noisy)
            mse_32 = F.mse_loss(noise_pred_32, noise_32)

            total_mse_64 += mse_64.item() * batch_size
            total_mse_32 += mse_32.item() * batch_size
            total_mse += (mse_64.item() + mse_32.item()) / 2 * batch_size
            count += batch_size

    return {
        'total_mse': total_mse / count,
        'mse_64': total_mse_64 / count,
        'mse_32': total_mse_32 / count,
    }


def visualize_denoising(model, scheduler, dataset, device, timestep, num_images, output_path):
    """Create visualization of denoising results."""
    model.eval()

    indices = np.random.choice(len(dataset), num_images, replace=False)
    images = torch.stack([dataset[int(i)] for i in indices], dim=0).to(device)
    images_32 = F.interpolate(images, size=(32, 32), mode='bilinear', align_corners=False)

    t_noisy = torch.tensor([timestep], device=device).expand(num_images)
    t_clean = torch.tensor([0], device=device).expand(num_images)

    with torch.no_grad():
        # Test 1: Both noisy
        noisy_64_both, _ = add_noise_at_timestep(images, timestep, scheduler)
        noisy_32_both, _ = add_noise_at_timestep(images_32, timestep, scheduler)
        pred_64_both, pred_32_both = model(noisy_64_both, noisy_32_both, t_noisy, t_noisy)

        # Denoise (simple one-step)
        alpha_bar = scheduler.alphas_cumprod[timestep]
        denoised_64_both = (noisy_64_both - (1 - alpha_bar).sqrt() * pred_64_both) / alpha_bar.sqrt()

        # Test 2a: Clean 32 + Noisy 64
        noisy_64_a, _ = add_noise_at_timestep(images, timestep, scheduler)
        pred_64_a, _ = model(noisy_64_a, images_32, t_noisy, t_clean)
        denoised_64_a = (noisy_64_a - (1 - alpha_bar).sqrt() * pred_64_a) / alpha_bar.sqrt()

        # Test 2b: Clean 64 + Noisy 32
        noisy_32_b, _ = add_noise_at_timestep(images_32, timestep, scheduler)
        _, pred_32_b = model(images, noisy_32_b, t_clean, t_noisy)
        denoised_32_b = (noisy_32_b - (1 - alpha_bar).sqrt() * pred_32_b) / alpha_bar.sqrt()

    # Create figure
    fig, axes = plt.subplots(num_images, 7, figsize=(21, 3 * num_images))

    for i in range(num_images):
        # Original
        axes[i, 0].imshow(to_image(images[i]))
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Original', fontsize=10)

        # Test 1: Both noisy
        axes[i, 1].imshow(to_image(noisy_64_both[i]))
        axes[i, 1].axis('off')
        axes[i, 2].imshow(to_image(denoised_64_both[i]))
        axes[i, 2].axis('off')
        if i == 0:
            axes[i, 1].set_title('Both Noisy', fontsize=10)
            axes[i, 2].set_title('Denoised', fontsize=10)

        # Test 2a: Noisy 64
        axes[i, 3].imshow(to_image(noisy_64_a[i]))
        axes[i, 3].axis('off')
        axes[i, 4].imshow(to_image(denoised_64_a[i]))
        axes[i, 4].axis('off')
        if i == 0:
            axes[i, 3].set_title('Noisy 64×64\n(Clean 32×32)', fontsize=10)
            axes[i, 4].set_title('Denoised', fontsize=10)

        # Test 2b: Noisy 32
        noisy_32_up = F.interpolate(noisy_32_b[i:i+1], size=(64, 64), mode='nearest')[0]
        denoised_32_up = F.interpolate(denoised_32_b[i:i+1], size=(64, 64), mode='nearest')[0]
        axes[i, 5].imshow(to_image(noisy_32_up))
        axes[i, 5].axis('off')
        axes[i, 6].imshow(to_image(denoised_32_up))
        axes[i, 6].axis('off')
        if i == 0:
            axes[i, 5].set_title('Noisy 32×32\n(Clean 64×64)', fontsize=10)
            axes[i, 6].set_title('Denoised', fontsize=10)

    plt.suptitle(f'Denoising Results (timestep = {timestep})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()


def plot_results_bar(results, output_path):
    """Create bar chart of test results."""
    fig, ax = plt.subplots(figsize=(10, 6))

    tests = ['Test 1\n(Both Noisy)', 'Test 2a\n(Clean32+Noisy64)', 'Test 2b\n(Clean64+Noisy32)']
    mse_64 = [results['test1_both_noisy']['mse_64'],
              results['test2a_clean32_noisy64']['mse_64'],
              results['test2b_clean64_noisy32']['mse_64']]
    mse_32 = [results['test1_both_noisy']['mse_32'],
              results['test2a_clean32_noisy64']['mse_32'],
              results['test2b_clean64_noisy32']['mse_32']]

    x = np.arange(len(tests))
    width = 0.35

    bars1 = ax.bar(x - width/2, mse_64, width, label='MSE 64×64', color='steelblue')
    bars2 = ax.bar(x + width/2, mse_32, width, label='MSE 32×32', color='coral')

    ax.set_ylabel('MSE')
    ax.set_title('Test Results: Per-Scale MSE (Noise Prediction)')
    ax.set_xticks(x)
    ax.set_xticklabels(tests)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Results chart saved to: {output_path}")
    plt.close()


def save_summary(results, args, output_path):
    """Save text summary of results."""
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("LIFT Dual Timestep Model Test Results\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test timestep: {args.timestep}\n")
        f.write(f"Number of samples: {args.num_samples}\n\n")

        f.write("-" * 60 + "\n")
        f.write("Test 1: Both Scales Noisy (Denoising Ability)\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Total MSE: {results['test1_both_noisy']['total_mse']:.6f}\n")
        f.write(f"  MSE 64×64: {results['test1_both_noisy']['mse_64']:.6f}\n")
        f.write(f"  MSE 32×32: {results['test1_both_noisy']['mse_32']:.6f}\n\n")

        f.write("-" * 60 + "\n")
        f.write("Test 2a: Clean 32×32 + Noisy 64×64 (Conditioning Ability)\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Total MSE: {results['test2a_clean32_noisy64']['total_mse']:.6f}\n")
        f.write(f"  MSE 64×64 (noisy): {results['test2a_clean32_noisy64']['mse_64']:.6f}\n")
        f.write(f"  MSE 32×32 (clean): {results['test2a_clean32_noisy64']['mse_32']:.6f}\n\n")

        f.write("-" * 60 + "\n")
        f.write("Test 2b: Clean 64×64 + Noisy 32×32 (Conditioning Ability)\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Total MSE: {results['test2b_clean64_noisy32']['total_mse']:.6f}\n")
        f.write(f"  MSE 64×64 (clean): {results['test2b_clean64_noisy32']['mse_64']:.6f}\n")
        f.write(f"  MSE 32×32 (noisy): {results['test2b_clean64_noisy32']['mse_32']:.6f}\n\n")

        f.write("=" * 60 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test 1 (Both Noisy):        MSE = {results['test1_both_noisy']['total_mse']:.6f}\n")
        f.write(f"Test 2a (Clean32+Noisy64):  MSE = {results['test2a_clean32_noisy64']['total_mse']:.6f}\n")
        f.write(f"Test 2b (Clean64+Noisy32):  MSE = {results['test2b_clean64_noisy32']['total_mse']:.6f}\n")

    print(f"Summary saved to: {output_path}")


def main():
    args = parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    hidden_dims = checkpoint.get('hidden_dims', [64, 128, 256, 512])
    print(f"Hidden dims: {hidden_dims}")
    print(f"Training mode: {checkpoint.get('training_mode', 'N/A')}")

    # Create model
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
    print("Loading AFHQ64 dataset...")
    dataset = AFHQ64Dataset(split='train', cache_dir=args.cache_dir)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Testing with timestep = {args.timestep}")
    print(f"{'='*60}\n")

    # Run tests
    print("Test 1: Both scales noisy (denoising ability)")
    results_both = test_both_noisy(model, scheduler, dataloader, device, args.timestep, args.num_samples)
    print(f"  Total MSE: {results_both['total_mse']:.6f}")
    print(f"  MSE 64×64: {results_both['mse_64']:.6f}")
    print(f"  MSE 32×32: {results_both['mse_32']:.6f}")

    print("\nTest 2a: Clean 32×32 + Noisy 64×64 (conditioning ability)")
    results_clean32 = test_condition_clean32_noisy64(model, scheduler, dataloader, device, args.timestep, args.num_samples)
    print(f"  Total MSE: {results_clean32['total_mse']:.6f}")
    print(f"  MSE 64×64 (noisy): {results_clean32['mse_64']:.6f}")
    print(f"  MSE 32×32 (clean): {results_clean32['mse_32']:.6f}")

    print("\nTest 2b: Clean 64×64 + Noisy 32×32 (conditioning ability)")
    results_clean64 = test_condition_clean64_noisy32(model, scheduler, dataloader, device, args.timestep, args.num_samples)
    print(f"  Total MSE: {results_clean64['total_mse']:.6f}")
    print(f"  MSE 64×64 (clean): {results_clean64['mse_64']:.6f}")
    print(f"  MSE 32×32 (noisy): {results_clean64['mse_32']:.6f}")

    # Compile results
    results = {
        'timestep': args.timestep,
        'num_samples': args.num_samples,
        'test1_both_noisy': results_both,
        'test2a_clean32_noisy64': results_clean32,
        'test2b_clean64_noisy32': results_clean64,
    }

    # Save results
    results_path = os.path.join(args.output_dir, 'test_results.pth')
    torch.save(results, results_path)
    print(f"\nResults saved to: {results_path}")

    # Save text summary
    summary_path = os.path.join(args.output_dir, 'test_summary.txt')
    save_summary(results, args, summary_path)

    # Create visualizations
    print("\nCreating visualizations...")

    vis_path = os.path.join(args.output_dir, 'test_visualization.png')
    visualize_denoising(model, scheduler, dataset, device, args.timestep, args.num_vis, vis_path)

    chart_path = os.path.join(args.output_dir, 'test_results_chart.png')
    plot_results_bar(results, chart_path)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Test 1 (Both Noisy):        MSE = {results_both['total_mse']:.6f}")
    print(f"Test 2a (Clean32+Noisy64):  MSE = {results_clean32['total_mse']:.6f}")
    print(f"Test 2b (Clean64+Noisy32):  MSE = {results_clean64['total_mse']:.6f}")


if __name__ == "__main__":
    main()
