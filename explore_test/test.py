#!/usr/bin/env python3
"""
Test Script for explore_test models

测试场景：
1. 64×64 noisy + 32×32 noisy (随机t) - 标准去噪能力
2. 64×64 noisy + 32×32 clean - 32×32 提供干净信息
3. 64×64 noisy + 32×32 pure random noise - 32×32 是纯噪声

对比：看 32×32 输入是否对 64×64 预测有帮助

Usage:
    python test.py --checkpoint checkpoints/single_t_final.pth --device 0
    python test.py --checkpoint checkpoints/no_t_final.pth --device 0
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scheduler import DDIMScheduler
from data import AFHQ64Dataset

from model_single_t import SingleTimestepModel
from model_no_t import NoTimestepModel


def parse_args():
    parser = argparse.ArgumentParser(description='Test explore models')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of test samples')
    parser.add_argument('--device', type=int, default=0, help='CUDA device')
    parser.add_argument('--timestep', type=int, default=500, help='Timestep for 64×64 (0-999)')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    return parser.parse_args()


def add_noise_at_timestep(x, t, scheduler):
    """Add noise at given timestep."""
    noise = torch.randn_like(x)
    t_tensor = torch.tensor([t], device=x.device).expand(x.shape[0])
    noisy = scheduler.add_noise(x, noise, t_tensor)
    return noisy, noise


def test_noisy_32(model, model_type, scheduler, dataloader, device, timestep_64, num_samples):
    """
    Test 1: 64×64 noisy (t=timestep_64) + 32×32 noisy (随机t)
    标准训练场景
    """
    model.eval()
    total_mse = 0
    count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Test 1: 64 noisy + 32 noisy(random t)"):
            if count >= num_samples:
                break

            images = batch.to(device)
            batch_size = images.shape[0]
            images_32 = F.interpolate(images, size=(32, 32), mode='bilinear', align_corners=False)

            # 64×64: 固定 timestep
            noisy_64, noise_64 = add_noise_at_timestep(images, timestep_64, scheduler)

            # 32×32: 随机 timestep
            t_32_random = torch.randint(0, 1000, (batch_size,), device=device)
            noise_32 = torch.randn_like(images_32)
            alphas = scheduler.alphas_cumprod[t_32_random.cpu()].to(device)
            while len(alphas.shape) < len(images_32.shape):
                alphas = alphas.unsqueeze(-1)
            noisy_32 = alphas.sqrt() * images_32 + (1 - alphas).sqrt() * noise_32

            # Predict
            t_64_tensor = torch.tensor([timestep_64], device=device).expand(batch_size)
            if model_type == 'single_t':
                noise_pred_64 = model(noisy_64, noisy_32, t_64_tensor)
            else:
                noise_pred_64 = model(noisy_64, noisy_32)

            mse = F.mse_loss(noise_pred_64, noise_64)
            total_mse += mse.item() * batch_size
            count += batch_size

    return total_mse / count


def test_clean_32(model, model_type, scheduler, dataloader, device, timestep_64, num_samples):
    """
    Test 2: 64×64 noisy + 32×32 clean
    32×32 提供干净的低分辨率信息
    """
    model.eval()
    total_mse = 0
    count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Test 2: 64 noisy + 32 clean"):
            if count >= num_samples:
                break

            images = batch.to(device)
            batch_size = images.shape[0]
            images_32 = F.interpolate(images, size=(32, 32), mode='bilinear', align_corners=False)

            # 64×64: noisy
            noisy_64, noise_64 = add_noise_at_timestep(images, timestep_64, scheduler)

            # 32×32: clean (no noise)
            clean_32 = images_32

            # Predict
            t_64_tensor = torch.tensor([timestep_64], device=device).expand(batch_size)
            if model_type == 'single_t':
                noise_pred_64 = model(noisy_64, clean_32, t_64_tensor)
            else:
                noise_pred_64 = model(noisy_64, clean_32)

            mse = F.mse_loss(noise_pred_64, noise_64)
            total_mse += mse.item() * batch_size
            count += batch_size

    return total_mse / count


def test_pure_random_32(model, model_type, scheduler, dataloader, device, timestep_64, num_samples):
    """
    Test 3: 64×64 noisy + 32×32 pure random noise
    32×32 是纯噪声，不包含任何图像信息
    """
    model.eval()
    total_mse = 0
    count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Test 3: 64 noisy + 32 pure random"):
            if count >= num_samples:
                break

            images = batch.to(device)
            batch_size = images.shape[0]

            # 64×64: noisy
            noisy_64, noise_64 = add_noise_at_timestep(images, timestep_64, scheduler)

            # 32×32: pure random noise (no image info)
            random_32 = torch.randn(batch_size, 3, 32, 32, device=device)

            # Predict
            t_64_tensor = torch.tensor([timestep_64], device=device).expand(batch_size)
            if model_type == 'single_t':
                noise_pred_64 = model(noisy_64, random_32, t_64_tensor)
            else:
                noise_pred_64 = model(noisy_64, random_32)

            mse = F.mse_loss(noise_pred_64, noise_64)
            total_mse += mse.item() * batch_size
            count += batch_size

    return total_mse / count


def plot_results(results, model_type, output_path):
    """Create bar chart of test results."""
    fig, ax = plt.subplots(figsize=(8, 5))

    tests = ['32 noisy\n(random t)', '32 clean', '32 pure\nrandom']
    mse_values = [results['test1_noisy_32'], results['test2_clean_32'], results['test3_random_32']]

    bars = ax.bar(tests, mse_values, color=['steelblue', 'seagreen', 'coral'])

    ax.set_ylabel('MSE (64×64 noise prediction)')
    ax.set_title(f'Test Results: {model_type} model\n(64×64 timestep = {results["timestep"]})')
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, mse_values):
        ax.annotate(f'{val:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Results chart saved to: {output_path}")
    plt.close()


def main():
    args = parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hidden_dims = checkpoint.get('hidden_dims', [64, 128, 256, 512])
    model_type = checkpoint.get('model_type', 'single_t')

    print(f"Model type: {model_type}")
    print(f"Hidden dims: {hidden_dims}")

    # Create model
    if model_type == 'single_t':
        model = SingleTimestepModel(hidden_dims=hidden_dims)
    else:
        model = NoTimestepModel(hidden_dims=hidden_dims)

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
    dataset = AFHQ64Dataset(split='train')
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Testing {model_type} model with 64×64 timestep = {args.timestep}")
    print(f"{'='*60}\n")

    # Run tests
    mse1 = test_noisy_32(model, model_type, scheduler, dataloader, device, args.timestep, args.num_samples)
    print(f"Test 1 (32 noisy, random t): MSE = {mse1:.6f}")

    mse2 = test_clean_32(model, model_type, scheduler, dataloader, device, args.timestep, args.num_samples)
    print(f"Test 2 (32 clean):           MSE = {mse2:.6f}")

    mse3 = test_pure_random_32(model, model_type, scheduler, dataloader, device, args.timestep, args.num_samples)
    print(f"Test 3 (32 pure random):     MSE = {mse3:.6f}")

    # Compile results
    results = {
        'model_type': model_type,
        'timestep': args.timestep,
        'num_samples': args.num_samples,
        'test1_noisy_32': mse1,
        'test2_clean_32': mse2,
        'test3_random_32': mse3,
    }

    # Save results
    results_path = os.path.join(args.output_dir, f'test_results_{model_type}.pth')
    torch.save(results, results_path)
    print(f"\nResults saved to: {results_path}")

    # Plot
    chart_path = os.path.join(args.output_dir, f'test_chart_{model_type}.png')
    plot_results(results, model_type, chart_path)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Test 1 (32 noisy, random t): MSE = {mse1:.6f}")
    print(f"Test 2 (32 clean):           MSE = {mse2:.6f}")
    print(f"Test 3 (32 pure random):     MSE = {mse3:.6f}")
    print()
    print("解读：")
    print("- 如果 Test2 < Test1 < Test3: 32×32 提供了有用信息")
    print("- 如果 Test1 ≈ Test2 ≈ Test3: 模型忽略了 32×32 输入")


if __name__ == "__main__":
    main()
