#!/usr/bin/env python3
"""
Test Pipeline for explore_test models

完整测试流程：
1. 运行 MSE 测试（test.py 的三个场景）
2. 生成样本图像并画 grid 对比

Usage:
    python test_pipeline.py --single_t_ckpt checkpoints/single_t_final.pth \
                            --no_t_ckpt checkpoints/no_t_final.pth \
                            --device 0
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scheduler import DDIMScheduler
from data import AFHQ64Dataset

# 确保从当前目录导入
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from model_single_t import SingleTimestepModel
from model_no_t import NoTimestepModel
from test import test_noisy_32, test_clean_32, test_pure_random_32

# 直接从 generate.py 导入生成函数
import importlib.util
spec = importlib.util.spec_from_file_location("generate_local", os.path.join(current_dir, "generate.py"))
generate_local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate_local)
generate_batch_single_t = generate_local.generate_batch_single_t
generate_batch_no_t = generate_local.generate_batch_no_t


def parse_args():
    parser = argparse.ArgumentParser(description='Test pipeline for explore models')
    parser.add_argument('--single_t_ckpt', type=str, default='checkpoints/single_t_final.pth',
                        help='Path to single_t checkpoint')
    parser.add_argument('--no_t_ckpt', type=str, default='checkpoints/no_t_final.pth',
                        help='Path to no_t checkpoint')
    parser.add_argument('--device', type=int, default=0, help='CUDA device')
    parser.add_argument('--timestep', type=int, default=500, help='Timestep for MSE test')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of MSE test samples')
    parser.add_argument('--num_gen', type=int, default=16, help='Number of images to generate')
    parser.add_argument('--num_steps', type=int, default=50, help='DDIM steps for generation')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden_dims = checkpoint.get('hidden_dims', [64, 128, 256, 512])
    model_type = checkpoint.get('model_type', 'single_t')

    if model_type == 'single_t':
        model = SingleTimestepModel(hidden_dims=hidden_dims)
    else:
        model = NoTimestepModel(hidden_dims=hidden_dims)

    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()

    return model, model_type


def run_mse_tests(model, model_type, scheduler, dataloader, device, timestep, num_samples):
    """Run all three MSE tests."""
    print(f"\n{'='*50}")
    print(f"MSE Tests for {model_type} model (timestep={timestep})")
    print(f"{'='*50}")

    mse1 = test_noisy_32(model, model_type, scheduler, dataloader, device, timestep, num_samples)
    print(f"Test 1 (32 noisy, random t): MSE = {mse1:.6f}")

    mse2 = test_clean_32(model, model_type, scheduler, dataloader, device, timestep, num_samples)
    print(f"Test 2 (32 clean):           MSE = {mse2:.6f}")

    mse3 = test_pure_random_32(model, model_type, scheduler, dataloader, device, timestep, num_samples)
    print(f"Test 3 (32 pure random):     MSE = {mse3:.6f}")

    return {
        'test1_noisy_32': mse1,
        'test2_clean_32': mse2,
        'test3_random_32': mse3,
    }


@torch.no_grad()
def generate_samples(model, model_type, scheduler, num_images, num_steps, device, seed):
    """Generate sample images."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if model_type == 'single_t':
        images = generate_batch_single_t(model, scheduler, num_images, num_steps, device)
    else:
        images = generate_batch_no_t(model, scheduler, num_images, num_steps, device)

    return images


def plot_comparison_grid(images_single_t, images_no_t, output_path):
    """Plot side-by-side comparison grid."""
    num_images = min(len(images_single_t), len(images_no_t), 8)

    fig, axes = plt.subplots(2, num_images, figsize=(2 * num_images, 4.5))

    for i in range(num_images):
        # single_t row
        img = images_single_t[i].cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('single_t', fontsize=12)

        # no_t row
        img = images_no_t[i].cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[1, i].imshow(img)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('no_t', fontsize=12)

    # Add row labels on the left
    axes[0, 0].text(-0.15, 0.5, 'single_t', transform=axes[0, 0].transAxes,
                    fontsize=12, fontweight='bold', va='center', ha='right')
    axes[1, 0].text(-0.15, 0.5, 'no_t', transform=axes[1, 0].transAxes,
                    fontsize=12, fontweight='bold', va='center', ha='right')

    plt.suptitle('Generated Samples Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comparison grid saved to: {output_path}")
    plt.close()


def plot_mse_comparison(results_single_t, results_no_t, output_path):
    """Plot MSE comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))

    tests = ['32 noisy\n(random t)', '32 clean', '32 pure\nrandom']
    x = np.arange(len(tests))
    width = 0.35

    mse_single_t = [results_single_t['test1_noisy_32'],
                    results_single_t['test2_clean_32'],
                    results_single_t['test3_random_32']]
    mse_no_t = [results_no_t['test1_noisy_32'],
                results_no_t['test2_clean_32'],
                results_no_t['test3_random_32']]

    bars1 = ax.bar(x - width/2, mse_single_t, width, label='single_t', color='steelblue')
    bars2 = ax.bar(x + width/2, mse_no_t, width, label='no_t', color='coral')

    ax.set_ylabel('MSE (64×64 noise prediction)')
    ax.set_title('MSE Comparison: single_t vs no_t')
    ax.set_xticks(x)
    ax.set_xticklabels(tests)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"MSE comparison chart saved to: {output_path}")
    plt.close()


def main():
    args = parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Check if checkpoints exist
    if not os.path.exists(args.single_t_ckpt):
        print(f"Error: {args.single_t_ckpt} not found. Train the model first.")
        return
    if not os.path.exists(args.no_t_ckpt):
        print(f"Error: {args.no_t_ckpt} not found. Train the model first.")
        return

    # Load models
    print(f"\nLoading single_t model: {args.single_t_ckpt}")
    model_single_t, _ = load_model(args.single_t_ckpt, device)
    print(f"Parameters: {sum(p.numel() for p in model_single_t.parameters()):,}")

    print(f"\nLoading no_t model: {args.no_t_ckpt}")
    model_no_t, _ = load_model(args.no_t_ckpt, device)
    print(f"Parameters: {sum(p.numel() for p in model_no_t.parameters()):,}")

    # Create scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="cosine",
        clip_sample=True
    )

    # Load dataset for MSE tests
    print("\nLoading AFHQ64 dataset...")
    from torch.utils.data import DataLoader
    dataset = AFHQ64Dataset(split='train')
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ========== Part 1: MSE Tests ==========
    print("\n" + "="*60)
    print("PART 1: MSE TESTS")
    print("="*60)

    results_single_t = run_mse_tests(
        model_single_t, 'single_t', scheduler, dataloader,
        device, args.timestep, args.num_samples
    )

    results_no_t = run_mse_tests(
        model_no_t, 'no_t', scheduler, dataloader,
        device, args.timestep, args.num_samples
    )

    # Plot MSE comparison
    mse_chart_path = os.path.join(args.output_dir, 'mse_comparison.png')
    plot_mse_comparison(results_single_t, results_no_t, mse_chart_path)

    # ========== Part 2: Generate Samples ==========
    print("\n" + "="*60)
    print("PART 2: GENERATE SAMPLES")
    print("="*60)

    print(f"\nGenerating {args.num_gen} samples with single_t model...")
    images_single_t = generate_samples(
        model_single_t, 'single_t', scheduler,
        args.num_gen, args.num_steps, device, args.seed
    )

    print(f"Generating {args.num_gen} samples with no_t model...")
    images_no_t = generate_samples(
        model_no_t, 'no_t', scheduler,
        args.num_gen, args.num_steps, device, args.seed
    )

    # Plot comparison grid
    grid_path = os.path.join(args.output_dir, 'generated_comparison.png')
    plot_comparison_grid(images_single_t, images_no_t, grid_path)

    # ========== Summary ==========
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\nMSE Results:")
    print(f"{'Test':<25} {'single_t':>12} {'no_t':>12}")
    print("-" * 50)
    print(f"{'32 noisy (random t)':<25} {results_single_t['test1_noisy_32']:>12.6f} {results_no_t['test1_noisy_32']:>12.6f}")
    print(f"{'32 clean':<25} {results_single_t['test2_clean_32']:>12.6f} {results_no_t['test2_clean_32']:>12.6f}")
    print(f"{'32 pure random':<25} {results_single_t['test3_random_32']:>12.6f} {results_no_t['test3_random_32']:>12.6f}")

    print("\n解读：")
    print("- single_t: 有 timestep 信息，应该能更好地预测 noise")
    print("- no_t: 无 timestep 信息，是'盲去噪器'")
    print("- 如果 single_t MSE << no_t MSE: 证明 timestep 很重要")
    print("- 如果 Test2 < Test1 < Test3: 说明 32×32 提供了有用信息")

    # Save all results
    all_results = {
        'single_t': results_single_t,
        'no_t': results_no_t,
        'timestep': args.timestep,
        'num_samples': args.num_samples,
    }
    results_path = os.path.join(args.output_dir, 'pipeline_results.pth')
    torch.save(all_results, results_path)
    print(f"\nAll results saved to: {results_path}")

    print(f"\nOutput files:")
    print(f"  - {mse_chart_path}")
    print(f"  - {grid_path}")
    print(f"  - {results_path}")


if __name__ == "__main__":
    main()
