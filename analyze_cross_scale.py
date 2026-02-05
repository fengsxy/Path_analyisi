#!/usr/bin/env python3
"""
分析跨尺度耦合机制

研究问题：为什么64×64误差和γ₁（32×32 SNR）独立？

分析方法：
1. 计算跨尺度Jacobian: ∂(noise_pred_64)/∂z_32
2. 比较同尺度vs跨尺度的Jacobian范数
3. 分析不同SNR下的耦合强度
4. 消融实验：把32×32输入置零
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.func import jvp, jacrev
from tqdm import tqdm

from model import LIFTDualTimestepModel
from scheduler import DDIMScheduler
from data import AFHQ64Dataset


def compute_jacobian_norms(model, x_64, x_32, t_64, t_32, num_samples=10):
    """
    计算四种Jacobian的Frobenius范数：
    - J_64_64: ∂(noise_pred_64)/∂z_64  (同尺度)
    - J_64_32: ∂(noise_pred_64)/∂z_32  (跨尺度)
    - J_32_64: ∂(noise_pred_32)/∂z_64  (跨尺度)
    - J_32_32: ∂(noise_pred_32)/∂z_32  (同尺度)

    使用Hutchinson估计器近似Frobenius范数
    """
    model.eval()

    # 定义四个函数
    def f_64_from_64(z):
        out_64, _ = model(z, x_32, t_64, t_32)
        return out_64

    def f_64_from_32(z):
        out_64, _ = model(x_64, z, t_64, t_32)
        return out_64

    def f_32_from_64(z):
        _, out_32 = model(z, x_32, t_64, t_32)
        return out_32

    def f_32_from_32(z):
        _, out_32 = model(x_64, z, t_64, t_32)
        return out_32

    def estimate_frobenius_norm(f, z, K=num_samples):
        """用Hutchinson估计器估计 ||J||_F^2"""
        z = z.detach().requires_grad_(True)
        total = 0.0
        for _ in range(K):
            v = torch.randn_like(z)
            _, jvp_out = jvp(f, (z,), (v,))
            total += (jvp_out ** 2).sum().item()
        return total / K

    with torch.no_grad():
        # 估计四种Jacobian的范数
        norm_64_64 = estimate_frobenius_norm(f_64_from_64, x_64)
        norm_64_32 = estimate_frobenius_norm(f_64_from_32, x_32)
        norm_32_64 = estimate_frobenius_norm(f_32_from_64, x_64)
        norm_32_32 = estimate_frobenius_norm(f_32_from_32, x_32)

    return {
        'J_64_64': np.sqrt(norm_64_64),
        'J_64_32': np.sqrt(norm_64_32),
        'J_32_64': np.sqrt(norm_32_64),
        'J_32_32': np.sqrt(norm_32_32),
    }


def ablation_zero_input(model, x_64, x_32, t_64, t_32):
    """
    消融实验：把某个输入置零，看输出变化
    """
    model.eval()

    with torch.no_grad():
        # 正常输出
        out_64_normal, out_32_normal = model(x_64, x_32, t_64, t_32)

        # 把32×32置零
        x_32_zero = torch.zeros_like(x_32)
        out_64_zero32, out_32_zero32 = model(x_64, x_32_zero, t_64, t_32)

        # 把64×64置零
        x_64_zero = torch.zeros_like(x_64)
        out_64_zero64, out_32_zero64 = model(x_64_zero, x_32, t_64, t_32)

        # 计算变化
        delta_64_when_zero32 = (out_64_normal - out_64_zero32).abs().mean().item()
        delta_32_when_zero32 = (out_32_normal - out_32_zero32).abs().mean().item()
        delta_64_when_zero64 = (out_64_normal - out_64_zero64).abs().mean().item()
        delta_32_when_zero64 = (out_32_normal - out_32_zero64).abs().mean().item()

        # 输出的绝对值（作为参考）
        out_64_mag = out_64_normal.abs().mean().item()
        out_32_mag = out_32_normal.abs().mean().item()

    return {
        'delta_64_when_zero32': delta_64_when_zero32,
        'delta_32_when_zero32': delta_32_when_zero32,
        'delta_64_when_zero64': delta_64_when_zero64,
        'delta_32_when_zero64': delta_32_when_zero64,
        'out_64_magnitude': out_64_mag,
        'out_32_magnitude': out_32_mag,
        # 相对变化
        'rel_delta_64_when_zero32': delta_64_when_zero32 / out_64_mag,
        'rel_delta_32_when_zero32': delta_32_when_zero32 / out_32_mag,
        'rel_delta_64_when_zero64': delta_64_when_zero64 / out_64_mag,
        'rel_delta_32_when_zero64': delta_32_when_zero64 / out_32_mag,
    }


def analyze_coupling_vs_snr(model, scheduler, dataset, device, num_snr_points=10):
    """
    分析跨尺度耦合强度随SNR的变化
    """
    # 准备数据
    indices = np.random.choice(len(dataset), 4, replace=False)
    x_batch = torch.stack([dataset[int(i)] for i in indices], dim=0).to(device)
    x_batch_32 = F.interpolate(x_batch, size=(32, 32), mode='bilinear', align_corners=False)

    # SNR范围
    snr_values = np.logspace(-2, 2, num_snr_points)

    results = {
        'snr': snr_values,
        'J_64_64': [],
        'J_64_32': [],
        'J_32_64': [],
        'J_32_32': [],
        'coupling_ratio_64': [],  # J_64_32 / J_64_64
        'coupling_ratio_32': [],  # J_32_64 / J_32_32
    }

    for snr in tqdm(snr_values, desc="Analyzing coupling vs SNR"):
        # 转换SNR到timestep
        alpha_bar = snr / (snr + 1.0)
        alphas_cumprod = scheduler.alphas_cumprod.cpu().numpy()
        t = np.argmin(np.abs(alphas_cumprod - alpha_bar))

        # 添加噪声
        noise_64 = torch.randn_like(x_batch)
        noise_32 = torch.randn_like(x_batch_32)
        t_tensor = torch.tensor([t], device=device).expand(x_batch.shape[0])

        z_64 = scheduler.add_noise(x_batch, noise_64, t_tensor)
        z_32 = scheduler.add_noise(x_batch_32, noise_32, t_tensor)

        # 计算Jacobian范数
        norms = compute_jacobian_norms(model, z_64, z_32, t_tensor, t_tensor, num_samples=8)

        results['J_64_64'].append(norms['J_64_64'])
        results['J_64_32'].append(norms['J_64_32'])
        results['J_32_64'].append(norms['J_32_64'])
        results['J_32_32'].append(norms['J_32_32'])
        results['coupling_ratio_64'].append(norms['J_64_32'] / (norms['J_64_64'] + 1e-10))
        results['coupling_ratio_32'].append(norms['J_32_64'] / (norms['J_32_32'] + 1e-10))

    return results


def plot_coupling_analysis(results, output_path):
    """绘制耦合分析结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    snr = results['snr']

    # 1. Jacobian范数 vs SNR
    ax = axes[0, 0]
    ax.loglog(snr, results['J_64_64'], 'b-o', label='J_64_64 (同尺度)', markersize=4)
    ax.loglog(snr, results['J_64_32'], 'b--s', label='J_64_32 (跨尺度)', markersize=4)
    ax.loglog(snr, results['J_32_32'], 'r-o', label='J_32_32 (同尺度)', markersize=4)
    ax.loglog(snr, results['J_32_64'], 'r--s', label='J_32_64 (跨尺度)', markersize=4)
    ax.set_xlabel('SNR')
    ax.set_ylabel('||J||_F')
    ax.set_title('Jacobian Frobenius范数 vs SNR')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 耦合比例 vs SNR
    ax = axes[0, 1]
    ax.semilogx(snr, results['coupling_ratio_64'], 'b-o', label='64×64: J_64_32/J_64_64', markersize=4)
    ax.semilogx(snr, results['coupling_ratio_32'], 'r-o', label='32×32: J_32_64/J_32_32', markersize=4)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='完全耦合 (ratio=1)')
    ax.set_xlabel('SNR')
    ax.set_ylabel('跨尺度/同尺度 Jacobian比值')
    ax.set_title('跨尺度耦合强度 vs SNR')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(max(results['coupling_ratio_64']), max(results['coupling_ratio_32'])) * 1.2)

    # 3. 64×64输出的Jacobian分解
    ax = axes[1, 0]
    j64_64 = np.array(results['J_64_64'])
    j64_32 = np.array(results['J_64_32'])
    total_64 = np.sqrt(j64_64**2 + j64_32**2)

    ax.fill_between(snr, 0, j64_64**2 / total_64**2 * 100, alpha=0.7, label='来自z_64', color='blue')
    ax.fill_between(snr, j64_64**2 / total_64**2 * 100, 100, alpha=0.7, label='来自z_32', color='orange')
    ax.set_xscale('log')
    ax.set_xlabel('SNR')
    ax.set_ylabel('贡献比例 (%)')
    ax.set_title('64×64输出的Jacobian来源分解')
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # 4. 32×32输出的Jacobian分解
    ax = axes[1, 1]
    j32_32 = np.array(results['J_32_32'])
    j32_64 = np.array(results['J_32_64'])
    total_32 = np.sqrt(j32_32**2 + j32_64**2)

    ax.fill_between(snr, 0, j32_32**2 / total_32**2 * 100, alpha=0.7, label='来自z_32', color='red')
    ax.fill_between(snr, j32_32**2 / total_32**2 * 100, 100, alpha=0.7, label='来自z_64', color='green')
    ax.set_xscale('log')
    ax.set_xlabel('SNR')
    ax.set_ylabel('贡献比例 (%)')
    ax.set_title('32×32输出的Jacobian来源分解')
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    plt.suptitle('LIFT模型跨尺度耦合分析', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/lift_full_random_final.pth')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--cache_dir', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hidden_dims = checkpoint.get('hidden_dims', [64, 128, 256, 512])

    model = LIFTDualTimestepModel(hidden_dims=hidden_dims)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()

    # 创建scheduler
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="cosine", clip_sample=True)

    # 加载数据
    print("Loading dataset...")
    dataset = AFHQ64Dataset(split='train', cache_dir=args.cache_dir)

    # 准备测试数据
    indices = np.random.choice(len(dataset), 4, replace=False)
    x_batch = torch.stack([dataset[int(i)] for i in indices], dim=0).to(device)
    x_batch_32 = F.interpolate(x_batch, size=(32, 32), mode='bilinear', align_corners=False)

    # 测试在中等SNR下
    t_mid = 500
    t_tensor = torch.tensor([t_mid], device=device).expand(4)

    noise_64 = torch.randn_like(x_batch)
    noise_32 = torch.randn_like(x_batch_32)
    z_64 = scheduler.add_noise(x_batch, noise_64, t_tensor)
    z_32 = scheduler.add_noise(x_batch_32, noise_32, t_tensor)

    print("\n" + "="*60)
    print("实验1: 消融实验 - 把某个输入置零")
    print("="*60)

    ablation = ablation_zero_input(model, z_64, z_32, t_tensor, t_tensor)

    print(f"\n当把32×32输入置零时:")
    print(f"  64×64输出变化: {ablation['delta_64_when_zero32']:.4f} (相对: {ablation['rel_delta_64_when_zero32']*100:.1f}%)")
    print(f"  32×32输出变化: {ablation['delta_32_when_zero32']:.4f} (相对: {ablation['rel_delta_32_when_zero32']*100:.1f}%)")

    print(f"\n当把64×64输入置零时:")
    print(f"  64×64输出变化: {ablation['delta_64_when_zero64']:.4f} (相对: {ablation['rel_delta_64_when_zero64']*100:.1f}%)")
    print(f"  32×32输出变化: {ablation['delta_32_when_zero64']:.4f} (相对: {ablation['rel_delta_32_when_zero64']*100:.1f}%)")

    print("\n" + "="*60)
    print("实验2: Jacobian范数比较 (t=500)")
    print("="*60)

    norms = compute_jacobian_norms(model, z_64, z_32, t_tensor, t_tensor, num_samples=16)

    print(f"\n同尺度Jacobian:")
    print(f"  ||∂(out_64)/∂z_64||_F = {norms['J_64_64']:.4f}")
    print(f"  ||∂(out_32)/∂z_32||_F = {norms['J_32_32']:.4f}")

    print(f"\n跨尺度Jacobian:")
    print(f"  ||∂(out_64)/∂z_32||_F = {norms['J_64_32']:.4f}")
    print(f"  ||∂(out_32)/∂z_64||_F = {norms['J_32_64']:.4f}")

    print(f"\n耦合比例:")
    print(f"  64×64输出: 跨尺度/同尺度 = {norms['J_64_32']/norms['J_64_64']:.4f}")
    print(f"  32×32输出: 跨尺度/同尺度 = {norms['J_32_64']/norms['J_32_32']:.4f}")

    print("\n" + "="*60)
    print("实验3: 耦合强度随SNR变化")
    print("="*60)

    coupling_results = analyze_coupling_vs_snr(model, scheduler, dataset, device, num_snr_points=12)

    # 绘图
    import os
    os.makedirs('results', exist_ok=True)
    plot_coupling_analysis(coupling_results, 'results/cross_scale_coupling_analysis.png')

    # 保存结果
    torch.save(coupling_results, 'results/cross_scale_coupling_analysis.pth')
    print("Results saved to: results/cross_scale_coupling_analysis.pth")

    print("\n" + "="*60)
    print("结论")
    print("="*60)

    avg_coupling_64 = np.mean(coupling_results['coupling_ratio_64'])
    avg_coupling_32 = np.mean(coupling_results['coupling_ratio_32'])

    print(f"\n平均跨尺度耦合比例:")
    print(f"  64×64输出: {avg_coupling_64:.4f} (即跨尺度Jacobian是同尺度的{avg_coupling_64*100:.1f}%)")
    print(f"  32×32输出: {avg_coupling_32:.4f} (即跨尺度Jacobian是同尺度的{avg_coupling_32*100:.1f}%)")

    if avg_coupling_64 < 0.1:
        print(f"\n→ 64×64输出几乎不依赖32×32输入 (耦合<10%)")
        print(f"  这解释了为什么64×64误差与γ₁独立")

    if avg_coupling_32 > 0.5:
        print(f"\n→ 32×32输出显著依赖64×64输入 (耦合>{avg_coupling_32*100:.0f}%)")
        print(f"  这是合理的：低分辨率需要参考高分辨率信息")


if __name__ == "__main__":
    main()
