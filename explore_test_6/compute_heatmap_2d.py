"""Compute 2D error heatmap for 2-scale SLOT model.

Computes Hutchinson-estimated discretization error at each (σ_orig, σ_2x)
pair on a G×G grid. Uses finite-difference JVP.

The model has 2 scales (orig 32×32, 2x 16×16), each with 6 torus channels.
"""
import argparse
import io
import math
import os
import pickle
import sys
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, "/home/ylong030/slot")
import dnnlib

from generate_and_fid_2d import convert_torus_2d


# ── Data loading ──────────────────────────────────────────────────────────

def load_cifar_images(num_images, seed=42):
    """Load CIFAR-10 images as [N, 3, 32, 32] in [-1, 1]."""
    from PIL import Image
    dataset_zip = "/home/ylong030/slot/datasets/cifar10-32x32.zip"
    rng = np.random.RandomState(seed)
    with zipfile.ZipFile(dataset_zip) as z:
        names = sorted([n for n in z.namelist() if n.endswith('.png')])
        indices = rng.choice(len(names), num_images, replace=False)
        images = []
        for idx in indices:
            with z.open(names[idx]) as f:
                img = Image.open(io.BytesIO(f.read())).convert('RGB')
                images.append(np.array(img))
    images = np.stack(images)
    t = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    return t
def build_multiscale_torus_2d(images):
    """Build 2-scale torus input from 32×32 RGB images.

    Returns [B, 12, 32, 32]: 2 scales × 6 torus channels.
    """
    B, C, H, W = images.shape
    x_orig = images
    x_2x_small = F.interpolate(images, scale_factor=0.5, mode='area')
    x_2x = F.interpolate(x_2x_small, size=(H, W), mode='nearest')
    x_ms = torch.cat([x_orig, x_2x], dim=1)
    return convert_torus_2d(x_ms)


def add_noise_edm_2d(x_torus, sigma_vec):
    """Add EDM-style noise: x_noisy = x + sigma * noise (2-scale)."""
    B, C, H, W = x_torus.shape
    num_scales = 2
    cps = C // num_scales
    sv = sigma_vec.view(B, num_scales, 1, 1, 1).expand(B, num_scales, cps, H, W)
    sv = sv.reshape(B, C, H, W)
    noise = torch.randn_like(x_torus)
    return x_torus + sv * noise


# ── Hutchinson estimator (finite-difference) ─────────────────────────────

def get_vHv_2scale_fd(f, z, K=4, eps=1e-3):
    """Compute vHv per scale using Hutchinson + finite differences.

    Returns [2] tensor: vHv for orig, 2x.
    """
    B, C, H, W = z.shape
    num_scales = 2
    cps = C // num_scales  # 6

    v = 1.0 / (H * W)
    scale = math.sqrt(v)

    per_scale_accum = torch.zeros(2, device=z.device)

    for _ in range(K):
        u = torch.empty_like(z).bernoulli_(0.5).mul_(2).add_(-1) * scale
        out_plus = f(z + eps * u)
        out_minus = f(z - eps * u)
        Ju = (out_plus - out_minus) / (2 * eps)
        Ju_sq = Ju ** 2 * v
        Ju_sq_view = Ju_sq.view(B, num_scales, cps, H, W)
        per_scale = Ju_sq_view.sum(dim=(2, 3, 4)).mean(dim=0)  # [2]
        per_scale_accum += per_scale

    return per_scale_accum / K


def chain_rule_factor_edm(sigma, sigma_data=0.5):
    snr = sigma_data ** 2 / (sigma ** 2 + 1e-20)
    return 1.0 / (snr * (1.0 + snr))
def compute_2d_heatmap(net, x_torus, sigma_grid, device, K=4):
    """Compute 2D error heatmap on G×G grid."""
    G = len(sigma_grid)
    error_orig = torch.zeros(G, G, device=device)
    error_2x = torch.zeros(G, G, device=device)

    B = x_torus.shape[0]
    pbar = tqdm(total=G ** 2, desc=f"Computing {G}² heatmap")

    for i in range(G):
        s_orig = sigma_grid[i]
        gamma_orig = chain_rule_factor_edm(s_orig)
        for j in range(G):
            s_2x = sigma_grid[j]
            gamma_2x = chain_rule_factor_edm(s_2x)

            sv = torch.tensor([[s_orig, s_2x]], device=device)
            sv = sv.expand(B, 2)
            z = add_noise_edm_2d(x_torus, sv)

            def f(z_in):
                return net(z_in, sv, force_fp32=True)

            with torch.no_grad():
                vhv = get_vHv_2scale_fd(f, z, K=K)

            error_orig[i, j] = vhv[0] * gamma_orig
            error_2x[i, j] = vhv[1] * gamma_2x
            pbar.update(1)

    pbar.close()
    return error_orig.cpu(), error_2x.cpu()


def build_sigma_grid(G, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    idx = torch.arange(G, dtype=torch.float64)
    s = (sigma_max ** (1/rho) + idx/(G-1) * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
    return s.to(torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None,
                        help='.pkl or .pth checkpoint for 2-scale model')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--grid_size', type=int, default=30,
                        help='Grid points per axis (30 = 900 total)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Images for Hutchinson estimation')
    parser.add_argument('--K', type=int, default=4, help='Hutchinson samples')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='results')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f'cuda:{args.device}')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    from generate_and_fid_2d import load_2scale_model
    print(f"Loading model...")
    net = load_2scale_model(args.model_path, device)

    # Load data and build torus input
    print(f"Loading {args.batch_size} CIFAR images...")
    images = load_cifar_images(args.batch_size, seed=args.seed).to(device)
    x_torus = build_multiscale_torus_2d(images)
    print(f"Torus input: {x_torus.shape}")

    # Build sigma grid
    G = args.grid_size
    sigma_grid = build_sigma_grid(G).to(device)
    sigma_data = 0.5
    snr_grid = sigma_data ** 2 / sigma_grid ** 2
    log_snr = torch.log(snr_grid)
    print(f"\nSigma grid ({G} points): [{sigma_grid[0]:.2f}, ..., {sigma_grid[-1]:.4f}]")
    print(f"logSNR range: [{log_snr[0]:.2f}, {log_snr[-1]:.2f}]")
    print(f"Total evaluations: {G**2}")

    # Compute 2D heatmap
    print(f"\nComputing 2D heatmap (K={args.K}, finite-diff)...")
    error_orig, error_2x = compute_2d_heatmap(
        net, x_torus, sigma_grid, device, K=args.K)
    error_total = error_orig + error_2x

    # Save
    results = {
        'sigma_grid': sigma_grid.cpu(),
        'log_snr': log_snr.cpu(),
        'error_orig': error_orig,
        'error_2x': error_2x,
        'error_total': error_total,
        'grid_size': G,
        'args': vars(args),
    }
    out_path = os.path.join(args.output_dir, f'heatmap_2d_{G}.pth')
    torch.save(results, out_path)
    print(f"\nSaved: {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Error ranges (with chain-rule factor)")
    print("=" * 60)
    for name, e in [('orig', error_orig), ('2x', error_2x)]:
        print(f"  {name:5s}: [{e.min():.4e}, {e.max():.4e}], "
              f"log10 range: [{np.log10(e.min().item()+1e-20):.1f}, "
              f"{np.log10(e.max().item()+1e-20):.1f}]")


if __name__ == '__main__':
    main()
