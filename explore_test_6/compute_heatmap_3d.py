"""Phase 2: Compute 3D error heatmap for SLOT model.

Computes Hutchinson-estimated discretization error at each (σ_orig, σ_2x, σ_4x)
triplet on a coarse grid. This gives the error landscape for 3D DP optimization.

The model has 3 scales (orig 32×32, 2x 16×16, 4x 8×8), each with 6 torus channels.
We compute per-scale error using JVP through the full model.
"""
import argparse
import io
import math
import os
import sys
import pickle
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jvp, vmap
from tqdm import tqdm

sys.path.insert(0, "/home/ylong030/slot")
import dnnlib

from generate_and_fid import convert_torus


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
    images = np.stack(images)  # [N, 32, 32, 3]
    # to [N, 3, 32, 32] float [-1, 1]
    t = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    return t

def build_multiscale_torus(images):
    """Build 3-scale torus input from 32×32 RGB images.

    Returns [B, 18, 32, 32]: 3 scales × 6 torus channels.
    """
    B, C, H, W = images.shape  # [B, 3, 32, 32]
    # Scale 0: orig 32×32
    x_orig = images
    # Scale 1: 2x (16×16 → 32×32)
    x_2x_small = F.interpolate(images, scale_factor=0.5, mode='area')
    x_2x = F.interpolate(x_2x_small, size=(H, W), mode='nearest')
    # Scale 2: 4x (8×8 → 32×32)
    x_4x_small = F.interpolate(images, scale_factor=0.25, mode='area')
    x_4x = F.interpolate(x_4x_small, size=(H, W), mode='nearest')
    # Concatenate and convert to torus
    x_ms = torch.cat([x_orig, x_2x, x_4x], dim=1)  # [B, 9, 32, 32]
    return convert_torus(x_ms)  # [B, 18, 32, 32]


def add_noise_edm(x_torus, sigma_vec):
    """Add EDM-style noise with per-scale sigma.

    x_noisy = x + sigma * noise  (EDM convention)

    Args:
        x_torus: [B, 18, 32, 32]
        sigma_vec: [B, 3] per-scale sigmas
    """
    B, C, H, W = x_torus.shape
    num_scales = 3
    cps = C // num_scales  # 6
    sv = sigma_vec.view(B, num_scales, 1, 1, 1).expand(B, num_scales, cps, H, W)
    sv = sv.reshape(B, C, H, W)
    noise = torch.randn_like(x_torus)
    return x_torus + sv * noise


# ── Hutchinson estimator ─────────────────────────────────────────────────

def get_vHv_3scale(f, z, K=4):
    """Compute vHv per scale using Hutchinson estimator.

    Returns [3] tensor: vHv for orig, 2x, 4x.
    """
    z = z.detach()
    B, C, H, W = z.shape
    num_scales = 3
    cps = C // num_scales  # 6

    # Uniform variance normalized by spatial size
    v = torch.ones(1, C, H, W, device=z.device) / (H * W)
    scale = torch.sqrt(v)

    eps = torch.empty((K, *z.shape), device=z.device, dtype=z.dtype)
    eps.bernoulli_(0.5).mul_(2).add_(-1)
    u_batch = eps * scale

    def single_jvp(u):
        return jvp(f, (z,), (u,))[1]

    batched_out = vmap(single_jvp, chunk_size=None)(u_batch)
    # batched_out: [K, B, C, H, W]
    hv = (batched_out ** 2).mean(dim=0)  # [B, C, H, W]
    hv = hv * v.expand_as(hv)

    # Sum per scale
    hv_view = hv.view(B, num_scales, cps, H, W)
    per_scale = hv_view.sum(dim=(2, 3, 4))  # [B, 3]
    return per_scale.mean(dim=0)  # [3]


def chain_rule_factor_edm(sigma, sigma_data=0.5):
    """Chain-rule factor for EDM: convert from x-space to logSNR-space.

    SNR = sigma_data^2 / sigma^2
    factor = 1 / (SNR * (1 + SNR))
    """
    snr = sigma_data ** 2 / (sigma ** 2 + 1e-20)
    return 1.0 / (snr * (1.0 + snr))

def compute_3d_heatmap(net, x_torus, sigma_grid, device, K=4):
    """Compute 3D error heatmap.

    Args:
        net: SLOT model
        x_torus: [B, 18, 32, 32] clean torus data
        sigma_grid: [G] sigma values for the grid
        K: Hutchinson samples

    Returns:
        error_orig: [G, G, G]
        error_2x: [G, G, G]
        error_4x: [G, G, G]
    """
    G = len(sigma_grid)
    error_orig = torch.zeros(G, G, G, device=device)
    error_2x = torch.zeros(G, G, G, device=device)
    error_4x = torch.zeros(G, G, G, device=device)

    B = x_torus.shape[0]
    total = G ** 3
    pbar = tqdm(total=total, desc=f"Computing {G}³ heatmap")

    for i in range(G):
        s_orig = sigma_grid[i]
        gamma_orig = chain_rule_factor_edm(s_orig)
        for j in range(G):
            s_2x = sigma_grid[j]
            gamma_2x = chain_rule_factor_edm(s_2x)
            for k_idx in range(G):
                s_4x = sigma_grid[k_idx]
                gamma_4x = chain_rule_factor_edm(s_4x)

                sv = torch.tensor([[s_orig, s_2x, s_4x]], device=device)
                sv = sv.expand(B, 3)

                z = add_noise_edm(x_torus, sv)

                def f(z_in):
                    return net(z_in, sv, force_fp32=True)

                with torch.no_grad():
                    vhv = get_vHv_3scale(f, z, K=K)  # [3]

                error_orig[i, j, k_idx] = vhv[0] * gamma_orig
                error_2x[i, j, k_idx] = vhv[1] * gamma_2x
                error_4x[i, j, k_idx] = vhv[2] * gamma_4x

                pbar.update(1)

    pbar.close()
    return error_orig.cpu(), error_2x.cpu(), error_4x.cpu()


def build_sigma_grid(G, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    """Build sigma grid matching the rho schedule spacing."""
    idx = torch.arange(G, dtype=torch.float64)
    s = (sigma_max ** (1/rho) + idx/(G-1) * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
    return s.to(torch.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/home/ylong030/slot/network-snapshot-052685.pkl')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--grid_size', type=int, default=10,
                        help='Grid points per axis (10 = 1000 total)')
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
    print(f"Loading model...")
    with dnnlib.util.open_url(args.model_path) as f:
        data = pickle.load(f)
    net = data['ema'].eval().to(device)

    # Load data and build torus input
    print(f"Loading {args.batch_size} CIFAR images...")
    images = load_cifar_images(args.batch_size, seed=args.seed).to(device)
    x_torus = build_multiscale_torus(images)
    print(f"Torus input: {x_torus.shape}")

    # Build sigma grid
    G = args.grid_size
    sigma_grid = build_sigma_grid(G).to(device)
    sigma_data = 0.5
    snr_grid = sigma_data ** 2 / sigma_grid ** 2
    log_snr = torch.log(snr_grid)
    print(f"\nSigma grid ({G} points): [{sigma_grid[0]:.2f}, ..., {sigma_grid[-1]:.4f}]")
    print(f"logSNR range: [{log_snr[0]:.2f}, {log_snr[-1]:.2f}]")
    print(f"Total evaluations: {G**3}")

    # Compute 3D heatmap
    print(f"\nComputing 3D heatmap (K={args.K})...")
    error_orig, error_2x, error_4x = compute_3d_heatmap(
        net, x_torus, sigma_grid, device, K=args.K)

    error_total = error_orig + error_2x + error_4x

    # Save
    results = {
        'sigma_grid': sigma_grid.cpu(),
        'log_snr': log_snr.cpu(),
        'error_orig': error_orig,
        'error_2x': error_2x,
        'error_4x': error_4x,
        'error_total': error_total,
        'grid_size': G,
        'args': vars(args),
    }
    out_path = os.path.join(args.output_dir, f'heatmap_3d_{G}.pth')
    torch.save(results, out_path)
    print(f"\nSaved: {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Error ranges (with chain-rule factor)")
    print("=" * 60)
    for name, e in [('orig', error_orig), ('2x', error_2x), ('4x', error_4x)]:
        print(f"  {name:5s}: [{e.min():.4e}, {e.max():.4e}], "
              f"log10 range: [{np.log10(e.min().item()+1e-20):.1f}, "
              f"{np.log10(e.max().item()+1e-20):.1f}]")

    # Check separability: does error_orig depend mainly on sigma_orig?
    print("\n" + "=" * 60)
    print("Separability analysis (CV across other axes)")
    print("=" * 60)
    eo = error_orig.numpy()
    e2 = error_2x.numpy()
    e4 = error_4x.numpy()
    # error_orig: fix i (sigma_orig), compute CV across j,k
    cv_orig = []
    for i in range(G):
        vals = eo[i, :, :].flatten()
        if vals.mean() > 0:
            cv_orig.append(vals.std() / vals.mean())
    print(f"  error_orig: mean CV across (σ_2x, σ_4x) = {np.mean(cv_orig):.4f}")
    # error_2x: fix j, compute CV across i,k
    cv_2x = []
    for j in range(G):
        vals = e2[:, j, :].flatten()
        if vals.mean() > 0:
            cv_2x.append(vals.std() / vals.mean())
    print(f"  error_2x:   mean CV across (σ_orig, σ_4x) = {np.mean(cv_2x):.4f}")
    # error_4x: fix k, compute CV across i,j
    cv_4x = []
    for k in range(G):
        vals = e4[:, :, k].flatten()
        if vals.mean() > 0:
            cv_4x.append(vals.std() / vals.mean())
    print(f"  error_4x:   mean CV across (σ_orig, σ_2x) = {np.mean(cv_4x):.4f}")


if __name__ == '__main__':
    main()
