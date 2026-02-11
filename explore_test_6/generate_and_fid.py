"""Generate CIFAR images with SLOT model and compute FID.

Supports multiple sigma schedule modes:
  sync, coarse_to_fine, multi_offset — from notebook
  orig, total — DP optimal paths from heatmap.pth
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
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/ylong030/slot")
import dnnlib

# ── Torus helpers ──────────────────────────────────────────────────────────
def convert_torus(u):
    scales = u.chunk(3, dim=1)
    out = []
    for s in scales:
        s = (math.pi / 2) * torch.clamp(s, -1, 1)
        out.append(torch.cat([torch.cos(s), torch.sin(s)], dim=1))
    return torch.cat(out, dim=1)

def convert_from_torus(t):
    chunks = t.chunk(3, dim=1)
    out = []
    for c in chunks:
        cos_part, sin_part = c.chunk(2, dim=1)
        angle = torch.atan2(sin_part, cos_part)
        out.append(angle / (math.pi / 2))
    return torch.cat(out, dim=1)

# ── Sigma schedule builders ───────────────────────────────────────────────
def _rho_schedule(n, sigma_max, sigma_min, rho, device):
    if n <= 1:
        return torch.full((n,), sigma_min, device=device, dtype=torch.float32)
    idx = torch.arange(n, dtype=torch.float64, device=device)
    s = (sigma_max ** (1/rho) + idx/(n-1) * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
    return s.to(torch.float32)

def _rho_interp(u, sigma_max, sigma_min, rho):
    return (sigma_max ** (1/rho) + u * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho

def build_sigma_schedule(num_steps, mode, sigma_min=0.002, sigma_max=80.0, rho=7.0, device='cpu'):
    sigmas = torch.zeros(num_steps, 3, device=device, dtype=torch.float32)
    def _sched(n):
        return _rho_schedule(n, sigma_max, sigma_min, rho, device)

    if mode == "sync":
        t = _sched(num_steps)
        sigmas[:] = t.unsqueeze(1).expand(-1, 3)
    elif mode == "coarse_to_fine":
        n4 = max(1, num_steps // 3)
        n2 = max(1, (num_steps - n4) // 2)
        n0 = num_steps - n4 - n2
        sigmas[:n4, 2] = _sched(n4); sigmas[:n4, 1] = sigma_max; sigmas[:n4, 0] = sigma_max
        sigmas[n4:n4+n2, 2] = sigma_min; sigmas[n4:n4+n2, 1] = _sched(n2); sigmas[n4:n4+n2, 0] = sigma_max
        sigmas[n4+n2:, 2] = sigma_min; sigmas[n4+n2:, 1] = sigma_min; sigmas[n4+n2:, 0] = _sched(n0)
    elif mode == "multi_offset":
        for k in range(num_steps):
            u4 = k / max(1, num_steps - 1)
            sigmas[k, 2] = _rho_interp(u4, sigma_max, sigma_min, rho)
            start2 = int(num_steps * 0.15)
            if k >= start2:
                u2 = (k - start2) / max(1, num_steps - 1 - start2)
                sigmas[k, 1] = _rho_interp(u2, sigma_max, sigma_min, rho)
            else:
                sigmas[k, 1] = sigma_max
            start0 = int(num_steps * 0.35)
            if k >= start0:
                u0 = (k - start0) / max(1, num_steps - 1 - start0)
                sigmas[k, 0] = _rho_interp(u0, sigma_max, sigma_min, rho)
            else:
                sigmas[k, 0] = sigma_max
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return sigmas

def build_dp_schedule(heatmap_path, path_type, device='cpu'):
    """Build sigma schedule from DP path in heatmap.pth."""
    data = torch.load(heatmap_path, map_location='cpu', weights_only=False)
    sigma_grid = data['sigma_grid'].numpy()
    if path_type == 'orig':
        path = data['path_orig']
    elif path_type == 'total':
        path = data['path_total']
    else:
        raise ValueError(f"Unknown path_type: {path_type}")
    # path is list of (i_orig, i_2x) grid indices
    # Skip the first point (both at sigma_max) — it's the starting point
    # The schedule has len(path)-1 steps (transitions between path points)
    num_steps = len(path) - 1
    sigmas = torch.zeros(num_steps, 3, device=device, dtype=torch.float32)
    for step in range(num_steps):
        i_orig, i_2x = path[step + 1]
        s_orig = float(sigma_grid[i_orig])
        s_2x = float(sigma_grid[i_2x])
        # Scale 4x follows 2x (same sigma)
        sigmas[step, 0] = s_orig
        sigmas[step, 1] = s_2x
        sigmas[step, 2] = s_2x
    return sigmas

# ── Heun sampler (from notebook) ──────────────────────────────────────────
def edm_sampler_torus_heun(net, latents, sigmas_schedule, class_labels=None,
                           S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):
    device_local = latents.device
    num_scales = getattr(net, "num_scales", 3)
    cps = getattr(net, "channels_per_scale", latents.shape[1] // num_scales)
    sigmas_schedule = sigmas_schedule.to(device_local)
    num_steps = sigmas_schedule.shape[0]

    if hasattr(net, 'round_sigma'):
        sigmas_schedule = net.round_sigma(sigmas_schedule)

    t_steps = torch.cat([sigmas_schedule, torch.zeros_like(sigmas_schedule[:1])], dim=0)

    def _expand(sigma_vec, ref):
        # sigma_vec: [1, num_scales] or [num_scales] -> expand to [B, C, H, W]
        if sigma_vec.dim() == 1:
            sigma_vec = sigma_vec.unsqueeze(0)
        sigma_vec = sigma_vec.expand(ref.shape[0], -1)  # [B, num_scales]
        sv = sigma_vec.view(ref.shape[0], num_scales, 1, 1, 1)
        sv = sv.expand(ref.shape[0], num_scales, cps, ref.shape[2], ref.shape[3])
        return sv.reshape(ref.shape[0], num_scales * cps, ref.shape[2], ref.shape[3])

    x_next = latents.to(torch.float64) * 1.4

    for i in range(num_steps):
        t_cur = t_steps[i]; t_nxt = t_steps[i+1]
        sc = _expand(t_cur.view(1,-1), x_next)
        x_scaled = x_next * torch.sqrt(1 + sc**2)

        t_scalar = float(torch.max(t_cur).item())
        gamma = min(S_churn/num_steps, math.sqrt(2)-1) if S_min <= t_scalar <= S_max else 0.0
        t_hat = t_cur * (1 + gamma)
        if hasattr(net, 'round_sigma'):
            t_hat = net.round_sigma(t_hat)

        if gamma > 0:
            sh = _expand(t_hat.view(1,-1), x_scaled)
            ns = torch.sqrt(torch.clamp(sh**2 - sc**2, min=0))
            x_scaled_hat = x_scaled + ns * S_noise * torch.randn_like(x_scaled)
        else:
            sh = sc; x_scaled_hat = x_scaled

        x_hat = x_next if gamma == 0 else x_scaled_hat / torch.sqrt(1 + sh**2)
        sb = t_hat.view(1,-1).expand(x_hat.shape[0], -1)
        denoised = net(x_hat, sb, class_labels).to(torch.float64)
        d_cur = torch.where(sh > 0, (x_scaled_hat - denoised) / sh, torch.zeros_like(x_scaled_hat))

        sn = _expand(t_nxt.view(1,-1), x_scaled_hat)
        x_scaled_next = x_scaled_hat + (sn - sh) * d_cur

        if i < num_steps - 1:
            snb = t_nxt.view(1,-1).expand(x_hat.shape[0], -1)
            x_next_pred = torch.where(sn > 0, x_scaled_next / torch.sqrt(1 + sn**2), x_scaled_next)
            den2 = net(x_next_pred, snb, class_labels).to(torch.float64)
            d_prime = torch.where(sn > 0, (x_scaled_next - den2) / sn, torch.zeros_like(x_scaled_next))
            x_scaled_next = x_scaled_hat + (sn - sh) * (0.5 * d_cur + 0.5 * d_prime)

        x_next = torch.where(sn > 0, x_scaled_next / torch.sqrt(1 + sn**2), x_scaled_next)

    return x_next.to(torch.float32)

# ── Generation + FID ──────────────────────────────────────────────────────
@torch.no_grad()
def generate_batch(net, batch_size, sigmas_schedule, seed, device):
    """Generate a batch of CIFAR images using the SLOT model."""
    rng = torch.Generator(device=device).manual_seed(seed)
    C = net.img_channels  # 18
    H = W = net.img_resolution  # 32
    # Generate latents and normalize to torus manifold (cos, sin pairs with unit norm)
    latents = torch.randn(batch_size, C, H, W, generator=rng, device=device)
    cos_part, sin_part = latents.chunk(2, dim=1)
    norm = torch.sqrt(cos_part ** 2 + sin_part ** 2 + 1e-8)
    latents = torch.cat([cos_part / norm, sin_part / norm], dim=1)
    x_out = edm_sampler_torus_heun(net, latents, sigmas_schedule)
    # Convert from torus to data space
    images = convert_from_torus(x_out)  # [B, 9, H, W]
    # Extract orig scale (first 3 channels)
    return images[:, :3]  # [B, 3, 32, 32]

def load_cifar_real_features(cache_path, device):
    """Load or compute real CIFAR-10 features for FID."""
    if os.path.exists(cache_path):
        print(f"Real features already cached: {cache_path}")
        data = np.load(cache_path)
        return data['mu'], data['sigma']

    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import calculate_activation_statistics

    print("Computing real CIFAR-10 features...")
    dataset_zip = "/home/ylong030/slot/datasets/cifar10-32x32.zip"
    images = []
    with zipfile.ZipFile(dataset_zip) as z:
        names = sorted([n for n in z.namelist() if n.endswith('.png')])
        for name in tqdm(names, desc="Loading CIFAR"):
            with z.open(name) as f:
                img = Image.open(io.BytesIO(f.read())).convert('RGB')
                images.append(np.array(img))
    images = np.stack(images)  # [N, 32, 32, 3]

    # Compute inception features
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device).eval()

    # Convert to torch: [N, 3, 32, 32] float [0,1]
    imgs_t = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
    # Resize to 299x299 for inception
    imgs_t = F.interpolate(imgs_t, size=(299, 299), mode='bilinear', align_corners=False)

    acts = []
    bs = 64
    for i in tqdm(range(0, len(imgs_t), bs), desc="Inception features"):
        batch = imgs_t[i:i+bs].to(device)
        pred = model(batch)[0]
        acts.append(pred.squeeze(-1).squeeze(-1).cpu().numpy())
    acts = np.concatenate(acts, axis=0)
    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, mu=mu, sigma=sigma)
    print(f"Saved real features to {cache_path}")
    return mu, sigma

def compute_fid(gen_images_list, real_mu, real_sigma, device):
    """Compute FID between generated images and real statistics."""
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import calculate_frechet_distance

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device).eval()

    # gen_images_list: list of [B, 3, 32, 32] tensors in [-1, 1]
    acts = []
    for batch in tqdm(gen_images_list, desc="FID features"):
        imgs = (batch.clamp(-1, 1) + 1) / 2  # to [0, 1]
        imgs = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
        pred = model(imgs.to(device))[0]
        acts.append(pred.squeeze(-1).squeeze(-1).cpu().numpy())
    acts = np.concatenate(acts, axis=0)
    gen_mu = np.mean(acts, axis=0)
    gen_sigma = np.cov(acts, rowvar=False)
    fid = calculate_frechet_distance(real_mu, real_sigma, gen_mu, gen_sigma)
    return fid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_images', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_steps', type=int, default=18)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--path_type', type=str, default='sync',
                        choices=['sync', 'coarse_to_fine', 'multi_offset', 'orig', 'total'])
    parser.add_argument('--model_path', type=str,
                        default='/home/ylong030/slot/network-snapshot-052685.pkl')
    parser.add_argument('--heatmap_path', type=str,
                        default='/home/ylong030/slot/simple_diffusion_clean/explore_test_6/results/heatmap.pth')
    parser.add_argument('--output_dir', type=str,
                        default='/home/ylong030/slot/simple_diffusion_clean/explore_test_6/results')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    with dnnlib.util.open_url(args.model_path) as f:
        data = pickle.load(f)
    net = data['ema'].eval().to(device)

    # Build sigma schedule
    if args.path_type in ('orig', 'total'):
        sigmas = build_dp_schedule(args.heatmap_path, args.path_type, device=device)
        num_steps = sigmas.shape[0]
    else:
        num_steps = args.num_steps
        sigmas = build_sigma_schedule(num_steps, args.path_type, device=device)

    print(f"Path type: {args.path_type}")
    s_orig = [f"{s:.3f}" for s in sigmas[:, 0].cpu().tolist()]
    s_2x = [f"{s:.3f}" for s in sigmas[:, 1].cpu().tolist()]
    print(f"sigma_orig: {s_orig[:5]}...{s_orig[-3:]}" if len(s_orig) > 8 else f"sigma_orig: {s_orig}")
    print(f"sigma_2x:   {s_2x[:5]}...{s_2x[-3:]}" if len(s_2x) > 8 else f"sigma_2x:   {s_2x}")

    # Load real features
    cache_path = os.path.join(args.output_dir, 'cifar10_real_features.npz')
    real_mu, real_sigma = load_cifar_real_features(cache_path, device)

    # Generate images
    print(f"\nGenerating {args.num_images} images...")
    num_batches = (args.num_images + args.batch_size - 1) // args.batch_size
    all_images = []
    for i in tqdm(range(num_batches), desc="Generating"):
        bs = min(args.batch_size, args.num_images - i * args.batch_size)
        seed_i = args.seed + i
        imgs = generate_batch(net, bs, sigmas, seed_i, device)
        all_images.append(imgs.cpu())

    all_images_cat = torch.cat(all_images, dim=0)[:args.num_images]
    print(f"Generated {len(all_images_cat)} images, range [{all_images_cat.min():.2f}, {all_images_cat.max():.2f}]")

    # Compute FID
    print("\nComputing FID...")
    # Re-chunk for FID computation
    fid_batches = [all_images_cat[i:i+args.batch_size] for i in range(0, len(all_images_cat), args.batch_size)]
    fid = compute_fid(fid_batches, real_mu, real_sigma, device)
    print(f"\nFID ({args.path_type}, {args.num_images} images): {fid:.4f}")

    # Save results
    csv_path = os.path.join(args.output_dir, 'fid_results.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a') as f:
        if write_header:
            f.write('path_type,num_images,num_steps,seed,fid\n')
        f.write(f'{args.path_type},{args.num_images},{num_steps},{args.seed},{fid:.4f}\n')
    print(f"Appended to {csv_path}")

    # Save sample grid
    from torchvision.utils import save_image
    grid_path = os.path.join(args.output_dir, f'samples_{args.path_type}.png')
    save_image((all_images_cat[:64].clamp(-1,1)+1)/2, grid_path, nrow=8)
    print(f"Saved sample grid: {grid_path}")

if __name__ == '__main__':
    main()
