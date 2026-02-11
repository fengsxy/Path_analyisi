"""Phase 1: Independent lead search for 3 scales.

Decouples 2x and 4x lead steps to find optimal (lead_2x, lead_4x) combination.
Currently both are tied together — this explores whether 4x should lead even
further ahead than 2x, or if they should stay coupled.

Grid search over (lead_2x, lead_4x) ∈ [0, max_lead] × [0, max_lead].
"""
import argparse
import os
import sys
import pickle

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, "/home/ylong030/slot")
import dnnlib

from generate_and_fid import (
    convert_torus, convert_from_torus,
    _rho_schedule,
    edm_sampler_torus_heun,
    load_cifar_real_features, compute_fid,
)


def load_model(path, device):
    with dnnlib.util.open_url(path) as f:
        data = pickle.load(f)
    return data['ema'].eval().to(device)


def build_3d_lead_schedule(num_steps, lead_2x, lead_4x,
                           sigma_min=0.002, sigma_max=80.0, rho=7.0, device='cpu'):
    """Build sigma schedule with independent leads for 2x and 4x.

    Args:
        lead_2x: how many steps 2x leads ahead of orig
        lead_4x: how many steps 4x leads ahead of orig
    """
    max_lead = max(lead_2x, lead_4x)
    total = num_steps + max_lead
    full_sched = _rho_schedule(total, sigma_max, sigma_min, rho, device)

    sigmas = torch.zeros(num_steps, 3, device=device, dtype=torch.float32)
    for k in range(num_steps):
        sigmas[k, 0] = full_sched[k]
        sigmas[k, 1] = full_sched[min(k + lead_2x, total - 1)]
        sigmas[k, 2] = full_sched[min(k + lead_4x, total - 1)]
    return sigmas


@torch.no_grad()
def generate_images(net, sigmas, num_images, batch_size, seed, device):
    num_batches = (num_images + batch_size - 1) // batch_size
    all_images = []
    for i in range(num_batches):
        bs = min(batch_size, num_images - i * batch_size)
        rng = torch.Generator(device=device).manual_seed(seed + i)
        latents = torch.randn(bs, 18, 32, 32, generator=rng, device=device)
        cos_part, sin_part = latents.chunk(2, dim=1)
        norm = torch.sqrt(cos_part ** 2 + sin_part ** 2 + 1e-8)
        latents = torch.cat([cos_part / norm, sin_part / norm], dim=1)
        x_out = edm_sampler_torus_heun(net, latents, sigmas)
        images = convert_from_torus(x_out)
        all_images.append(images[:, :3].cpu())
    return torch.cat(all_images, dim=0)[:num_images]

def eval_fid(net, sigmas, num_images, batch_size, seed, real_mu, real_sigma, device):
    imgs = generate_images(net, sigmas, num_images, batch_size, seed, device)
    batches = [imgs[i:i+batch_size] for i in range(0, len(imgs), batch_size)]
    return compute_fid(batches, real_mu, real_sigma, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/home/ylong030/slot/network-snapshot-052685.pkl')
    parser.add_argument('--num_images', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--max_lead', type=int, default=10,
                        help='Max lead steps to search')
    parser.add_argument('--num_steps', type=int, default=18)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}')
    os.makedirs(args.output_dir, exist_ok=True)
    net = load_model(args.model_path, device)

    cache_path = os.path.join(args.output_dir, 'cifar10_real_features.npz')
    real_mu, real_sigma = load_cifar_real_features(cache_path, device)

    max_lead = args.max_lead
    results = np.full((max_lead + 1, max_lead + 1), np.nan)

    csv_path = os.path.join(args.output_dir, 'lead_3d_search.csv')
    with open(csv_path, 'w') as f:
        f.write('lead_2x,lead_4x,fid\n')

    total = (max_lead + 1) ** 2
    print(f"Grid search: (lead_2x, lead_4x) in [0, {max_lead}] x [0, {max_lead}]")
    print(f"Total: {total} evaluations, {args.num_images} images each")
    print()

    pbar = tqdm(total=total, desc="3D lead search")
    for l2 in range(max_lead + 1):
        for l4 in range(max_lead + 1):
            sigmas = build_3d_lead_schedule(
                args.num_steps, l2, l4, device=device)
            fid = eval_fid(net, sigmas, args.num_images, args.batch_size,
                          args.seed, real_mu, real_sigma, device)
            results[l2, l4] = fid
            with open(csv_path, 'a') as f:
                f.write(f'{l2},{l4},{fid:.4f}\n')
            pbar.set_postfix(lead_2x=l2, lead_4x=l4, fid=f'{fid:.2f}')
            pbar.update(1)
    pbar.close()

    # Save results matrix
    np.save(os.path.join(args.output_dir, 'lead_3d_matrix.npy'), results)

    # Print heatmap
    print("\n" + "=" * 60)
    print("FID matrix (rows=lead_2x, cols=lead_4x)")
    print("=" * 60)
    header = "      " + "".join(f"  4x={j:<5d}" for j in range(max_lead + 1))
    print(header)
    for i in range(max_lead + 1):
        row = f"2x={i:<2d} "
        for j in range(max_lead + 1):
            row += f"  {results[i, j]:7.2f}"
        print(row)

    # Find best
    best_idx = np.unravel_index(np.nanargmin(results), results.shape)
    print(f"\nBest: lead_2x={best_idx[0]}, lead_4x={best_idx[1]}, "
          f"FID={results[best_idx]:.4f}")

    # Compare with coupled lead (diagonal)
    print("\nCoupled (lead_2x == lead_4x):")
    for l in range(max_lead + 1):
        print(f"  lead={l}: FID={results[l, l]:.2f}")

    # Plot heatmap
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(results, origin='lower', cmap='viridis_r',
                       extent=[-0.5, max_lead+0.5, -0.5, max_lead+0.5])
        ax.set_xlabel('lead_4x', fontsize=12)
        ax.set_ylabel('lead_2x', fontsize=12)
        ax.set_title(f'FID vs (lead_2x, lead_4x) — {args.num_images} images', fontsize=13)
        plt.colorbar(im, ax=ax, label='FID')
        # Mark best
        ax.plot(best_idx[1], best_idx[0], 'r*', markersize=15, label=f'Best: ({best_idx[0]},{best_idx[1]})')
        # Mark diagonal
        diag = list(range(max_lead + 1))
        ax.plot(diag, diag, 'w--', linewidth=1.5, alpha=0.5, label='Coupled (2x=4x)')
        ax.legend(loc='upper right')
        ax.set_xticks(range(max_lead + 1))
        ax.set_yticks(range(max_lead + 1))
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'lead_3d_heatmap.png'), dpi=150)
        print(f"\nSaved heatmap: {args.output_dir}/lead_3d_heatmap.png")
        plt.close()
    except Exception as e:
        print(f"Plot failed: {e}")


if __name__ == '__main__':
    main()
