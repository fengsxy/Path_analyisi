"""Evaluate top path candidates with more images + extended lead-step search.

Phase 1: Quick scan of lead_6..lead_10 with 1000 images to find optimal lead
Phase 2: Run top-5 paths with 10000 images for reliable FID
"""
import argparse
import os
import sys
import pickle

import torch
from tqdm import tqdm

sys.path.insert(0, "/home/ylong030/slot")
import dnnlib

from generate_and_fid import (
    convert_torus, convert_from_torus,
    _rho_schedule, _rho_interp,
    edm_sampler_torus_heun,
    load_cifar_real_features, compute_fid,
)
from explore_paths import build_lead_follow_schedule


def load_model(path, device):
    with dnnlib.util.open_url(path) as f:
        data = pickle.load(f)
    return data['ema'].eval().to(device)


@torch.no_grad()
def generate_images(net, sigmas, num_images, batch_size, seed, device):
    """Generate images and return as list of batches."""
    num_batches = (num_images + batch_size - 1) // batch_size
    all_images = []
    for i in tqdm(range(num_batches), desc="Generating", leave=False):
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
    """Generate + compute FID."""
    imgs = generate_images(net, sigmas, num_images, batch_size, seed, device)
    batches = [imgs[i:i+batch_size] for i in range(0, len(imgs), batch_size)]
    return compute_fid(batches, real_mu, real_sigma, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/home/ylong030/slot/network-snapshot-052685.pkl')
    parser.add_argument('--num_images_scan', type=int, default=1000, help='Images for quick scan')
    parser.add_argument('--num_images_final', type=int, default=10000, help='Images for final eval')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--skip_scan', action='store_true', help='Skip phase 1 scan')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}')
    os.makedirs(args.output_dir, exist_ok=True)
    net = load_model(args.model_path, device)

    cache_path = os.path.join(args.output_dir, 'cifar10_real_features.npz')
    real_mu, real_sigma = load_cifar_real_features(cache_path, device)

    all_results = []

    # Phase 1: Extended lead-step scan (1000 images)
    if not args.skip_scan:
        print("=" * 60)
        print(f"Phase 1: Extended lead-step scan ({args.num_images_scan} images)")
        print("=" * 60)
        for lead in range(0, 13):
            name = f"lead_{lead}" if lead > 0 else "sync"
            sigmas = build_lead_follow_schedule(18, lead, device=device)
            fid = eval_fid(net, sigmas, args.num_images_scan, args.batch_size,
                          args.seed, real_mu, real_sigma, device)
            print(f"  {name:12s}: FID = {fid:.2f}")
            all_results.append((name, args.num_images_scan, fid))

        # Save scan results
        scan_csv = os.path.join(args.output_dir, 'lead_scan.csv')
        with open(scan_csv, 'w') as f:
            f.write('path_type,num_images,fid\n')
            for name, n, fid in all_results:
                f.write(f'{name},{n},{fid:.4f}\n')
        print(f"\nScan results saved to {scan_csv}")

        # Find optimal lead
        best_name, _, best_fid = min(all_results, key=lambda x: x[2])
        print(f"\nBest scan result: {best_name} (FID={best_fid:.2f})")

    # Phase 2: Final eval with more images
    print("\n" + "=" * 60)
    print(f"Phase 2: Final evaluation ({args.num_images_final} images)")
    print("=" * 60)

    final_paths = [
        ("sync", 0),
        ("lead_3", 3),
        ("lead_4", 4),
        ("lead_5", 5),
        ("lead_6", 6),
        ("lead_7", 7),
        ("lead_8", 8),
    ]

    final_results = []
    for name, lead in final_paths:
        sigmas = build_lead_follow_schedule(18, lead, device=device)
        fid = eval_fid(net, sigmas, args.num_images_final, args.batch_size,
                      args.seed, real_mu, real_sigma, device)
        print(f"  {name:12s}: FID = {fid:.2f}")
        final_results.append((name, args.num_images_final, fid))

    # Save final results
    final_csv = os.path.join(args.output_dir, 'final_path_eval.csv')
    with open(final_csv, 'w') as f:
        f.write('path_type,num_images,fid\n')
        for name, n, fid in final_results:
            f.write(f'{name},{n},{fid:.4f}\n')
    print(f"\nFinal results saved to {final_csv}")

    best_name, _, best_fid = min(final_results, key=lambda x: x[2])
    print(f"\nBest final result: {best_name} (FID={best_fid:.2f})")


if __name__ == '__main__':
    main()
