"""Phase 3: 3D DP path optimization for SLOT model.

Finds optimal 18-step path from (0,0,0) to (G-1,G-1,G-1) in 3D sigma space.
Uses the 3D error heatmap from compute_heatmap_3d.py.

Cost per step = Σ_s (e_s[cur] + e_s[next])/2 × |Δλ_s|
where λ = logSNR and s ∈ {orig, 2x, 4x}.
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm


def find_optimal_path_3d(error_orig, error_2x, error_4x, log_snr,
                         num_steps=18, max_jump=3, max_diag_dist=None):
    """Find optimal 3D path using DP.

    Args:
        error_orig: [G, G, G] error for orig scale
        error_2x: [G, G, G] error for 2x scale
        error_4x: [G, G, G] error for 4x scale
        log_snr: [G] logSNR values
        num_steps: number of steps
        max_jump: max grid cells per step per dimension
        max_diag_dist: if set, max L-inf distance from diagonal (i==j==k)

    Returns:
        path: list of (i, j, k) tuples, length num_steps+1
        cost: total cost
    """
    eo = error_orig.numpy() if torch.is_tensor(error_orig) else error_orig
    e2 = error_2x.numpy() if torch.is_tensor(error_2x) else error_2x
    e4 = error_4x.numpy() if torch.is_tensor(error_4x) else error_4x
    lsnr = log_snr.numpy() if torch.is_tensor(log_snr) else log_snr
    G = eo.shape[0]

    INF = float('inf')
    # dp[i, j, k, step] = min cost to reach (i,j,k) in `step` steps
    dp = np.full((G, G, G, num_steps + 1), INF, dtype=np.float64)
    # parent stores (pi, pj, pk) for backtracking
    parent = np.full((G, G, G, num_steps + 1, 3), -1, dtype=np.int16)

    dp[0, 0, 0, 0] = 0.0

    for step in tqdm(range(num_steps), desc="3D DP"):
        remaining = num_steps - step - 1
        for i in range(G):
            for j in range(G):
                for k in range(G):
                    if dp[i, j, k, step] == INF:
                        continue
                    for ni in range(i, min(i + max_jump + 1, G)):
                        for nj in range(j, min(j + max_jump + 1, G)):
                            for nk in range(k, min(k + max_jump + 1, G)):
                                if ni == i and nj == j and nk == k:
                                    continue

                                # Diagonal proximity constraint
                                if max_diag_dist is not None:
                                    if (abs(ni - nj) > max_diag_dist or
                                        abs(ni - nk) > max_diag_dist or
                                        abs(nj - nk) > max_diag_dist):
                                        continue

                                # Reachability check
                                dist = (G-1-ni) + (G-1-nj) + (G-1-nk)
                                if dist > remaining * 3 * max_jump:
                                    continue
                                if remaining == 0:
                                    if ni != G-1 or nj != G-1 or nk != G-1:
                                        continue

                                # Cost: trapezoidal integral in λ-space
                                dl_o = abs(lsnr[ni] - lsnr[i])
                                dl_2 = abs(lsnr[nj] - lsnr[j])
                                dl_4 = abs(lsnr[nk] - lsnr[k])
                                co = (eo[i,j,k] + eo[ni,nj,nk]) / 2 * dl_o
                                c2 = (e2[i,j,k] + e2[ni,nj,nk]) / 2 * dl_2
                                c4 = (e4[i,j,k] + e4[ni,nj,nk]) / 2 * dl_4
                                new_cost = dp[i,j,k,step] + co + c2 + c4

                                if new_cost < dp[ni,nj,nk,step+1]:
                                    dp[ni,nj,nk,step+1] = new_cost
                                    parent[ni,nj,nk,step+1] = [i, j, k]

    end = G - 1
    if dp[end, end, end, num_steps] == INF:
        print(f"Warning: Cannot reach ({end},{end},{end}) in {num_steps} steps "
              f"with max_jump={max_jump}")
        if max_jump < G - 1:
            print("Retrying with larger max_jump...")
            return find_optimal_path_3d(
                error_orig, error_2x, error_4x, log_snr,
                num_steps, max_jump + 1, max_diag_dist)
        return None, INF

    # Backtrack
    path = []
    ci, cj, ck, cs = end, end, end, num_steps
    while cs >= 0:
        path.append((ci, cj, ck))
        if cs == 0:
            break
        pi, pj, pk = parent[ci, cj, ck, cs]
        ci, cj, ck, cs = int(pi), int(pj), int(pk), cs - 1
    path = path[::-1]

    return path, dp[end, end, end, num_steps]

def path_to_sigma_schedule(path, sigma_grid, device='cpu'):
    """Convert 3D grid path to sigma schedule [num_steps, 3]."""
    sg = sigma_grid.numpy() if torch.is_tensor(sigma_grid) else sigma_grid
    num_steps = len(path) - 1
    sigmas = torch.zeros(num_steps, 3, device=device, dtype=torch.float32)
    for step in range(num_steps):
        i, j, k = path[step + 1]
        sigmas[step, 0] = float(sg[i])
        sigmas[step, 1] = float(sg[j])
        sigmas[step, 2] = float(sg[k])
    return sigmas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--heatmap', default='results/heatmap_3d_10.pth')
    parser.add_argument('--num_steps', type=int, default=18)
    parser.add_argument('--max_jump', type=int, default=3)
    parser.add_argument('--max_diag_dist', type=int, default=None,
                        help='Max L-inf distance from diagonal (None=unconstrained)')
    parser.add_argument('--output_dir', default='results')
    # FID evaluation
    parser.add_argument('--eval', action='store_true', help='Evaluate path FID')
    parser.add_argument('--model_path', default='/home/ylong030/slot/network-snapshot-052685.pkl')
    parser.add_argument('--num_images', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load heatmap
    print(f"Loading heatmap: {args.heatmap}")
    data = torch.load(args.heatmap, map_location='cpu', weights_only=False)
    error_orig = data['error_orig']
    error_2x = data['error_2x']
    error_4x = data['error_4x']
    log_snr = data['log_snr']
    sigma_grid = data['sigma_grid']
    G = data['grid_size']
    print(f"Grid: {G}×{G}×{G}, logSNR: [{log_snr[0]:.2f}, {log_snr[-1]:.2f}]")

    # Run DP variants
    configs = [
        ("unconstrained", None),
        ("diag_2", 2),
        ("diag_3", 3),
        ("diag_4", 4),
    ]

    results = {}
    for name, diag_dist in configs:
        print(f"\n{'='*50}")
        print(f"DP path: {name} (max_jump={args.max_jump}, "
              f"max_diag_dist={diag_dist})")
        print(f"{'='*50}")

        path, cost = find_optimal_path_3d(
            error_orig, error_2x, error_4x, log_snr,
            num_steps=args.num_steps, max_jump=args.max_jump,
            max_diag_dist=diag_dist)

        if path is None:
            print("  FAILED — no valid path found")
            continue

        print(f"  Cost: {cost:.6e}")
        print(f"  Path: {path[0]} -> {path[-1]}")

        # Print steps
        for s in range(len(path) - 1):
            i0, j0, k0 = path[s]
            i1, j1, k1 = path[s + 1]
            print(f"    step {s:2d}: ({i0},{j0},{k0}) -> ({i1},{j1},{k1})  "
                  f"Δ=({i1-i0},{j1-j0},{k1-k0})")

        sigmas = path_to_sigma_schedule(path, sigma_grid)
        results[name] = {'path': path, 'cost': cost, 'sigmas': sigmas}

    # Save all paths
    save_path = os.path.join(args.output_dir, 'dp_3d_paths.pth')
    torch.save(results, save_path)
    print(f"\nSaved paths: {save_path}")

    # Evaluate FID if requested
    if args.eval:
        import pickle
        sys.path.insert(0, "/home/ylong030/slot")
        import dnnlib
        from generate_and_fid import (
            convert_from_torus, edm_sampler_torus_heun,
            load_cifar_real_features, compute_fid,
        )

        device = torch.device(f'cuda:{args.device}')
        print(f"\nLoading model for FID evaluation...")
        with dnnlib.util.open_url(args.model_path) as f:
            model_data = pickle.load(f)
        net = model_data['ema'].eval().to(device)

        cache_path = os.path.join(args.output_dir, 'cifar10_real_features.npz')
        real_mu, real_sigma = load_cifar_real_features(cache_path, device)

        csv_path = os.path.join(args.output_dir, 'dp_3d_fid.csv')
        with open(csv_path, 'w') as f:
            f.write('path_type,num_images,fid,cost\n')

        for name, res in results.items():
            sigmas = res['sigmas'].to(device)
            print(f"\nEvaluating {name}...")
            num_batches = (args.num_images + args.batch_size - 1) // args.batch_size
            all_imgs = []
            for bi in tqdm(range(num_batches), desc=f"  {name}"):
                bs = min(args.batch_size, args.num_images - bi * args.batch_size)
                rng = torch.Generator(device=device).manual_seed(args.seed + bi)
                latents = torch.randn(bs, 18, 32, 32, generator=rng, device=device)
                cos_p, sin_p = latents.chunk(2, dim=1)
                norm = torch.sqrt(cos_p**2 + sin_p**2 + 1e-8)
                latents = torch.cat([cos_p/norm, sin_p/norm], dim=1)
                x_out = edm_sampler_torus_heun(net, latents, sigmas)
                imgs = convert_from_torus(x_out)
                all_imgs.append(imgs[:, :3].cpu())
            all_imgs = torch.cat(all_imgs, dim=0)[:args.num_images]
            batches = [all_imgs[i:i+args.batch_size]
                      for i in range(0, len(all_imgs), args.batch_size)]
            fid = compute_fid(batches, real_mu, real_sigma, device)
            print(f"  {name}: FID = {fid:.4f}")
            with open(csv_path, 'a') as f:
                f.write(f'{name},{args.num_images},{fid:.4f},{res["cost"]:.6e}\n')

        print(f"\nFID results saved: {csv_path}")


if __name__ == '__main__':
    main()
