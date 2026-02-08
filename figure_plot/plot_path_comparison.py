#!/usr/bin/env python3
"""
Plot path comparison across epochs using pre-computed paths from heatmap .pth files.
Ensures consistency with individual heatmap plots.

Usage:
    python figure_plot/plot_path_comparison.py                    # non-EMA
    python figure_plot/plot_path_comparison.py --ema              # EMA
    python figure_plot/plot_path_comparison.py --ema --epochs 400 800 1200 2000
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--epochs', type=int, nargs='+',
                        default=[400, 800, 1200, 1600, 2000])
    parser.add_argument('--results_dir', type=str, default='results')
    args = parser.parse_args()

    prefix = 'heatmap_30_ema_' if args.ema else 'heatmap_30_'
    tag = 'EMA ' if args.ema else ''

    # Load all data
    data_list = []
    valid_epochs = []
    for epoch in args.epochs:
        path = os.path.join(args.results_dir, f'{prefix}{epoch}ep.pth')
        if not os.path.exists(path):
            print(f"Missing: {path}, skipping")
            continue
        data = torch.load(path, map_location='cpu', weights_only=False)
        data_list.append(data)
        valid_epochs.append(epoch)
        print(f"Loaded {tag}{epoch}ep")

    if not data_list:
        print("No data found!")
        return

    n_epochs = len(valid_epochs)
    fig, axes = plt.subplots(2, n_epochs, figsize=(5 * n_epochs, 10))
    if n_epochs == 1:
        axes = axes.reshape(2, 1)

    for col, (epoch, data) in enumerate(zip(valid_epochs, data_list)):
        t_grid = data['t_grid']
        t_np = t_grid.numpy() if torch.is_tensor(t_grid) else t_grid
        extent = [t_np[0], t_np[-1], t_np[0], t_np[-1]]

        error_64 = data['error_64']
        error_total = data['error_total']
        samples_64 = data['samples_64']
        samples_total = data['samples_total']

        rows = [
            (error_64, samples_64, '64×64 Error', 'DP-64'),
            (error_total, samples_total, 'Total Error', 'DP-Total'),
        ]

        for row_idx, (error, samples, err_label, path_label) in enumerate(rows):
            ax = axes[row_idx, col]
            error_np = error.numpy() if torch.is_tensor(error) else error

            im = ax.imshow(np.log10(error_np + 1e-10).T, origin='lower',
                           extent=extent, aspect='auto', cmap='viridis')

            # Diagonal reference
            ax.plot([t_np[0], t_np[-1]], [t_np[0], t_np[-1]],
                    'w--', linewidth=1.5, alpha=0.5)

            # DP path (red line + yellow points) — directly from .pth
            s_t64 = [t_np[p[0]] for p in samples]
            s_t32 = [t_np[p[1]] for p in samples]
            ax.plot(s_t64, s_t32, 'r-', linewidth=2, alpha=0.8)
            ax.scatter(s_t64, s_t32, c='yellow', s=50, zorder=5,
                       edgecolors='black', linewidths=1)

            if row_idx == 0:
                ax.set_title(f'{tag}{epoch}ep', fontsize=13, fontweight='bold')
            ax.set_xlabel('t_64', fontsize=11)
            if col == 0:
                ax.set_ylabel(f'{err_label}\nt_32', fontsize=12, fontweight='bold')
            else:
                ax.set_yticklabels([])

    # Colorbars
    fig.colorbar(axes[0, -1].images[0], ax=axes[0, :], location='right',
                 label='log₁₀(64×64 Error)', pad=0.02, aspect=30)
    fig.colorbar(axes[1, -1].images[0], ax=axes[1, :], location='right',
                 label='log₁₀(Total Error)', pad=0.02, aspect=30)

    plt.tight_layout(rect=[0, 0, 0.97, 0.96])

    suffix = 'ema_' if args.ema else ''
    output = os.path.join(args.results_dir, f'path_comparison_{suffix}across_epochs.png')
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


if __name__ == '__main__':
    main()
