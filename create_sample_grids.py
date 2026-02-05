#!/usr/bin/env python3
"""
Create sample grids for visualization.

Usage:
    python create_sample_grids.py \
        --baseline_dir results/fid_baseline \
        --diagonal_dir results/fid_lift_diagonal \
        --dp_total_dir results/fid_lift_dp_total \
        --dp_64_dir results/fid_lift_dp_64 \
        --output_dir figures
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def create_grid(image_dir, num_images=16, grid_size=(4, 4)):
    """Create a grid of images from a directory."""
    images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])[:num_images]

    rows, cols = grid_size
    img_size = 64

    grid = np.zeros((rows * img_size, cols * img_size, 3), dtype=np.uint8)

    for i, img_name in enumerate(images):
        if i >= rows * cols:
            break
        img = Image.open(os.path.join(image_dir, img_name))
        img_np = np.array(img)

        row = i // cols
        col = i % cols
        grid[row*img_size:(row+1)*img_size, col*img_size:(col+1)*img_size] = img_np

    return grid


def parse_args():
    parser = argparse.ArgumentParser(description='Create sample grids')
    parser.add_argument('--baseline_dir', type=str, default='results/fid_baseline')
    parser.add_argument('--diagonal_dir', type=str, default='results/fid_lift_diagonal')
    parser.add_argument('--dp_total_dir', type=str, default='results/fid_lift_dp_total')
    parser.add_argument('--dp_64_dir', type=str, default='results/fid_lift_dp_64')
    parser.add_argument('--output_dir', type=str, default='figures')
    parser.add_argument('--num_images', type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Create individual grids
    dirs = {
        'baseline': args.baseline_dir,
        'diagonal': args.diagonal_dir,
        'dp_total': args.dp_total_dir,
        'dp_64': args.dp_64_dir,
    }

    grids = {}
    for name, dir_path in dirs.items():
        if os.path.exists(dir_path):
            print(f"Creating grid for {name}...")
            grid = create_grid(dir_path, args.num_images)
            grids[name] = grid

            # Save individual grid
            output_path = os.path.join(args.output_dir, f'generated_{name}.png')
            Image.fromarray(grid).save(output_path)
            print(f"  Saved to: {output_path}")

    # Create comparison figure
    if len(grids) >= 2:
        print("\nCreating comparison figure...")
        fig, axes = plt.subplots(1, len(grids), figsize=(5*len(grids), 5))

        titles = {
            'baseline': 'Baseline (Non-LIFT)',
            'diagonal': 'LIFT Diagonal',
            'dp_total': 'LIFT DP Total',
            'dp_64': 'LIFT DP 64×64',
        }

        for ax, (name, grid) in zip(axes, grids.items()):
            ax.imshow(grid)
            ax.set_title(titles.get(name, name))
            ax.axis('off')

        plt.tight_layout()
        comparison_path = os.path.join(args.output_dir, 'comparison_all.png')
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison to: {comparison_path}")


if __name__ == "__main__":
    main()
