#!/usr/bin/env python3
"""
Generate sample grids from 100 epoch models.
"""

import os
from PIL import Image
import numpy as np

def create_grid(image_dir, output_path, num_images=16, grid_size=(4, 4)):
    """Create a grid of sample images."""
    images = []
    for i in range(num_images):
        img_path = os.path.join(image_dir, f'{i:05d}.png')
        if os.path.exists(img_path):
            img = Image.open(img_path)
            images.append(img)
        else:
            print(f"Warning: {img_path} not found")
            break

    if len(images) < num_images:
        print(f"Only found {len(images)} images, adjusting grid...")
        num_images = len(images)
        grid_size = (4, len(images) // 4)

    # Get image size
    img_width, img_height = images[0].size

    # Create grid
    grid_width = grid_size[1] * img_width
    grid_height = grid_size[0] * img_height
    grid = Image.new('RGB', (grid_width, grid_height))

    for idx, img in enumerate(images):
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        x = col * img_width
        y = row * img_height
        grid.paste(img, (x, y))

    grid.save(output_path)
    print(f"✓ Saved: {output_path}")

def main():
    os.makedirs('figures', exist_ok=True)

    # Generate grids for 100 epoch models
    models = [
        ('results/fid_baseline_100ep', 'figures/generated_baseline_100ep.png', 'Baseline 100ep'),
        ('results/fid_lift_diagonal_100ep', 'figures/generated_diagonal_100ep.png', 'LIFT Diagonal 100ep'),
        ('results/fid_lift_dp64_100ep', 'figures/generated_dp64_100ep.png', 'LIFT DP-64 100ep'),
        ('results/fid_lift_dp_total_100ep', 'figures/generated_dp_total_100ep.png', 'LIFT DP-Total 100ep'),
    ]

    for image_dir, output_path, name in models:
        if os.path.exists(image_dir):
            print(f"\nGenerating {name}...")
            create_grid(image_dir, output_path)
        else:
            print(f"⚠️  {image_dir} not found, skipping...")

if __name__ == "__main__":
    main()
