#!/usr/bin/env python3
"""
Prepare real images from AFHQ dataset for FID evaluation.

Usage:
    python prepare_fid_real.py --output_dir results/fid_real --num_images 1000
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from data import AFHQ64Dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare real images for FID')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_images', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)

    # Load dataset
    print("Loading AFHQ64 dataset...")
    dataset = AFHQ64Dataset(split='train')
    print(f"Dataset size: {len(dataset)}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Sample random indices
    num_images = min(args.num_images, len(dataset))
    indices = np.random.choice(len(dataset), num_images, replace=False)

    # Save images
    print(f"Saving {num_images} images to {args.output_dir}...")
    for i, idx in enumerate(tqdm(indices, desc="Saving")):
        img_tensor = dataset[idx]
        # Convert from [-1, 1] to [0, 255]
        img_np = ((img_tensor.numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        img_pil.save(os.path.join(args.output_dir, f'{i:05d}.png'))

    print(f"\nSaved {num_images} real images to {args.output_dir}")


if __name__ == "__main__":
    main()
