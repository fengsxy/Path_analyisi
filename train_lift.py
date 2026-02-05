#!/usr/bin/env python3
"""
Training Script for LIFT Dual Timestep Model on AFHQ64

- Dataset: AFHQ64 (64×64 RGB images + 32×32 downsampled)
- Model: LIFT Dual Timestep (independent timesteps for each scale)
- Training: 30 epochs with random (t_64, t_32) pairs

Usage:
    python train_lift.py --epochs 30 --batch_size 64 --device 0
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import time

from model import LIFTDualTimestepModel, count_parameters
from scheduler import DDIMScheduler
from data import AFHQ64Dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train LIFT Dual Timestep on AFHQ64')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--hidden_dims', type=str, default='64,128,256,512',
                        help='Hidden dimensions (comma-separated)')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='HuggingFace cache directory')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    return parser.parse_args()


class NoiseSchedule:
    """Noise schedule for training"""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.alphas_cumprod = scheduler.alphas_cumprod

    def add_noise(self, x, noise, timesteps):
        """Add noise to image at given timesteps"""
        alphas = self.alphas_cumprod[timesteps.cpu()].to(x.device)
        while len(alphas.shape) < len(x.shape):
            alphas = alphas.unsqueeze(-1)

        noisy = alphas.sqrt() * x + (1 - alphas).sqrt() * noise
        return noisy


def train_one_epoch(model, dataloader, optimizer, noise_schedule,
                    device, epoch, scaler=None):
    """Train for one epoch with random independent timesteps"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch in pbar:
        images_64 = batch.to(device)  # [B, 3, 64, 64]
        batch_size = images_64.shape[0]

        # Downsample to 32×32
        images_32 = F.interpolate(images_64, size=(32, 32), mode='bilinear', align_corners=False)

        # Sample INDEPENDENT random timesteps for each scale
        t_64 = torch.randint(0, 1000, (batch_size,), device=device).long()
        t_32 = torch.randint(0, 1000, (batch_size,), device=device).long()

        # Sample noise for each scale
        noise_64 = torch.randn_like(images_64)
        noise_32 = torch.randn_like(images_32)

        # Add noise at respective timesteps
        noisy_64 = noise_schedule.add_noise(images_64, noise_64, t_64)
        noisy_32 = noise_schedule.add_noise(images_32, noise_32, t_32)

        # Forward pass
        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                noise_pred_64, noise_pred_32 = model(noisy_64, noisy_32, t_64, t_32)
                loss_64 = F.mse_loss(noise_pred_64, noise_64)
                loss_32 = F.mse_loss(noise_pred_32, noise_32)
                loss = loss_64 + loss_32

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            noise_pred_64, noise_pred_32 = model(noisy_64, noisy_32, t_64, t_32)
            loss_64 = F.mse_loss(noise_pred_64, noise_64)
            loss_32 = F.mse_loss(noise_pred_32, noise_32)
            loss = loss_64 + loss_32

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'loss_64': f"{loss_64.item():.4f}",
            'loss_32': f"{loss_32.item():.4f}"
        })

    return total_loss / num_batches


def main():
    args = parse_args()

    # Setup
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Parse hidden dims
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    print(f"Hidden dims: {hidden_dims}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print("Loading AFHQ64 dataset...")
    dataset = AFHQ64Dataset(split='train', cache_dir=args.cache_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")

    # Create model
    model = LIFTDualTimestepModel(hidden_dims=hidden_dims)
    model = model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(dataloader), eta_min=1e-6
    )

    # Create noise schedule
    ddim_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="cosine")
    noise_schedule = NoiseSchedule(ddim_scheduler)

    # Mixed precision
    scaler = GradScaler() if args.fp16 else None

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    start_time = time.time()

    losses = []
    for epoch in range(args.epochs):
        epoch_loss = train_one_epoch(
            model, dataloader, optimizer, noise_schedule,
            device, epoch, scaler
        )
        losses.append(epoch_loss)
        lr_scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_loss:.4f}")

    # Save final model
    save_path = os.path.join(args.output_dir, 'lift_dual_timestep_final.pth')
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'hidden_dims': hidden_dims,
        'losses': losses,
        'epochs': args.epochs
    }, save_path)
    print(f"\nModel saved to: {save_path}")

    # Print summary
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    print(f"Final loss: {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
