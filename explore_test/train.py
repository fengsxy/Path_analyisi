#!/usr/bin/env python3
"""
训练脚本：支持两种实验架构

架构 A (single_t): 单 timestep，32×32 noisy image（噪声水平未知）
架构 B (no_t): 无 timestep，32×32 noisy image（噪声水平未知）

Usage:
    python train.py --model single_t --epochs 10 --device 0
    python train.py --model no_t --epochs 10 --device 1
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# 引用外部文件
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scheduler import DDIMScheduler
from data import AFHQ64Dataset

from model_single_t import SingleTimestepModel, count_parameters
from model_no_t import NoTimestepModel


def parse_args():
    parser = argparse.ArgumentParser(description='Train explore models')
    parser.add_argument('--model', type=str, required=True, choices=['single_t', 'no_t'],
                        help='Model architecture: single_t or no_t')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--hidden_dims', type=str, default='64,128,256,512',
                        help='Hidden dimensions (comma-separated)')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--save_every', type=int, default=None,
                        help='Save checkpoint every N epochs')
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


def train_one_epoch_single_t(model, dataloader, optimizer, noise_schedule, device, epoch):
    """Train single_t model for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch in pbar:
        images_64 = batch.to(device)
        batch_size = images_64.shape[0]

        # Downsample to 32×32
        images_32 = F.interpolate(images_64, size=(32, 32), mode='bilinear', align_corners=False)

        # Sample timesteps
        t_64 = torch.randint(0, 1000, (batch_size,), device=device).long()
        t_32 = torch.randint(0, 1000, (batch_size,), device=device).long()  # 随机，但不传给模型

        # Sample noise
        noise_64 = torch.randn_like(images_64)
        noise_32 = torch.randn_like(images_32)

        # Add noise
        noisy_64 = noise_schedule.add_noise(images_64, noise_64, t_64)
        noisy_32 = noise_schedule.add_noise(images_32, noise_32, t_32)  # 噪声水平未知

        # Forward pass - 只传 t_64，不传 t_32
        optimizer.zero_grad()
        noise_pred_64 = model(noisy_64, noisy_32, t_64)
        loss = F.mse_loss(noise_pred_64, noise_64)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / num_batches


def train_one_epoch_no_t(model, dataloader, optimizer, noise_schedule, device, epoch):
    """Train no_t model for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch in pbar:
        images_64 = batch.to(device)
        batch_size = images_64.shape[0]

        # Downsample to 32×32
        images_32 = F.interpolate(images_64, size=(32, 32), mode='bilinear', align_corners=False)

        # Sample timesteps (用于加噪，但不传给模型)
        t_64 = torch.randint(0, 1000, (batch_size,), device=device).long()
        t_32 = torch.randint(0, 1000, (batch_size,), device=device).long()

        # Sample noise
        noise_64 = torch.randn_like(images_64)
        noise_32 = torch.randn_like(images_32)

        # Add noise
        noisy_64 = noise_schedule.add_noise(images_64, noise_64, t_64)
        noisy_32 = noise_schedule.add_noise(images_32, noise_32, t_32)

        # Forward pass - 不传任何 t
        optimizer.zero_grad()
        noise_pred_64 = model(noisy_64, noisy_32)
        loss = F.mse_loss(noise_pred_64, noise_64)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / num_batches


def main():
    args = parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model: {args.model}")

    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    print(f"Hidden dims: {hidden_dims}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print("Loading AFHQ64 dataset...")
    dataset = AFHQ64Dataset(split='train')
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
    if args.model == 'single_t':
        model = SingleTimestepModel(hidden_dims=hidden_dims)
        train_fn = train_one_epoch_single_t
    else:
        model = NoTimestepModel(hidden_dims=hidden_dims)
        train_fn = train_one_epoch_no_t

    model = model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Noise schedule
    ddim_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="cosine")
    noise_schedule = NoiseSchedule(ddim_scheduler)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    start_time = time.time()

    losses = []
    for epoch in range(args.epochs):
        epoch_loss = train_fn(model, dataloader, optimizer, noise_schedule, device, epoch)
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_loss:.4f}")

        if args.save_every is not None and (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f'{args.model}_{epoch+1}ep.pth')
            torch.save({
                'model_state': model.state_dict(),
                'hidden_dims': hidden_dims,
                'losses': losses,
                'epochs': epoch + 1,
                'model_type': args.model
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Save final model
    save_path = os.path.join(args.output_dir, f'{args.model}_final.pth')
    torch.save({
        'model_state': model.state_dict(),
        'hidden_dims': hidden_dims,
        'losses': losses,
        'epochs': args.epochs,
        'model_type': args.model
    }, save_path)
    print(f"\nModel saved to: {save_path}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    print(f"Final loss: {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
