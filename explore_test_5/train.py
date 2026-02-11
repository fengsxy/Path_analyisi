#!/usr/bin/env python3
"""
Training Script for SingleTimestepModel with EMA — 3 Training Regimes

Regimes:
  same_t:     t_32 = t_64 (same timestep for both scales)
  dp_path:    t_32 = interp(DP-64 path, t_64) from LIFT EMA 400ep
  heuristic:  t_32 = int(t_64 * 0.8)

All use SingleTimestepModel (receives only t_64) with EMA (decay=0.9999).

Usage:
    python train.py --model same_t --epochs 2000 --batch_size 256 --device 0
    python train.py --model dp_path --epochs 2000 --batch_size 256 --device 1
    python train.py --model heuristic --epochs 2000 --batch_size 256 --device 2
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scheduler import DDIMScheduler
from data import AFHQ64Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'explore_test'))
from model_single_t import SingleTimestepModel, count_parameters


class EMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )
    def apply(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        return {'decay': self.decay, 'shadow': self.shadow}

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']

    class _AverageContext:
        def __init__(self, ema):
            self.ema = ema
        def __enter__(self):
            self.ema.apply()
            return self
        def __exit__(self, *args):
            self.ema.restore()

    def average_parameters(self):
        return self._AverageContext(self)


class NoiseSchedule:
    """Noise schedule for training."""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.alphas_cumprod = scheduler.alphas_cumprod

    def add_noise(self, x, noise, timesteps):
        alphas = self.alphas_cumprod[timesteps.cpu()].to(x.device)
        while len(alphas.shape) < len(x.shape):
            alphas = alphas.unsqueeze(-1)
        return alphas.sqrt() * x + (1 - alphas).sqrt() * noise


# DP-64 path from LIFT EMA 400ep (results/heatmap_30_ema_400ep.pth)
DP64_PATH_T64 = [999, 964, 930, 895, 861, 826, 792, 757, 688, 620, 516, 378, 206, 34, 0, 0, 0, 0, 0]
DP64_PATH_T32 = [999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 861, 688, 516, 344, 172, 0]

def build_dp_interpolator():
    """Build interpolation function: given t_64, return t_32 from DP-64 path."""
    t64 = np.array(DP64_PATH_T64[::-1], dtype=np.float64)  # ascending
    t32 = np.array(DP64_PATH_T32[::-1], dtype=np.float64)

    # Remove duplicate t_64=0 entries, keep only the one mapping to t_32=0
    mask = np.ones(len(t64), dtype=bool)
    seen_zero = False
    for i in range(len(t64)):
        if t64[i] == 0:
            if seen_zero:
                mask[i] = False
            seen_zero = True
    t64 = t64[mask]
    t32 = t32[mask]

    def interpolate(t_64_batch):
        """t_64_batch: tensor [B]. Returns t_32: tensor [B]."""
        t64_np = t_64_batch.cpu().numpy().astype(np.float64)
        t32_np = np.interp(t64_np, t64, t32)
        return torch.from_numpy(t32_np).long().clamp(0, 999).to(t_64_batch.device)

    return interpolate


dp_interpolate = build_dp_interpolator()


def train_one_epoch_same_t(model, dataloader, optimizer, noise_schedule, ema, device, epoch):
    """Train with t_32 = t_64."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch in pbar:
        images_64 = batch.to(device)
        batch_size = images_64.shape[0]
        images_32 = F.interpolate(images_64, size=(32, 32), mode='bilinear', align_corners=False)

        t_64 = torch.randint(0, 1000, (batch_size,), device=device)
        t_32 = t_64  # same timestep

        noise_64 = torch.randn_like(images_64)
        noise_32 = torch.randn_like(images_32)

        noisy_64 = noise_schedule.add_noise(images_64, noise_64, t_64)
        noisy_32 = noise_schedule.add_noise(images_32, noise_32, t_32)

        noise_pred_64 = model(noisy_64, noisy_32, t_64.float())
        loss = F.mse_loss(noise_pred_64, noise_64)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=loss.item())

    return total_loss / num_batches


def train_one_epoch_dp_path(model, dataloader, optimizer, noise_schedule, ema, device, epoch):
    """Train with t_32 = dp_path_interpolate(t_64)."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch in pbar:
        images_64 = batch.to(device)
        batch_size = images_64.shape[0]
        images_32 = F.interpolate(images_64, size=(32, 32), mode='bilinear', align_corners=False)

        t_64 = torch.randint(0, 1000, (batch_size,), device=device)
        t_32 = dp_interpolate(t_64)

        noise_64 = torch.randn_like(images_64)
        noise_32 = torch.randn_like(images_32)

        noisy_64 = noise_schedule.add_noise(images_64, noise_64, t_64)
        noisy_32 = noise_schedule.add_noise(images_32, noise_32, t_32)

        noise_pred_64 = model(noisy_64, noisy_32, t_64.float())
        loss = F.mse_loss(noise_pred_64, noise_64)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=loss.item())

    return total_loss / num_batches


def train_one_epoch_heuristic(model, dataloader, optimizer, noise_schedule, ema, device, epoch):
    """Train with t_32 = int(t_64 * 0.8)."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch in pbar:
        images_64 = batch.to(device)
        batch_size = images_64.shape[0]
        images_32 = F.interpolate(images_64, size=(32, 32), mode='bilinear', align_corners=False)

        t_64 = torch.randint(0, 1000, (batch_size,), device=device)
        t_32 = (t_64.float() * 0.8).long().clamp(0, 999)

        noise_64 = torch.randn_like(images_64)
        noise_32 = torch.randn_like(images_32)

        noisy_64 = noise_schedule.add_noise(images_64, noise_64, t_64)
        noisy_32 = noise_schedule.add_noise(images_32, noise_32, t_32)

        noise_pred_64 = model(noisy_64, noisy_32, t_64.float())
        loss = F.mse_loss(noise_pred_64, noise_64)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=loss.item())

    return total_loss / num_batches


def parse_args():
    parser = argparse.ArgumentParser(description='Train SingleTimestepModel with EMA — 3 regimes')
    parser.add_argument('--model', type=str, required=True, choices=['same_t', 'dp_path', 'heuristic'],
                        help='Training regime')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--hidden_dims', type=str, default='64,128,256,512')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--save_every', type=int, default=200)
    parser.add_argument('--cache_dir', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]

    print("=" * 50)
    print(f"Training: {args.model} (SingleTimestepModel + EMA)")
    print("=" * 50)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"EMA decay: {args.ema_decay}")
    print(f"Hidden dims: {hidden_dims}")
    print(f"Device: {device}")
    print("")

    model = SingleTimestepModel(hidden_dims=hidden_dims).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    ema = EMA(model, decay=args.ema_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="cosine", clip_sample=True)
    noise_schedule = NoiseSchedule(scheduler)

    dataset = AFHQ64Dataset(split='train', cache_dir=args.cache_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True)

    train_fns = {
        'same_t': train_one_epoch_same_t,
        'dp_path': train_one_epoch_dp_path,
        'heuristic': train_one_epoch_heuristic,
    }
    train_fn = train_fns[args.model]

    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()
    losses = []

    for epoch in range(args.epochs):
        epoch_loss = train_fn(model, dataloader, optimizer, noise_schedule, ema, device, epoch)
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_loss:.4f}")

        if args.save_every and (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f'{args.model}_ema_{epoch+1}ep.pth')
            torch.save({
                'model_state': model.state_dict(),
                'ema_state': ema.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'hidden_dims': hidden_dims,
                'losses': losses,
                'epochs': epoch + 1,
                'model_type': args.model,
                'ema_decay': args.ema_decay,
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    # Save final
    final_path = os.path.join(args.output_dir, f'{args.model}_ema_final.pth')
    torch.save({
        'model_state': model.state_dict(),
        'ema_state': ema.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'hidden_dims': hidden_dims,
        'losses': losses,
        'epochs': args.epochs,
        'model_type': args.model,
        'ema_decay': args.ema_decay,
    }, final_path)
    print(f"\nFinal model saved: {final_path}")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.1f} minutes")
    print(f"Final loss: {losses[-1]:.4f}")


if __name__ == "__main__":
    main()