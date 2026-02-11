"""Train a 2-scale SLOT model on CIFAR-10 (orig + 2x only).

Usage:
    python train_slot_2d.py --epochs 2000 --device 0
    python train_slot_2d.py --epochs 2000 --device 0 --resume checkpoints/slot_2scale_400ep.pth
"""
import argparse
import copy
import io
import math
import os
import pickle
import sys
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, "/home/ylong030/slot")
import dnnlib
from torch_utils import persistence

# ── Dataset ───────────────────────────────────────────────────────────────
class CIFAR10ZipDataset(Dataset):
    """Load CIFAR-10 from zip file."""
    def __init__(self, zip_path, resolution=32):
        from PIL import Image
        self.images = []
        with zipfile.ZipFile(zip_path) as z:
            names = sorted([n for n in z.namelist() if n.endswith('.png')])
            for name in names:
                with z.open(name) as f:
                    img = Image.open(io.BytesIO(f.read())).convert('RGB')
                    self.images.append(np.array(img))
        self.images = np.stack(self.images)  # [N, H, W, 3]
        self.resolution = resolution

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # [H, W, 3] uint8
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1  # [3, H, W] in [-1, 1]
        return img
# ── Loss function (2-scale version) ───────────────────────────────────────
class SLOTLoss2Scale:
    """2-scale torus EDM loss (orig + 2x only)."""
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, coupled=False):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.coupled = coupled

    def convert_torus(self, u):
        scales = u.chunk(2, dim=1)
        out = []
        for s in scales:
            s = (math.pi / 2) * torch.clamp(s, -1, 1)
            out.append(torch.cat([torch.cos(s), torch.sin(s)], dim=1))
        return torch.cat(out, dim=1)

    def build_multiscale_input(self, images):
        B, C, H, W = images.shape
        x0 = images
        x2 = F.interpolate(F.interpolate(images, scale_factor=0.5, mode='area'),
                           size=(H, W), mode='nearest')
        return torch.cat([x0, x2], dim=1)

    def add_noise(self, x_torus, sigmas_vec):
        B, C, H, W = x_torus.shape
        x_view = x_torus.view(B, 2, -1, H, W)
        sigmas_bc = sigmas_vec.view(B, 2, 1, 1, 1)
        snr = 1.0 / (sigmas_bc ** 2)
        alpha = torch.sqrt(snr / (snr + 1.0))
        beta = 1.0 / torch.sqrt(snr + 1.0)
        eps = torch.randn_like(x_view)
        z = (alpha * x_view + beta * eps).view(B, C, H, W)
        return z

    def __call__(self, net, images, augment_pipe=None):
        B = images.shape[0]
        device = images.device
        if self.coupled:
            rnd = torch.randn([B, 1, 1, 1], device=device)
            sigmas = (rnd * self.P_std + self.P_mean).exp().repeat(1, 2, 1, 1)
        else:
            rnd = torch.randn([B, 2, 1, 1], device=device)
            sigmas = (rnd * self.P_std + self.P_mean).exp()
        weights = (sigmas**2 + self.sigma_data**2) / (sigmas * self.sigma_data)**2
        y, aug_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        y_ms = self.build_multiscale_input(y)
        y_torus = self.convert_torus(y_ms)
        z_torus = self.add_noise(y_torus, sigmas)
        sigma_input = sigmas.view(B, 2)
        D_yn = net(z_torus, sigma_input, augment_labels=aug_labels)
        raw_loss = (D_yn - y_torus) ** 2
        raw_view = raw_loss.view(B, 2, -1, images.shape[2], images.shape[3])
        weighted = raw_view * weights.unsqueeze(2)
        return weighted.mean(dim=[1, 2, 3, 4])

# ── Augmentation ──────────────────────────────────────────────────────────
class SimpleAugment(nn.Module):
    def __init__(self, p=0.12):
        super().__init__()
        self.p = p

    def forward(self, images):
        B = images.shape[0]
        if torch.rand(1).item() < self.p:
            images = torch.flip(images, dims=[3])
        aug_labels = torch.zeros(B, 9, device=images.device)
        return images, aug_labels
# ── Training loop ─────────────────────────────────────────────────────────
def train(args):
    device = torch.device(f'cuda:{args.device}')
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("Loading CIFAR-10...")
    dataset = CIFAR10ZipDataset("/home/ylong030/slot/datasets/cifar10-32x32.zip")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)
    print(f"Dataset: {len(dataset)} images, {len(loader)} batches/epoch")

    from training.networks import EDMPrecondSlot
    net = EDMPrecondSlot(
        img_resolution=32,
        img_channels=12,  # 2 scales × 6 (cos/sin per RGB)
        label_dim=0,
        sigma_data=0.5,
        num_scales=2,
        model_type='SongUNet',
        model_channels=128,
        channel_mult=[2, 2, 2],
        augment_dim=9,
        dropout=0.13,
        embedding_type='positional',
        encoder_type='standard',
        decoder_type='standard',
        channel_mult_noise=1,
        resample_filter=[1, 1],
    ).to(device)
    net.torus = True
    net.original_img_channels = 3

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Model params: {total_params/1e6:.1f}M")

    ema = copy.deepcopy(net).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999))
    loss_fn = SLOTLoss2Scale(P_mean=-1.2, P_std=1.2, sigma_data=0.5, coupled=False)
    augment_pipe = SimpleAugment(p=0.12).to(device)

    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        net.load_state_dict(ckpt['net'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed at epoch {start_epoch}")

    ema_halflife_kimg = 500
    ema_rampup_ratio = 0.05
    images_per_epoch = len(dataset)

    print(f"\nTraining for {args.epochs} epochs (from epoch {start_epoch})...")
    for epoch in range(start_epoch, args.epochs):
        net.train()
        epoch_loss = 0
        num_batches = 0
        cur_nimg = epoch * images_per_epoch

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for batch in pbar:
            images = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            lr = args.lr * min(cur_nimg / max(10000 * 1000, 1e-8), 1)
            for g in optimizer.param_groups:
                g['lr'] = lr
            loss = loss_fn(net, images, augment_pipe=augment_pipe)
            loss.mean().backward()
            for p in net.parameters():
                if p.grad is not None:
                    torch.nan_to_num(p.grad, nan=0, posinf=1e5, neginf=-1e5, out=p.grad)
            optimizer.step()
            ema_halflife_nimg = ema_halflife_kimg * 1000
            if ema_rampup_ratio is not None:
                ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
            ema_beta = 0.5 ** (args.batch_size / max(ema_halflife_nimg, 1e-8))
            for p_ema, p_net in zip(ema.parameters(), net.parameters()):
                p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
            cur_nimg += args.batch_size
            epoch_loss += loss.mean().item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.mean().item():.4f}", lr=f"{lr:.2e}")

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{args.epochs} — loss: {avg_loss:.4f}, lr: {lr:.2e}")

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.checkpoint_dir, f"slot_2scale_{epoch+1}ep.pth")
            torch.save({
                'epoch': epoch,
                'net': net.state_dict(),
                'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_path)
            ema_path = os.path.join(args.checkpoint_dir, f"slot_2scale_ema_{epoch+1}ep.pkl")
            ema_copy = copy.deepcopy(ema)
            with open(ema_path, 'wb') as f:
                pickle.dump({'ema': ema_copy}, f)
            print(f"  Saved: {ckpt_path} + {ema_path}")

    print("Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--save_every', type=int, default=200)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str,
                        default='/home/ylong030/slot/simple_diffusion_clean/explore_test_6/checkpoints')
    args = parser.parse_args()
    train(args)
