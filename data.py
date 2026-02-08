"""
Data utilities for Simple Diffusion Clean.

Provides dataset loading and image processing utilities.
"""

import math
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import datasets


def unnormalize_to_zero_to_one(t):
    """Convert from [-1, 1] to [0, 1]."""
    return (t + 1) * 0.5


def normalize_to_neg_one_to_one(t):
    """Convert from [0, 1] to [-1, 1]."""
    return t * 2 - 1


def to_image(tensor):
    """Convert tensor to numpy image for display."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    img = tensor.detach().cpu()
    img = unnormalize_to_zero_to_one(img)
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()
    return img


class AFHQ64Dataset(Dataset):
    """AFHQ 64x64 dataset from HuggingFace."""

    def __init__(self, split='train', cache_dir=None):
        self.dataset = datasets.load_dataset(
            "huggan/AFHQv2",
            split=split,
            cache_dir=cache_dir
        )
        self.transform = self._get_transform()

    def _get_transform(self):
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return self.transform(img)


class CelebAHQ64Dataset(Dataset):
    """CelebA-HQ 64x64 dataset from HuggingFace."""

    def __init__(self, split='train', cache_dir=None):
        self.dataset = datasets.load_dataset(
            "mattymchen/celeba-hq",
            split=split,
            cache_dir=cache_dir
        )
        self.transform = self._get_transform()

    def _get_transform(self):
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return self.transform(img)


class MNIST64Dataset(Dataset):
    """MNIST upscaled to 64x64."""

    def __init__(self, split='train', cache_dir=None):
        self.dataset = datasets.load_dataset(
            "mnist",
            split=split,
            cache_dir=cache_dir
        )
        self.transform = self._get_transform()

    def _get_transform(self):
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert to 3 channels
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]['image']
        return self.transform(img)
