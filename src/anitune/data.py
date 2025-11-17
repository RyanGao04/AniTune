from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@dataclass
class DataConfig:
    root: Path
    img_size: int = 224
    train_split: float = 0.9
    batch_size: int = 32
    num_workers: int = 4
    use_grayscale: bool = False


def build_transforms(img_size: int, use_grayscale: bool = False):
    channels = 1 if use_grayscale else 3
    normalize = transforms.Normalize(
        mean=[0.5] * channels,
        std=[0.5] * channels,
    )
    train_tfms = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.1)),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.1)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tfms, eval_tfms


def build_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader]:
    train_tfms, eval_tfms = build_transforms(cfg.img_size, cfg.use_grayscale)

    dataset = datasets.ImageFolder(cfg.root, transform=train_tfms)
    n_train = int(len(dataset) * cfg.train_split)
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    # Apply eval transforms to val subset
    val_ds.dataset.transform = eval_tfms

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
